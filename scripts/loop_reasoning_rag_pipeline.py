# -*- coding: utf-8 -*-
import os, csv, re, time, requests
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

from pythainlp.tokenize import word_tokenize
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

load_dotenv()
THAILLM_API_KEY = os.environ.get("THAILLM_API_KEY", "1p1dC5cHTY9G4myMrb48gixAdDcFA55L")

# === CONFIGURATION ===
N_QUESTIONS = 100
DATA_DIR = "./data"
KB_DIR = f"{DATA_DIR}/knowledge_base"

# === CHECKPOINT FILES ===
EMBEDDINGS_CACHE = "embeddings_cache_loop.npy"
BACKUP_SUBMISSION = "backup_submission_loop.csv"

# === 0. LLM UTILS ===
def ask_llm(messages, model="typhoon", max_retries=5, temperature=0.3):
    url = f"http://thaillm.or.th/api/{model}/v1/chat/completions"
    headers = {"Content-Type": "application/json", "apikey": THAILLM_API_KEY}
    payload = {
        "model": "/model",
        "messages": messages,
        "max_tokens": 1500,
        "temperature": temperature,
    }
    for attempt in range(max_retries):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=60)
            if resp.status_code == 429:
                time.sleep(min(2 ** attempt, 10))
                continue
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except requests.exceptions.RequestException:
            time.sleep(min(2 ** attempt, 10))
    return ""

def parse_answer(text):
    if not text: return 9
    clean = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    m = re.search(r"ANSWER:\s*(\d+)", clean)
    if m: return int(m.group(1))
    for d in re.findall(r"\b(\d{1,2})\b", clean):
        if 1 <= int(d) <= 10: return int(d)
    return 9

def parse_final(text):
    if not text: return 9
    clean = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    m = re.search(r"FINAL_ANSWER:\s*(\d+)", clean)
    if m: return int(m.group(1))
    for d in re.findall(r"\b(\d{1,2})\b", clean):
        if 1 <= int(d) <= 10: return int(d)
    return 9
    
# === 1. DATA PREPARATION ===
def get_or_build_chunks():
    kb_dir = Path(KB_DIR)
    documents = []
    if kb_dir.exists():
        for fp in sorted(kb_dir.rglob("*.md")):
            documents.append({
                "path": str(fp.relative_to(kb_dir)), 
                "text": fp.read_text(encoding="utf-8")
            })

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    raw_chunks = []
    for doc in documents:
        for text in text_splitter.split_text(doc["text"]):
            raw_chunks.append((doc, text))
            
    chunks = []
    print(f"Found {len(documents)} documents, split into {len(raw_chunks)} chunks total.")
    
    for doc, text in tqdm(raw_chunks, desc="Processing Raw Chunks"):
        chunks.append({"text": text, "source": doc["path"], "original_text": text})
            
    return chunks

# === 2. INDEXING ===
print("=== PHASE 1: PREPARATION ===")
chunks = get_or_build_chunks()

print("\nLoading BAAI/bge-m3 for Vector Search...")
embed_model = SentenceTransformer("BAAI/bge-m3")
chunk_texts = [c["text"] for c in chunks]

if os.path.exists(EMBEDDINGS_CACHE):
    print(f"Loading dense embeddings from {EMBEDDINGS_CACHE}...")
    chunk_embeddings = np.load(EMBEDDINGS_CACHE)
else:
    print("Computing dense embeddings and saving to checkpoint...")
    chunk_embeddings = embed_model.encode(chunk_texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
    np.save(EMBEDDINGS_CACHE, chunk_embeddings)

print("\nBuilding BM25 Index for Keyword Search...")
tokenized_chunks = [word_tokenize(c["text"], engine="newmm") for c in chunks]
bm25 = BM25Okapi(tokenized_chunks)

print("\nLoading Cross-Encoder for Reranking (BAAI)...")
cross_encoder = CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=512)


# === 3. INFERENCE PIPELINE ===
print("\n=== PHASE 2: LOOPING REASONING INFERENCE ===")
def rewrite_query(question):
    prompt = f"คำถามจากลูกค้า: '{question}'\n\nจงเขียนคำถามซ้ำ (Rewrite) ขยายความและแต่งประโยคให้ค้นหาข้อมูลร้านขายเครื่องใช้ไฟฟ้าง่ายขึ้น ตอบเฉพาะคำถามที่ถูกเกลาแล้ว ไม่ต้องมีคำอธิบายเพิ่ม"
    rewritten = ask_llm([{"role": "user", "content": prompt}], temperature=0.1)
    return rewritten if rewritten else question

def retrieve_top_k(query_rewritten, fetch_k=40):
    q_emb = embed_model.encode([query_rewritten], normalize_embeddings=True)
    dense_scores = np.dot(chunk_embeddings, q_emb.T).flatten()
    dense_idx = np.argsort(dense_scores)[::-1][:fetch_k]
    
    tokens = word_tokenize(query_rewritten, engine="newmm")
    bm25_scores = bm25.get_scores(tokens)
    bm25_idx = np.argsort(bm25_scores)[::-1][:fetch_k]
    
    rrf_k = 60
    rrf_scores = {}
    for rank, idx in enumerate(dense_idx, 1):
        rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (rrf_k + rank)
    for rank, idx in enumerate(bm25_idx, 1):
        rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (rrf_k + rank)
        
    merged_idx = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:fetch_k]
    return merged_idx

def rerank_to_top_10(query_rewritten, candidate_indices, top_k=10):
    pairs = [[query_rewritten, chunks[idx]["text"]] for idx in candidate_indices]
    cross_scores = cross_encoder.predict(pairs)
    top_indices = np.argsort(cross_scores)[::-1][:top_k]
    return [candidate_indices[i] for i in top_indices]

# === 4. EXECUTION LOOP ===
questions = []
try:
    with open(f"{DATA_DIR}/questions.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            choices = {str(i): row[f"choice_{i}"] for i in range(1, 11)}
            questions.append({"id": int(row["id"]), "question": row["question"], "choices": choices})
except FileNotFoundError:
    print("Warning: data/questions.csv not found. Skipping QA loop.")

results = []
done_qids = set()
if os.path.exists(BACKUP_SUBMISSION):
    print(f"Resuming generation from {BACKUP_SUBMISSION}...")
    with open(BACKUP_SUBMISSION, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if row:
                qid_val = int(row[0])
                results.append([qid_val, int(row[1])])
                done_qids.add(qid_val)
else:
    with open(BACKUP_SUBMISSION, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "answer"])

print(f"Running pipeline on {min(N_QUESTIONS, len(questions))} questions...")
for q in questions[:N_QUESTIONS]:
    q_id = q["id"]
    original_q = q["question"]
    
    if q_id in done_qids:
        continue
    
    # 1. RETRIEVAL
    rewritten_q = rewrite_query(original_q)
    top40_idx = retrieve_top_k(rewritten_q, fetch_k=40)
    final_top10_idx = rerank_to_top_10(rewritten_q, top40_idx, top_k=10)
    retrieved_chunks = [chunks[i] for i in final_top10_idx]
    
    context_text = "\n\n".join(f"--- ข้อมูล {i+1} ---\n{c['text']}" for i, c in enumerate(retrieved_chunks))
    choices_text = "\n".join(f"{k}. {v}" for k, v in q["choices"].items())
    
    # ==========================
    # 2. LOOPING REASONING
    # ==========================
    sys_prompt_agent = "คุณคือ AI ระดับหัวกะทิของร้านฟ้าใหม่ จงตอบคำถามอย่างมีเหตุผลอิงจากข้อมูลที่ให้ไว้เท่านั้น"
    
    # Initial Draft
    draft_prompt = (
        f"[ข้อมูลอ้างอิง]\n{context_text}\n\n"
        f"[คำถาม]\n{original_q}\n\n"
        f"[ตัวเลือก]\n{choices_text}\n\n"
        f"จงวิเคราะห์ข้อมูลอ้างอิงทีละบรรทัดเพื่อหาคำตอบที่ถูกต้องที่สุด\n"
        f"อธิบายเหตุผลสั้นๆ ว่าทำไมถึงเลือกข้อนี้ (หรือถ้าไม่มีข้อมูลให้ตอบ 9)\n"
        f"จบบรรทัดสุดท้ายล่างสุดด้วยคำว่า ANSWER: X (X คือตัวเลข 1-10)"
    )
    draft_raw = ask_llm([
        {"role": "system", "content": sys_prompt_agent},
        {"role": "user", "content": draft_prompt}
    ], temperature=0.3)
    current_ans = parse_answer(draft_raw)
    current_reasoning = draft_raw
    
    MAX_LOOPS = 3
    final_ans = current_ans
    loops_taken = 0
    
    for loop_num in range(1, MAX_LOOPS + 1):
        loops_taken = loop_num
        verify_prompt = (
            f"[ข้อมูลอ้างอิง]\n{context_text}\n\n"
            f"[คำถาม]\n{original_q}\n\n"
            f"[ตัวเลือก]\n{choices_text}\n\n"
            f"คำตอบที่พิจารณาอยู่คือ: ข้อ {current_ans}\n"
            f"เหตุผล: {current_reasoning}\n\n"
            f"จงตรวจสอบอย่างตบตาว่า 'คำตอบข้อ {current_ans}' ถูกต้อง 100% แน่นอนหรือไม่?\n"
            f"- หากประเมินแล้ว ถูกต้อง แน่นอน ให้พิมพ์คำว่า 'CORRECT' ในบรรทัดแรก และจบบรรทัดสุดท้ายด้วย FINAL_ANSWER: {current_ans}\n"
            f"- หากพบว่า ผิด หรือมีข้ออื่นที่ชัวร์กว่า ให้พิมพ์คำว่า 'WRONG' ในบรรทัดแรก อธิบายเหตุผลที่แก้ใหม่ และจบบรรทัดสุดท้ายด้วย FINAL_ANSWER: Y"
        )
        
        verify_raw = ask_llm([
            {"role": "system", "content": sys_prompt_agent},
            {"role": "user", "content": verify_prompt}
        ], temperature=0.1)
        
        final_ans = parse_final(verify_raw)
        
        if "CORRECT" in verify_raw.upper() or final_ans == current_ans:
            print(f"  Q{q_id:>3}: Found Answer={final_ans} (Approved in Loop {loop_num}) ✅")
            break
        else:
            # Re-assign and loop again
            current_ans = final_ans
            current_reasoning = verify_raw
            if loop_num == MAX_LOOPS:
                print(f"  Q{q_id:>3}: Found Answer={final_ans} (Forced stop after {MAX_LOOPS} Loops) ⚠️")
    
    results.append([q_id, final_ans])
    
    with open(BACKUP_SUBMISSION, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([q_id, final_ans])

with open("loop_reasoning_submission.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "answer"])
    writer.writerows(results)

print("LOOP REASONING pipeline complete! Saved to loop_reasoning_submission.csv")
