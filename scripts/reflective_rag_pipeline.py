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
EMBEDDINGS_CACHE = "embeddings_cache_reflective.npy"
BACKUP_SUBMISSION = "backup_submission_reflective.csv"

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

def parse_draft(text):
    if not text: return 9
    clean = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    m = re.search(r"DRAFT_ANSWER:\s*(\d+)", clean)
    if m: return int(m.group(1))
    for d in re.findall(r"\b(\d{1,2})\b", clean):
        if 1 <= int(d) <= 10: return int(d)
    return 9

def parse_final(text):
    if not text: return 9
    clean = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    m = re.search(r"FINAL_ANSWER:\s*(\d+)", clean)
    if m: return int(m.group(1))
    # Fallback to the first digit found if exact word not matched
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
print("\n=== PHASE 2: REFLECTIVE INFERENCE ===")
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

sys_prompt_draft = """คุณคือผู้ช่วย AI ของร้านขายเครื่องใช้ไฟฟ้า 'ฟ้าใหม่' (FahMai)
จงร่างคำตอบตามข้อมูลที่ได้รับ 

[กฎการตอบ]
- ยึดข้อมูลอ้างอิงเป็นหลัก ห้ามเดาเอง
- มีคำตอบในบรบท -> เลขข้อ 1-8
- หาไม่เจอ/ข้อมูลไม่พอ -> ข้อ 9
- ไม่เกี่ยวกับร้านหรือสินค้า -> ข้อ 10"""

sys_prompt_critic = """คุณคือ AI ผู้ตรวจสอบคำตอบ (Reviewer) ของร้านฟ้าใหม่
หน้าที่ของคุณคือประเมินว่า 'ร่างคำตอบ' ถูกต้องตามบริบทข้อมูลหรือไม่
- หากร่างคำตอบถูกต้องอยู่แล้ว ให้ยืนยันตัวเลขเดิม
- หากผิดพลาดหรือโดนหลอก ให้แก้ไขเป็นตัวเลขที่ถูกต้อง
- ตอบ 9 เฉพาะเมื่อไม่มีข้อมูลอ้างอิงในบริบทเลยจริงๆ เท่านั้น"""

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
    # 2. DRAFT ROUND (ACTOR)
    # ==========================
    draft_prompt = (
        f"[บริบทข้อมูล]\n{context_text}\n\n"
        f"[คำถามจริง]\n{original_q}\n\n"
        f"[ตัวเลือก]\n{choices_text}\n\n"
        f"จงวิเคราะห์หาคำตอบ อธิบายเหตุผลสั้นๆ 1-2 ประโยค\n"
        f"จากนั้นเขียนบรรทัดสุดท้ายสรุปเป็นคำว่า DRAFT_ANSWER: X (X คือรหัส 1-10)"
    )
    
    draft_raw = ask_llm([
        {"role": "system", "content": sys_prompt_draft},
        {"role": "user", "content": draft_prompt}
    ], temperature=0.3)
    
    draft_ans = parse_draft(draft_raw)
    
    # ==========================
    # 3. VERIFICATION ROUND (CRITIC)
    # ==========================
    critic_prompt = (
        f"[บริบทอ้างอิง]\n{context_text}\n\n"
        f"[คำถาม]\n{original_q}\n\n"
        f"[ตัวเลือก]\n{choices_text}\n\n"
        f"ร่างคำตอบเดิมคือ: ข้อ {draft_ans}\n"
        f"เหตุผลเดิม: {draft_raw}\n\n"
        f"หน้าที่ของคุณคือ 'ประเมินความถูกต้อง (Review)':\n"
        f"1. ข้อมูลในบริบทอ้างอิง สนับสนุนคำตอบข้อ {draft_ans} จริงหรือไม่?\n"
        f"2. หากถูกต้องชัดเจน ให้ยืนยันคำตอบเดิมเป็น FINAL_ANSWER: {draft_ans}\n"
        f"3. หากพบว่าร่างคำตอบตีความผิด ให้เปลี่ยนเป็นตัวเลขที่ถูกต้อง\n"
        f"สรุปเหตุผลสั้นๆ แล้วจบบรรทัดล่างสุดด้วย FINAL_ANSWER: Y (Y คือตัวเลข 1-10)"
    )
    
    final_raw = ask_llm([
        {"role": "system", "content": sys_prompt_critic},
        {"role": "user", "content": critic_prompt}
    ], temperature=0.1)
    
    final_ans = parse_final(final_raw)
    results.append([q_id, final_ans])
    
    # Print status indicating if it changed its mind
    changed = "🔄 CHANGED!" if final_ans != draft_ans else "✅ Verified"
    print(f"  Q{q_id:>3}: Draft={draft_ans} -> Final={final_ans} {changed}")
    
    with open(BACKUP_SUBMISSION, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([q_id, final_ans])

with open("reflective_submission.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "answer"])
    writer.writerows(results)

print("REFLECTIVE RAG pipeline complete! Saved to reflective_submission.csv")
