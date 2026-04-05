# -*- coding: utf-8 -*-
import os, csv, re, time, requests, json
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
USE_CONTEXTUAL_ENRICHMENT = False # Keeps pipeline fast 

# === CHECKPOINT FILES ===
CONTEXTUAL_CACHE = "contextual_chunks_checkpoint_improved.json"
EMBEDDINGS_CACHE = "embeddings_cache_improved.npy"
BACKUP_SUBMISSION = "backup_submission_improved.csv"

# === 0. LLM UTILS ===
def ask_llm(messages, model="typhoon", max_retries=5, temperature=0):
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
    return 9 # Fallback if not sure
    
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

    # [IMPROVEMENT 1: Increase Chunk Size from 400 to 800]
    # This prevents Thai spec tables from being split in half
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
print("\n=== PHASE 2: INFERENCE ===")
def rewrite_query(question):
    # Reduced temperature to 0.1 for more conservative, stable rewrites
    prompt = f"คำถามจากลูกค้า: '{question}'\n\nจงเขียนคำถามซ้ำ (Rewrite) ขยายความและแต่งประโยคให้ค้นหาข้อมูลร้านขายเครื่องใช้ไฟฟ้าง่ายขึ้น ตอบเฉพาะคำถามที่ถูกเกลาแล้ว ไม่ต้องมีคำอธิบายเพิ่ม"
    rewritten = ask_llm([{"role": "user", "content": prompt}], temperature=0.1)
    return rewritten if rewritten else question

def retrieve_top_k(query_rewritten, fetch_k=40):
    # [IMPROVEMENT 2: Increase fetch_k from 30 to 40]
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
    # [IMPROVEMENT 3: Increase LLM context window from 5 to 10]
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

sys_prompt = """คุณคือผู้ช่วย AI ของร้านขายเครื่องใช้ไฟฟ้า 'ฟ้าใหม่' (FahMai)
จงตอบคำถามโดยเลือกหมายเลขตัวเลือก (1-10) ที่ถูกต้องที่สุดเพียงหมายเลขเดียว โดยอิงจากบริบทที่ให้มาเท่านั้น

[กฎการตอบ]
- หากมีคำตอบในบริบท ให้เลือกหมายเลขของคำตอบนั้น (1-8)
- หากข้อมูลในบริบท ไม่เพียงพอ หรือ ไม่มีการบรรยายถึงเรื่องนี้ ให้ตอบ 9
- หากคำถาม ไม่เกี่ยวกับร้านฟ้าใหม่ หรือเครื่องใช้ไฟฟ้า ให้ตอบ 10
- ตอบจำกัดแค่ตัวเลข ไม่ต้องอธิบาย และนำหน้าด้วย ANSWER: X"""

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
    
    rewritten_q = rewrite_query(original_q)
    top40_idx = retrieve_top_k(rewritten_q, fetch_k=40)
    final_top10_idx = rerank_to_top_10(rewritten_q, top40_idx, top_k=10)
    retrieved_chunks = [chunks[i] for i in final_top10_idx]
    
    context_text = "\n\n".join(f"--- ข้อมูล {i+1} ---\n{c['text']}" for i, c in enumerate(retrieved_chunks))
    choices_text = "\n".join(f"{k}. {v}" for k, v in q["choices"].items())
    
    final_prompt = (
        f"[บริบทข้อมูล]\n{context_text}\n\n"
        f"[คำถามจริง]\n{original_q}\n\n"
        f"[ตัวเลือก]\n{choices_text}\n\n"
        f"ตอบ ANSWER: X (X คือรหัสตัวเลือก)"
    )
    
    raw = ask_llm([
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": final_prompt}
    ], temperature=0.1)
    
    ans = parse_answer(raw)
    results.append([q_id, ans])
    print(f"  Q{q_id:>3}: Rewrote=({rewritten_q[:30]}...) | Ans={ans} | Context={len(context_text)} chars")
    
    with open(BACKUP_SUBMISSION, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([q_id, ans])

with open("improved_advanced_submission.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "answer"])
    writer.writerows(results)

print("Improved Advanced pipeline complete! Saved to improved_advanced_submission.csv")
