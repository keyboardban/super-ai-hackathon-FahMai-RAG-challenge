# -*- coding: utf-8 -*-
import os, csv, re, time, requests, json
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

from pythainlp.tokenize import word_tokenize
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

load_dotenv()
THAILLM_API_KEY = os.environ.get("THAILLM_API_KEY", "1p1dC5cHTY9G4myMrb48gixAdDcFA55L")

# === CONFIGURATION ===
N_QUESTIONS = 100
DATA_DIR = "./data"
KB_DIR = f"{DATA_DIR}/knowledge_base"
USE_CONTEXTUAL_ENRICHMENT = True # Set to False to skip the heavy generating baseline.

# === CHECKPOINT FILES ===
CONTEXTUAL_CACHE = "contextual_chunks_checkpoint.json"
EMBEDDINGS_CACHE = "embeddings_cache.npy"
BACKUP_SUBMISSION = "backup_submission_typhoon.csv"

# === 0. LLM UTILS & ULTIMATE PARSERS ===
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
    
    # ULTIMATE STAGE: Try parsing Strict JSON Output
    try:
        json_str = re.sub(r'```json|```', '', text).strip()
        data = json.loads(json_str)
        if "answer" in data:
            return int(data["answer"])
    except json.JSONDecodeError:
        pass
        
    # BACKUP: Regex formatting
    clean = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    m = re.search(r'"answer"\s*:\s*(\d+)', clean, re.IGNORECASE)
    if m: return int(m.group(1))
    
    m = re.search(r"ANSWER:\s*(\d+)", clean, re.IGNORECASE)
    if m: return int(m.group(1))
    
    for d in re.findall(r"\b(\d{1,2})\b", clean):
        if 1 <= int(d) <= 10: return int(d)
        
    return 9 # Fallback if not sure
    
def unroll_markdown_tables(text):
    """
    ULTIMATE STAGE 1: Table-to-Text Unrolling
    Converts Markdown tables into readable sentences for better embeddings.
    """
    lines = text.split('\n')
    unrolled_lines = []
    in_table = False
    headers = []
    
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('|') and stripped.endswith('|'):
            cells = [c.strip() for c in stripped.strip('|').split('|')]
            if all(set(c) <= set('- :') for c in cells):
                continue
            if not in_table:
                in_table = True
                headers = cells
            else:
                row_info = ", ".join(f"{h}: {c}" for h, c in zip(headers, cells) if h and c)
                unrolled_lines.append(f"• ข้อมูลตาราง: {row_info}")
        else:
            in_table = False
            unrolled_lines.append(line)
            
    return '\n'.join(unrolled_lines)
    
def generate_contextual_chunk(doc_text, chunk_text):
    system_prompt = "คุณคือ AI ผู้ช่วยสรุปข้อมูล จงเขียนบริบทสั้นๆ 1-2 ประโยคอธิบายว่าส่วนย่อย (Chunk) นี้พูดถึงอะไรในภาพรวมของเอกสารหลัก เพื่อนำไปใช้เป็นข้อมูลบริบทปะหน้าในการค้นหา"
    user_prompt = f"[เอกสารฉบับเต็มบางส่วน]\n{doc_text[:1500]}...\n\n[ส่วนย่อย (Chunk)]\n{chunk_text}\n\nจงสรุปบริบทสั้นๆ ปะหน้าข้อความส่วนย่อยนี้:"
    
    context = ask_llm([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ], temperature=0.3)
    
    if context:
        return f"[บริบท: {context}]\n{chunk_text}"
    return chunk_text

# === 1. DATA PREPARATION (PHASE 1) ===
def get_or_build_chunks():
    kb_dir = Path(KB_DIR)
    documents = []
    if kb_dir.exists():
        for fp in sorted(kb_dir.rglob("*.md")):
            raw_text = fp.read_text(encoding="utf-8")
            unrolled_text = unroll_markdown_tables(raw_text) # Apply Table Unrolling
            documents.append({
                "path": str(fp.relative_to(kb_dir)), 
                "text": unrolled_text
            })

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    
    # 1. Flatten into (doc, text) pairs
    raw_chunks = []
    for doc in documents:
        for text in text_splitter.split_text(doc["text"]):
            raw_chunks.append((doc, text))
            
    chunks = []
    print(f"Found {len(documents)} documents, split into {len(raw_chunks)} chunks total.")
    
    if USE_CONTEXTUAL_ENRICHMENT:
        print("Starting Contextual Enrichment...")
        # CHECKPOINT 1: Contextual Cache Recovery
        if os.path.exists(CONTEXTUAL_CACHE):
            print(f"Resuming from partial cache: {CONTEXTUAL_CACHE}...")
            with open(CONTEXTUAL_CACHE, "r", encoding="utf-8") as f:
                chunks = json.load(f)
        
        done_texts = set(c.get("original_text", c["text"]) for c in chunks)
        
        for doc, text in tqdm(raw_chunks, desc="Enriching Chunks via API"):
            if text in done_texts:
                continue
                
            enriched_text = generate_contextual_chunk(doc["text"], text)
            new_chunk = {"text": enriched_text, "source": doc["path"], "original_text": text}
            chunks.append(new_chunk)
            
            with open(CONTEXTUAL_CACHE, "w", encoding="utf-8") as f:
                json.dump(chunks, f, ensure_ascii=False, indent=2)
                
            time.sleep(0.5)
    else:
        print("Skipping API Contextual Enrichment (USE_CONTEXTUAL_ENRICHMENT=False).")
        for doc, text in tqdm(raw_chunks, desc="Processing Raw Chunks"):
            chunks.append({"text": text, "source": doc["path"], "original_text": text})
            
    return chunks

# === 2. INDEXING ===
print("=== PHASE 1: PREPARATION ===")
chunks = get_or_build_chunks()

print("\nLoading BAAI/bge-m3 for Vector Search...")
embed_model = SentenceTransformer("BAAI/bge-m3")
chunk_texts = [c["text"] for c in chunks]

# CHECKPOINT 2: Embeddings Disk Cache
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

print("\nHybrid Search initialized. (Cross-Encoder Reranking disabled)")


# === 3. INFERENCE PIPELINE (PHASE 2) ===
print("\n=== PHASE 2: INFERENCE ===")
def rewrite_query(question):
    # Disabled Query Rewriting to save inference time and support smaller models
    return question

def retrieve_top_k(query_rewritten, fetch_k=30):
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

def rerank_to_top_5(query_rewritten, candidate_indices, top_k=5):
    # Reranking disabled, fallback to relying strictly on Hybrid Search RRF ranking
    return candidate_indices[:top_k]


# === 4. EXECUTION LOOP ===
questions = []
try:
    with open(f"{DATA_DIR}/questions.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            choices = {str(i): row[f"choice_{i}"] for i in range(1, 11)}
            questions.append({"id": int(row["id"]), "question": row["question"], "choices": choices})
except FileNotFoundError:
    print("Warning: data/questions.csv not found. Skipping QA loop.")

sys_prompt = """คุณคือ AI ผู้ช่วยร้านฟ้าใหม่ จงตอบคำถามโดยอ้างอิงจากข้อมูลต่อไปนี้เป็นหลัก
หากมีข้อมูลที่สอดคล้องหรือใกล้เคียงให้เลือกตอบข้อ 1-8 ที่ตรงที่สุด
แต่ถ้าข้อมูลไม่เกี่ยวข้องกันเลยจริงๆ หรือไม่มีระบุไว้เลย ค่อยตอบ 9 หรือ 10"""

# CHECKPOINT 3: Generation Backup Recovery
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
    top30_idx = retrieve_top_k(rewritten_q, fetch_k=30)
    final_top5_idx = rerank_to_top_5(rewritten_q, top30_idx, top_k=5)
    retrieved_chunks = [chunks[i] for i in final_top5_idx]
    
    context_text = "\n\n".join(f"--- ข้อมูล {i+1} ---\n{c['text']}" for i, c in enumerate(retrieved_chunks))
    choices_text = "\n".join(f"{k}. {v}" for k, v in q["choices"].items())
    
    final_prompt = (
        f"[ข้อมูลอ้างอิง]\n{context_text}\n\n"
        f"[คำถามจากลูกค้า]\n{original_q}\n\n"
        f"[ตัวเลือก]\n{choices_text}\n\n"
        f"ตอบนำหน้าด้วย ANSWER: X (X คือหมายเลข 1-10)"
    )
    
    raw = ask_llm([
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": final_prompt}
    ], temperature=0.1)
    
    ans = parse_answer(raw)
    results.append([q_id, ans])
    print(f"  Q{q_id:>3}: Ans={ans}")
    
    with open(BACKUP_SUBMISSION, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([q_id, ans])

# Write Final Output
with open("typhoon_submission.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "answer"])
    writer.writerows(results)

print("TYPHOON pipeline complete! Saved to typhoon_submission.csv")
