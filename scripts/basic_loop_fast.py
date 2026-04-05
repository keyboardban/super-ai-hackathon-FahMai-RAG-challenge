# -*- coding: utf-8 -*-
"""
basic_loop_fast.py
เวอร์ชันเร่งสปีดของ basic_loop_starter_kit.py
- ใช้ Multi-threading รัน 5 ข้อพร้อมกัน (ปรับ MAX_WORKERS ได้)
- ยังคงกระบวนการ Loop Reasoning เหมือนเดิมทุกประการ (MAX_LOOPS=3)
- ยังคง Hybrid Retrieval (Dense + BM25 + RRF) เหมือนเดิม
- เพิ่ม Progress Bar แสดงความคืบหน้า
- เพิ่ม Thread Lock ป้องกันไฟล์ CSV พัง
"""
import os, csv, re, time, requests
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
import concurrent.futures
import threading

from pythainlp.tokenize import word_tokenize
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

load_dotenv()
THAILLM_API_KEY = os.environ.get("THAILLM_API_KEY", "1p1dC5cHTY9G4myMrb48gixAdDcFA55L")

# === CONFIGURATION (เหมือนเดิมทุกอย่าง) ===
N_QUESTIONS = 100
DATA_DIR = "./data"
KB_DIR = f"{DATA_DIR}/knowledge_base"
TOP_K = 5
MAX_LOOPS = 3          # จำนวนรอบทบทวนคำตอบ (เท่าเดิม)
MAX_WORKERS = 5        # จำนวนข้อที่รันพร้อมกัน (ปรับได้ ยิ่งเยอะยิ่งเร็ว แต่ระวัง Rate Limit)

# === CHECKPOINT FILES ===
EMBEDDINGS_CACHE = "embeddings_cache_basic_loop.npy"  # ใช้แคชเดิมได้เลย
BACKUP_SUBMISSION = "backup_basic_loop_fast.csv"
FINAL_SUBMISSION = "basic_loop_fast_submission.csv"

csv_lock = threading.Lock()
embed_lock = threading.Lock()  # ป้องกัน SentenceTransformer พังเวลาหลาย Thread เรียกพร้อมกัน

# === 0. LLM UTILS (เหมือนเดิมทุกอย่าง) ===
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

# === 1. DATA PREPARATION (เหมือนเดิมทุกอย่าง) ===
def get_or_build_chunks():
    kb_dir = Path(KB_DIR)
    documents = []
    if kb_dir.exists():
        for fp in sorted(kb_dir.rglob("*.md")):
            documents.append({
                "path": str(fp.relative_to(kb_dir)),
                "text": fp.read_text(encoding="utf-8")
            })

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    raw_chunks = []
    for doc in documents:
        for text in text_splitter.split_text(doc["text"]):
            raw_chunks.append((doc, text))

    chunks = []
    print(f"Found {len(documents)} documents, split into {len(raw_chunks)} chunks total.")
    for doc, text in tqdm(raw_chunks, desc="Processing Raw Chunks"):
        chunks.append({"text": text, "source": doc["path"], "original_text": text})

    return chunks

# === 2. INDEXING (เหมือนเดิมทุกอย่าง) ===
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

# === 3. RETRIEVAL (เหมือนเดิมทุกอย่าง) ===
def dense_retrieve(query, chunk_embs, k=TOP_K):
    with embed_lock:
        q_emb = embed_model.encode([query], normalize_embeddings=True)
    scores = np.dot(chunk_embs, q_emb.T).flatten()
    top_idx = np.argsort(scores)[::-1][:k]
    return top_idx, scores[top_idx]

def bm25_retrieve(query, k=TOP_K):
    tokens = word_tokenize(query, engine="newmm")
    scores = bm25.get_scores(tokens)
    top_idx = np.argsort(scores)[::-1][:k]
    return top_idx, scores[top_idx]

def hybrid_retrieve(query, chunk_embs, k=TOP_K, rrf_k=60):
    fetch_k = k * 2
    d_idx, _ = dense_retrieve(query, chunk_embs, k=fetch_k)
    b_idx, _ = bm25_retrieve(query, k=fetch_k)

    rrf_scores = {}
    for rank, idx in enumerate(d_idx, 1):
        rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (rrf_k + rank)
    for rank, idx in enumerate(b_idx, 1):
        rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (rrf_k + rank)

    sorted_idx = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:k]
    return sorted_idx

# === 4. WORKER FUNCTION (Logic เหมือนเดิมทุกอย่าง แค่ห่อเป็นฟังก์ชัน) ===
def process_question(q):
    q_id = q["id"]
    original_q = q["question"]

    # 1. STARTER KIT RETRIEVAL (เหมือนเดิม)
    h_idx = hybrid_retrieve(original_q, chunk_embeddings, k=TOP_K)
    retrieved_chunks = [chunks[i] for i in h_idx]

    context_text = "\n\n".join(f"--- ข้อมูล {i+1} ---\n{c['text']}" for i, c in enumerate(retrieved_chunks))
    choices_text = "\n".join(f"{k}. {v}" for k, v in q["choices"].items())

    # 2. LOOPING REASONING (เหมือนเดิมทุกอย่าง)
    sys_prompt_agent = "คุณคือ AI ระดับหัวกะทิของร้านฟ้าใหม่ จงตอบคำถามอย่างมีเหตุผลอิงจากข้อมูลที่ให้ไว้เท่านั้น"

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

    final_ans = current_ans

    for loop_num in range(1, MAX_LOOPS + 1):
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
            current_ans = final_ans
            current_reasoning = verify_raw
            if loop_num == MAX_LOOPS:
                print(f"  Q{q_id:>3}: Found Answer={final_ans} (Forced stop after {MAX_LOOPS} Loops) ⚠️")

    # บันทึกลงไฟล์อย่างปลอดภัย
    with csv_lock:
        with open(BACKUP_SUBMISSION, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([q_id, final_ans])
            f.flush()
            os.fsync(f.fileno())

    return [q_id, final_ans]

# === 5. EXECUTION LOOP (เปลี่ยนจาก for loop เป็น ThreadPool) ===
if __name__ == "__main__":
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

    questions_to_process = [q for q in questions[:N_QUESTIONS] if q["id"] not in done_qids]

    print(f"\n=== PHASE 2: BASIC RETRIEVAL + LOOP REASONING (x{MAX_WORKERS} Concurrent) ===")
    print(f"Running pipeline on {len(questions_to_process)} questions with {MAX_WORKERS} workers...")

    if questions_to_process:
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [executor.submit(process_question, q) for q in questions_to_process]
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing"):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as exc:
                    print(f"Exception: {exc}")

    results.sort(key=lambda x: x[0])

    with open(FINAL_SUBMISSION, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "answer"])
        writer.writerows(results)

    print(f"\nBASIC LOOP (FAST) pipeline complete! Saved to {FINAL_SUBMISSION}")
