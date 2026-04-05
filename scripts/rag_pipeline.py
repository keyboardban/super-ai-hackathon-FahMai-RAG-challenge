import os
import re
import pandas as pd
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import faiss
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# ==========================================
# 1. CONFIGURATION
# ==========================================
# Update these paths once you have downloaded the Kaggle dataset
DATA_DIR = "./data" # Path to extracted Kaggle data
KNOWLEDGE_BASE_PATH = os.path.join(DATA_DIR, "knowledge_base") # Path to knowledge_base directory
TEST_QUESTIONS_PATH = os.path.join(DATA_DIR, "questions.csv")
SUBMISSION_PATH = "submission.csv"

# Model Configurations
EMBEDDING_MODEL_NAME = "BAAI/bge-m3" # Excellent multilingual & Thai support
LLM_MODEL_NAME = "scb10x/Typhoon-S-ThaiLLM-8B-Instruct" # One of the 4 allowed models

MAX_SEQ_LENGTH = 512
TOP_K_RETRIEVAL = 3 # Number of chunks to retrieve per question

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# ==========================================
# 2. DATA PREPARATION & CHUNKING
# ==========================================
def load_knowledge_base(dir_path):
    print(f"Loading Knowledge Base from {dir_path}...")
    documents = []
    
    if not os.path.exists(dir_path):
        print(f"Directory {dir_path} not found. Returning a dummy Knowledge Base for testing.")
        documents.append(Document(page_content="ร้านฟ้าใหม่เปิดทำการทุกวันเวลา 09:00 ถึง 20:00 น."))
    else:
        for root, _, files in os.walk(dir_path):
            for filename in files:
                if filename.endswith(".md") or filename.endswith(".txt"):
                    filepath = os.path.join(root, filename)
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            text = f.read()
                            if text.strip():
                                documents.append(Document(page_content=text, metadata={"source": filename}))
                    except Exception as e:
                        print(f"Error reading {filepath}: {e}")
                        
    if not documents:
        documents.append(Document(page_content="ไม่มีข้อมูลในฐานข้อมูลวิเคราะห์"))
    
    # Chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Total documents chunked into {len(chunks)} chunks.")
    return [chunk.page_content for chunk in chunks]

def load_test_questions(file_path):
    print(f"Loading Test Questions from {file_path}...")
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"File {file_path} not found. Returning dummy test questions.")
        return pd.DataFrame({
            "id": [1, 2, 3],
            "question": ["ร้านฟ้าใหม่เปิดกี่โมง?", "มีขายเครื่องบินไหม?", "ประเทศฝรั่งเศสเมืองหลวงคืออะไร?"],
            "choice_1": ["08:00", "มี", "ปารีส"],
            "choice_2": ["09:00", "ไม่มี", "ลอนดอน"],
            "choice_3": ["10:00", "กำลังจัดหา", "โรม"],
            "choice_4": ["11:00", "ไม่แน่ใจ", "โตเกียว"],
            "choice_5": ["12:00", "รอยืนยัน", "โซล"],
            "choice_6": ["13:00", "สั่งได้", "มาดริด"],
            "choice_7": ["14:00", "หมด", "ปักกิ่ง"],
            "choice_8": ["15:00", "เลิกผลิต", "กรุงเทพ"]
        })

# ==========================================
# 3. VECTOR DATABASE & RETRIEVAL
# ==========================================
def build_vector_store(chunks):
    print(f"Loading Embedding Model: {EMBEDDING_MODEL_NAME}")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
    
    print("Encoding chunks...")
    embeddings = embedding_model.encode(chunks, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
    
    print("Building FAISS Index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension) # Inner product for normalized embeddings = Cosine Similarity
    index.add(embeddings)
    return embedding_model, index

def retrieve_context(query, index, embedding_model, chunks, top_k=TOP_K_RETRIEVAL):
    query_vector = embedding_model.encode([query], normalize_embeddings=True)
    distances, indices = index.search(query_vector, top_k)
    
    retrieved_texts = [chunks[idx] for idx in indices[0]]
    context = "\n---\n".join(retrieved_texts)
    return context

# ==========================================
# 4. LLM INFERENCE (GENERATION)
# ==========================================
def setup_llm():
    print(f"Loading LLM: {LLM_MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME, token="1p1dC5cHTY9G4myMrb48gixAdDcFA55L")
    model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL_NAME,
    token="1p1dC5cHTY9G4myMrb48gixAdDcFA55L",
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True,
    device_map=device
    )


    generator = pipeline(
        "text-generation", 
        model=model, 
        tokenizer=tokenizer, 
        max_new_tokens=20, # We only need an integer representation
        temperature=0.1,   # Low temp for deterministic answers
        do_sample=False
    )
    return generator

def format_prompt(context, question, choices_dict):
    """
    Constructs the prompt adhering strictly to competition rules.
    """
    choices_str = ""
    for i in range(1, 11):
        choice_text = choices_dict.get(f"choice_{i}", "")
        if pd.notna(choice_text) and choice_text != "":
            choices_str += f"{i}:{choice_text}\n"

    prompt = f"""คุณคือผู้ช่วย AI ของร้านขายเครื่องใช้ไฟฟ้า 'ฟ้าใหม่' (FahMai)
จงตอบคำถามโดยเลือกหมายเลขตัวเลือก (1-10) ที่ถูกต้องที่สุดเพียงหมายเลขเดียว โดยอิงจากบริบทที่ให้มาเท่านั้น

[กฎการตอบ]
- หากมีคำตอบในบริบท ให้เลือกหมายเลขของคำตอบนั้น (1-8)
- หากข้อมูลในบริบท ไม่เพียงพอ หรือ ไม่มีการบรรยายถึงเรื่องนี้ในฐานข้อมูล ให้ตอบ 9
- หากคำถาม ไม่เกี่ยวกับร้านฟ้าใหม่ หรือไม่เกี่ยวกับเครื่องใช้ไฟฟ้า/นโยบายเลย ให้ตอบ 10
- ตอบเพียงตัวเลข 1 ถึง 10 เท่านั้น ห้ามมีข้อความอื่นเจือปน

[บริบท]
{context}

[คำถาม]
{question}

[ตัวเลือก]
{choices_str}

ตอบ (รหัสตัวเลือก):"""
    return prompt

def extract_answer(generated_text):
    # Use regex to find the first integer between 1 and 10 in the generated text
    matches = re.findall(r'\b(10|[1-9])\b', generated_text)
    if matches:
         return int(matches[0])
    return 9 # Fallback if parsing fails

# ==========================================
# 5. MAIN EXECUTION PIPELINE
# ==========================================
def main():
    # 1. Load Data
    chunks = load_knowledge_base(KNOWLEDGE_BASE_PATH)
    test_df = load_test_questions(TEST_QUESTIONS_PATH)
    
    # 2. Setup Retrieval
    embedding_model, index = build_vector_store(chunks)
    
    # 3. Setup LLM
    generator = setup_llm()
    
    results = []
    
    # 4. Run Evaluation
    print("Running QA pipeline...")
    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        q_id = row['id']
        question = row['question']
        
        # Build choices dict (assuming columns are named choice_1, choice_2... choice_10)
        choices_dict = {f"choice_{i}": row.get(f"choice_{i}", "") for i in range(1, 11)}
        
        # Retrieve Context
        context = retrieve_context(question, index, embedding_model, chunks)
        
        # Generate Prompt
        prompt = format_prompt(context, question, choices_dict)
        
        # Inference
        out = generator(prompt, return_full_text=False)
        generated_text = out[0]['generated_text']
        
        # Parse Answer
        ans_idx = extract_answer(generated_text)
        
        results.append({
            "id": q_id,
            "answer": ans_idx
        })
    
    # 5. Save Submission
    submission_df = pd.DataFrame(results)
    submission_df.to_csv(SUBMISSION_PATH, index=False)
    print(f"Pipeline complete! Submission saved to {SUBMISSION_PATH}")
    print(submission_df.head(10))

if __name__ == "__main__":
    main()
