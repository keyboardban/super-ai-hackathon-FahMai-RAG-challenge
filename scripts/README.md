# 📑 คู่มือเจาะลึกโค้ด RAG Pipeline (Scripts Detailed Documentation)

ไฟล์โค้ด Python (`.py`) ภายในโฟลเดอร์นี้ถูกออกแบบมาเพื่อทดสอบสมมติฐานและปรับแต่ง (Fine-tuning) กระบวนการทำ RAG (Retrieval-Augmented Generation) ซึ่งในแต่ละไฟล์มีการตั้งค่า Hyper-parameters และเทคนิคที่ต่างกันออกไป 

*(หมายเหตุ: นำโค้ดกลุ่มทดสอบย่อยเช่น `oracle` และ `stealth` ออกจากรายการเปรียบเทียบแล้ว เพื่อลดความซ้ำซ้อน)*

การทำงานหลักจะอิงกับโมเดล **Typhoon (ThaiLLM API)** สำหรับการตอบคำถาม และใช้ **BAAI/bge-m3** สำหรับทำ Text Embeddings

---

## 🛠 เทคนิคหลักที่ใช้สลับไปมาในโค้ดแต่ละชุด
ก่อนจะไปดูว่าแต่ละไฟล์ต่างกันอย่างไร เรามาทำความเข้าใจ 4 เทคนิคเบื้องหลังโค้ดเหล่านี้กันก่อน:
1. **Chunk Size (`400/50` vs `800/150`)**: 
   - `400`: เล็ก แม่นยำ ดึงข้อความกระชับ
   - `800`: ใหญ่ขึ้น ช่วยให้อ่านเอกสารทีละมากๆ ได้ดี ลดปัญหาข้อมูลขาดตอนแต่เปลือง Token
2. **Query Rewrite**: ให้ LLM ช่วยเกลาคำถามของลูกค้าที่อาจจะคลุมเครือ ให้กลายเป็นคำถามที่ตรงประเด็น ง่ายต่อการค้นหาในระบบ
3. **Cross-Encoder Reranker (`BAAI/bge-reranker-v2-m3`)**: เทคนิคจัดเรียงเอกสารชั้นสูง หลังจากค้นหาด้วย Vector เสร็จ จะส่งคำถามและเอกสารที่หาเจอไปให้โมเดล Reranker ให้คะแนนความสอดคล้องกันอีกรอบนึง (แม่นขึ้นมาก แต่ช้าลง)
4. **Contextual Enrichment (`generate_contextual_chunk`)**: เป็นท่าระดับ SOTA โดยให้ LLM ช่วยสรุปใจความสำคัญของหัวข้อมา "ปะหน้า" Chunk ทุกชิ้นก่อนนำไปฝัง (Embed) ทำให้อัตราการหาเจอ (Retrieval Hit Rate) พุ่งทะยาน แต่ใช้เวลาประมวลผลฐานข้อมูลนานถึง 15-16 ชั่วโมง

---

## 📋 เจาะลึกความแตกต่างแบบเรียงตัว

### 👶 1. กลุ่มระดับพื้นฐานตระกูล Loop (เน้นไวและความเรียบง่าย)
กลุ่มนี้จะ **ไม่มี** Reranker, **ไม่มี** Query Rewrite และ **ไม่มี** Contextual Enrichment
- **`rag_pipeline.py` & `copy_of_starter_kit_fahmai_rag.py`**: โค้ด Starter ดั้งเดิม เป็นโครงสร้าง Langchain พื้นฐาน
- **`basic_loop_starter_kit.py`**: รื้อ Langchain ออกแล้วเขียนเป็น Loop ควบคุมเอง (Chunks=400/50, Temp=0.3/0.1)
- **`basic_loop_fast.py` & `fast_concurrent_loop_pipeline.py`**: สิ่งที่เพิ่มขึ้นมาคือระบบ **Multithreading** โดยใช้ `ThreadPoolExecutor` และระบบ Lock ช่วยดึงข้อมูลและยิง API หลายๆ ข้อบรรทัดพร้อมกัน *ทำให้รันทุกข้อเสร็จไวมากๆ เหมาะสำหรับการรันเป็น Baseline เพื่อเทียบเวลา!*

### ⚖️ 2. กลุ่มระดับกลาง (ปิด Enrichment เพื่อความไว เน้นปรับ Chunk / Prompt)
กลุ่มนี้จะมี Reranker และมี Query Rewrite แต่ปิด Enrichment
- **`improved_advanced_rag_pipeline.py`**: ขยาย Chunk Size เป็น `800/150` เพื่อกวาดบริบทให้กว้างที่สุด ใช้ X-Encoder Reranker แต่ใช้ `USE_CONTEXTUAL_ENRICHMENT = False` วิ่งด้วยอุณหภูมิต่ำ (Temp=0.1) เพื่อลดอาการหลอน (Hallucination)
- **`loop_reasoning_rag_pipeline.py`**: รันด้วย Chunk `800/150` เหมือนกัน แต่อุณหภูมิตอนดึงบริบทจะปรับเป็น 0.3 เพื่อเปิดความอิสระให้โมเดลประมวลผลตรรกะเหตุผล ก่อนจะสรุปด้วย Temp=0.1

### 🚀 3. กลุ่มตัวเต็ม Contextual เต็มสูบ (ไม่มี Reranker)
กลุ่มนี้จะเปิดใช้ฟีเจอร์กินเวลาอย่าง **Contextual Enrichment** ก่อนฝังข้อมูล แต่ตัด Reranker ออกเพื่อไม่ให้ตอนถามตอบช้าเกินไป
- **`typhoon_rag_pipeline.py`**: ใช้ Chunk `400/50` และทำการ Contextual Enrichment พร้อมกับทำ Query Rewrite กำหนดอุณหภูมิคำตอบไฟนอลที่ Temp = 0
- **`ultimate_rag_pipeline.py`**: เหมือน Typhoon ด้านบนทุกประการ แต่ปรับ Chunk ให้หนาขึ้นเป็น `800/150` (ครอบคลุมบริบทได้ลึกขึ้น)

### 🥇 4. กลุ่มตัวท็อป (ท่าไม้ตาย Full Pipeline)
เปิดทุกฟังก์ชันที่มีในระบบ กินเวลาเยอะที่สุด แต่หวังความแม่นยำสูงที่สุด
- **`advanced_rag_pipeline.py`**: ใช้งานทั้ง Reranker, Rewrite และ Contextual Enrichment บน Chunk ขนาดกระชับ `400/50` 
- **`sota_contextual_rag_pipeline.py`**: **(✨ The SOTA Feature)** เปิดฟีเจอร์ทุกอย่างและดัน Chunk Size ไปที่ `800/150` โมเดลนี้จะใช้เวลาสร้าง Cache แบบมหาศาลและหาข้อมูลแบบเจาะจงลึกสุด เหมาะรันทิ้งไว้ข้ามคืนเพื่อเอา Top Score บน Leaderboard

### 🤔 5. กลุ่มทดลองโมเดลแปลกใหม่ (Experimental Models)
- **`pathumma_rag_pipeline.py`**: ฉีกกรอบด้วยการไม่ใช้ Typhoon แต่ไปใช้ `Pathumma-ThaiLLM-qwen3-8b-think-3.0.0` ซึ่งเป็นโมเดลสำหรับ Deep Thinking โดยเฉพาะ โค้ดมีการวางระบบตัดแท็ก `<think>` ที่ Pathumma ให้เหตุผลในใจก่อนตอบออกให้อัตโนมัติ (ปิด Contextual Enrichment, ใช้ Reranker)
- **`reflective_rag_pipeline.py`**: ปรับแต่ง Prompt ในฝั่งของ Query Rewrite และการคัดกรองคำตอบ เพื่อให้ LLM เกิดกระบวนการสะท้อนความคิดตนเองก่อนจะฟันธงตอบ (Chunk 800/150, ใช้ Reranker)

### 🛠 6. สคริปต์อรรถประโยชน์
- **`evaluate_and_iterate_pipeline.py`**: สคริปต์รัน Loop พิเศษที่สามารถจับ Metric เพื่อประเมินคะแนนของลูปต่างๆ โดยอัตโนมัติ (Automated evaluation)
- **`compare_submissions.py`**: ตัวเช็คความถูกต้องไฟล์ CSV (ทำงานด้วยหลักการเดียวกับ Jupyter Notebook แต่รันด้วย Terminal)
