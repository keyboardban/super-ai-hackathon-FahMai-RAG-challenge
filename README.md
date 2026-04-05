# 🏆 FahMai RAG Challenge: From Baseline to SOTA Contextual Pipeline

**Author:** Teerachot Saenthong (ID: 601367)  
**Background:** Computer Science, Chulalongkorn University  
**Passion:** Machine Learning in Speech & Audio Processing, ASR (Automatic Speech Recognition), and Signal Processing.

> *"For me, the charm of AI isn't about having the model with the most parameters. It's the ability to analyze errors systematically and fine-tune architectures to solve problems precisely under constraints."*

---

## 🎯 The Problem Statement
The challenge was to build an AI Customer Service bot for an IT shop named **"FahMai" (ฟ้าใหม่)**. 
Large Language Models alone are prone to hallucination. **Retrieval-Augmented Generation (RAG)** is absolutely necessary here to provide strict **Grounding**—connecting the LLM directly to FahMai's localized database, ensuring verifiable answers, and completely eliminating hallucinations.

---

## 🚀 The Evolution of the Pipeline

### 🟢 Iteration 1: The Baseline (Speed over Accuracy)
**The Naive Approach:** I started with a fundamental pipeline prioritizing speed and simplicity to create a baseline. 
- **What I did:** Implemented both standard LangChain structures (`rag_pipeline.py`) and Custom Python Loops (`basic_loop_fast.py`). Documents were split using `RecursiveCharacterTextSplitter` at exactly **400 tokens**. I stripped away reranking and query expansion to test raw execution time.
- **The Result:** The inference was lightning fast thanks to parallel processing (`ThreadPoolExecutor`), achieving minimal latency. However, it suffered heavily from **Context Loss** and **Semantic Fragmentation**. The model hallucinated frequently when context was cut off mid-sentence, failing to answer questions that required cross-document reasoning.

### 🟡 Iteration 2: The Optimization (Fixing the Blind Spots)
**Deep Error Analysis:** Instead of blindly changing models, I plotted Confusion Matrices (`visualize_error_analysis.py`) and analyzed misclassified pairs. I discovered system blind spots regarding transliterations (e.g., "SaiFah" vs. "ซายฟ้า"), ambiguities between Option 9 (No Info) and Option 10 (Irrelevant), and failing math/comparison queries.
- **What I did:** 
  - Expanded Chunk Size to **800** to retain a wider context (`improved_advanced_rag_pipeline.py`).
  - Added **Query Expansion/Rewriting** (`rewrite_query`) to clarify user intent and correct spelling *before* searching.
  - Integrated the **BAAI Cross-Encoder** (`BAAI/bge-reranker-v2-m3`) to rerank and validate context relevance.
- **Engineering Feat:** Handled severe API bottlenecks while maintaining inference speed by orchestrating asynchronous Multithreading. 

### 🔴 Iteration 3: The Ultimate SOTA Contextual Pipeline (The Masterpiece)
**The SOTA Pipeline:** Even with rerankers, semantic dilution was still an issue. To fix this, I developed the ultimate version: `sota_contextual_rag_pipeline.py`.
- **Contextual Enrichment:** I engineered a feature where an LLM reads the full document and generates a mini-summary to prepend to *every single chunk* (`generate_contextual_chunk`). This created **High Information Density** and solved the Context Loss problem permanently.
- **Hybrid Search Architecture:** Shifted from naive dense search to a Hybrid Retrieval system. Used **Dense Retrieval** (`BAAI/bge-m3`) merged with **Sparse Retrieval** (`BM25Okapi` + PyThaiNLP Tokenizer) and fused their scores mathematically using **RRF (Reciprocal Rank Fusion)**.
- **Safety Nets:** Implemented Self-Reflection Prompting and heuristic Rule-based logic directly inside the Query Rewriter to catch edge-case anomalies (e.g., hardcoding conditions for "Option 9 vs 10") before they reached the main generation phase.

---

## 🔬 Best Configuration & Ablation
Through extensive ablation studies in this 24-hour window, the supreme configuration emerged as:
**[Reranker + Query Rewrite + Contextual Enrichment]** on **Chunk Size 800** (with Chunk 400 as a strong secondary variant like in `advanced_rag_pipeline.py`). 

*Note: I originally conceptualized a rigorous "Chain-of-Thought (CoT)" implementation, but due to severe execution-time constraints and API rate limits during the Hackathon, I could not finalize it in the ultimate codebase.*

---

## 🎙️ Vision & Application: The Future of Audio RAG
My core passion has always been Speech & Audio Processing. Watching this text-based RAG pipeline evolve inspired my ultimate vision—merging highly contextual RAG with Audio:
1. **Real-time Voice-Driven RAG:** Building ultra-low latency voice bots for customer support, directly injecting retrieved context into spoken generation.
2. **Audio-Native RAG (CLAP):** Moving beyond speech-to-text translations by using **Contrastive Language-Audio Pretraining (CLAP)** to retrieve audio cues or tonal context directly from raw audio databases recordings.
3. **RAG-Powered Error Correction:** Using contextual Knowledge Bases to dynamically correct ASR (Automatic Speech Recognition) errors on the fly, dramatically improving transcription accuracy for highly specialized IT domain terms.

*This hackathon proved that data is not just an asset; it’s a compass. The right pragmatic engineering makes AI genuinely revolutionary.*
