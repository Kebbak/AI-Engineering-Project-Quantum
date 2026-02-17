# Design and Evaluation Documentation

## Design Choices

### Embedding Model
- **Model:** all-MiniLM-L6-v2 (local SentenceTransformer)
- **Justification:**
  - Free, open-source, and can run locally without API costs.
  - Provides strong performance for semantic search and retrieval tasks.
  - 384-dimensional embeddings are efficient for storage and computation.

### Chunking
- **Method:** Fixed-size character window (300 chars) with 50-character overlap
- **Justification:**
  - Ensures context continuity and avoids splitting important information.
  - Overlap helps preserve meaning across chunk boundaries.
  - Simple, deterministic, and reproducible.

### Retrieval Top-k (k)
- **Value:** k = 5 (default)
- **Justification:**
  - Balances recall and context window size for LLM input.
  - Empirically, 3–5 chunks usually provide enough relevant evidence for policy Q&A.

### Prompt Format
- **Method:**
  - System prompt instructs the LLM to answer only from provided context and refuse out-of-corpus questions.
  - User prompt includes the question and concatenated retrieved chunks as context.
  - Output is limited in length and does not include citations in the answer text (citations are shown separately).
- **Justification:**
  - Reduces hallucination by restricting LLM to retrieved evidence.
  - Keeps answers concise and focused.
  - Guardrails ensure the system does not answer outside the corpus.

### Vector Store
- **Choice:** ChromaDB (local persistent storage)
- **Justification:**
  - Free, open-source, and easy to set up locally.
  - Supports fast similarity search and metadata storage.
  - No external dependencies or cloud costs.

---

## Evaluation Summary
- See evaluation_questions.py for the question set.
- Metrics: Groundedness, Citation Accuracy, Latency (p50/p95).
- See README for setup and run instructions.

---

*This document justifies the main design and architecture decisions for the RAG LLM-based company policy QA system.*
