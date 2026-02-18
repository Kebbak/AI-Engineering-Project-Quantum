### Project Steps Completed

This project fulfills the following requirements:

1. **Environment and Reproducibility**
   - Virtual environment created
   - Dependencies listed in requirements.txt
   - README with setup and run instructions
   - Fixed seeds set where applicable
2. **Ingestion and Indexing**
   - Documents parsed and cleaned (PDF, HTML, Markdown, TXT)
   - Documents chunked
   - Chunks embedded using a free embedding model
   - Embedded chunks stored in ChromaDB
3. **Retrieval and Generation (RAG)**
   - Top-k retrieval implemented
   - Prompting strategy injects retrieved chunks and citations
   - Guardrails: answers limited to corpus, output length restricted, citations included
4. **Web Application**
   - Flask web app with endpoints: / (chat UI), /chat (API), /health (status)
5. **CI/CD**
   - GitHub Actions workflow for dependency install and build/start check
   - Automated tests included
6. **Evaluation**
   - Evaluation set of questions covering policy topics
   - Metrics reported: groundedness, citation accuracy, latency
7. **Design Documentation**
   - Design choices justified in design-and-evaluation.md

---
## Project Structure

- **Description:** End-to-end RAG pipeline for document ingestion, embedding, indexing, and retrieval for company policy Q&A.
- **Features:**
   - Synthetic corpus creation (Markdown, HTML, TXT, PDF)
   - Document parsing and chunking
   - Embedding with local model (sentence-transformers)
   - Vector database (ChromaDB) setup and querying
   - Web app for Q&A with citations
   - Python scripts for ingestion and retrieval
- **Technologies Used:**
   - Python 3.x
   - sentence-transformers (all-MiniLM-L6-v2)
   - ChromaDB
   - python-dotenv
   - fpdf (for PDF generation)
## Getting Started

1. **Clone the repository:**
   ```sh
   git clone git@github.com:Kebbak/AI-Engineering-Project-Quantum.git
   cd AI-Engineering-Project-Quantum
   ```
2. **Create and activate the virtual environment:**
   ```sh
   python3 -m venv venv
   source venv/bin/activate
   ```
3. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
4. **Run ingestion:**
   ```sh
   python3 ingest.py
   ```
5. **Start the web app:**
   ```sh
   python3 app.py
   ```
6. **Query the vector database (optional):**
   ```sh
   python rag_qa_py -q "Which holidays are observed as paid holidays by the company?"
   ```
