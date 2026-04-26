# rag-qa-agent

A fully local, privacy-preserving Retrieval-Augmented Generation (RAG) system that lets you upload PDF documents and ask natural language questions. Answers are grounded in your documents with page-level citations - no hallucinations, no external API calls.
Built with LangChain, Ollama (Llama 3.1 / Mistral), FAISS, and HuggingFace Sentence Transformers. 

rag-qa-agent/
├── app/
│   ├── __init__.py
│   ├── ingest.py          # PDF loading and chunking
│   ├── vectorstore.py     # FAISS store build and load
│   ├── retriever.py       # similarity search
│   ├── chain.py           # LangChain RAG chain
│   └── prompt.py          # prompt templates
├── api/
│   ├── __init__.py
│   └── main.py            # FastAPI endpoints
├── ui/
│   └── streamlit_app.py   # Streamlit chat interface
├── docs/                  # put your PDFs here
├── tests/
│   └── test_chain.py
├── main.py                # CLI entry point
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example

1. Choose one — Llama 3.1 is recommended
ollama pull llama3.1

# Or Mistral for a smaller, faster option
ollama pull mistral

2. Verify Ollama is running:
ollama list

3. Create and activate a virtual environment
python -m venv venv

# Mac / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate

4. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

5. Configure environment variables
cp .env.example .env

Edit .env if needed — default settings work out of the box with Ollama:
envOLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
CHUNK_SIZE=500
CHUNK_OVERLAP=50
TOP_K=4
MLFLOW_TRACKING_URI=./mlflow_tracking

6. Add a PDF
Place any PDF into the docs/ folder:
cp your-document.pdf docs/

Running the Project
Option A — CLI (quickest test)
python main.py
The system indexes your PDFs, then opens an interactive prompt:
RAG Agent ready. Type 'quit' to exit.
Question: What is this document about?
Answer: ...

Option B — Streamlit UI
streamlit run ui/streamlit_app.py
Open http://localhost:8501 in your browser. Upload PDFs from the sidebar, configure retrieval settings, and chat.

How It Works
PDF documents
     │
     ▼
Text chunking (RecursiveCharacterTextSplitter)
     │  chunk_size=500, overlap=50
     ▼
Sentence Transformer embeddings (all-MiniLM-L6-v2)
     │
     ▼
FAISS vector index (persisted to disk)
     │
     │  ◄── User question (embedded with same model)
     ▼
Cosine similarity retrieval (top-k chunks)
     │
     ▼
LangChain RAG chain
     │  context = retrieved chunks + page numbers
     ▼
Ollama LLM (Llama 3.1 / Mistral) — local inference
     │
     ▼
Answer with page citations
