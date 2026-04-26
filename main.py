from dotenv import load_dotenv
load_dotenv()

from app.ingest import load_documents, split_documents
from app.vectorstore import get_embedding_model, build_vectorstore, load_vectorstore
from app.retriever import build_retriever
from app.chain import build_rag_chain
import os

def main():
    PDF_PATH = "docs/"          # put your PDFs here
    VS_PATH  = "./vectorstore"

    emb = get_embedding_model(use_openai=True)

    if os.path.exists(VS_PATH):
        print("Loading existing vectorstore...")
        vs = load_vectorstore(emb, VS_PATH)
    else:
        print("Building vectorstore from scratch...")
        docs   = load_documents(PDF_PATH)
        chunks = split_documents(docs)
        vs     = build_vectorstore(chunks, emb, VS_PATH)

    retriever = build_retriever(vs, k=4)
    chain     = build_rag_chain(retriever)

    print("\nRAG Agent ready. Type 'quit' to exit.\n")
    while True:
        q = input("Question: ").strip()
        if q.lower() in ("quit", "exit"): break
        if q:
            answer = chain.invoke(q)
            print(f"\nAnswer: {answer}\n")

if __name__ == "__main__":
    main()