
from dotenv import load_dotenv
import os
load_dotenv()

from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS, Chroma

def get_embedding_model(use_openai=True):
    if use_openai:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
           model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    # Free alternative — no API key needed
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

def build_vectorstore(chunks, embedding_model, store_path="./vectorstore"):
    vectorstore = FAISS.from_documents(chunks, embedding_model)
    vectorstore.save_local(store_path)
    print(f"Saved {len(chunks)} vectors to {store_path}")
    return vectorstore

def load_vectorstore(embedding_model, store_path="./vectorstore"):
    return FAISS.load_local(store_path, embedding_model,
                            allow_dangerous_deserialization=True)