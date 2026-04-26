import os
from dotenv import load_dotenv

load_dotenv()

print("API KEY LOADED:", os.getenv("OPENAI_API_KEY"))


import streamlit as st
from app.ingest import load_documents, split_documents
from app.vectorstore import get_embedding_model, build_vectorstore
from app.retriever import build_retriever
from app.chain import build_rag_chain
import tempfile, os

st.set_page_config(page_title="RAG Document Q&A", page_icon="📄")
st.title("Document Q&A Agent")

# Session state for chain persistence
if "chain" not in st.session_state:
    st.session_state.chain = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar — upload
with st.sidebar:
    st.header("Upload Documents")
    uploaded = st.file_uploader("Upload PDFs", type=["pdf"],
                                 accept_multiple_files=True)
    k_val = st.slider("Chunks to retrieve (k)", 2, 8, 4)
    model = st.selectbox("LLM", ["gpt-4o-mini", "gpt-4o"])

    if st.button("Process Documents") and uploaded:
        with st.spinner("Indexing..."):
            with tempfile.TemporaryDirectory() as tmp:
                paths = []
                for f in uploaded:
                    p = os.path.join(tmp, f.name)
                    with open(p, "wb") as out: out.write(f.read())
                    paths.append(p)
                docs = []
                for p in paths:
                    docs.extend(load_documents(p))
                chunks = split_documents(docs)
                emb = get_embedding_model(use_openai=True)
                vs = build_vectorstore(chunks, emb)
                ret = build_retriever(vs, k=k_val)
                st.session_state.chain = build_rag_chain(ret, model=model)
        st.success(f"Indexed {len(chunks)} chunks!")

# Chat interface
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("Ask a question about your documents..."):
    if not st.session_state.chain:
        st.warning("Please upload and process documents first.")
    else:
        st.session_state.messages.append({"role":"user","content":prompt})
        with st.chat_message("user"):
            st.write(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = st.session_state.chain.invoke(prompt)
            st.write(answer)
        st.session_state.messages.append({"role":"assistant","content":answer})