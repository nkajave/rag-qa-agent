from langchain_core.prompts import ChatPromptTemplate

RAG_PROMPT = ChatPromptTemplate.from_template("""
You are a precise document assistant. Answer the question using ONLY
the provided context. If the answer is not in the context, say:
"I don't have enough information in the provided documents to answer this."

Do NOT use any outside knowledge. Cite the source page number
for each fact you state using [Page N] format.

Context:
{context}

Question: {question}

Answer (with page citations):
""")

# For conversational RAG — adds chat history
CONVERSATIONAL_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful document assistant. Use ONLY the
    context below to answer. Cite page numbers as [Page N].
    If unsure, say so — do not hallucinate.\n\nContext:\n{context}"""),
    ("placeholder", "{chat_history}"),
    ("human", "{question}"),
])