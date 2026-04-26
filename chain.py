from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from app.prompt import RAG_PROMPT
import mlflow



def format_docs(docs):
    return "\n\n".join(
        f"[Page {doc.metadata.get('page', '?')}]\n{doc.page_content}"
        for doc in docs
    )

def build_rag_chain(retriever, model="gpt-4o-mini", temperature=0):
    llm = ChatOllama(
    model="llama3",
    temperature=0
   )

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )
    return chain

def query_with_tracking(chain, question: str, run_name="rag_query"):
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("question", question)
        answer = chain.invoke(question)
        mlflow.log_param("answer_length", len(answer))
        return answer