from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_documents(path: str):
    if path.endswith(".pdf"):
        loader = PyPDFLoader(path)
    else:
        loader = DirectoryLoader(path, glob="**/*.pdf",
                                 loader_cls=PyPDFLoader)
    return loader.load()

def split_documents(docs, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks from {len(docs)} documents")
    return chunks