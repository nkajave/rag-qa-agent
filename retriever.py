def build_retriever(vectorstore, k=4, search_type="similarity"):
    """
    search_type options:
      'similarity'        — pure cosine similarity (default)
      'mmr'               — max marginal relevance (diversity)
      'similarity_score_threshold' — filter by score
    """
    return vectorstore.as_retriever(
        search_type=search_type,
        search_kwargs={
            "k": k,
            "score_threshold": 0.5  # only for score_threshold mode
        }
    )

# Test retrieval directly
def test_retrieval(retriever, query: str):
    docs = retriever.invoke(query)
    for i, doc in enumerate(docs):
        print(f"\n--- Chunk {i+1} (page {doc.metadata.get('page','?')}) ---")
        print(doc.page_content[:300])
    return docs