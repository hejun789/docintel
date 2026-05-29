import chromadb
from sentence_transformers import SentenceTransformer
from config import (
    CHROMA_DIR, CHROMA_COLLECTION, EMBEDDING_MODEL, RERANKER_MODEL,
    TOP_K_RETRIEVE, TOP_K_FINAL, RELEVANCE_THRESHOLD, USE_RERANKER,
)

_embedder = None
_reranker = None
_collection = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder


def _get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder(RERANKER_MODEL)
    return _reranker


def _get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        _collection = client.get_or_create_collection(CHROMA_COLLECTION)
    return _collection


def retrieve(question: str, top_k: int = TOP_K_RETRIEVE, source_filter: str = None) -> dict:
    """
    1. Embed the question and fetch top_k candidate chunks from ChromaDB.
    2. Re-rank candidates with a cross-encoder.
    3. Return the top TOP_K_FINAL chunks with their re-ranker scores.

    Returns:
        {"chunks": [...], "warning": str or None}
    """
    embedder = _get_embedder()
    collection = _get_collection()

    query_embedding = embedder.encode([question]).tolist()
    query_params = {"query_embeddings": query_embedding, "n_results": top_k}
    if source_filter:
        query_params["where"] = {"source": source_filter}

    results = collection.query(**query_params)

    candidates = []
    for i in range(len(results["documents"][0])):
        meta = results["metadatas"][0][i]
        candidates.append({
            "text": results["documents"][0][i],
            "source": meta["source"],
            "page_number": meta["page_number"],
            "chunk_index": meta["chunk_index"],
        })

    if not candidates:
        return {"chunks": [], "warning": "No documents have been uploaded yet."}

    if USE_RERANKER:
        reranker = _get_reranker()
        pairs = [(question, c["text"]) for c in candidates]
        scores = reranker.predict(pairs).tolist()
        for chunk, score in zip(candidates, scores):
            chunk["score"] = round(score, 4)
        ranked = sorted(candidates, key=lambda c: c["score"], reverse=True)
    else:
        # fallback: use bi-encoder distance converted to similarity score
        for i, chunk in enumerate(candidates):
            distance = results["distances"][0][i]
            chunk["score"] = round(max(0.0, 1 - distance / 2), 4)
        ranked = sorted(candidates, key=lambda c: c["score"], reverse=True)

    top_chunks = ranked[:TOP_K_FINAL]

    warning = None
    if top_chunks[0]["score"] < RELEVANCE_THRESHOLD:
        warning = "No relevant content found in the uploaded documents for this question."

    return {"chunks": top_chunks, "warning": warning}


if __name__ == "__main__":
    import sys

    question = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "what is the network topology used?"
    print(f"Question: {question}\n")

    result = retrieve(question)

    if result["warning"]:
        print(f"Warning: {result['warning']}\n")

    for c in result["chunks"]:
        print(f"Score: {c['score']} | Page {c['page_number']} | {c['source']}")
        print(c["text"][:300])
        print()
