from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from config import (
    CHROMA_DIR, CHROMA_COLLECTION, EMBEDDING_MODEL, RERANKER_MODEL,
    TOP_K_RETRIEVE, TOP_K_FINAL, RELEVANCE_THRESHOLD, USE_RERANKER,
)

_embedder = None
_reranker = None
_vectorstore = None

# Broad "summarize the whole document" style questions don't map to any single
# passage, so the cross-encoder scores every chunk low and the relevance gate
# would wrongly reject them. Detect these and skip the gate.
_SUMMARY_PATTERNS = (
    "summar", "main point", "main idea", "overall", "key point", "key takeaway",
    "what is this document about", "what is the document about",
    "what is this paper about", "what's this about", "tl;dr", "tldr",
    "gist", "in a nutshell", "high level", "high-level", "overview",
)


def _is_summary_question(question: str) -> bool:
    q = question.lower()
    return any(p in q for p in _SUMMARY_PATTERNS)


def _get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return _embedder


def _get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder(RERANKER_MODEL)
    return _reranker


def _get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = Chroma(
            collection_name=CHROMA_COLLECTION,
            embedding_function=_get_embedder(),
            persist_directory=CHROMA_DIR,
        )
    return _vectorstore


def retrieve(question: str, top_k: int = TOP_K_RETRIEVE, source_filter: str = None) -> dict:
    """
    1. Embed the question and fetch top_k candidate chunks from ChromaDB.
    2. Re-rank candidates with a cross-encoder.
    3. Return the top TOP_K_FINAL chunks with their re-ranker scores.

    Returns:
        {"chunks": [...], "warning": str or None}
    """
    vectorstore = _get_vectorstore()

    search_kwargs = {"k": top_k}
    if source_filter:
        search_kwargs["filter"] = {"source": source_filter}

    # LangChain Chroma returns (Document, score) tuples — score is L2 distance
    results = vectorstore.similarity_search_with_score(question, **search_kwargs)

    if not results:
        return {"chunks": [], "warning": "No documents have been uploaded yet."}

    candidates = []
    for doc, distance in results:
        candidates.append({
            "text": doc.page_content,
            "source": doc.metadata["source"],
            "page_number": doc.metadata["page_number"],
            "chunk_index": doc.metadata["chunk_index"],
        })

    if USE_RERANKER:
        reranker = _get_reranker()
        pairs = [(question, c["text"]) for c in candidates]
        scores = reranker.predict(pairs).tolist()
        for chunk, score in zip(candidates, scores):
            chunk["score"] = round(score, 4)
        ranked = sorted(candidates, key=lambda c: c["score"], reverse=True)
    else:
        for chunk, (_, distance) in zip(candidates, results):
            chunk["score"] = round(max(0.0, 1 - distance / 2), 4)
        ranked = sorted(candidates, key=lambda c: c["score"], reverse=True)

    top_chunks = ranked[:TOP_K_FINAL]

    # Summary-type questions score low across the board, so bypass the relevance
    # gate for them and let the LLM synthesize from the top chunks.
    warning = None
    if not _is_summary_question(question) and top_chunks[0]["score"] < RELEVANCE_THRESHOLD:
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
