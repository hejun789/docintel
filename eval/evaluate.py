"""
Retrieval evaluation harness for DocIntel.

Measures how well the retrieval pipeline surfaces the relevant chunk for a set
of labeled questions, and quantifies the value of the cross-encoder re-ranker
by comparing Hit@k with re-ranking ON vs OFF.

Run from the project root (with the eval document already ingested):
    python eval/evaluate.py

Metrics:
  - Recall@N      : a gold chunk appears anywhere in the N bi-encoder candidates.
  - Hit@k (bi)    : a gold chunk is in the top-k by bi-encoder distance alone.
  - Hit@k (rerank): a gold chunk is in the top-k after cross-encoder re-ranking.
  - MRR           : mean reciprocal rank of the first gold chunk (rerank order).
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import TOP_K_RETRIEVE
from retriever import _get_vectorstore, _get_reranker

EVAL_SET = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_set.json")
# Fixed at 3 so Hit@3 stays a stable, comparable metric independent of the
# production TOP_K_FINAL setting (which can be tuned without changing the eval).
K = 3


def _is_gold(chunk_text: str, expects: list[str]) -> bool:
    """A chunk is the relevant (gold) chunk if it contains any expected substring."""
    low = chunk_text.lower()
    return any(e.lower() in low for e in expects)


def _first_gold_rank(chunks: list[str], expects: list[str]) -> int | None:
    """1-based rank of the first gold chunk in an ordered list, or None if absent."""
    for i, text in enumerate(chunks):
        if _is_gold(text, expects):
            return i + 1
    return None


def evaluate():
    with open(EVAL_SET, encoding="utf-8") as f:
        data = json.load(f)

    questions = data["questions"]
    vectorstore = _get_vectorstore()
    reranker = _get_reranker()

    if vectorstore._collection.count() == 0:
        sys.exit("No chunks in the vector store. Ingest the eval document first.")

    rows = []
    recall_hits = bi_hits = rerank_hits = 0
    rr_sum = 0.0  # for MRR over rerank order

    for q in questions:
        question, expects = q["question"], q["expects"]

        # Stage 1: bi-encoder candidates (vector distance order)
        results = vectorstore.similarity_search_with_score(question, k=TOP_K_RETRIEVE)
        cand_texts = [doc.page_content for doc, _ in results]

        # Stage 2: cross-encoder re-ranking of those same candidates
        scores = reranker.predict([(question, t) for t in cand_texts]).tolist()
        reranked = [t for t, _ in sorted(zip(cand_texts, scores), key=lambda p: p[1], reverse=True)]

        recall_rank = _first_gold_rank(cand_texts, expects)        # anywhere in candidates
        bi_rank = _first_gold_rank(cand_texts[:K], expects)        # bi-encoder top-K
        rr_rank = _first_gold_rank(reranked, expects)              # full reranked order
        rerank_topk = _first_gold_rank(reranked[:K], expects)      # reranked top-K

        recall_hits += recall_rank is not None
        bi_hits += bi_rank is not None
        rerank_hits += rerank_topk is not None
        rr_sum += (1.0 / rr_rank) if rr_rank else 0.0

        rows.append({
            "q": question,
            "page": q.get("page", "?"),
            "recall": "✓" if recall_rank else "✗",
            "bi": bi_rank if bi_rank else "–",
            "rr": rerank_topk if rerank_topk else "–",
        })

    n = len(questions)
    print(f"\nRetrieval evaluation — {n} questions on '{data['document']}'")
    print(f"Pipeline: top-{TOP_K_RETRIEVE} bi-encoder candidates -> cross-encoder rerank -> top-{K} to LLM\n")

    print(f"{'Q (page)':<58}{'Recall@'+str(TOP_K_RETRIEVE):<12}{'bi top'+str(K):<10}{'rr top'+str(K):<10}")
    print("-" * 90)
    for r in rows:
        label = f"{r['q'][:48]}  (p{r['page']})"
        print(f"{label:<58}{r['recall']:<12}{str(r['bi']):<10}{str(r['rr']):<10}")

    print("-" * 90)
    print(f"\nRecall@{TOP_K_RETRIEVE}              : {recall_hits}/{n}  ({recall_hits/n:.0%})  — gold chunk retrieved at all")
    print(f"Hit@{K} (bi-encoder)   : {bi_hits}/{n}  ({bi_hits/n:.0%})  — gold in top-{K} WITHOUT reranking")
    print(f"Hit@{K} (re-ranked)    : {rerank_hits}/{n}  ({rerank_hits/n:.0%})  — gold in top-{K} WITH reranking")
    print(f"MRR (re-ranked)      : {rr_sum/n:.3f}")

    delta = rerank_hits - bi_hits
    sign = "+" if delta >= 0 else ""
    print(f"\nRe-ranker effect on Hit@{K}: {sign}{delta} questions "
          f"({bi_hits/n:.0%} -> {rerank_hits/n:.0%})")


if __name__ == "__main__":
    evaluate()
