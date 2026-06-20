"""Faithfulness / answer-quality eval using an LLM-as-judge.

Runs the agent on a labeled question set and asks a judge model to score each answer
on groundedness, relevance, and correctness. Prints an aggregate report.

Run from the project root (with the eval document ingested):
    python eval/faithfulness.py
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import agent
import llm
from config import JUDGE_MODEL

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "faithfulness_set.json")

JUDGE_PROMPT = """You are grading a document-grounded answer.

Question: {question}
Reference answer: {reference}
Model answer: {answer}

Score 0.0-1.0 on three axes and reply with ONLY a JSON object:
- groundedness: is every claim consistent with the reference (no invented facts)?
- relevance: does it address the question?
- correctness: does it match the reference answer?

Example: {{"groundedness": 0.9, "relevance": 1.0, "correctness": 0.8}}"""


def parse_judge_scores(text):
    """Extract the three scores from the judge's reply, tolerating surrounding prose."""
    match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    data = json.loads(match.group(0)) if match else {}
    return {k: float(data.get(k, 0.0)) for k in ("groundedness", "relevance", "correctness")}


def judge(question, reference, answer):
    reply = llm.chat(
        [{"role": "user", "content": JUDGE_PROMPT.format(
            question=question, reference=reference, answer=answer)}],
        model=JUDGE_MODEL,
    )
    return parse_judge_scores(reply["content"] or "")


def evaluate():
    with open(DATA, encoding="utf-8") as f:
        data = json.load(f)

    rows, totals = [], {"groundedness": 0.0, "relevance": 0.0, "correctness": 0.0}
    for q in data["questions"]:
        result = agent.run(q["question"])
        scores = judge(q["question"], q["reference"], result["answer"])
        for k in totals:
            totals[k] += scores[k]
        rows.append((q["question"], scores, result["refused"]))

    n = len(data["questions"])
    print(f"\nFaithfulness eval - {n} questions on '{data['document']}'\n")
    print(f"{'Question':<52}{'Ground':<9}{'Relev':<9}{'Correct':<9}")
    print("-" * 79)
    for question, s, refused in rows:
        tag = "  [refused]" if refused else ""
        print(f"{question[:50]:<52}{s['groundedness']:<9.2f}{s['relevance']:<9.2f}{s['correctness']:<9.2f}{tag}")
    print("-" * 79)
    print(f"{'AVERAGE':<52}{totals['groundedness']/n:<9.2f}{totals['relevance']/n:<9.2f}{totals['correctness']/n:<9.2f}")


if __name__ == "__main__":
    evaluate()
