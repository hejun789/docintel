"""Tests for the summary-question detection that bypasses the relevance gate.

Broad "summarize the document" questions score low against every individual
chunk, so the relevance gate must NOT reject them. _is_summary_question is the
logic that distinguishes those from specific factual questions.
"""
import pytest

from retriever import _is_summary_question


@pytest.mark.parametrize("question", [
    "Summarize this document",
    "Can you summarize what the document is talking about?",
    "What is the main point of this paper?",
    "Give me an overview of the document",
    "What is this document about?",
    "tl;dr",
    "What are the key takeaways?",
])
def test_summary_questions_detected(question):
    assert _is_summary_question(question) is True


@pytest.mark.parametrize("question", [
    "What is explainability in the framework?",
    "Which classifier gives over 0.99 accuracy?",
    "How does the NLP preprocessing pipeline work?",
    "What does the Decision Fusion Layer combine?",
    "Who is the author of the paper?",
])
def test_specific_questions_not_flagged(question):
    assert _is_summary_question(question) is False


def test_detection_is_case_insensitive():
    assert _is_summary_question("SUMMARIZE THIS") is True
    assert _is_summary_question("Main Point?") is True
