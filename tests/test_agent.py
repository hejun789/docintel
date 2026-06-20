import json

import agent


# ── Task 3: tools & schemas ──────────────────────────────────────────────────

def test_retrieve_tool_marks_strong_results(monkeypatch):
    monkeypatch.setattr(agent, "_retrieve", lambda query, source_filter=None: {
        "chunks": [{"text": "t", "source": "d.pdf", "page_number": 2, "chunk_index": 0, "score": 7.0}],
        "warning": None,
    })
    out = agent.retrieve_tool("q")
    assert out["top_score"] == 7.0
    assert out["results"][0]["page"] == 2
    assert "Relevant" in out["hint"]


def test_retrieve_tool_marks_weak_results(monkeypatch):
    monkeypatch.setattr(agent, "_retrieve", lambda query, source_filter=None: {
        "chunks": [{"text": "t", "source": "d.pdf", "page_number": 1, "chunk_index": 0, "score": -9.0}],
        "warning": "weak",
    })
    out = agent.retrieve_tool("q")
    assert out["top_score"] == -9.0
    assert "reformulate" in out["hint"].lower()


def test_list_documents_tool(monkeypatch):
    monkeypatch.setattr(agent, "_list_documents", lambda: ["a.pdf", "b.pdf"])
    assert agent.list_documents_tool() == {"documents": ["a.pdf", "b.pdf"]}


def test_tool_schemas_present():
    names = {s["function"]["name"] for s in agent.TOOL_SCHEMAS}
    assert names == {"retrieve", "list_documents", "finish"}


# ── Task 4: control loop (also the agent-behaviour eval) ─────────────────────

def _tc(name, args, id="1"):
    return {"id": id, "name": name, "arguments": json.dumps(args)}


def _scripted_chat(script):
    """Returns a chat() that yields each scripted response in turn."""
    it = iter(script)

    def chat(messages, tools=None, model=None):
        return next(it)

    return chat


def _stub_retrieve(monkeypatch, score):
    monkeypatch.setattr(agent, "_retrieve", lambda query, source_filter=None: {
        "chunks": [{"text": "ctx", "source": "d.pdf", "page_number": 1, "chunk_index": 0, "score": score}],
        "warning": None if score >= 0 else "weak",
    })


def test_single_hop_then_finish(monkeypatch):
    _stub_retrieve(monkeypatch, 7.0)
    script = [
        {"content": None, "tool_calls": [_tc("retrieve", {"query": "x"})]},
        {"content": None, "tool_calls": [_tc("finish", {"answer": "A", "citations": [{"source": "d.pdf", "page": 1}]})]},
    ]
    out = agent.run("q", chat=_scripted_chat(script))
    assert out["answer"] == "A"
    assert out["refused"] is False
    assert sum(1 for t in out["trace"] if t["tool"] == "retrieve") == 1


def test_multi_hop_retrieves_twice(monkeypatch):
    _stub_retrieve(monkeypatch, 7.0)
    script = [
        {"content": None, "tool_calls": [_tc("retrieve", {"query": "part1"})]},
        {"content": None, "tool_calls": [_tc("retrieve", {"query": "part2"})]},
        {"content": None, "tool_calls": [_tc("finish", {"answer": "A", "citations": [{"source": "d.pdf", "page": 1}]})]},
    ]
    out = agent.run("q", chat=_scripted_chat(script))
    assert sum(1 for t in out["trace"] if t["tool"] == "retrieve") == 2


def test_off_topic_refusal(monkeypatch):
    _stub_retrieve(monkeypatch, -9.0)
    script = [
        {"content": None, "tool_calls": [_tc("retrieve", {"query": "quantum"})]},
        {"content": None, "tool_calls": [_tc("finish", {"answer": "Not in your documents.", "citations": []})]},
    ]
    out = agent.run("q", chat=_scripted_chat(script))
    assert out["refused"] is True
    assert out["citations"] == []


def test_iteration_cap_forces_refusal(monkeypatch):
    _stub_retrieve(monkeypatch, -9.0)
    always_retrieve = {"content": None, "tool_calls": [_tc("retrieve", {"query": "x"})]}
    script = [always_retrieve] * 10
    out = agent.run("q", chat=_scripted_chat(script), max_iters=3)
    assert out["refused"] is True
    assert sum(1 for t in out["trace"] if t["tool"] == "retrieve") == 3
