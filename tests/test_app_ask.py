import app as app_module


def test_ask_streams_agent_answer(monkeypatch):
    monkeypatch.setattr(app_module.agent, "run", lambda question, source_filter=None, history=None: {
        "answer": "Grounded answer.",
        "citations": [{"source": "d.pdf", "page": 3}],
        "trace": [{"tool": "retrieve", "args": {"query": "q"}, "result": {}}],
        "refused": False,
    })
    client = app_module.app.test_client()
    resp = client.post("/ask", json={"question": "What is X?"})
    body = resp.get_data(as_text=True)

    assert "Grounded answer." in body
    assert "d.pdf" in body
    compact = body.replace(" ", "")
    assert '"done":true' in compact


def test_ask_rejects_empty_question():
    client = app_module.app.test_client()
    resp = client.post("/ask", json={"question": "   "})
    assert resp.status_code == 400


def test_ask_rejects_missing_question():
    client = app_module.app.test_client()
    resp = client.post("/ask", json={})
    assert resp.status_code == 400


def test_ask_handles_agent_error_gracefully(monkeypatch):
    def boom(question, source_filter=None, history=None):
        raise RuntimeError("OpenRouter 429")

    monkeypatch.setattr(app_module.agent, "run", boom)
    client = app_module.app.test_client()
    resp = client.post("/ask", json={"question": "What is X?"})

    # Should NOT 500 — returns a streamed, graceful message instead.
    assert resp.status_code == 200
    assert "went wrong" in resp.get_data(as_text=True)
