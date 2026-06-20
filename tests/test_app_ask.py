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
