# Agentic DocIntel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace DocIntel's fixed `retrieve → rerank → generate` pipeline with a tool-using agent that decides at runtime which tools to call, multi-hops on complex questions, self-corrects on weak retrieval, and refuses when the documents lack the answer.

**Architecture:** A hand-written agent control loop (`agent.py`) drives the model through the OpenRouter native function-calling API (`llm.py`). The agent calls three tools — `retrieve`, `list_documents`, `finish` — that wrap the existing, already-tested retrieval core. Evaluation adds a faithfulness (LLM-as-judge) layer and deterministic agent-behaviour tests. No agent framework (no LangGraph/AgentExecutor).

**Tech Stack:** Python, Flask, OpenAI SDK (pointed at OpenRouter), existing LangChain/Chroma/sentence-transformers retrieval core, pytest.

## Global Constraints

- Strictly document-grounded: every non-refusal answer must carry page citations; the agent never invents facts. (verbatim from spec)
- No agent framework — custom loop on native function-calling. (verbatim from spec)
- Reuse existing modules (`retriever.py`, `ingest.py`) as tools; do not rewrite them. (verbatim from spec)
- Agent loop has an iteration cap (default 5) to bound cost. (verbatim from spec)
- Do NOT add a `Co-Authored-By: Claude` trailer to any commit in this repo (user removed it deliberately; re-adding re-creates the GitHub contributor entry).
- Existing retrieval eval (`eval/evaluate.py`) and unit tests must still pass.
- Python deps pinned in `requirements.txt`; dev-only deps in `requirements-dev.txt`.

## File Structure

- **Create `llm.py`** — thin OpenRouter chat client supporting tool-calling; returns a normalized dict `{content, tool_calls}`. Single responsibility: talk to the model.
- **Create `agent.py`** — tool wrappers, tool schemas, tool registry, and the `run()` control loop. Single responsibility: agent orchestration.
- **Modify `config.py`** — add `AGENT_MODEL`, `JUDGE_MODEL`, `AGENT_MAX_ITERS`.
- **Modify `app.py`** — `/ask` delegates to `agent.run()` and streams the result over the existing SSE contract.
- **Create `eval/faithfulness_set.json`** — labeled Q&A with reference answers.
- **Create `eval/faithfulness.py`** — LLM-as-judge faithfulness eval + score parser + report.
- **Create `tests/test_llm.py`** — unit test the client (mocked SDK).
- **Create `tests/test_agent.py`** — unit tests for tools + the control loop (these double as the agent-behaviour eval, using a scripted fake chat).
- **Create `tests/test_app_ask.py`** — Flask test-client test for `/ask` (mocked `agent.run`).
- **Create `tests/test_faithfulness.py`** — unit test the judge-score parser (mocked judge output).
- **Modify `requirements.txt` / `requirements-dev.txt`** — add `openai`.
- **Modify `README.md`** — document the agent architecture, eval, run steps.

---

### Task 1: Config additions

**Files:**
- Modify: `config.py`

**Interfaces:**
- Produces: `AGENT_MODEL: str`, `JUDGE_MODEL: str`, `AGENT_MAX_ITERS: int`

- [ ] **Step 1: Add config values**

Append to `config.py` (after the existing `OPENROUTER_MODEL` line):

```python
# Agent / judge models (default to the main model; override per-env for tool-calling-capable models)
AGENT_MODEL = os.getenv("AGENT_MODEL", OPENROUTER_MODEL)
JUDGE_MODEL = os.getenv("JUDGE_MODEL", OPENROUTER_MODEL)
AGENT_MAX_ITERS = int(os.getenv("AGENT_MAX_ITERS", "5"))
```

- [ ] **Step 2: Verify it imports**

Run: `python -c "import config; print(config.AGENT_MODEL, config.AGENT_MAX_ITERS)"`
Expected: prints the model name and `5` with no error.

- [ ] **Step 3: Commit**

```bash
git add config.py
git commit -m "feat: add agent/judge model and iteration-cap config"
```

---

### Task 2: LLM client with tool-calling (`llm.py`)

**Files:**
- Create: `llm.py`
- Test: `tests/test_llm.py`

**Interfaces:**
- Consumes: `config.OPENROUTER_API_KEY`, `config.AGENT_MODEL`
- Produces: `chat(messages: list[dict], tools: list[dict] | None = None, model: str | None = None) -> dict` returning `{"content": str | None, "tool_calls": list[{"id": str, "name": str, "arguments": str}]}`

- [ ] **Step 1: Write the failing test**

Create `tests/test_llm.py`:

```python
import llm


class _FakeFunction:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class _FakeToolCall:
    def __init__(self, id, name, arguments):
        self.id = id
        self.type = "function"
        self.function = _FakeFunction(name, arguments)


class _FakeMessage:
    def __init__(self, content, tool_calls):
        self.content = content
        self.tool_calls = tool_calls


class _FakeChoice:
    def __init__(self, message):
        self.message = message


class _FakeResponse:
    def __init__(self, message):
        self.choices = [_FakeChoice(message)]


class _FakeClient:
    def __init__(self, message):
        self._message = message

        class _Completions:
            def create(inner_self, **kwargs):
                self.kwargs = kwargs
                return _FakeResponse(self._message)

        class _Chat:
            completions = _Completions()

        self.chat = _Chat()


def test_chat_normalizes_tool_calls(monkeypatch):
    msg = _FakeMessage(None, [_FakeToolCall("call_1", "retrieve", '{"query": "x"}')])
    fake = _FakeClient(msg)
    monkeypatch.setattr(llm, "_get_client", lambda: fake)

    out = llm.chat([{"role": "user", "content": "hi"}], tools=[{"type": "function"}])

    assert out["content"] is None
    assert out["tool_calls"] == [
        {"id": "call_1", "name": "retrieve", "arguments": '{"query": "x"}'}
    ]


def test_chat_normalizes_plain_content(monkeypatch):
    msg = _FakeMessage("hello", None)
    fake = _FakeClient(msg)
    monkeypatch.setattr(llm, "_get_client", lambda: fake)

    out = llm.chat([{"role": "user", "content": "hi"}])

    assert out["content"] == "hello"
    assert out["tool_calls"] == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_llm.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'llm'`

- [ ] **Step 3: Write minimal implementation**

Create `llm.py`:

```python
"""Thin OpenRouter chat client with native tool-calling, normalized to plain dicts."""
from openai import OpenAI

from config import OPENROUTER_API_KEY, AGENT_MODEL

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)
    return _client


def chat(messages, tools=None, model=None):
    """Call the model once. Returns {'content': str|None, 'tool_calls': [...]}.

    Each tool_call is {'id', 'name', 'arguments'} where arguments is a raw JSON string.
    """
    kwargs = {"model": model or AGENT_MODEL, "messages": messages}
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"

    message = _get_client().chat.completions.create(**kwargs).choices[0].message
    tool_calls = [
        {"id": tc.id, "name": tc.function.name, "arguments": tc.function.arguments}
        for tc in (message.tool_calls or [])
    ]
    return {"content": message.content, "tool_calls": tool_calls}
```

- [ ] **Step 4: Install the SDK and pin it**

Run: `pip install openai`
Then: `python -c "import openai, re; print(openai.__version__)"`
Add the printed version to `requirements.txt` as `openai==<version>` (e.g. `openai==1.59.0`).

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_llm.py -v`
Expected: PASS (2 passed)

- [ ] **Step 6: Commit**

```bash
git add llm.py tests/test_llm.py requirements.txt
git commit -m "feat: add OpenRouter tool-calling client (llm.py)"
```

---

### Task 3: Agent tools + schemas (`agent.py` part 1)

**Files:**
- Create: `agent.py`
- Test: `tests/test_agent.py`

**Interfaces:**
- Consumes: `retriever.retrieve`, `ingest.list_documents`, `config.RELEVANCE_THRESHOLD`
- Produces:
  - `retrieve_tool(query: str, source_filter: str | None = None) -> dict` → `{"results": [{"text","source","page","score"}], "top_score": float|None, "hint": str}`
  - `list_documents_tool() -> dict` → `{"documents": list[str]}`
  - `TOOL_SCHEMAS: list[dict]` (retrieve, list_documents, finish)
  - `TOOLS: dict[str, callable]` (retrieve, list_documents)

- [ ] **Step 1: Write the failing test**

Create `tests/test_agent.py`:

```python
import agent


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_agent.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'agent'`

- [ ] **Step 3: Write minimal implementation**

Create `agent.py`:

```python
"""Tool-using agent: tools, schemas, and the control loop (added in Task 4)."""
from config import RELEVANCE_THRESHOLD
from retriever import retrieve as _retrieve
from ingest import list_documents as _list_documents


def retrieve_tool(query, source_filter=None):
    """Two-stage retrieval over the uploaded documents."""
    result = _retrieve(query, source_filter=source_filter)
    chunks = result["chunks"]
    top_score = chunks[0]["score"] if chunks else None
    weak = top_score is None or top_score < RELEVANCE_THRESHOLD
    return {
        "results": [
            {"text": c["text"], "source": c["source"], "page": c["page_number"], "score": c["score"]}
            for c in chunks
        ],
        "top_score": top_score,
        "hint": (
            "Low relevance — if these passages don't answer the question, reformulate the "
            "query and call retrieve again."
            if weak else
            "Relevant passages found."
        ),
    }


def list_documents_tool():
    """List the filenames currently available to answer from."""
    return {"documents": _list_documents()}


TOOLS = {"retrieve": retrieve_tool, "list_documents": list_documents_tool}

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "retrieve",
            "description": (
                "Search the uploaded documents for passages relevant to a query. "
                "Call multiple times with different queries for multi-part questions. "
                "If results are low-relevance, reformulate and call again."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "A focused search query."},
                    "source_filter": {"type": "string", "description": "Optional filename to restrict the search."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_documents",
            "description": "List the filenames of all uploaded documents available to answer from.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": (
                "Provide the final answer. Every claim must be supported by retrieved passages. "
                "If the documents do not contain the answer, set answer to an explicit refusal and "
                "citations to an empty list."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {"type": "string", "description": "The grounded answer, or an explicit refusal."},
                    "citations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "source": {"type": "string"},
                                "page": {"type": "integer"},
                            },
                            "required": ["source", "page"],
                        },
                        "description": "Source+page for each passage used. Empty if refusing.",
                    },
                },
                "required": ["answer", "citations"],
            },
        },
    },
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_agent.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add agent.py tests/test_agent.py
git commit -m "feat: add agent tools and tool schemas"
```

---

### Task 4: Agent control loop (`agent.py` part 2) — also the agent-behaviour eval

**Files:**
- Modify: `agent.py`
- Test: `tests/test_agent.py`

**Interfaces:**
- Consumes: `llm.chat` (injectable), `TOOLS`, `TOOL_SCHEMAS`, `config.AGENT_MAX_ITERS`
- Produces: `run(question: str, source_filter: str | None = None, history: list | None = None, chat=llm.chat, max_iters=AGENT_MAX_ITERS) -> {"answer": str, "citations": list, "trace": list, "refused": bool}`

- [ ] **Step 1: Write the failing tests (these are the agent-behaviour eval)**

Append to `tests/test_agent.py`:

```python
import json
import agent


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
    # Always asks to retrieve, never finishes.
    always_retrieve = {"content": None, "tool_calls": [_tc("retrieve", {"query": "x"})]}
    script = [always_retrieve] * 10
    out = agent.run("q", chat=_scripted_chat(script), max_iters=3)
    assert out["refused"] is True
    assert sum(1 for t in out["trace"] if t["tool"] == "retrieve") == 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_agent.py -k "hop or refusal or cap" -v`
Expected: FAIL with `AttributeError: module 'agent' has no attribute 'run'`

- [ ] **Step 3: Write the control loop**

Append to `agent.py`:

```python
import json

import llm
from config import AGENT_MAX_ITERS

SYSTEM_PROMPT = """You are a document-grounded research assistant.

You answer ONLY from the uploaded documents. Use the `retrieve` tool to find relevant
passages; for multi-part questions, call it several times with different queries. If a
retrieval is low-relevance, reformulate the query and retrieve again before giving up.

When you can answer, call `finish` with the answer and a citation (source + page) for
every passage you used. If, after searching, the documents do not contain the answer,
call `finish` with an explicit refusal and an empty citations list. Never use outside
knowledge. Never invent citations."""

_REFUSAL = "I couldn't find enough relevant information in your documents to answer that."


def _assistant_message(resp):
    msg = {"role": "assistant", "content": resp["content"]}
    if resp["tool_calls"]:
        msg["tool_calls"] = [
            {"id": t["id"], "type": "function",
             "function": {"name": t["name"], "arguments": t["arguments"]}}
            for t in resp["tool_calls"]
        ]
    return msg


def run(question, source_filter=None, history=None, chat=llm.chat, max_iters=AGENT_MAX_ITERS):
    """Drive the agent loop. Returns {answer, citations, trace, refused}."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages += history[-6:]
    messages.append({"role": "user", "content": question})

    trace = []
    for _ in range(max_iters):
        resp = chat(messages, tools=TOOL_SCHEMAS)

        if not resp["tool_calls"]:
            # Model answered without calling finish — accept its content as the answer.
            return {"answer": resp["content"] or _REFUSAL, "citations": [],
                    "trace": trace, "refused": not resp["content"]}

        messages.append(_assistant_message(resp))

        for call in resp["tool_calls"]:
            name = call["name"]
            try:
                args = json.loads(call["arguments"] or "{}")
            except json.JSONDecodeError:
                args = {}

            if name == "finish":
                citations = args.get("citations", [])
                return {"answer": args.get("answer", _REFUSAL), "citations": citations,
                        "trace": trace, "refused": len(citations) == 0}

            func = TOOLS.get(name)
            result = func(**args) if func else {"error": f"unknown tool: {name}"}
            trace.append({"tool": name, "args": args, "result": result})
            messages.append({"role": "tool", "tool_call_id": call["id"],
                             "name": name, "content": json.dumps(result)})

    # Iteration cap reached without finishing → grounded refusal.
    return {"answer": _REFUSAL, "citations": [], "trace": trace, "refused": True}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_agent.py -v`
Expected: PASS (8 passed — 4 from Task 3 + 4 here)

- [ ] **Step 5: Commit**

```bash
git add agent.py tests/test_agent.py
git commit -m "feat: add agent control loop with multi-hop, refusal, and iteration cap"
```

---

### Task 5: Wire the agent into `/ask`

**Files:**
- Modify: `app.py` (the `/ask` route)
- Test: `tests/test_app_ask.py`

**Interfaces:**
- Consumes: `agent.run`
- Produces: SSE stream — one `{"token": answer}` event then one `{"done": true, "sources": citations, "trace": trace}` event.

- [ ] **Step 1: Write the failing test**

Create `tests/test_app_ask.py`:

```python
import json
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
    assert '"done": true' in body.lower().replace(" ", "") or '"done":true' in body.replace(" ", "")


def test_ask_rejects_empty_question():
    client = app_module.app.test_client()
    resp = client.post("/ask", json={"question": "   "})
    assert resp.status_code == 400
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_app_ask.py -v`
Expected: FAIL (current `/ask` calls `retrieve`/`generate_answer_stream`, not `agent.run`; `app_module.agent` does not exist yet)

- [ ] **Step 3: Update the `/ask` route**

In `app.py`, add to the imports at the top:

```python
import agent
```

Replace the existing `ask()` view function body with:

```python
@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()

    if not data or "question" not in data:
        return jsonify({"error": "Request body must include a 'question' field"}), 400

    question = data["question"].strip()
    if not question:
        return jsonify({"error": "Question cannot be empty"}), 400

    history = data.get("history", [])
    source_filter = data.get("document") or None

    result = agent.run(question, source_filter=source_filter, history=history)

    def stream():
        import json
        yield f"data: {json.dumps({'token': result['answer']})}\n\n"
        yield f"data: {json.dumps({'done': True, 'sources': result['citations'], 'trace': result['trace']})}\n\n"

    return Response(stream_with_context(stream()), content_type="text/event-stream")
```

(The old `from retriever import retrieve` / `from generator import generate_answer_stream` imports may stay; they are now unused by `/ask` but harmless.)

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_app_ask.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Run the full suite to confirm nothing broke**

Run: `python -m pytest tests/ -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add app.py tests/test_app_ask.py
git commit -m "feat: route /ask through the agent loop"
```

---

### Task 6: Faithfulness eval (LLM-as-judge)

**Files:**
- Create: `eval/faithfulness_set.json`
- Create: `eval/faithfulness.py`
- Test: `tests/test_faithfulness.py`

**Interfaces:**
- Consumes: `agent.run`, `llm.chat`, `config.JUDGE_MODEL`
- Produces: `parse_judge_scores(text: str) -> dict` → `{"groundedness": float, "relevance": float, "correctness": float}`; `evaluate()` prints an aggregate report.

- [ ] **Step 1: Create the labeled dataset**

Create `eval/faithfulness_set.json` (reference answers from the sample document):

```json
{
  "document": "HeJun(TP086047) Assignment Part1.docx",
  "questions": [
    {
      "question": "What three detection techniques does PhishGuard-LLM combine?",
      "reference": "Fine-tuned RoBERTa classification, zero-shot GPT reasoning, and AI-content fingerprinting (perplexity/burstiness)."
    },
    {
      "question": "How does the system detect AI-generated content before classification?",
      "reference": "It measures perplexity and burstiness to flag machine-generated text before any classifier runs."
    },
    {
      "question": "What does the Decision Fusion Layer do?",
      "reference": "It combines the three detection modules' outputs via a weighted confidence-based scoring system."
    },
    {
      "question": "What is the average cost of a phishing data breach in 2024?",
      "reference": "USD 4.88 million."
    },
    {
      "question": "What explainability methods does the framework use?",
      "reference": "SHAP values and attention-based visualisation."
    }
  ]
}
```

- [ ] **Step 2: Write the failing test for the parser**

Create `tests/test_faithfulness.py`:

```python
import eval_faithfulness_import as fa  # noqa  -- placeholder, replaced below
```

Replace that file's contents with:

```python
import importlib.util
import os

# Load eval/faithfulness.py as a module (the eval/ dir is not a package).
_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "eval", "faithfulness.py")
_spec = importlib.util.spec_from_file_location("faithfulness", _path)
faithfulness = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(faithfulness)


def test_parse_judge_scores_extracts_floats():
    text = '{"groundedness": 0.9, "relevance": 1.0, "correctness": 0.8}'
    out = faithfulness.parse_judge_scores(text)
    assert out == {"groundedness": 0.9, "relevance": 1.0, "correctness": 0.8}


def test_parse_judge_scores_tolerates_surrounding_text():
    text = 'Here is my rating:\n{"groundedness": 0.5, "relevance": 0.6, "correctness": 0.7}\nThanks.'
    out = faithfulness.parse_judge_scores(text)
    assert out["groundedness"] == 0.5
    assert out["correctness"] == 0.7
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/test_faithfulness.py -v`
Expected: FAIL (file `eval/faithfulness.py` does not exist)

- [ ] **Step 4: Write the eval script**

Create `eval/faithfulness.py`:

```python
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

Score 0.0–1.0 on three axes and reply with ONLY a JSON object:
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
    print(f"\nFaithfulness eval — {n} questions on '{data['document']}'\n")
    print(f"{'Question':<52}{'Ground':<9}{'Relev':<9}{'Correct':<9}")
    print("-" * 79)
    for question, s, refused in rows:
        tag = "  [refused]" if refused else ""
        print(f"{question[:50]:<52}{s['groundedness']:<9.2f}{s['relevance']:<9.2f}{s['correctness']:<9.2f}{tag}")
    print("-" * 79)
    print(f"{'AVERAGE':<52}{totals['groundedness']/n:<9.2f}{totals['relevance']/n:<9.2f}{totals['correctness']/n:<9.2f}")


if __name__ == "__main__":
    evaluate()
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_faithfulness.py -v`
Expected: PASS (2 passed)

- [ ] **Step 6: Commit**

```bash
git add eval/faithfulness.py eval/faithfulness_set.json tests/test_faithfulness.py
git commit -m "feat: add faithfulness LLM-as-judge eval"
```

---

### Task 7: Manual end-to-end verification + README

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: everything above. No new code interfaces.

- [ ] **Step 0 (smoke test FIRST): confirm the model actually emits tool_calls**

Before relying on it, verify the free model returns structured tool calls (some advertise
tools but emit them flakily):
```bash
python -c "import llm; r=llm.chat([{'role':'user','content':'List my documents.'}], tools=__import__('agent').TOOL_SCHEMAS); print(r)"
```
Expected: the response dict contains a non-empty `tool_calls` list. If it's empty/erratic,
switch `AGENT_MODEL` to another free tool-caller (e.g. `openai/gpt-oss-20b:free`,
`google/gemma-4-31b-it:free`) and retry.

- [ ] **Step 1: Set a tool-calling model and run the app**

Ensure `.env` has a function-calling-capable FREE model:
```
AGENT_MODEL=openai/gpt-oss-120b:free
JUDGE_MODEL=openai/gpt-oss-20b:free
AGENT_MAX_ITERS=4
```
Note: agents make several calls per question, so on the free tier (50 requests/day with no
credits, 1000/day with a one-time $10 top-up) keep the eval set small if you stay at $0.
Then run:
```bash
python app.py
```
Upload the sample document and ask: (a) a specific question, (b) a multi-part question, (c) an off-topic question. Confirm: specific answers with citations, multi-part triggers multiple retrievals (check the trace in the SSE response), off-topic refuses.

- [ ] **Step 2: Run the faithfulness eval**

Run: `python eval/faithfulness.py`
Expected: a table of groundedness/relevance/correctness scores and an average row. Note the averages for the README.

- [ ] **Step 3: Run the full test suite**

Run: `python -m pytest tests/ -q`
Expected: all pass (existing + new agent/llm/app/faithfulness tests).

- [ ] **Step 4: Update the README**

Add an "Agent" section to `README.md` (after the "How it works" section) describing: the agent loop, the three tools, multi-hop + self-correction + grounded refusal, and "no agent framework — native function-calling + custom loop." Add the faithfulness averages from Step 2 to the Evaluation section. Add a note that the agent-behaviour cases are covered by `tests/test_agent.py`.

- [ ] **Step 5: Commit**

```bash
git add README.md
git commit -m "docs: document agent architecture and faithfulness eval"
```

---

## Self-Review

**Spec coverage:**
- Agent control loop → Task 4 ✓
- Three tools (retrieve/list_documents/finish) → Task 3 ✓
- Native function-calling, no framework → Task 2 (`llm.py`) + Task 4 ✓
- Self-correcting retrieval (weak-score hint → reformulate) → Task 3 `retrieve_tool` hint + Task 4 loop ✓
- Grounded refusal → Task 4 (`finish` empty citations + iteration cap) ✓
- Faithfulness self-check v1 (finish requires citations; system prompt enforces grounding) → Task 3 schema + Task 4 prompt ✓
- Trace capture → Task 4 ✓
- Retrieval eval (existing) → unchanged, re-run in Task 7 ✓
- Faithfulness eval (LLM-judge) → Task 6 ✓
- Agent-behaviour eval (trace assertions) → Task 4 tests ✓
- Cost bound (iteration cap) → Task 1 config + Task 4 ✓
- No Co-Authored-By trailer → noted in Global Constraints; commit messages above omit it ✓

**Placeholder scan:** No "TBD"/"handle edge cases"/"similar to". The `test_faithfulness.py` Step 2 placeholder line is explicitly replaced in the same step. Each code step shows full code.

**Type consistency:** `chat()` returns `{content, tool_calls}` (Task 2) and is consumed identically in Task 4. `retrieve_tool`/`list_documents_tool` shapes (Task 3) match what the loop serializes (Task 4). `agent.run` return shape `{answer, citations, trace, refused}` (Task 4) matches `/ask` consumption (Task 5) and the faithfulness eval (Task 6).

**Out of scope (per spec):** web fallback, prompt-injection defense, observability dashboard, v2 blocking judge — intentionally deferred; extension points (tool registry, trace, judge) are in place.
