# Agentic DocIntel — Design Spec

**Date:** 2026-06-20
**Status:** Approved, ready for implementation planning
**Scope:** Focused upgrade (~1 week), architected to extend toward Flagship features.

## Goal

Evolve DocIntel from a **fixed RAG pipeline** (`retrieve → rerank → generate`, the same
steps every request) into a **tool-using agent** that decides at runtime what to do:
which tools to call, how many times, with what reformulated queries, and when it has
enough to answer or must refuse. Target outcome is an industry-relevant project that
demonstrates AI agent application development for an internship search.

### Why this matters

A fixed pipeline breaks on real questions: multi-part questions need more than one
retrieval; out-of-scope questions get hallucinated answers or unhelpful refusals. An
agent handles these by *reasoning about what to do*. This is the capability the target
internship hires for, and the project should demonstrate it end to end — including the
robustness (self-correction, grounded refusal) and evaluation (faithfulness, agent
behaviour) that separate an industry system from a demo.

## Key decisions (from brainstorming)

| Decision | Choice | Rationale |
|---|---|---|
| Scope | Focused upgrade, extensible | Complete & defensible without the 3-week trap; leaves PhishGuard/CyberSentinel time. |
| Agent mechanism | Native function-calling + hand-written control loop | Industry-standard tool interface, but the reasoning loop is visible/explainable (not a framework black box). |
| Grounding | Strictly document-grounded | Preserves the "every claim traceable to an uploaded doc" guarantee. Web fallback is a later add-on. |
| Eval | Retrieval (done) + faithfulness (LLM-judge) + agent-behaviour (trace) | Most agent projects have zero eval; this is the differentiator. |

## Architecture

Replace the fixed body of `/ask` with an **agent control loop**:

```
ask(question)
  └─> AGENT LOOP (max N iterations, e.g. 5)
        LLM sees: question + tool results so far
        LLM decides ONE of:
          - call a tool  -> run it, append result, loop again
          - finish       -> return grounded answer + citations, exit
        Guardrails:
          - iteration cap (no infinite loops)
          - weak retrieval (top score < RELEVANCE_THRESHOLD) -> instruct reformulate
  └─> stream final answer over existing SSE endpoint + capture trace
```

The model chooses whether to retrieve, how many times, and with what query — control flow
is decided at runtime by the model. That is what makes it an agent rather than RAG.

**Framework choice:** a custom ~80-line loop instead of LangChain `AgentExecutor` /
LangGraph. Frameworks are faster to wire but hide the reasoning loop, which hurts
debugging and interview explainability. The spec documents the framework alternatives so
they can be discussed.

## Components & boundaries

New module **`agent.py`** owns the control loop + tool registry. Existing modules stay
single-purpose and are called *into* (not rewritten):

- `retriever.py` — two-stage retrieval (unchanged)
- `ingest.py` — ingestion + `list_documents()` (unchanged)
- `generator.py` — LLM call; extended to support tool-calling messages
- `app.py` — `/ask` delegates to `agent.run()` instead of the fixed pipeline

### Tools

Minimal, sharply-bounded toolset (fewer tools = more reliable agent behaviour):

| Tool | Signature | Purpose | Reuses |
|---|---|---|---|
| `retrieve` | `retrieve(query, source_filter=None) -> list[chunk]` | Two-stage retrieval; callable multiple times for multi-part questions. | `retriever.retrieve()` (built) |
| `list_documents` | `list_documents() -> list[str]` | Lets the agent know what's available before answering. | `ingest.list_documents()` (built) |
| `finish` | `finish(answer, citations) -> final` | Agent's "done" signal; forces structured, cited output. | new, tiny |

Tool descriptions are part of the engineering: they teach the model *when* to call a tool
multiple times. Tool-description tuning is an explicit agent-dev skill demonstrated here.

## Robustness: self-correction & grounding

1. **Self-correcting retrieval (multi-hop).** After each `retrieve`, the loop checks the
   top reranker score against `RELEVANCE_THRESHOLD`. Weak context → the agent is told to
   reformulate the query and retrieve again. Capped at N iterations.
2. **Grounded refusal.** If no relevant context is found after attempts, the agent must
   `finish` with the explicit "not in your documents" answer — never invented. Preserves
   strict grounding, now as a reasoned decision rather than a hardcoded gate.
3. **Faithfulness self-check (defense in depth).**
   - v1 (this scope): `finish` requires citations; system prompt enforces "every sentence
     must trace to a provided chunk." No extra LLM call.
   - v2 (Flagship): separate LLM-as-judge groundedness pass that blocks/flags unfaithful
     answers — reuses the eval judge.

**Trace.** Each loop records tools called, arguments, returns, and scores. The trace backs
both observability (why did it answer that way) and evaluation (Layer 3 grades traces). It
can also be surfaced in the UI as the agent's reasoning steps.

**Cost/latency tradeoff (acknowledged).** Hard questions may take 3–4 LLM calls vs 1.
Mitigation: cap iterations; multi-hop only when retrieval is weak. This bounded-cost
reasoning is a deliberate, explainable design choice.

## Evaluation (three layers)

1. **Retrieval eval — built.** `eval/evaluate.py` (Recall@20, Hit@3, MRR) validates the
   `retrieve` tool in isolation. Unchanged.
2. **Faithfulness / answer-quality eval — new.** LLM-as-judge over a labeled Q&A set,
   scoring **groundedness** (claims supported by cited chunks — catches hallucination),
   **answer relevance**, and **correctness** vs a reference answer. Produces a scored
   report for the README/resume.
3. **Agent-behaviour eval — new, small.** Trace assertions proving the agency works:
   - simple question → retrieves once
   - multi-part question → retrieves multiple times (multi-hop)
   - off-topic question → refuses (grounding holds)
   - weak-first-retrieval → reformulates (self-correction fires)

**Judge model:** a capable model (same provider acceptable). LLM-as-judge is standard
practice; its biases are documented as a known limitation.

## Out of scope (Flagship add-ons, designed to bolt on later)

- Web-search fallback tool (`web_search`) with mixed-source labeling
- Prompt-injection defense (high-value given the author's security background)
- Observability/tracing dashboard (e.g. LangSmith / OpenTelemetry)
- v2 faithfulness judge as a blocking gate in the live path

These require no rework: the tool registry, trace capture, and judge are the extension
points.

## Success criteria

- `/ask` runs through the agent loop; simple questions answer in one hop, multi-part
  questions trigger multiple retrievals, off-topic questions are refused.
- Strict grounding preserved: every non-refusal answer carries page citations.
- Faithfulness eval produces a groundedness/relevance/correctness report.
- Agent-behaviour eval asserts the four trajectory cases above.
- Existing retrieval eval and unit tests still pass; new agent logic has unit tests.
- README documents the agent architecture, the eval results, and the tradeoffs.
