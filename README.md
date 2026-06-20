---
title: DocIntel
emoji: 📄
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# DocIntel — AI Document Intelligence System

A RAG (Retrieval-Augmented Generation) pipeline that lets you upload documents and ask natural language questions against them. Answers are grounded in your documents, not the internet — with source citations down to the page number.

![Python](https://img.shields.io/badge/Python-3.13-blue) ![Flask](https://img.shields.io/badge/Flask-3.1-lightgrey) ![ChromaDB](https://img.shields.io/badge/ChromaDB-1.5-orange) ![License](https://img.shields.io/badge/license-MIT-green)

---

## Demo

**Live demo:** [https://huggingface.co/spaces/hejun123/docintel](https://huggingface.co/spaces/hejun123/docintel)

Upload a PDF → Ask a question → Get a grounded answer with page citations.

---

## How it works

```
Document → Extract text → Chunk (512 chars, 100 overlap)
        → Embed (all-MiniLM-L6-v2) → Store in ChromaDB

Question → Embed → Retrieve top 20 candidates from ChromaDB
        → Re-rank with cross-encoder (ms-marco-MiniLM-L-6-v2)
        → Keep top 3 → Generate grounded answer via LLM
```

The two-stage retrieval is the key engineering decision: a **bi-encoder** (fast, approximate) fetches 20 candidates, then a **cross-encoder** (slower, precise) re-ranks them by scoring the question and each chunk jointly. This catches relevant chunks that vector similarity alone would miss.

---

## Agent

Answering does not run a fixed pipeline — it runs a **tool-using agent** that decides at runtime what to do:

```
Question
  └─> AGENT LOOP (max N iterations)
        model sees question + results so far, and chooses ONE:
          - call retrieve(query)      → search the documents (multi-hop: call again with new queries)
          - call list_documents()     → see what's available
          - call finish(answer, cites)→ return a grounded, cited answer, or refuse
        weak retrieval → the tool result hints the model to reformulate and retry
  └─> grounded answer + page citations + a trace of the agent's steps
```

- **Multi-hop retrieval** — multi-part questions trigger several `retrieve` calls with different queries; simple ones use a single call.
- **Self-correction** — when a retrieval scores below the relevance threshold, the agent reformulates the query and tries again.
- **Grounded refusal** — if the documents don't contain the answer (or the iteration cap is hit), the agent returns an explicit "not in your documents" with no citations, never an invented answer.
- **No agent framework** — the control loop is hand-written directly on the model's native function-calling API (via OpenRouter), not LangChain `AgentExecutor` or LangGraph, so the reasoning loop is fully visible and the per-question cost is bounded by an iteration cap.

The agent's tools (`retrieve`, `list_documents`) wrap the existing retrieval core; `finish` forces structured, cited output.

---

## Features

- Upload PDF, DOCX, TXT, and Markdown files
- Two-stage retrieval: bi-encoder + cross-encoder re-ranking
- Grounded answers with page-level source citations
- Persistent document library across server restarts
- Delete documents (removes chunks from vector store)
- Relevance threshold — explicitly says "I don't know" rather than hallucinating
- Clean two-panel UI: document manager + chat interface

---

## Tech stack

| Layer | Technology | Why |
|---|---|---|
| Backend | Python + Flask | Lightweight, fast to iterate |
| PDF parsing | PyMuPDF | Handles messy PDFs better than PyPDF2 |
| Text chunking | LangChain RecursiveCharacterTextSplitter | Respects paragraph/sentence boundaries |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) | Free, runs locally, 384-dim vectors |
| Re-ranking | sentence-transformers (ms-marco-MiniLM-L-6-v2) | Cross-encoder, significantly better precision |
| Vector database | ChromaDB | Local, persistent, no cloud account needed |
| LLM | OpenRouter (any free model) | Flexible model selection, free tier available |
| Frontend | HTML / CSS / Vanilla JS | No framework overhead for this scope |

---

## Project structure

```
docintel/
├── app.py          # Flask routes: /upload, /ask, /documents, /document/<name>
├── agent.py        # Tool-using agent: tools, schemas, and the control loop
├── llm.py          # OpenRouter native function-calling client
├── ingest.py       # Extract → chunk → embed → store pipeline
├── retriever.py    # Two-stage retrieval: bi-encoder + cross-encoder re-ranking
├── config.py       # Model names, chunk parameters, thresholds
├── requirements.txt
├── eval/           # Retrieval, faithfulness, and agent-behaviour evaluation
├── tests/          # pytest unit + behaviour tests
├── templates/
│   └── index.html
└── static/
    ├── style.css
    └── app.js
```

---

## Setup

**1. Clone and install dependencies**
```bash
git clone https://github.com/hejun789/docintel.git
cd docintel
pip install -r requirements.txt
```

**2. Create a `.env` file**
```
OPENROUTER_API_KEY=your_openrouter_key_here
OPENROUTER_MODEL=nvidia/nemotron-3-super-120b-a12b:free
```

Get a free API key at [openrouter.ai](https://openrouter.ai). Any model listed as free works.

**3. Run**
```bash
python app.py
```

Open `http://127.0.0.1:5000` in your browser.

---

## API endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Frontend UI |
| POST | `/upload` | Upload and ingest a document |
| POST | `/ask` | Ask a question, returns answer + sources |
| GET | `/documents` | List all ingested documents |
| DELETE | `/document/<filename>` | Remove a document and its chunks |

---

## Key design decisions

**Why chunk overlap?**
If an answer spans a chunk boundary, overlap ensures the complete sentence appears in at least one chunk. Without it, split sentences produce incomplete, confusing context for the LLM.

**Why a cross-encoder re-ranker?**
Bi-encoder similarity scores everything independently — fast but imprecise. A cross-encoder sees the question and chunk together, scoring their relevance jointly. The result is noticeably better precision, especially for specific technical questions.

**Why `all-MiniLM-L6-v2` for embeddings?**
Runs entirely locally at no cost, produces 384-dimensional vectors, and performs competitively with larger models on semantic similarity tasks. The cross-encoder re-ranker compensates for any retrieval imprecision.

---

## Evaluation

Retrieval is measured against a hand-labeled question set (`eval/eval_set.json`), where each question is tagged with a distinctive phrase that must appear in the retrieved chunk. `eval/evaluate.py` reports recall and quantifies the value of the re-ranking stage:

```
python eval/evaluate.py
```

Results on a 14-question set (sample research paper):

| Metric | Score | Meaning |
|---|---|---|
| Recall@20 | 93% | Gold chunk retrieved among bi-encoder candidates |
| Hit@3 (bi-encoder only) | 79% | Gold chunk in top-3 **without** re-ranking |
| Hit@3 (with re-ranker) | **93%** | Gold chunk in top-3 **with** cross-encoder re-ranking |
| MRR | 0.93 | Mean reciprocal rank after re-ranking |

The cross-encoder re-ranker lifts Hit@3 from **79% → 93%** — concrete evidence that the second retrieval stage earns its cost by pulling the genuinely relevant chunk into the top-3 that reach the LLM.

### Answer faithfulness (LLM-as-judge)

Retrieval recall measures whether the right chunk is *found*; it does not measure whether the *answer* is faithful. `eval/faithfulness.py` runs the agent end-to-end on a labeled Q&A set and uses a judge model to score each answer on groundedness, relevance, and correctness:

```
python eval/faithfulness.py
```

Results on a 5-question set (`gpt-oss-120b:free` agent, `gpt-oss-20b:free` judge):

| Metric | Score | Meaning |
|---|---|---|
| Groundedness | 0.80 | Claims consistent with the source (no invented facts) |
| Relevance | 1.00 | Answer addresses the question |
| Correctness | 0.80 | Answer matches the reference |

### Agent behaviour

The agent's *decisions* (not just its answers) are asserted in `tests/test_agent.py` against the step trace: simple questions retrieve once, multi-part questions retrieve multiple times, off-topic questions are refused, and weak first retrievals trigger reformulation.

---

## Tests

Unit tests cover the highest-risk logic — the agent control loop (multi-hop, refusal, iteration cap, via a scripted model), the tool-calling client, the `/ask` route, the summary-question gate, the chunking pipeline, and the upload filter:

```bash
pip install -r requirements-dev.txt
pytest
```

---

## Planned improvements

- Source passage highlighting (show exact text used, not just page number)
- Table extraction (PyMuPDF skips tables in technical PDFs)
- HyDE retrieval (embed a hypothetical answer for better candidate recall)
- Semantic chunking (split at meaning boundaries instead of fixed character count)
- Multi-language support (Bahasa Malaysia, Chinese)

---

## License

MIT
