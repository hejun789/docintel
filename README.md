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
├── ingest.py       # Extract → chunk → embed → store pipeline
├── retriever.py    # Two-stage retrieval: bi-encoder + cross-encoder re-ranking
├── generator.py    # Prompt construction + LLM answer generation via OpenRouter
├── config.py       # Model names, chunk parameters, thresholds
├── requirements.txt
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

---

## Tests

Unit tests cover the highest-risk pure logic — the summary-question gate, the chunking pipeline, and the upload filter:

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
