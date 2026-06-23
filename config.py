import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=True)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct:free")

# Agent / judge models (default to the main model; override per-env for tool-calling-capable models)
AGENT_MODEL = os.getenv("AGENT_MODEL", OPENROUTER_MODEL)
JUDGE_MODEL = os.getenv("JUDGE_MODEL", OPENROUTER_MODEL)
AGENT_MAX_ITERS = int(os.getenv("AGENT_MAX_ITERS", "6"))
# Low temperature → more deterministic, consistent answers (better for grounded
# factual Q&A than the provider's default randomness).
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
USE_RERANKER = os.getenv("USE_RERANKER", "true").lower() == "true"

CHROMA_DIR = "chroma_db"
CHROMA_COLLECTION = "documents"

# Chroma Cloud (optional): if these are set, documents persist in the cloud and
# survive container restarts/redeploys. If unset, falls back to the local
# ephemeral chroma_db/ directory.
CHROMA_API_KEY = os.getenv("CHROMA_API_KEY", "")
CHROMA_TENANT = os.getenv("CHROMA_TENANT", "")
CHROMA_DATABASE = os.getenv("CHROMA_DATABASE", "")

CHUNK_SIZE = 512
CHUNK_OVERLAP = 100
TOP_K_RETRIEVE = 20
TOP_K_FINAL = 5  # send more chunks to the LLM so large tables (e.g. rubrics) are less fragmented
# Cross-encoder cutoff for "relevant enough to answer from". -5.0 accepts
# borderline-but-relevant content (e.g. dense table/rubric chunks score ~-3)
# while still rejecting genuinely off-topic queries (which score ~-9 to -11).
RELEVANCE_THRESHOLD = -5.0 if USE_RERANKER else 0.3
