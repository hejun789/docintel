import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=True)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct:free")

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
USE_RERANKER = os.getenv("USE_RERANKER", "true").lower() == "true"

CHROMA_DIR = "chroma_db"
CHROMA_COLLECTION = "documents"

CHUNK_SIZE = 512
CHUNK_OVERLAP = 100
TOP_K_RETRIEVE = 20
TOP_K_FINAL = 3
RELEVANCE_THRESHOLD = -2.0 if USE_RERANKER else 0.3
