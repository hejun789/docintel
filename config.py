import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=True)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct:free")

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

CHROMA_DIR = "chroma_db"
CHROMA_COLLECTION = "documents"

CHUNK_SIZE = 512
CHUNK_OVERLAP = 100
TOP_K_RETRIEVE = 20   # how many chunks to fetch from ChromaDB
TOP_K_FINAL = 3       # how many to keep after re-ranking and send to LLM
RELEVANCE_THRESHOLD = -2.0  # cross-encoder logit; below this = no relevant content
