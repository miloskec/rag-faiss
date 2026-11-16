import os
from pathlib import Path

from dotenv import load_dotenv


WORKSPACE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = WORKSPACE_DIR.parent
load_dotenv(PROJECT_DIR / ".env")
DATA_DIR = WORKSPACE_DIR / "data"
FEED_DIR = WORKSPACE_DIR / "feed"
SYNTHETIC_CORPUS_DIR = WORKSPACE_DIR / "examples" / "synthetic_corpus"
LOG_DIR = Path(os.getenv("RAG_LOG_DIR", WORKSPACE_DIR / "logs"))

EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME", "intfloat/multilingual-e5-large"
)
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "1024"))
RERANKER_MODEL_NAME = os.getenv(
    "RERANKER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "1200"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "150"))
MAX_CONTEXT_CHARS = int(os.getenv("RAG_MAX_CONTEXT_CHARS", "6000"))
MAX_PROMPT_TOKENS = int(os.getenv("RAG_MAX_PROMPT_TOKENS", "2048"))
MAX_NEW_TOKENS = int(os.getenv("RAG_MAX_NEW_TOKENS", "512"))
DEFAULT_TOP_K = int(os.getenv("RAG_TOP_K", "5"))
DEFAULT_RAG_MODE = os.getenv("RAG_MODE", "rag_v0").strip().lower()
DEFAULT_MODEL = os.getenv("MODEL", "llama-7b-hf").strip()


def env_flag(name, default=False):
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def normalize_rag_mode(mode):
    normalized = (mode or DEFAULT_RAG_MODE).strip().lower()
    aliases = {
        "v0": "rag_v0",
        "basic": "rag_v0",
        "rag_v0": "rag_v0",
        "v1": "rag_v1",
        "rerank": "rag_v1",
        "reranked": "rag_v1",
        "rag_v1": "rag_v1",
    }
    if normalized not in aliases:
        raise ValueError("mode must be one of: rag_v0, rag_v1")
    return aliases[normalized]
