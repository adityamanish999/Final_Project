"""Central configuration: paths, model names, defaults. Loads .env once on import."""

from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# Data locations
DATA_DIR = PROJECT_ROOT / "data"
SAMPLE_PDFS_DIR = DATA_DIR / "sample_pdfs"
UPLOADS_DIR = DATA_DIR / "uploads"
VECTOR_STORE_DIR = DATA_DIR / "vector_store"

# Models
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
GROQ_MODEL_NAME = "llama-3.3-70b-versatile"

# Chunking defaults
DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 150

# Retrieval defaults
DEFAULT_FETCH_K = 20
DEFAULT_TOP_K = 5

# Tool-calling loop
MAX_TOOL_ITERATIONS = 4

GROQ_API_KEY_ENV_VAR = "GROQ_API_KEY"
