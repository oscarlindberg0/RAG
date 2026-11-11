from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DOCUMENTS_DIR = BASE_DIR / "documents"
CHROMA_DIR = BASE_DIR / "chroma_db"

LLM_MODEL = "granite4.2"
EMBEDDING_MODEL = "nomic-embed-text"

COLLECTION_NAME = "knowledge_base"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

TOP_K = 4