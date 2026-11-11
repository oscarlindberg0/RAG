from pathlib import Path
from pypdf import PdfReader
from src.config import (
    DOCUMENTS_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP
)

# Get text from document
def load_document(path: Path) -> str:
    suffix = path.suffix.lower()

    if suffix in [".txt", ".md"]:
        return path.read_text(encoding="utf-8")

    elif suffix == ".pdf":
        reader = PdfReader(path)