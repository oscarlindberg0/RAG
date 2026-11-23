from pathlib import Path
from pypdf import PdfReader
try:
    from .config import (
        DOCUMENTS_DIR,
        CHUNK_SIZE,
        CHUNK_OVERLAP
    )
except ImportError:
    from config import (
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

        return "\n".join(
            page.extract_text() or ""
            for page in reader.pages
        )

    raise ValueError(f"unsupported file type: {suffix}")

# Split text into overlapping chunks
def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP
) -> list[str]:

    text = " ".join(text.split())

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        chunks.append(text[start:end])

        if end == len(text):
            break

        start = end - overlap

    return chunks

# Load and chunk documents
def load_all_documents():

    documents = []

    for path in DOCUMENTS_DIR.rglob("*"):

        if path.suffix.lower() not in [".txt", ".md", ".pdf"]:
            continue

        text = load_document(path)

        chunks = chunk_text(text)

        for index, chunk in enumerate(chunks):
            documents.append({
                "id": f"{path.relative_to(DOCUMENTS_DIR)}_{index}",
                "text": chunk,
                "source": str(path.relative_to(DOCUMENTS_DIR))
            })

    return documents