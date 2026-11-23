import ollama
import json
import math
import sqlite3
try:
    from .config import(
        CHROMA_DIR,
        COLLECTION_NAME,
        EMBEDDING_MODEL
    )
except ImportError:
    from config import(
        CHROMA_DIR,
        COLLECTION_NAME,
        EMBEDDING_MODEL
    )

class VectorStore:

    def __init__(self):
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        self.db_path = CHROMA_DIR / f"{COLLECTION_NAME}.sqlite3"
        self.connection = sqlite3.connect(str(self.db_path))
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                source TEXT NOT NULL,
                embedding TEXT NOT NULL
            )
            """
        )
        self.connection.commit()

    # Generate embeddings for vectordb
    def generate_embedding(self, text: str) -> list[float]:

        response = ollama.embed(
            model=EMBEDDING_MODEL,
            input=text
        )

        return response["embeddings"][0]

    # Embed documents and add to db
    def add_documents(self, documents: list[dict]):
        if not documents:
            return

        for document in documents:
            embedding = self.generate_embedding(
                document["text"]
            )
            self.connection.execute(
                """
                INSERT INTO documents (id, text, source, embedding)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    text = excluded.text,
                    source = excluded.source,
                    embedding = excluded.embedding
                """,
                (
                    document["id"],
                    document["text"],
                    document["source"],
                    json.dumps(embedding),
                ),
            )

        self.connection.commit()

        print(f"indexed {len(documents)} chunks")

    # Similarity search
    def search(self, query: str, top_k: int = 4):
        query_embedding = self.generate_embedding(query)

        retrieved_documents = []
        query_norm = math.sqrt(sum(value * value for value in query_embedding))

        rows = self.connection.execute(
            "SELECT text, source, embedding FROM documents"
        )

        for text, source, encoded_embedding in rows:
            embedding = json.loads(encoded_embedding)
            document_norm = math.sqrt(sum(value * value for value in embedding))

            if query_norm == 0 or document_norm == 0:
                similarity = 0.0
            else:
                dot_product = sum(
                    query_value * document_value
                    for query_value, document_value in zip(query_embedding, embedding)
                )
                similarity = dot_product / (query_norm * document_norm)

            retrieved_documents.append({
                "text": text,
                "source": source,
                "distance": 1.0 - similarity,
            })

        retrieved_documents.sort(key=lambda document: document["distance"])
        return retrieved_documents[:max(top_k, 0)]