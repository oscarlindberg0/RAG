import chromadb
import ollama
from src.config import(
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL
)

class VectorStore:

    # Init db and create a collection
    def __init__(self):
        self.client = chromadb.PersistentClient(
            path=str(CHROMA_DIR)
        )

        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )

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

        embeddings = []

        for document in documents:
            embedding = self.generate_embedding(
                document["text"]
            )

            embeddings.append(embedding)

        self.collection.upsert(
            ids=[doc["id"] for doc in documents],

            documents=[
                doc["text"] for doc in documents
            ],

            metadatas=[
                {"source": doc["source"]}
                for doc in documents
            ],

            embeddings=embeddings
        )

        print(f"indexed {len(documents)} chunks")

    # Similarity search
    def search(self, query: str, top_k: int = 4):
        query_embedding = self.generate_embedding(query)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"]
        )

        retrieved_documents = []

        for i, document in enumerate(results["documents"][0]):

            retrieved_documents.append({
                "text": document,
                "source": results["metadatas"][0][i]["source"],
                "distance": results["distances"][0][i]
            })

        return retrieved_documents