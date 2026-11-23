import ollama
try:
    from .vector_store import VectorStore
    from .config import LLM_MODEL, TOP_K
except ImportError:
    from vector_store import VectorStore
    from config import LLM_MODEL, TOP_K

class RAG:
    def __init__(self):
        self.vector_store = VectorStore()

    # Get context documents
    def retrieve(self, question: str):
        return self.vector_store.search(
            query=question,
            top_k=TOP_K
        )

    def generate_answer(
            self,
            question: str,
            context_documents: list[dict]
    ):
        context = "\n\n".join(
            f"Source: {doc['source']}\n{doc['text']}"
            for doc in context_documents
        )

        system_prompt = """
        You are a helpful AI assistant.

        Answer the user's question using the provided context.

        Rules:
        - Only use information supported by the context.
        - If the context does not contain the answer, say so.
        - Do not invent facts or information.
        - Mention relevant source filenames in your answer.
        """

        user_prompt = f"""
        Context:

        {context}

        Question:

        {question}
        """

        response = ollama.chat(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ]
        )

        return response["message"]["content"]

    def ask(self, question: str):
        documents = self.retrieve(question)

        if not documents:
            return "no relevant docs found"

        answer = self.generate_answer(
            question,
            documents
        )

        return {
            "answer": answer,
            "sources": [
                {
                    "source": doc["source"],
                    "distance": round(doc["distance"], 4)
                }
                for doc in documents
            ]
        }