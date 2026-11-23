try:
    from .ingestion import load_all_documents
    from .vector_store import VectorStore
    from .rag import RAG
except ImportError:
    from ingestion import load_all_documents
    from vector_store import VectorStore
    from rag import RAG

def index_documents():
    documents = load_all_documents()

    vector_store = VectorStore()

    vector_store.add_documents(documents)

def chat():
    rag = RAG()

    print("\nGranite rag chat")
    print("Type 'exit' to quit\n")

    while True:

        question = input("You: ")

        if question.lower() in ["exit", "quit"]:
            break

        result = rag.ask(question)

        print("\nAssistant:")

        if isinstance(result, str):
            print(result)
        else:
            print(result["answer"])

            print("\nSources:")

            for source in result["sources"]:
                print(
                    f"- {source['source']} "
                    f"(distance: {source['distance']})"
                )

        print("\n" + "-" * 50 + "\n")


if __name__ == "__main__":

    print("1. Index documents")
    print("2. Chat")

    choice = input("Select option: ")

    if choice == "1":
        index_documents()

    elif choice == "2":
        chat()