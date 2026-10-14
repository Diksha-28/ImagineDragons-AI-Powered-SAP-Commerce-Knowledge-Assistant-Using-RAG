import sys
from retrieval.retriever import retrieve
from llm.answer import generate_answer


def run(query: str):
    print(f"\nQuery: {query}")
    print("-" * 50)

    print("Retrieving relevant chunks...")
    chunks = retrieve(query)
    print(f"Found {len(chunks)} chunks\n")

    print("Generating answer...")
    result = generate_answer(query, chunks)

    print("\nAnswer:")
    print(result["answer"])
    print("\nSources:")
    for source in result["sources"]:
        print(f"  - {source}")


if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else "What is this documentation about?"
    run(query)