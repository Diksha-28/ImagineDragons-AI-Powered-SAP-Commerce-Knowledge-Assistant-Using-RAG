import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from ingest.chunker import chunk_docs

COLLECTION_NAME = "commerce_docs"
CHROMA_PATH = "./chroma_db"


def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    embedding_fn = SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
    )
    return collection


def run_ingestion():
    chunks = chunk_docs()
    collection = get_collection()

    ids = [c["id"] for c in chunks]
    texts = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        collection.upsert(
            ids=ids[i:i + batch_size],
            documents=texts[i:i + batch_size],
            metadatas=metadatas[i:i + batch_size],
        )
        print(f"Uploaded batch {i // batch_size + 1}")

    print(f"Ingestion complete. {len(chunks)} chunks stored.")


if __name__ == "__main__":
    run_ingestion()