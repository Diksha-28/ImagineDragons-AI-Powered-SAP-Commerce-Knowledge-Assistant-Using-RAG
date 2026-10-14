import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

COLLECTION_NAME = "commerce_docs"
CHROMA_PATH = "./chroma_db"
TOP_K = 5


def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    embedding_fn = SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    collection = client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
    )
    return collection


def retrieve(query: str, top_k: int = TOP_K) -> list[dict]:
    collection = get_collection()
    results = collection.query(
        query_texts=[query],
        n_results=top_k,
    )

    chunks = []
    for i in range(len(results["documents"][0])):
        chunks.append({
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "score": results["distances"][0][i],
        })

    return chunks