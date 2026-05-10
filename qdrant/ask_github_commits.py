from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import os


load_dotenv()


QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or None
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "github_commits")

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)

DEFAULT_TOP_K = int(os.getenv("TOP_K", "5"))


model = SentenceTransformer(EMBEDDING_MODEL)

qdrant = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)


def search_profiles(question: str, top_k: int = DEFAULT_TOP_K):
    query_vector = model.encode(question).tolist()

    response = qdrant.query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_vector,
        limit=top_k,
        with_payload=True,
    )

    return response.points


def print_results(question: str, top_k: int = DEFAULT_TOP_K):
    results = search_profiles(question, top_k)

    print("")
    print(f"Question: {question}")
    print(f"Top K: {top_k}")
    print("=" * 80)

    if not results:
        print("No results found.")
        return

    for index, result in enumerate(results, start=1):
        payload = result.payload or {}

        print("")
        print(f"Result #{index}")
        print(f"Score: {result.score}")
        print(f"Section: {payload.get('section')}")
        print(f"Company: {payload.get('company')}")
        print(f"Title: {payload.get('title')}")
        print("-" * 80)
        print(payload.get("text"))
        print("=" * 80)


if __name__ == "__main__":
    question = "How can I create a github actions workflow?"
    print_results(question, top_k=8)