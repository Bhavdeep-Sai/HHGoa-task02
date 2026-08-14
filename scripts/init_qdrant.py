import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.app.vector_store import get_qdrant_store
from backend.app.embeddings import get_embedding_provider


def main():
    print("Initializing Qdrant collections...")
    embeddings = get_embedding_provider()
    store = get_qdrant_store()
    store.init_collections(vector_size=embeddings.dimension)
    print(f"Collections initialized successfully with vector dimension = {embeddings.dimension}.")


if __name__ == "__main__":
    main()
