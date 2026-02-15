"""
query_chromadb.py

Query the persisted ChromaDB collection built by ingest.py.

- Uses the same persistent store: ./chroma_db
- Uses the same collection name: policy_chunks
- Computes query embeddings locally with ./all-MiniLM-L6-v2 (to avoid EF conflicts)
- Allows passing a query via CLI: python3 query_chromadb.py -q "your question"
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List
import chromadb
from chromadb.errors import NotFoundError
from sentence_transformers import SentenceTransformer

# ---------------------------
# Config (must match ingest.py)
# ---------------------------
CHROMA_DB_DIR = "chroma_db"
COLLECTION_NAME = "policy_chunks"
EMBEDDING_MODEL = "./all-MiniLM-L6-v2"
DEFAULT_QUERY = "What is the VPN policy for contractors?"
TOP_K = 5


def embed_query(texts: List[str], model_path: str) -> List[List[float]]:
    """
    Compute embeddings for query texts using the same SentenceTransformer model
    as used for ingestion.
    """
    model = SentenceTransformer(model_path)
    vecs = model.encode(texts, show_progress_bar=False)
    return vecs.tolist()


def pretty_print_results(results: dict, top_k: int) -> None:
    docs = results.get("documents", [[]])
    metas = results.get("metadatas", [[]])
    dists = results.get("distances", [[]])

    if not docs or not docs[0]:
        print("No results.")
        return

    print("Top matches:")
    for i, (doc, meta, dist) in enumerate(zip(docs[0], metas[0], dists[0])):
        if i >= top_k:
            break
        print("—" * 60)
        print(f"Rank: {i+1}")
        print(f"Distance: {dist:.6f}")
        print(f"Source: {meta.get('source', 'unknown')}")
        print("Snippet:")
        print(doc.strip())


def main() -> None:
    # CLI
    parser = argparse.ArgumentParser(description="Query ChromaDB collection built by ingest.py")
    parser.add_argument("-q", "--query", type=str, default=DEFAULT_QUERY, help="Your question / query text")
    parser.add_argument("-k", "--top_k", type=int, default=TOP_K, help="Number of results to return")
    args = parser.parse_args()

    # Init persistent client
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)

    # Debug: list collections
    print("Debug: listing collections...")
    for c in client.list_collections():
        t = getattr(c, "tenant", None)
        d = getattr(c, "database", None)
        print(f"- name={c.name} tenant={t} database={d}")

    # Get collection WITHOUT embedding_function to avoid conflicts
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
    except NotFoundError:
        raise SystemExit(
            f"Collection '{COLLECTION_NAME}' not found at {Path(CHROMA_DB_DIR).resolve()}.\n"
            f"Did you run ingest.py successfully?"
        )

    count = collection.count()
    print(f"Collection '{COLLECTION_NAME}' count: {count}")

    # Compute query embedding locally (matches ingest model)
    query_text = args.query
    print(f"\nQuery: {query_text}\n")
    q_vec = embed_query([query_text], EMBEDDING_MODEL)

    # Query using vector (no embedding function attached)
    results = collection.query(
        query_embeddings=q_vec,
        n_results=max(1, args.top_k),
        include=["documents", "metadatas", "distances"],
    )

    pretty_print_results(results, args.top_k)


if __name__ == "__main__":
    main()