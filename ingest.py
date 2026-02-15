"""
ingest.py

Script to parse, chunk, embed, and index policy documents for a RAG application.

Pipeline:
- Reads documents from ./corpus via doc_utils.read_documents (expects [{'source': str, 'content': str}, ...])
- Chunks documents with overlap (character-based)
- Embeds chunks locally with SentenceTransformers (./all-MiniLM-L6-v2)
- Persists to ChromaDB at ./chroma_db in collection 'policy_chunks'

Run:
    python3 ingest.py

Notes:
- The 'UNEXPECTED embeddings.position_ids' notice from SentenceTransformers for all-MiniLM-L6-v2 is harmless.
- This script precomputes embeddings and stores them; no embedding function is attached to the collection to avoid conflicts during query.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any
import math
import chromadb
from sentence_transformers import SentenceTransformer
from doc_utils import read_documents  # must return [{'source': str, 'content': str}, ...]

# ---------------------------
# Config
# ---------------------------
CORPUS_DIR = Path("corpus")
CHROMA_DB_DIR = "chroma_db"             # Will be created automatically
EMBEDDING_MODEL = "./all-MiniLM-L6-v2"  # Local directory or HF model name
COLLECTION_NAME = "policy_chunks"
CHUNK_SIZE = 300        # characters
CHUNK_OVERLAP = 50      # characters
BATCH_SIZE = 512        # how many chunks to embed per batch (tune for RAM/CPU)


# ---------------------------
# Helpers
# ---------------------------
def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Split text into overlapping chunks by character count.
    Ensures forward progress even for small texts.
    """
    if not text:
        return []
    chunks: List[str] = []
    start = 0
    step = max(1, chunk_size - overlap)
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += step
    return chunks


def embed_chunks_local(chunks: List[str], model_path: str, batch_size: int = 512) -> List[List[float]]:
    """
    Compute embeddings for a list of text chunks using a local SentenceTransformer model.
    all-MiniLM-L6-v2 produces 384-d embeddings.

    Args:
        chunks: list of strings to embed
        model_path: local dir or model name
        batch_size: embedding batch size

    Returns:
        List of embeddings (list of floats)
    """
    if not chunks:
        return []
    model = SentenceTransformer(model_path)

    # SentenceTransformer handles batching internally if we pass batch_size
    vectors = model.encode(
        chunks,
        show_progress_bar=True,
        batch_size=min(batch_size, max(1, len(chunks)))
    )
    return vectors.tolist()


def ensure_unique_ids(ids: List[str]) -> List[str]:
    """
    Guarantee uniqueness of IDs by suffixing duplicates with an incrementing counter.
    """
    seen: Dict[str, int] = {}
    unique_ids: List[str] = []
    for _id in ids:
        if _id not in seen:
            seen[_id] = 0
            unique_ids.append(_id)
        else:
            seen[_id] += 1
            unique_ids.append(f"{_id}_{seen[_id]}")
    return unique_ids


# ---------------------------
# Main
# ---------------------------
def main() -> None:
    # 1) Read documents
    docs = read_documents(CORPUS_DIR)
    print(f"Loaded {len(docs)} documents.")

    # 2) Chunk documents
    all_chunks: List[str] = []
    metadatas: List[Dict[str, Any]] = []
    ids: List[str] = []
    id_counter = 0

    for doc in docs:
        content = doc.get("content", "")
        source = doc.get("source", "unknown")

        if not content:
            # Skip empty docs, but continue
            continue

        chunks = chunk_text(content, CHUNK_SIZE, CHUNK_OVERLAP)
        for ch in chunks:
            all_chunks.append(ch)
            metadatas.append({"source": source})
            ids.append(f"doc_{id_counter}")
            id_counter += 1

    print(f"Total chunks: {len(all_chunks)}")
    if not all_chunks:
        print("No chunks to index. Ensure there are valid documents in the 'corpus' folder.")
        return

    # Ensure IDs are unique (defensive)
    ids = ensure_unique_ids(ids)

    # 3) Embed chunks locally
    embeddings = embed_chunks_local(all_chunks, EMBEDDING_MODEL, BATCH_SIZE)

    # 4) Persist to ChromaDB (no embedding_function to avoid conflicts during query)
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    # For very large datasets you could chunk adds; for 20 items this is fine:
    collection.add(
        ids=ids,
        documents=all_chunks,
        metadatas=metadatas,
        embeddings=embeddings,  # provide vectors directly
    )

    # Summary
    count = collection.count()
    print(f"Ingest complete. Collection '{COLLECTION_NAME}' now has {count} items.")
    print(f"Chroma DB path: {Path(CHROMA_DB_DIR).resolve()}")

    # Debug: list available collections
    print("Collections present:")
    for c in client.list_collections():
        t = getattr(c, "tenant", None)
        d = getattr(c, "database", None)
        print(f"- name={c.name} tenant={t} database={d}")


if __name__ == "__main__":
    main()