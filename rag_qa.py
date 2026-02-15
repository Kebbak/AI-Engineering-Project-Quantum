"""
rag_qa.py

End-to-end RAG pipeline over your ChromaDB collection:
- Top-k retrieval from ./chroma_db, collection 'policy_chunks'
- Optional cross-encoder re-ranking (SentenceTransformers)
- Prompt construction with injected chunks + citations
- Guardrails:
    * Refuse to answer outside the corpus
    * Limit output length
    * Always cite sources in the final answer

Usage examples:
    python3 rag_qa.py -q "What is the VPN policy for contractors?"
    python3 rag_qa.py -q "remote work expectations" -k 5 --rerank
    python3 rag_qa.py --interactive
    python3 rag_qa.py --file questions.txt -k 5 --rerank --min-score 0.25

Environment (any one provider):
    # No API key required for retrieval-only mode.

"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import sys
import json
import time

import chromadb
from chromadb.errors import NotFoundError
from sentence_transformers import SentenceTransformer

# Optional re-ranker
try:
    from sentence_transformers.cross_encoder import CrossEncoder
    HAS_CROSS_ENCODER = True
except Exception:
    HAS_CROSS_ENCODER = False

    # ...existing code...

import os
import requests

# ---------------------------
# Configuration (aligns with your ingest/query)
# ---------------------------
CHROMA_DB_DIR = "chroma_db"
COLLECTION_NAME = "policy_chunks"
EMBEDDING_MODEL = "./all-MiniLM-L6-v2"        # same model used for ingestion
DEFAULT_TOP_K = 5
DEFAULT_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # ~120MB

# Reasonable context/answer controls
MAX_CONTEXT_CHARS = 8000  # safety trimming to avoid massive prompts
DEFAULT_MAX_TOKENS = 300  # for OpenRouter; adjust as needed
DEFAULT_MAX_SENTENCES = 6


# ---------------------------
# Data classes
# ---------------------------
@dataclass
class RetrievedChunk:
    rank: int
    text: str
    source: str
    doc_id: str
    distance: Optional[float] = None
    rerank_score: Optional[float] = None


# ---------------------------
# Utilities
# ---------------------------
def embed_queries(texts: List[str], model_path: str) -> List[List[float]]:
    model = SentenceTransformer(model_path)
    vecs = model.encode(texts, show_progress_bar=False)
    return vecs.tolist()


def list_collections(client: chromadb.PersistentClient) -> None:
    print("Debug: listing collections...")
    for c in client.list_collections():
        t = getattr(c, "tenant", None)
        d = getattr(c, "database", None)
        print(f"- name={c.name} tenant={t} database={d}")


def retrieve(
    client: chromadb.PersistentClient,
    query: str,
    top_k: int,
    include_embeddings: bool = False,
) -> List[RetrievedChunk]:
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
    except NotFoundError:
        raise SystemExit(
            f"Collection '{COLLECTION_NAME}' not found at {Path(CHROMA_DB_DIR).resolve()}.\n"
            f"Did you run ingest.py successfully?"
        )

    q_vec = embed_queries([query], EMBEDDING_MODEL)[0]
    include = ["documents", "metadatas", "distances"]
    if include_embeddings:
        include.append("embeddings")

    result = collection.query(
        query_embeddings=[q_vec],
        n_results=max(1, top_k),
        include=include
    )

    docs = result.get("documents", [[]])[0]
    metas = result.get("metadatas", [[]])[0]
    dists = result.get("distances", [[]])[0] if "distances" in result else [None] * len(docs)
    ids   = result.get("ids", [[]])[0] if "ids" in result else [f"doc_{i}" for i in range(len(docs))]

    retrieved: List[RetrievedChunk] = []
    for i, (doc, meta, dist, did) in enumerate(zip(docs, metas, dists, ids), start=1):
        retrieved.append(RetrievedChunk(
            rank=i, text=doc or "", source=meta.get("source", "unknown"), doc_id=str(did), distance=dist
        ))
    return retrieved


def maybe_rerank(
    query: str,
    retrieved: List[RetrievedChunk],
    rerank: bool,
    cross_encoder_model: str,
) -> List[RetrievedChunk]:
    if not rerank:
        return retrieved

    if not HAS_CROSS_ENCODER:
        print("[warn] CrossEncoder not available; skipping rerank. Install sentence-transformers full extras.")
        return retrieved

    # Load CE lazily and score [query, chunk]
    try:
        ce = CrossEncoder(cross_encoder_model)
    except Exception as e:
        print(f"[warn] Failed to load cross-encoder '{cross_encoder_model}': {e}. Skipping rerank.")
        return retrieved

    pairs = [[query, r.text] for r in retrieved]
    scores = ce.predict(pairs).tolist()
    for r, s in zip(retrieved, scores):
        r.rerank_score = float(s)

    # Sort by score desc (higher is better)
    retrieved.sort(key=lambda x: x.rerank_score if x.rerank_score is not None else -1.0, reverse=True)
    # Re-assign ranks
    for i, r in enumerate(retrieved, start=1):
        r.rank = i
    return retrieved


def should_abstain(
    retrieved: List[RetrievedChunk],
    use_rerank_gate: bool,
    min_score: Optional[float] = None
) -> bool:
    """
    Guardrail: refuse to answer outside the corpus.
    If re-ranking is enabled and min_score is provided, abstain when top rerank score is below threshold.
    """
    if not retrieved:
        return True
    if use_rerank_gate and (min_score is not None):
        top = retrieved[0]
        if top.rerank_score is None:
            return False  # cannot gate w/out score
        return top.rerank_score < float(min_score)
    return False


def build_context(retrieved: List[RetrievedChunk], max_chars: int = MAX_CONTEXT_CHARS) -> Tuple[str, List[Dict[str, str]]]:
    """
    Build a single context string with clear separators and citations.
    Returns context and a compact sources list for final rendering.
    """
    parts: List[str] = []
    sources: List[Dict[str, str]] = []

    total_len = 0
    for r in retrieved:
        header = f"[Source: {r.source} | ID: {r.doc_id} | Rank: {r.rank}"
        if r.rerank_score is not None:
            header += f" | score={r.rerank_score:.3f}"
        if r.distance is not None:
            try:
                header += f" | dist={float(r.distance):.3f}"
            except Exception:
                header += f" | dist={r.distance}"
        header += "]"

        block = f"{header}\n{r.text.strip()}\n"
        if total_len + len(block) > max_chars:
            break

        parts.append(block)
        total_len += len(block)

        sources.append({"source": r.source, "id": r.doc_id})

    # Deduplicate sources (preserve order)
    seen = set()
    unique_sources: List[Dict[str, str]] = []
    for s in sources:
        key = (s["source"], s["id"])
        if key not in seen:
            seen.add(key)
            unique_sources.append(s)

    return "\n\n-----\n\n".join(parts), unique_sources


def build_messages(
    user_query: str,
    context: str,
    max_sentences: int = DEFAULT_MAX_SENTENCES
) -> List[Dict[str, str]]:
    """
    Build chat messages including system guardrails & user prompt.
    """
    system = (
        "You are a helpful assistant for company policy Q&A. "
        "Answer ONLY using the provided context. "
        "If the answer is not in the context, say: 'I can only answer about our policies based on the provided documents.' "
        f"Limit your answer to {max_sentences} sentences. "
        "Always cite sources inline as [source: <ID or title>]."
    )
    user = (
        f"Question:\n{user_query}\n\n"
        f"Context (use ONLY this to answer):\n{context}\n\n"
        "Instructions:\n"
        "- Provide a direct, concise answer.\n"
        "- Cite using [source: <ID or title>] inline.\n"
        "- If insufficient info, refuse as instructed.\n"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def call_openrouter_chat(model: str, messages: List[Dict[str, str]], max_tokens: int) -> str:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set")
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://local.tool",
        "X-Title": "Company Policy RAG",
    }
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.2,
    }
    r = requests.post(url, headers=headers, json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"].strip()


def generate_answer(
    model: str,
    user_query: str,
    context: str,
    max_tokens: int,
    max_sentences: int,
) -> str:
    messages = build_messages(user_query, context, max_sentences)
    return call_openrouter_chat(model, messages, max_tokens)


# ---------------------------
# Providers
# ---------------------------
    # ...existing code...


    # ...removed: no LLM provider...


    # ...existing code...


    # ...removed: no LLM provider...


    # ...removed: no LLM provider...


# ---------------------------
# Orchestration
# ---------------------------
def answer_query(
    query: str,
    top_k: int,
    rerank: bool,
    min_score: Optional[float],
    max_sentences: int,
    model: str = "openai/gpt-4o-mini",
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> None:
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    list_collections(client)
    retrieved = retrieve(client, query, top_k=top_k, include_embeddings=False)
    # Guardrail: abstain if below threshold (only if rerank & min_score provided)
    abstain = should_abstain(retrieved, rerank, min_score=min_score)
    if abstain:
        print("\nAnswer:")
        print("I can only answer about our policies based on the provided documents, and I don't have enough information.")
        print("\nSources: (none — insufficient evidence)")
        return
    # Build context
    context, sources = build_context(retrieved, max_chars=MAX_CONTEXT_CHARS)
    # Generate answer with LLM
    try:
        answer = generate_answer(
            model=model,
            user_query=query,
            context=context,
            max_tokens=max_tokens,
            max_sentences=max_sentences,
        )
    except Exception as e:
        print(f"[warn] LLM generation failed: {e}")
        print("\nFalling back to retrieval-only mode. Top chunks with citations:\n")
        for r in retrieved[:top_k]:
            print("—" * 60)
            print(f"[source: {r.source} | id: {r.doc_id}]")
            print(r.text.strip())
        return
    print("\nAnswer:")
    print(answer)
    if sources:
        uniq = []
        seen = set()
        for s in sources:
            key = (s["source"], s["id"])
            if key not in seen:
                uniq.append(s)
                seen.add(key)
        print("\nSources:")
        for s in uniq:
            print(f"- {s['source']} (ID: {s['id']})")
    else:
        print("\nSources: (none collected)")


def run_interactive(
    top_k: int,
    rerank: bool,
    min_score: Optional[float],
    max_sentences: int,
) -> None:
    print("\nEntering interactive mode. Type 'exit' to quit.\n")
    try:
        while True:
            try:
                q = input("Q> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting.")
                break
            if q.lower() in {"exit", "quit", ":q"}:
                print("Bye.")
                break
            if not q:
                continue
            answer_query(
                q, top_k, rerank, min_score, max_sentences
            )
            print("\n" + "=" * 80 + "\n")
    except KeyboardInterrupt:
        print("\nBye.")


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG over company policy corpus (ChromaDB).")
    parser.add_argument("-q", "--query", action="append", help="Question to ask. Repeat for multiple.")
    parser.add_argument("--file", type=str, help="Path to a file with one question per line.")
    parser.add_argument("--interactive", action="store_true", help="Interactive Q&A loop.")
    parser.add_argument("-k", "--top-k", type=int, default=DEFAULT_TOP_K, help=f"Top K docs to retrieve (default: {DEFAULT_TOP_K}).")
    parser.add_argument("--rerank", action="store_true", help="Enable cross-encoder reranking.")
    parser.add_argument("--min-score", type=float, default=None, help="If set WITH --rerank, abstain if top score < min-score (e.g., 0.25).")
    parser.add_argument("--model", type=str, default="openai/gpt-4o-mini", help="Model name for OpenRouter (default: openai/gpt-4o-mini)")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help=f"Max tokens for generation (default: {DEFAULT_MAX_TOKENS}).")
    parser.add_argument("--max-sentences", type=int, default=DEFAULT_MAX_SENTENCES, help=f"Sentence limit for the answer (default: {DEFAULT_MAX_SENTENCES}).")

    args = parser.parse_args()

    # Priority order: interactive > file > direct queries
    if args.interactive:
        run_interactive(args.top_k, args.rerank, args.min_score, args.max_sentences)
        return

    if args.file:
        path = Path(args.file)
        if not path.exists():
            raise SystemExit(f"File not found: {path}")
        with path.open("r", encoding="utf-8") as f:
            questions = [line.strip() for line in f if line.strip()]
        for q in questions:
            answer_query(q, args.top_k, args.rerank, args.min_score, args.max_sentences, args.model, args.max_tokens)
            print("\n" + "=" * 80 + "\n")
        return

    if args.query:
        for q in args.query:
            answer_query(q, args.top_k, args.rerank, args.min_score, args.max_sentences, args.model, args.max_tokens)
            print("\n" + "=" * 80 + "\n")
        return

    # If nothing provided:
    parser.print_help()
    print("\nTip: try --interactive or -q \"your question\"")

if __name__ == "__main__":
    main()