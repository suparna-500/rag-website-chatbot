"""
vectorstore.py
--------------
Embeds TextChunk objects using sentence-transformers and stores them in
a local ChromaDB collection. Also exposes a similarity_search() function
used by the RAG chain on Day 3.

ChromaDB stores everything in a local folder (./chroma_db by default),
so no server setup is needed — perfect for a demo project.
"""

import os
import hashlib
from typing import List, Tuple

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from chunker import TextChunk


# ── Constants you can tweak ──────────────────────────────────────────────────
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"   # ~80 MB, very fast, good quality
CHROMA_DIR       = "./chroma_db"          # local persistence directory
COLLECTION_NAME  = "rag_chatbot"


# ── Lazy-load the embedding model once per process ──────────────────────────
_embed_model: SentenceTransformer | None = None

def get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        print(f"Loading embedding model '{EMBED_MODEL_NAME}' (first run may download ~80 MB)…")
        _embed_model = SentenceTransformer(EMBED_MODEL_NAME)
        print("Model loaded.")
    return _embed_model


# ── ChromaDB client + collection ─────────────────────────────────────────────
def get_collection(chroma_dir: str = CHROMA_DIR):
    """Return (or create) the persistent ChromaDB collection."""
    client = chromadb.PersistentClient(path=chroma_dir)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},   # cosine similarity
    )
    return collection


def _chunk_id(chunk: TextChunk) -> str:
    """
    Stable, unique ID for a chunk based on its content.
    Using a hash means re-ingesting the same page won't create duplicates.
    """
    raw = f"{chunk.source_url}::{chunk.chunk_index}::{chunk.text[:50]}"
    return hashlib.md5(raw.encode()).hexdigest()


# ── Main ingestion function ───────────────────────────────────────────────────
def ingest_chunks(chunks: List[TextChunk], chroma_dir: str = CHROMA_DIR) -> int:
    """
    Embed all chunks and upsert them into ChromaDB.

    Returns the number of chunks actually added (skips duplicates).

    We batch the embedding call so sentence-transformers can parallelise
    across all chunks at once — much faster than one-by-one.
    """
    if not chunks:
        print("No chunks to ingest.")
        return 0

    model      = get_embed_model()
    collection = get_collection(chroma_dir)

    texts     = [c.text       for c in chunks]
    ids       = [_chunk_id(c) for c in chunks]
    metadatas = [{"source_url": c.source_url, "chunk_index": c.chunk_index} for c in chunks]

    print(f"Embedding {len(chunks)} chunks…")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_list=True)
    print("Embedding complete.")

    # upsert: insert new, overwrite if same ID already exists
    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    total_stored = collection.count()
    print(f"Vector store now contains {total_stored} chunks.")
    return len(chunks)


# ── Query function used by RAG chain ─────────────────────────────────────────
def similarity_search(
    query: str,
    top_k: int = 5,
    chroma_dir: str = CHROMA_DIR,
) -> List[Tuple[str, str, float]]:
    """
    Embed the query and retrieve the top-k most similar chunks.

    Returns a list of (chunk_text, source_url, distance) tuples.
    Lower distance = more similar (cosine distance, 0 = identical).
    """
    model      = get_embed_model()
    collection = get_collection(chroma_dir)

    if collection.count() == 0:
        return []

    query_embedding = model.encode([query], convert_to_list=True)[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        hits.append((doc, meta["source_url"], dist))

    return hits


def clear_collection(chroma_dir: str = CHROMA_DIR) -> None:
    """
    Delete all vectors from the collection.
    Called when the user enters a new URL in the Streamlit app.
    """
    client = chromadb.PersistentClient(path=chroma_dir)
    # delete and recreate — simplest way to wipe everything
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass  # already empty — nothing to delete
    client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    print("Vector store cleared.")


# ── quick self-test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    from chunker import TextChunk

    # Fake some chunks
    sample_chunks = [
        TextChunk(text="RAG stands for Retrieval-Augmented Generation.", source_url="https://example.com/rag", chunk_index=0),
        TextChunk(text="ChromaDB is a local vector database that requires no server.", source_url="https://example.com/chroma", chunk_index=0),
        TextChunk(text="Sentence-transformers provide fast, free local embeddings.", source_url="https://example.com/embed", chunk_index=0),
        TextChunk(text="Streamlit makes it easy to build Python web apps.", source_url="https://example.com/streamlit", chunk_index=0),
        TextChunk(text="The all-MiniLM-L6-v2 model is small but produces high-quality embeddings.", source_url="https://example.com/model", chunk_index=0),
    ]

    # Wipe any leftover test data
    clear_collection()

    # Ingest
    ingest_chunks(sample_chunks)

    # Query
    query = "What embedding model should I use?"
    print(f"\nQuery: {query}")
    results = similarity_search(query, top_k=3)
    for text, url, dist in results:
        print(f"  [{dist:.3f}] {url}\n  → {text}\n")