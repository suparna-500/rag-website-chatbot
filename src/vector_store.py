"""
vector_store.py
---------------
Stores website chunks in ChromaDB using Sentence Transformers embeddings.
Also provides similarity search for the RAG pipeline.
"""

import hashlib
from typing import List, Tuple

import chromadb
from sentence_transformers import SentenceTransformer

from chunker import TextChunk


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "rag_chatbot"


# ---------------------------------------------------------------------
# Lazy-load embedding model
# ---------------------------------------------------------------------

_embed_model = None


def get_embed_model():
    global _embed_model

    if _embed_model is None:
        print(f"Loading embedding model '{EMBED_MODEL_NAME}'...")
        _embed_model = SentenceTransformer(EMBED_MODEL_NAME)
        print("Embedding model loaded.")

    return _embed_model


# ---------------------------------------------------------------------
# Chroma Collection
# ---------------------------------------------------------------------

def get_collection(chroma_dir: str = CHROMA_DIR):

    client = chromadb.PersistentClient(path=chroma_dir)

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    return collection


# ---------------------------------------------------------------------
# Unique ID
# ---------------------------------------------------------------------

def _chunk_id(chunk: TextChunk):

    text = f"{chunk.source_url}_{chunk.chunk_index}_{chunk.text}"

    return hashlib.md5(text.encode()).hexdigest()


# ---------------------------------------------------------------------
# Ingest Chunks
# ---------------------------------------------------------------------

def ingest_chunks(
    chunks: List[TextChunk],
    chroma_dir: str = CHROMA_DIR,
):

    if not chunks:
        print("No chunks found.")
        return 0

    model = get_embed_model()

    collection = get_collection(chroma_dir)

    texts = [chunk.text for chunk in chunks]

    ids = [_chunk_id(chunk) for chunk in chunks]

    metadatas = [
        {
            "source_url": chunk.source_url,
            "chunk_index": chunk.chunk_index,
        }
        for chunk in chunks
    ]

    print(f"Embedding {len(texts)} chunks...")

    embeddings = model.encode(
        texts,
        show_progress_bar=True,
    ).tolist()

    print("Embedding complete.")

    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    print(f"Vector store now contains {collection.count()} chunks.")

    return len(chunks)


# ---------------------------------------------------------------------
# Similarity Search
# ---------------------------------------------------------------------

def similarity_search(
    query: str,
    top_k: int = 5,
    chroma_dir: str = CHROMA_DIR,
) -> List[Tuple[str, str, float]]:

    collection = get_collection(chroma_dir)

    if collection.count() == 0:
        return []

    model = get_embed_model()

    query_embedding = model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    retrieved = []

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]

    for doc, meta, dist in zip(docs, metas, dists):

        retrieved.append(
            (
                doc,
                meta["source_url"],
                dist,
            )
        )

    return retrieved


# ---------------------------------------------------------------------
# Clear Collection
# ---------------------------------------------------------------------

def clear_collection(chroma_dir: str = CHROMA_DIR):

    client = chromadb.PersistentClient(path=chroma_dir)

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    print("Vector store cleared.")


# ---------------------------------------------------------------------
# Self Test
# ---------------------------------------------------------------------

if __name__ == "__main__":

    from chunker import TextChunk

    sample_chunks = [

        TextChunk(
            text="RAG stands for Retrieval Augmented Generation.",
            source_url="https://example.com",
            chunk_index=0,
        ),

        TextChunk(
            text="ChromaDB is a vector database.",
            source_url="https://example.com",
            chunk_index=1,
        ),

        TextChunk(
            text="Sentence Transformers create embeddings.",
            source_url="https://example.com",
            chunk_index=2,
        ),
    ]

    clear_collection()

    ingest_chunks(sample_chunks)

    print()

    results = similarity_search(
        "What is RAG?",
        top_k=2,
    )

    for text, url, distance in results:

        print("--------------------------------")

        print("Source :", url)

        print("Distance :", distance)

        print(text)