"""
chunker.py
----------
Takes the list of scraped pages (from scraper.py) and splits their text
into overlapping chunks suitable for embedding.

Why overlap? So that a sentence that falls near a chunk boundary isn't
lost — both neighbouring chunks contain it, so retrieval still finds it.
"""

from dataclasses import dataclass
from typing import List
from scraper import ScrapedPage   # reuse the dataclass from Day 1


@dataclass
class TextChunk:
    """A single chunk ready to be embedded."""
    text: str          # the actual chunk text
    source_url: str    # which page it came from
    chunk_index: int   # position within that page (for debugging)


def split_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Recursive character text splitter — tries to split on paragraph
    boundaries first, then sentences, then words, then characters.

    chunk_size : target size in characters (not tokens)
    overlap    : how many characters to repeat at the start of the next chunk
    """
    # Priority order of separators — try the "nicest" split first
    separators = ["\n\n", "\n", ". ", " ", ""]

    def _split(text: str, separators: List[str]) -> List[str]:
        if not text.strip():
            return []

        sep = separators[0]
        remaining_seps = separators[1:]

        # If the whole text already fits in one chunk, we're done
        if len(text) <= chunk_size:
            return [text.strip()]

        # Try splitting on this separator
        if sep == "":
            # Last resort: hard character split
            parts = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size - overlap)]
            return [p.strip() for p in parts if p.strip()]

        raw_parts = text.split(sep)

        chunks: List[str] = []
        current = ""

        for part in raw_parts:
            candidate = (current + sep + part).strip() if current else part.strip()

            if len(candidate) <= chunk_size:
                current = candidate
            else:
                # Save what we have so far
                if current:
                    chunks.append(current)

                # If the part itself is too long, recurse with finer separators
                if len(part) > chunk_size and remaining_seps:
                    sub_chunks = _split(part, remaining_seps)
                    if sub_chunks:
                        # carry overlap from last sub-chunk into next current
                        current = sub_chunks[-1][-overlap:] if overlap else ""
                        chunks.extend(sub_chunks[:-1])
                    else:
                        current = ""
                else:
                    # Start fresh with overlap from previous chunk
                    overlap_text = current[-overlap:] if overlap and current else ""
                    current = (overlap_text + " " + part).strip() if overlap_text else part.strip()

        if current:
            chunks.append(current)

        return chunks

    return _split(text, separators)


def chunk_pages(pages: List[ScrapedPage], chunk_size: int = 500, overlap: int = 50) -> List[TextChunk]:
    """
    Convert a list of ScrapedPage objects into a flat list of TextChunk objects.

    Each chunk remembers which URL it came from so we can show sources
    in the chat UI later.
    """
    all_chunks: List[TextChunk] = []

    for page in pages:
        if not page.text.strip():
            continue  # skip empty pages

        page_chunks = split_text(page.text, chunk_size=chunk_size, overlap=overlap)

        for idx, chunk_text in enumerate(page_chunks):
            all_chunks.append(TextChunk(
                text=chunk_text,
                source_url=page.url,
                chunk_index=idx,
            ))

    return all_chunks


# ── quick self-test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Fake a scraped page so we can test chunker.py standalone
    from scraper import ScrapedPage

    dummy_text = (
        "Introduction to RAG\n\n"
        "Retrieval-Augmented Generation (RAG) is a technique that combines "
        "a retrieval system with a language model. "
        "The retrieval system fetches relevant documents from a vector store. "
        "The language model then uses those documents as context to generate an answer.\n\n"
        "Why RAG?\n\n"
        "RAG helps ground the model's answers in real documents, reducing hallucination. "
        "It also allows the model to answer questions about content it was never trained on, "
        "such as your own website or internal documentation.\n\n"
        "How it works\n\n"
        "Step 1: Scrape and chunk your documents.\n"
        "Step 2: Embed each chunk into a vector.\n"
        "Step 3: Store vectors in a vector database.\n"
        "Step 4: At query time, embed the question and retrieve the top-k similar chunks.\n"
        "Step 5: Pass the chunks + question to an LLM and return the answer."
    )

    fake_page = ScrapedPage(url="https://example.com/rag", title="RAG Overview", text=dummy_text)

    chunks = chunk_pages([fake_page], chunk_size=200, overlap=30)

    print(f"Total chunks produced: {len(chunks)}\n")
    for i, c in enumerate(chunks):
        print(f"--- Chunk {i} (len={len(c.text)}) ---")
        print(c.text)
        print()