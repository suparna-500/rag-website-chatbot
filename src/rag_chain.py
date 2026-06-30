"""
rag_chain.py
------------
Retrieves the most relevant chunks from ChromaDB and uses Gemini
to answer the user's question based ONLY on those chunks.
"""

import google.generativeai as genai

from config import GEMINI_API_KEY
from vector_store import similarity_search

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

# Fast and inexpensive model
model = genai.GenerativeModel("gemini-2.5-flash")


def build_prompt(question: str, retrieved_chunks: list) -> str:
    """
    Build the prompt sent to Gemini.
    """

    context = ""

    for i, (text, url, distance) in enumerate(retrieved_chunks, start=1):
        context += f"""
Source {i}
URL: {url}

{text}

----------------------------------------
"""

    prompt = f"""
You are a helpful AI assistant.

Answer the user's question ONLY using the website content provided below.

If the answer cannot be found in the provided context, reply:

"I couldn't find that information in the scraped website."

Website Context

{context}

User Question:
{question}

Answer:
"""

    return prompt


def ask_question(question: str, top_k: int = 5):
    """
    Complete RAG pipeline.

    Returns:
        answer
        retrieved_sources
    """

    # Retrieve similar chunks
    retrieved = similarity_search(question, top_k=top_k)

    if not retrieved:
        return (
            "The knowledge base is empty. Please scrape a website first.",
            [],
        )

    prompt = build_prompt(question, retrieved)

    response = model.generate_content(prompt)

    answer = response.text.strip()

    sources = []

    for _, url, _ in retrieved:
        if url not in sources:
            sources.append(url)

    return answer, sources


if __name__ == "__main__":

    while True:

        question = input("\nAsk a question (type 'exit' to quit): ")

        if question.lower() == "exit":
            break

        answer, sources = ask_question(question)

        print("\nAnswer")
        print("----------------------------------")
        print(answer)

        print("\nSources")
        print("----------------------------------")

        for src in sources:
            print(src)