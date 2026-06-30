from scraper import scrape_site
from chunker import chunk_pages
from vector_store import ingest_chunks, clear_collection

url = input("Website URL: ")

print("\nScraping website...")
pages = scrape_site(url)

print(f"\nScraped {len(pages)} pages")

print("\nChunking...")
chunks = chunk_pages(pages)

print(f"Created {len(chunks)} chunks")

print("\nClearing old database...")
clear_collection()

print("\nStoring embeddings...")
ingest_chunks(chunks)

print("\nDone!")