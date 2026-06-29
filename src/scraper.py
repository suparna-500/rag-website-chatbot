"""
scraper.py  (Day 1 — place your real scraper here)
This stub defines ScrapedPage so Day 2 modules can import it.
Replace with your full scraper.py from Day 1.
"""

from dataclasses import dataclass
from typing import List
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse


@dataclass
class ScrapedPage:
    url: str
    title: str
    text: str


def scrape_site(start_url: str, max_depth: int = 2, max_pages: int = 40) -> List[ScrapedPage]:
    """
    Recursively crawl start_url, staying within the same domain.
    Returns a list of ScrapedPage objects.
    """
    domain = urlparse(start_url).netloc
    visited = set()
    pages: List[ScrapedPage] = []
    queue = [(start_url, 0)]  # (url, depth)

    headers = {"User-Agent": "Mozilla/5.0 (compatible; RAGBot/1.0)"}

    while queue and len(pages) < max_pages:
        url, depth = queue.pop(0)

        if url in visited:
            continue
        visited.add(url)

        try:
            resp = requests.get(url, headers=headers, timeout=10)
            if "text/html" not in resp.headers.get("Content-Type", ""):
                continue
            resp.raise_for_status()
        except Exception as e:
            print(f"  [skip] {url} — {e}")
            continue

        soup = BeautifulSoup(resp.text, "html.parser")

        # Remove noise
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        title = soup.title.string.strip() if soup.title else url

        # Preserve structure: headings → lines, tables → pipe-separated
        lines = []
        for el in soup.find_all(["h1","h2","h3","h4","h5","h6","p","li","tr"]):
            if el.name.startswith("h"):
                lines.append(f"\n{'#'*(int(el.name[1]))} {el.get_text(' ', strip=True)}")
            elif el.name == "tr":
                cells = [td.get_text(" ", strip=True) for td in el.find_all(["th","td"])]
                lines.append(" | ".join(cells))
            else:
                t = el.get_text(" ", strip=True)
                if t:
                    lines.append(t)

        text = "\n".join(lines)
        pages.append(ScrapedPage(url=url, title=title, text=text))
        print(f"  [scraped] ({len(pages)}/{max_pages}) {url}")

        # Enqueue child links
        if depth < max_depth:
            for a in soup.find_all("a", href=True):
                link = urljoin(url, a["href"]).split("#")[0].split("?")[0]
                if urlparse(link).netloc == domain and link not in visited:
                    queue.append((link, depth + 1))

    print(f"\nDone. Scraped {len(pages)} pages.")
    return pages