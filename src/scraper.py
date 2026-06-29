"""
scraper.py
-----------
A simple recursive, same-domain web scraper.

What it does:
- Starts from a given URL
- Follows internal links (same domain only) up to a max depth
- Stops after collecting a max number of pages (to keep things fast)
- Extracts clean visible text from each page (headings, paragraphs, lists, tables)
- Skips non-HTML resources (images, PDFs, etc.)

This is intentionally kept simple for a 7-day project timeline.
"""

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time


class WebsiteScraper:
    def __init__(self, start_url: str, max_depth: int = 2, max_pages: int = 30, delay: float = 0.5):
        """
        :param start_url: The URL to start scraping from.
        :param max_depth: How many "clicks" deep from the start page to follow links.
        :param max_pages: Hard cap on total pages scraped (keeps it fast).
        :param delay: Seconds to wait between requests (be polite to the server).
        """
        self.start_url = start_url
        self.max_depth = max_depth
        self.max_pages = max_pages
        self.delay = delay

        self.domain = urlparse(start_url).netloc
        self.visited = set()
        self.pages = []  # list of dicts: {"url": ..., "title": ..., "text": ...}

        # Pretend to be a normal browser so sites don't block us outright
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) RAGChatbotScraper/1.0"
        }

    def is_same_domain(self, url: str) -> bool:
        """Check if a URL belongs to the same domain as the start URL."""
        return urlparse(url).netloc == self.domain

    def is_valid_page(self, url: str) -> bool:
        """Filter out non-HTML resources we don't want to scrape."""
        skip_extensions = (
            ".pdf", ".jpg", ".jpeg", ".png", ".gif", ".svg",
            ".zip", ".mp4", ".mp3", ".css", ".js", ".ico", ".woff", ".woff2"
        )
        path = urlparse(url).path.lower()
        return not path.endswith(skip_extensions)

    def clean_url(self, url: str) -> str:
        """Remove fragments (#section) so we don't treat anchors as separate pages."""
        parsed = urlparse(url)
        return parsed._replace(fragment="").geturl()

    def extract_text(self, soup: BeautifulSoup) -> str:
        """
        Extract readable text from the page, preserving some structure:
        - Headings get a newline before/after
        - Tables are converted into readable rows
        - Scripts/styles/nav/footer are removed (mostly noise)
        """
        # Remove elements that are rarely useful content
        for tag in soup(["script", "style", "noscript", "nav", "footer", "header", "form"]):
            tag.decompose()

        text_parts = []

        for element in soup.find_all(["h1", "h2", "h3", "h4", "p", "li", "table"]):
            if element.name == "table":
                # Convert table rows into readable "col1 | col2 | col3" lines
                rows = element.find_all("tr")
                for row in rows:
                    cols = [c.get_text(strip=True) for c in row.find_all(["td", "th"])]
                    if cols:
                        text_parts.append(" | ".join(cols))
            else:
                text = element.get_text(strip=True)
                if text:
                    if element.name in ["h1", "h2", "h3", "h4"]:
                        text_parts.append(f"\n## {text}\n")
                    else:
                        text_parts.append(text)

        return "\n".join(text_parts)

    def get_links(self, soup: BeautifulSoup, current_url: str):
        """Find all valid same-domain links on the page."""
        links = []
        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]
            full_url = urljoin(current_url, href)
            full_url = self.clean_url(full_url)

            if self.is_same_domain(full_url) and self.is_valid_page(full_url):
                links.append(full_url)
        return links

    def scrape_page(self, url: str):
        """Fetch and parse a single page. Returns (title, text, links) or None on failure."""
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()

            content_type = response.headers.get("Content-Type", "")
            if "text/html" not in content_type:
                return None

            soup = BeautifulSoup(response.text, "html.parser")
            title = soup.title.string.strip() if soup.title and soup.title.string else url
            text = self.extract_text(soup)
            links = self.get_links(soup, url)

            return title, text, links

        except requests.RequestException as e:
            print(f"[WARN] Failed to fetch {url}: {e}")
            return None

    def crawl(self):
        """
        Run a breadth-first recursive crawl starting from self.start_url.
        Stops when max_depth or max_pages is reached.
        """
        queue = [(self.clean_url(self.start_url), 0)]  # (url, depth)

        while queue and len(self.pages) < self.max_pages:
            current_url, depth = queue.pop(0)

            if current_url in self.visited:
                continue
            if depth > self.max_depth:
                continue

            print(f"[INFO] Scraping (depth {depth}): {current_url}")
            self.visited.add(current_url)

            result = self.scrape_page(current_url)
            if result is None:
                continue

            title, text, links = result

            if text.strip():  # only keep pages that actually have content
                self.pages.append({
                    "url": current_url,
                    "title": title,
                    "text": text
                })

            # Queue up new links for the next depth level
            if depth < self.max_depth:
                for link in links:
                    if link not in self.visited:
                        queue.append((link, depth + 1))

            time.sleep(self.delay)  # be polite, avoid hammering the server

        print(f"[INFO] Done. Scraped {len(self.pages)} pages.")
        return self.pages


if __name__ == "__main__":
    # Quick manual test - run: python scraper.py
    test_url = input("Enter a URL to scrape: ").strip()

    scraper = WebsiteScraper(start_url=test_url, max_depth=2, max_pages=10)
    pages = scraper.crawl()

    for page in pages:
        print("\n" + "=" * 80)
        print(f"URL: {page['url']}")
        print(f"TITLE: {page['title']}")
        print(f"TEXT PREVIEW: {page['text'][:300]}...")