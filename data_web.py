import requests
from bs4 import BeautifulSoup
import os
import json
from urllib.parse import urljoin, urlparse


def get_text_from_url(url, visited=None, depth=2, text_data=None):
    if visited is None:
        visited = set()
    if text_data is None:
        text_data = {}

    # Limit recursion depth to avoid downloading too many pages
    if depth == 0 or url in visited:
        return text_data

    visited.add(url)

    try:
        # Fetch the webpage content
        response = requests.get(url)

        # Skip non-UTF-8 content
        if response.encoding.lower() != "utf-8":
            print(f"Skipping non-UTF-8 content from {url}")
            return text_data

        if response.status_code != 200:
            return text_data

        # Parse the content
        soup = BeautifulSoup(response.content, "html.parser")
        # Extract text content, ignoring images, videos, gifs, etc.
        text = soup.get_text()

        # Store the text content in the dictionary
        text_data[url] = text

        # Find and follow links recursively
        for link in soup.find_all("a", href=True):
            link_url = urljoin(url, link["href"])
            parsed_link = urlparse(link_url)
            # Process only links on the same domain
            if parsed_link.netloc == urlparse(url).netloc:
                get_text_from_url(link_url, visited, depth - 1, text_data)

    except Exception as e:
        print(f"Not including: {url}")

    return text_data


def save_to_text(text_data, filename):
    """Save the text content to a .txt file."""
    with open(filename, "w", encoding="utf-8") as f:
        for url, text in text_data.items():
            f.write(f"URL: {url}\n")
            f.write(text)
            f.write("\n" + "=" * 80 + "\n")


def save_to_json(text_data, filename):
    """Save the text content to a .json file."""
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(text_data, f, ensure_ascii=False, indent=4)


# Example usage
text_data = get_text_from_url("https://www.vit.edu/", depth=10)
# Save to JSON file
save_to_json(text_data, "vit_edu.json")
