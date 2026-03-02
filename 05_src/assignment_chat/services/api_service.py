"""
Service 1: Open Library API
Fetches book/author information and rewrites it conversationally.
"""

import sys
import os
import requests
from langchain_core.messages import HumanMessage, SystemMessage

# Fix import path when running this file directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.llm_client import get_llm
from config import PERSONA, PROMPT_BOOK_LOOKUP


OPEN_LIBRARY_SEARCH = "https://openlibrary.org/search.json"
OPEN_LIBRARY_WORKS = "https://openlibrary.org/works/{}.json"
OPEN_LIBRARY_AUTHOR = "https://openlibrary.org/authors/{}.json"
GOOGLE_BOOKS = "https://www.googleapis.com/books/v1/volumes"

# Set DEBUG=true in your .env to see detailed logs
DEBUG = os.getenv("DEBUG", "false").lower() == "true"


def debug(msg: str):
    """Print debug messages only when DEBUG=true in environment."""
    if DEBUG:
        print(f"  [API] {msg}")


# ── Query Extraction ──────────────────────────────────────────────────────────

def extract_title_and_author(user_query: str) -> str:
    """
    Use the LLM to extract a clean title + author search string from the
    user's raw message before hitting the Open Library API.
    Prevents the wrong book being returned due to extra words in the query.
    """
    llm = get_llm(temperature=0)
    response = llm.invoke([
        SystemMessage(content="""Extract the book title and author name from the user's message 
        and return them as a clean search string like 'Title Author'. 
        If only a title is mentioned, return just the title.
        Return ONLY the search string, nothing else — no explanation, no punctuation."""),
        HumanMessage(content=user_query)
    ])
    cleaned = response.content.strip()
    debug(f"Extracted search query: '{cleaned}' from '{user_query}'")
    return cleaned


# ── API Fetch Functions ───────────────────────────────────────────────────────

def fetch_book_info(query: str) -> tuple[dict | None, str | None]:
    """
    Search Open Library for a book.
    Returns (data, error) — error is None on success.
    Distinguishes between not-found and unreachable.
    """
    params = {
        "q": query,
        "limit": 3,
        "fields": "key,title,author_name,first_publish_year,subject,number_of_pages_median,ratings_average"
    }
    debug(f"Calling Open Library search: '{query}'")
    try:
        response = requests.get(OPEN_LIBRARY_SEARCH, params=params, timeout=10)
        response.raise_for_status()
        docs = response.json().get("docs", [])
        if not docs:
            debug("No results returned from Open Library")
            return None, None                          # not found — no error
        debug(f"Top result: '{docs[0].get('title')}' by {docs[0].get('author_name', ['?'])[0]}")
        return docs[0], None                           # success
    except requests.exceptions.ConnectionError:
        debug("Connection error — could not reach Open Library")
        return None, "connection_error"
    except requests.exceptions.Timeout:
        debug("Request timed out")
        return None, "timeout"
    except requests.exceptions.HTTPError as e:
        debug(f"HTTP error: {e.response.status_code}")
        return None, f"http_error_{e.response.status_code}"
    except requests.RequestException as e:
        debug(f"Unexpected request error: {e}")
        return None, "request_error"


def fetch_book_description(work_key: str) -> str:
    """
    Fetch the description/summary for a specific work from Open Library.
    Returns empty string on any failure — description is optional enrichment.
    """
    debug(f"Fetching description for work key: {work_key}")
    try:
        url = OPEN_LIBRARY_WORKS.format(work_key.replace("/works/", ""))
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        desc = data.get("description", "")
        if isinstance(desc, dict):
            return desc.get("value", "")
        return desc or ""
    except requests.exceptions.ConnectionError:
        debug("Connection error fetching description — skipping")
        return ""
    except requests.exceptions.Timeout:
        debug("Timeout fetching description — skipping")
        return ""
    except requests.RequestException:
        return ""


def fetch_google_books_description(title: str, author: str = "") -> str:
    """
    Fallback: fetch description from Google Books.
    Returns empty string on any failure.
    """
    query = f"{title} {author}".strip()
    debug(f"Falling back to Google Books: '{query}'")
    params = {"q": query, "maxResults": 1}
    try:
        response = requests.get(GOOGLE_BOOKS, params=params, timeout=10)
        response.raise_for_status()
        items = response.json().get("items", [])
        if items:
            return items[0].get("volumeInfo", {}).get("description", "")
    except requests.exceptions.ConnectionError:
        debug("Connection error fetching Google Books description — skipping")
    except requests.exceptions.Timeout:
        debug("Timeout fetching Google Books description — skipping")
    except requests.RequestException:
        pass
    return ""


# ── User-Facing Error Messages ────────────────────────────────────────────────

ERROR_MESSAGES = {
    "connection_error": (
        "I'm sorry, I can't seem to reach Open Library right now — "
        "it may be a network issue. Please try again in a moment."
    ),
    "timeout": (
        "Open Library is taking too long to respond right now. "
        "Please try again shortly."
    ),
    "request_error": (
        "Something went wrong contacting Open Library. Please try again."
    ),
}

def get_error_message(error: str) -> str:
    """Return a user-facing message for a given error code."""
    if error.startswith("http_error_"):
        code = error.replace("http_error_", "")
        return f"Open Library returned an unexpected error (HTTP {code}). Please try again later."
    return ERROR_MESSAGES.get(error, "Something went wrong. Please try again.")


# ── Main Entry Point ──────────────────────────────────────────────────────────

def lookup_book(query: str) -> str:
    """
    Main entry point for Service 1.
    Cleans the query, fetches book data, and rewrites it conversationally via LLM.
    """
    search_query = extract_title_and_author(query)
    book_data, error = fetch_book_info(search_query)

    # Handle API errors distinctly from not-found
    if error:
        debug(f"Returning error message for: {error}")
        return get_error_message(error)

    # Handle not found
    if not book_data:
        return (
            f"I searched high and low but couldn't find anything on Open Library for "
            f"'{query}'. Could you try a slightly different title or author name?"
        )

    title = book_data.get("title", "Unknown Title")
    authors = book_data.get("author_name", ["Unknown Author"])
    year = book_data.get("first_publish_year", "unknown year")
    pages = book_data.get("number_of_pages_median")
    rating = book_data.get("ratings_average")
    subjects = book_data.get("subject", [])[:6]
    work_key = book_data.get("key", "")

    # Try to get a description — fail gracefully if unavailable
    description = ""
    if work_key:
        description = fetch_book_description(work_key)
    if not description:
        description = fetch_google_books_description(title, authors[0] if authors else "")
    debug(f"Description: {'found' if description else 'none'} ({len(description)} chars)")

    raw_info = f"""
Title: {title}
Author(s): {', '.join(authors)}
First Published: {year}
{"Pages: " + str(pages) if pages else ""}
{"Average Rating: " + f"{rating:.1f}/5" if rating else ""}
{"Subjects/Genres: " + ', '.join(subjects) if subjects else ""}
{"Description: " + description if description else "No description available."}
""".strip()

    llm = get_llm(temperature=0.7)
    messages = [
        SystemMessage(content=f"{PERSONA}\n\n{PROMPT_BOOK_LOOKUP}"),
        HumanMessage(content=f"Here's the raw data for a book the customer asked about:\n\n{raw_info}\n\nPlease tell them about it naturally.")
    ]

    response = llm.invoke(messages)
    return response.content


# ── Manual Tests ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import unittest.mock as mock

    os.environ["DEBUG"] = "true"
    DEBUG = True

    test_cases = [
        # (label, query, expected: 'found' | 'not_found' | 'error', simulate_error)
        ("Well-known classic",    "Tell me about Dune by Frank Herbert",       "found",     None),
        ("Classic no author",     "What is Moby Dick about?",                  "found",     None),
        ("Less obvious title",    "Tell me about Middlemarch by George Eliot", "found",     None),
        ("Ambiguous title",       "Tell me about The Road",                    "found",     None),
        ("Raw title + author",    "Dune Frank Herbert",                        "found",     None),
        ("Graceful not-found",    "a book called xyzxyzxyz by nobody",         "not_found", None),
        ("Timeout simulation",    "Dune Frank Herbert",                        "error",     requests.exceptions.Timeout),
        ("Connection simulation", "Dune Frank Herbert",                        "error",     requests.exceptions.ConnectionError),
    ]

    passed = 0
    failed = 0

    for label, query, expected, simulate_error in test_cases:
        print(f"\n{'═' * 55}")
        print(f"TEST: {label}")
        print(f"Query: '{query}'")
        print(f"{'─' * 55}")

        if simulate_error:
            # Patch requests.get to raise the simulated error
            with mock.patch("requests.get", side_effect=simulate_error):
                book_data, error = fetch_book_info(query)
        else:
            cleaned = extract_title_and_author(query)
            print(f"Cleaned query: '{cleaned}'")
            book_data, error = fetch_book_info(cleaned)

        # Show what the API returned
        if error:
            print(f"API error: '{error}'")
            print(f"User message: \"{get_error_message(error)}\"")
            actual = "error"
        elif book_data:
            print(f"API returned: '{book_data.get('title')}' by {book_data.get('author_name', ['?'])[0]}")
            desc = fetch_book_description(book_data.get("key", ""))
            if desc:
                print(f"Description: Open Library ({len(desc)} chars)")
            else:
                fallback = fetch_google_books_description(book_data.get("title", ""), "")
                print(f"Description: {'Google Books fallback' if fallback else 'none found'}")
            actual = "found"
        else:
            print("API returned: no results")
            actual = "not_found"

        # Full pipeline only for real API calls
        if not simulate_error:
            print("\nFull LLM response:")
            print(lookup_book(query))

        # Pass/fail
        if actual == expected:
            print(f"\n✓ PASS")
            passed += 1
        else:
            print(f"\n✗ FAIL (expected '{expected}', got '{actual}')")
            failed += 1

    print(f"\n{'═' * 55}")
    print(f"Results: {passed} passed, {failed} failed out of {len(test_cases)} tests")