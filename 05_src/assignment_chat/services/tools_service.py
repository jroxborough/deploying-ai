"""
Service 3: Function Calling Tools
Reading list management and book filtering via LLM function calling.
"""

import json
import os
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from typing import Optional
from services.llm_client import get_llm
from config import PERSONA, PROMPT_TOOLS


# In-memory reading list (persists for the session)
_reading_list: list[dict] = []


# ── Tool Definitions ──────────────────────────────────────────────────────────

@tool
def add_to_reading_list(title: str, author: str = "", notes: str = "") -> str:
    """Add a book to the user's reading list.
    
    Args:
        title: The title of the book to add.
        author: The author's name (optional).
        notes: Any personal notes about why you want to read it (optional).
    """
    entry = {"title": title, "author": author, "notes": notes}
    # Avoid duplicates
    for existing in _reading_list:
        if existing["title"].lower() == title.lower():
            return f"'{title}' is already on your reading list!"
    _reading_list.append(entry)
    return f"Added '{title}'{' by ' + author if author else ''} to your reading list."


@tool
def remove_from_reading_list(title: str) -> str:
    """Remove a book from the user's reading list.
    
    Args:
        title: The title of the book to remove.
    """
    global _reading_list
    original_count = len(_reading_list)
    _reading_list = [b for b in _reading_list if b["title"].lower() != title.lower()]
    if len(_reading_list) < original_count:
        return f"Removed '{title}' from your reading list."
    return f"I couldn't find '{title}' on your reading list."


@tool
def get_reading_list() -> str:
    """Show the user's current reading list."""
    if not _reading_list:
        return "Your reading list is empty! Ask me for recommendations and I'll help you fill it up."
    
    lines = ["Here's your reading list:\n"]
    for i, book in enumerate(_reading_list, 1):
        line = f"{i}. {book['title']}"
        if book.get("author"):
            line += f" — {book['author']}"
        if book.get("notes"):
            line += f"\n   📝 {book['notes']}"
        lines.append(line)
    return "\n".join(lines)


@tool
def clear_reading_list() -> str:
    """Clear all books from the user's reading list."""
    global _reading_list
    count = len(_reading_list)
    _reading_list = []
    return f"Cleared {count} book{'s' if count != 1 else ''} from your reading list."


@tool
def filter_books(
    max_pages: Optional[int] = None,
    min_rating: Optional[float] = None,
    genre: Optional[str] = None,
    mood: Optional[str] = None
) -> str:
    """Filter books from the semantic database by criteria like page count, rating, genre, or mood.
    This will trigger a filtered semantic search.
    
    Args:
        max_pages: Maximum number of pages (e.g. 300 for a quick read).
        min_rating: Minimum rating out of 5 (e.g. 4.0 for highly rated books).
        genre: Genre or category to filter by (e.g. 'mystery', 'sci-fi', 'romance').
        mood: Mood or vibe (e.g. 'cozy', 'dark', 'uplifting', 'thought-provoking').
    """
    # Build a natural language query from the filters for semantic search
    query_parts = []
    if mood:
        query_parts.append(mood)
    if genre:
        query_parts.append(genre)
    if max_pages:
        query_parts.append(f"short book under {max_pages} pages")
    if min_rating:
        query_parts.append(f"highly rated")

    if not query_parts:
        return "Please specify at least one filter (genre, mood, page count, or rating)."

    constructed_query = " ".join(query_parts)

    # Import here to avoid circular imports
    from services.semantic_service import semantic_search, format_search_results

    books = semantic_search(constructed_query, n_results=5)

    # Apply hard filters on metadata
    if max_pages:
        books = [b for b in books if not b.get("pages") or (str(b.get("pages", "")).isdigit() and int(b["pages"]) <= max_pages)]
    if min_rating:
        books = [b for b in books if not b.get("rating") or float(str(b.get("rating", 0))) >= min_rating]

    return format_search_results(constructed_query, books[:3])


@tool
def recommend_similar(title: str) -> str:
    """Find books similar to a given title using semantic search.
    
    Args:
        title: The title of the book to find similar books for.
    """
    from services.semantic_service import semantic_search, format_search_results
    from services.api_service import fetch_book_info

    # First try to get a description of the book to use as the query
    book_data = fetch_book_info(title)
    if book_data:
        subjects = book_data.get("subject", [])[:4]
        author = book_data.get("author_name", [""])[0]
        query = f"books like {title} by {author}: {' '.join(subjects)}"
    else:
        query = f"books similar to {title}"

    books = semantic_search(query, n_results=4)
    # Filter out the book itself
    books = [b for b in books if b["title"].lower() != title.lower()][:3]

    return format_search_results(f"books similar to {title}", books)


# ── Tool Registry ─────────────────────────────────────────────────────────────

TOOLS = [
    add_to_reading_list,
    remove_from_reading_list,
    get_reading_list,
    clear_reading_list,
    filter_books,
    recommend_similar,
]

TOOLS_BY_NAME = {t.name: t for t in TOOLS}


# ── Main Entry Point ──────────────────────────────────────────────────────────

def run_tool_agent(user_message: str, conversation_history: list = None) -> str:
    """
    Main entry point for Service 3.
    Runs a single LLM call with tools bound, executes any tool calls, 
    then returns the final response.
    """
    llm = get_llm(temperature=0.3).bind_tools(TOOLS)

    system_prompt = f"{PERSONA}\n\n{PROMPT_TOOLS}"

    messages = [SystemMessage(content=system_prompt)]
    if conversation_history:
        messages.extend(conversation_history[-6:])  # last 3 turns for context
    messages.append(HumanMessage(content=user_message))

    response = llm.invoke(messages)

    # Handle tool calls
    if response.tool_calls:
        messages.append(response)
        tool_results = []

        for tool_call in response.tool_calls:
            tool_fn = TOOLS_BY_NAME.get(tool_call["name"])
            if tool_fn:
                result = tool_fn.invoke(tool_call["args"])
                tool_results.append(
                    ToolMessage(content=str(result), tool_call_id=tool_call["id"])
                )

        messages.extend(tool_results)

        # Final response after tool execution
        final_llm = get_llm(temperature=0.7)
        final_response = final_llm.invoke(messages)
        return final_response.content

    return response.content