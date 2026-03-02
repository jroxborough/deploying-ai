"""
Service 2: Semantic Book Search
Embeds user queries and searches ChromaDB for matching book summaries.
Enriches results with metadata from a pandas CSV.
"""

import os
import pandas as pd
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from langchain_core.messages import HumanMessage, SystemMessage
from services.llm_client import get_llm
from config import PERSONA, PROMPT_RECOMMENDATIONS

# Paths
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
CHROMA_DIR = os.path.join(DATA_DIR, "embeddings")
METADATA_CSV = os.path.join(DATA_DIR, "books_metadata.csv")

COLLECTION_NAME = "book_summaries"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # fast, good quality, 384-dim


def get_chroma_collection():
    """Load the persistent ChromaDB collection."""
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
    return collection


def get_embedding_model():
    """Load the sentence transformer model."""
    return SentenceTransformer(EMBEDDING_MODEL)


def load_metadata() -> pd.DataFrame | None:
    """Load the book metadata CSV if it exists."""
    if os.path.exists(METADATA_CSV):
        return pd.read_csv(METADATA_CSV)
    return None


def semantic_search(query: str, n_results: int = 3) -> list[dict]:
    """
    Embed the query and retrieve the top N matching books from ChromaDB.
    Enriches results with metadata CSV where available.
    """
    model = get_embedding_model()
    query_embedding = model.encode(query).tolist()

    collection = get_chroma_collection()

    if collection.count() == 0:
        return []

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(n_results, collection.count()),
        include=["documents", "metadatas", "distances"]
    )

    books = []
    metadata_df = load_metadata()

    for i, doc in enumerate(results["documents"][0]):
        meta = results["metadatas"][0][i] if results["metadatas"] else {}
        distance = results["distances"][0][i] if results["distances"] else None
        similarity = round(1 - distance, 3) if distance is not None else None

        book = {
            "title": meta.get("title", "Unknown"),
            "author": meta.get("author", "Unknown"),
            "summary": doc,
            "similarity": similarity,
        }

        # Enrich from CSV metadata if available
        if metadata_df is not None and "title" in metadata_df.columns:
            match = metadata_df[metadata_df["title"].str.lower() == book["title"].lower()]
            if not match.empty:
                row = match.iloc[0]
                book["genres"] = row.get("genres", "")
                book["pages"] = row.get("pages", "")
                book["rating"] = row.get("rating", "")

        books.append(book)

    return books


def format_search_results(query: str, books: list[dict]) -> str:
    """
    Pass raw search results to LLM to write a natural recommendation response.
    """
    if not books:
        return "I searched my shelves but couldn't find anything that quite matched. Could you describe the mood or themes you're after a bit differently?"

    # Build context for the LLM
    books_text = ""
    for i, b in enumerate(books, 1):
        books_text += f"\n{i}. Title: {b['title']}\n"
        books_text += f"   Author: {b['author']}\n"
        if b.get("genres"):
            books_text += f"   Genres: {b['genres']}\n"
        if b.get("pages"):
            books_text += f"   Pages: {b['pages']}\n"
        if b.get("rating"):
            books_text += f"   Rating: {b['rating']}\n"
        books_text += f"   Summary excerpt: {b['summary'][:300]}...\n"

    llm = get_llm(temperature=0.7)

    messages = [
        SystemMessage(content=f"{PERSONA}\n\n{PROMPT_RECOMMENDATIONS}"),
        HumanMessage(content=f"The customer asked for: '{query}'\n\nHere are the books I found:\n{books_text}\n\nPlease recommend them naturally.")
    ]

    response = llm.invoke(messages)
    return response.content


def search_books(query: str) -> str:
    """Main entry point for Service 2."""
    books = semantic_search(query, n_results=3)
    return format_search_results(query, books)