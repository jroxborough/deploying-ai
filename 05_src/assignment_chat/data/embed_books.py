"""
embed_books.py
──────────────
One-time script to build the ChromaDB vector store from the CMU Book Summary Corpus.

Usage:
    python data/embed_books.py

Input:
    data/booksummaries.txt  ← download from:
    http://www.cs.cmu.edu/~dbamman/data/booksummaries.tar.gz

Output:
    data/embeddings/        ← ChromaDB persistent store (commit this to repo)
    data/books_metadata.csv ← structured metadata for enrichment

The CMU Book Summary Corpus format (tab-separated):
    Wikipedia ID | Freebase ID | Title | Author | Publication Date | Genres | Summary
"""

import os
import csv
import json
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import chromadb

# ── Config ────────────────────────────────────────────────────────────────────

DATA_DIR = os.path.dirname(__file__)
INPUT_FILE = os.path.join(DATA_DIR, "booksummaries.txt")
CHROMA_DIR = os.path.join(DATA_DIR, "embeddings")
METADATA_CSV = os.path.join(DATA_DIR, "books_metadata.csv")

COLLECTION_NAME = "book_summaries"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Limit dataset size to keep ChromaDB manageable and under github limit
MAX_BOOKS = 3000

# Minimum summary length to filter out stubs
MIN_SUMMARY_LENGTH = 100


# ── Parse CMU Corpus ──────────────────────────────────────────────────────────

def parse_cmu_corpus(filepath: str) -> list[dict]:
    """Parse the CMU Book Summary Corpus tab-separated file."""
    books = []
    seen_titles = set()

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 7:
                continue

            _, _, title, author, pub_date, genres_raw, summary = parts[:7]

            # Skip short summaries
            if len(summary) < MIN_SUMMARY_LENGTH:
                continue

            # Skip duplicates
            key = title.lower().strip()
            if key in seen_titles:
                continue
            seen_titles.add(key)

            # Parse genres from Freebase JSON-like format
            genres = []
            try:
                genres_dict = json.loads(genres_raw)
                genres = list(genres_dict.values())[:4]
            except (json.JSONDecodeError, AttributeError):
                pass

            books.append({
                "title": title.strip(),
                "author": author.strip(),
                "pub_date": pub_date.strip(),
                "genres": ", ".join(genres),
                "summary": summary.strip(),
            })

            if len(books) >= MAX_BOOKS:
                break

    return books


# ── Build ChromaDB ────────────────────────────────────────────────────────────

def build_chroma_store(books: list[dict]):
    """Embed book summaries and store in ChromaDB with file persistence."""
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print(f"Initialising ChromaDB at: {CHROMA_DIR}")
    os.makedirs(CHROMA_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Drop and recreate for a clean build
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    # Embed and insert in batches
    BATCH_SIZE = 100
    for i in tqdm(range(0, len(books), BATCH_SIZE), desc="Embedding books"):
        batch = books[i:i + BATCH_SIZE]

        documents = [b["summary"] for b in batch]
        metadatas = [{"title": b["title"], "author": b["author"], "genres": b["genres"]} for b in batch]
        ids = [f"book_{i + j}" for j in range(len(batch))]

        embeddings = model.encode(documents, show_progress_bar=False).tolist()

        collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )

    print(f"✓ Stored {collection.count()} books in ChromaDB")


# ── Save Metadata CSV ─────────────────────────────────────────────────────────

def save_metadata_csv(books: list[dict]):
    """Save structured metadata to CSV for pandas enrichment."""
    df = pd.DataFrame(books)[["title", "author", "pub_date", "genres"]]
    df.to_csv(METADATA_CSV, index=False)
    print(f"✓ Saved metadata CSV: {METADATA_CSV} ({len(df)} rows)")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not os.path.exists(INPUT_FILE):
        print(f"""
ERROR: Input file not found: {INPUT_FILE}

Please download the CMU Book Summary Corpus:
  1. Go to: http://www.cs.cmu.edu/~dbamman/booksummaries.html
  2. Download: booksummaries.tar.gz
  3. Extract booksummaries.txt into the data/ folder
  4. Re-run this script
        """)
        exit(1)

    print(f"Parsing CMU Book Summary Corpus (max {MAX_BOOKS} books)...")
    books = parse_cmu_corpus(INPUT_FILE)
    print(f"Parsed {len(books)} books")

    build_chroma_store(books)
    save_metadata_csv(books)

    print("\n✓ Done! You can now commit the data/embeddings/ folder to your repo.")
    print("  Evaluators won't need to re-run this script.")
