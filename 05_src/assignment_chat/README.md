# 📚 Lindsay's Bookshop — AI Book Recommendation Assistant

A conversational AI bookshop assistant built with LangGraph, ChromaDB, and Gradio. Lindsay is a warm, opinionated independent bookshop owner who helps users discover books, look up titles, and manage a personal reading list.

---

## Services

### Service 1 — Open Library API (api_service.py)
Fetches information about specific books or authors from the [Open Library API](https://openlibrary.org/developers/api), with Google Books as a fallback for descriptions. Raw API responses are rewritten conversationally by the LLM in Lindsay's voice.

### Service 2 — Semantic Book Search (semantic_service.py)
Handles vague, mood-based, or thematic queries (e.g. *"something dark and atmospheric set in the past"*) using semantic similarity search over a ChromaDB vector store of book summaries. Results are enriched with structured metadata from `data/books_metadata.csv`.

### Service 3 — Function Calling: Reading List & Filters (tools_service.py)
Uses LangChain tool calling to manage a session reading list and filter books by criteria. Available tools: `add_to_reading_list`, `remove_from_reading_list`, `get_reading_list`, `clear_reading_list`, `filter_books`, `recommend_similar`.

---

## Setup

### 1. Set up Environment and 
Dependent only deploying-ai-env environment; see SETUP.md in deploying-ai parent folder for details on creation and activation.

Ensure that the .secrets has your API_GATEWAY_KEY and the .env has the appropriate base_url. Also set the DEBUG flag in .env if you would like to see the Agent's routing decisions and/or error messages (printed to terminal).

### 2. Build the ChromaDB vector store
> ⚠️ **This step only needs to be run once.** The resulting `data/embeddings/` folder is committed to the repo so evaluators do not need to re-run this.

Download the CMU Book Summary Corpus:
1. Visit: http://www.cs.cmu.edu/~dbamman/booksummaries.html
2. Download `booksummaries.tar.gz`
3. Extract `booksummaries.txt` into the `data/` folder
4. Run:
```bash
python data/embed_books.py
```

#### Embedding Process
- **Dataset**: CMU Book Summary Corpus (~16k Wikipedia-sourced book plot summaries)
- **Subset used**: First 5,000 books with summaries longer than 100 characters, deduplicated by title
- **Embedding model**: `all-MiniLM-L6-v2` from `sentence-transformers` (384-dimensional embeddings)
- **Storage**: ChromaDB with file persistence at `data/embeddings/`, cosine similarity metric
- **Metadata**: Title, author, genres stored as ChromaDB metadata; full structured metadata saved to `data/books_metadata.csv` for pandas enrichment of query results

### 3. Run the app
```bash
python app.py
```
Then open http://localhost:7860 in your browser.

---

## Architecture

```
User (Gradio)
     ↓
LangGraph Agent
     ↓
  Router (LLM classifier)
  ↙         ↓         ↘
API      Semantic    Tool Call
Lookup   Search      (Function
(OL API) (ChromaDB)   Calling)
     ↘       ↓       ↙
      Response Node
           ↓
    Memory Manager
    (trim + summarize
     long conversations)
```

### Memory Management
Conversation history is managed by `MemoryManager` in `memory/memory_manager.py`. When the conversation exceeds 20 messages, the oldest messages are summarized into a rolling context note using the LLM, and the active window is trimmed to the last 12 messages. This keeps the context window manageable without losing important context from earlier in the conversation.

---

## Project Structure
```
bookshop/
├── app.py                   # Gradio UI entry point
├── agent.py                 # LangGraph agent + router
├── config.py                # Persona name, shop name, and all LLM prompt templates
├── services/
│   ├── llm_client.py        # Shared OpenAI client factory (AWS API Gateway)
│   ├── api_service.py       # Service 1: Open Library API
│   ├── semantic_service.py  # Service 2: ChromaDB semantic search
│   ├── tools_service.py     # Service 3: Function calling tools
│   └── guardrails.py        # Two-layer input guardrails (regex + LLM)
├── data/
│   ├── embed_books.py       # One-time embedding script
│   ├── books_metadata.csv   # Structured metadata (generated)
│   └── embeddings/          # ChromaDB persistent store (generated)
├── memory/
│   └── memory_manager.py    # Conversation memory + summarization
├── .env                     # OPENAI_MODEL (not committed)
├── .secrets                 # API_GATEWAY_KEY (not committed)
├── .env.example             # Template for .env
└── README.md
```

---

## Limitations

**Routing is not always accurate.** The LLM-based router classifies each message independently, which means ambiguous or context-dependent messages can be misrouted. For example, "tell me more about that one" after a recommendation may be sent to the wrong service because the router does not have access to the previous turn.

**The reading list is session-only.** The reading list managed by Service 3 lives in memory and is lost when the server restarts. There is no persistent storage between sessions.

**The semantic search dataset is static and dated.** The CMU Book Summary Corpus contains approximately 16,000 books sourced from Wikipedia (of which we take a subsample), with a bias towards older and more canonical titles. Recently published books are unlikely to appear in search results.

**Open Library descriptions are inconsistent.** Some books have rich descriptions via the Open Library API; others return very little, falling back to Google Books or no description at all. This affects the quality of Service 1 responses for less well-known titles.

**The guardrail LLM check adds latency.** Every message passes through a two-layer guardrail — a fast regex check followed by an LLM call. The LLM layer adds a small but noticeable delay to every response, including completely benign messages.

**The guardrails may be overly restrictive.** For example, books that may lexically overlap with restricted terms but not semantically can still be filtered out.

**Memory summarization can lose detail.** The rolling summarization strategy compresses older context into a few sentences. Specific details mentioned early in a conversation — such as a stated preference or a book title — may be lost or paraphrased imprecisely after summarization.

---

## Potential Next Steps

**Context-aware routing.** Pass the last one or two conversation turns to the router so it can correctly handle references like "add that to my list" or "tell me more about the second one." This would significantly reduce misrouting on follow-up messages.

**Hybrid search.** Augment the current semantic search with a BM25 keyword pass using LangChain's `EnsembleRetriever`. This would improve results for exact title or author name queries that pure semantic search handles poorly.

**Reading list persistence.** Save the reading list to a JSON file keyed by session ID so it survives server restarts.