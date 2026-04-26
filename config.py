# config.py — tweak these to change models / behaviour
import os

# ---------------- Google GenAI ----------------
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = "gemini-3.1-flash-lite-preview"

# ---------------- Embedding ----------------
EMBED_MODEL = "gemini-embedding-001"

# ---------------- Retrieval ----------------
TOP_K = 6              # chunks returned by retriever
MMR_LAMBDA = 0.7      # 0=max diversity, 1=max relevance

# ---------------- Knowledge Base Storage ----------------
CHROMA_INDEX_PATH = "./data/chroma_db"
CHROMA_INDEX_COLLECTION = "qna_docs"

# ---------------- Memory Storage ----------------
CHROMA_MEMORY_PATH = "./data/chroma_db"
CHROMA_MEMORY_COLLECTION = "conversation_memory"

# ---------------- Memory (Vector-backed + Hybrid) ----------------
MEMORY_TOKEN_LIMIT = 2000      # tokens for recent messages in context
MEMORY_TOP_K = 4               # vector search results for semantic retrieval
MEMORY_RECENT_COUNT = 5        # last N messages to always include (recent buffer)

# ---------------- Unstructured.io chunking params ----------------
CHUNK_MAX_CHARS = 2000      # hard ceiling per chunk
CHUNK_SOFT_LIMIT = 1000    # preferred split point
CHUNK_MIN_CHARS = 350      # merge sections smaller than this

# ---------------- Debugging ----------------
DEBUG_LLM = False
DEBUG_TIMING = False
