# agent/memory.py
# Hybrid Memory: Recent messages buffer + Semantic search
# Recent messages stored in-memory for guaranteed recency
# All messages stored in ChromaDB for semantic retrieval

from typing import List, Optional
import chromadb
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.vector_stores.chroma import ChromaVectorStore
import config


class ConversationMemory:
    """
    Hybrid vector-backed memory using ChromaDB + in-memory recent buffer.

    Memory retrieval strategy:
      1. Recent buffer: last N messages from in-memory cache (guaranteed recency)
      2. Semantic search: relevant past context via vector similarity
      3. Deduplication: skip semantic results if already covered by recent buffer
    """

    def __init__(self):
        self.chroma_client = chromadb.PersistentClient(path=config.CHROMA_MEMORY_PATH)
        self.collection = self.chroma_client.get_or_create_collection(config.CHROMA_MEMORY_COLLECTION)
        self.vector_store = ChromaVectorStore(chroma_collection=self.collection)
        self._recent_cache: List[dict] = []  # in-memory recent messages
        self._message_order = 0  # insertion counter
        self._initialized = False

    def initialize(self):
        from llama_index.llms.google_genai import GoogleGenAI
        from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

        if not config.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY not set")

        self.llm = GoogleGenAI(model=config.GEMINI_MODEL, api_key=config.GOOGLE_API_KEY, max_retries=0)
        self.embed_model = GoogleGenAIEmbedding(model_name=config.EMBED_MODEL, api_key=config.GOOGLE_API_KEY)
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model
        self._initialized = True
        print("[memory] Initialized")

    def get_recent_messages(self) -> List[dict]:
        """Return last N messages from recent cache, newest last."""
        return self._recent_cache[-config.MEMORY_RECENT_COUNT:]

    def _format_recent_context(self, recent: List[dict]) -> str:
        """Format recent messages for prompt."""
        if not recent:
            return ""
        lines = ["Recent conversation:"]
        for msg in recent:
            lines.append(f"- {msg['role']}: {msg['content']}")
        return "\n".join(lines)

    def _get_semantic_context(self, query: str) -> str:
        """Get relevant past context via semantic search."""
        try:
            index = VectorStoreIndex.from_vector_store(self.vector_store)
            retriever = index.as_retriever(similarity_top_k=config.MEMORY_TOP_K)
            nodes = retriever.retrieve(query)

            if nodes:
                lines = []
                for node in nodes:
                    text = node.text[:200] if len(node.text) > 200 else node.text
                    lines.append(f"- {text}")
                return "\n".join(lines)
        except Exception as e:
            print(f"[memory] Semantic search error: {e}")
        return ""

    def get_context(self, query: str) -> str:
        """
        Get combined memory context: recent buffer + semantic search.
        Recent messages are always included first.
        Semantic results are skipped if already covered by recent buffer.
        """
        if not self._initialized:
            return ""

        parts = []

        recent = self.get_recent_messages()
        if recent:
            recent_text = self._format_recent_context(recent)
            parts.append(recent_text)
            recent_combined = recent_text
        else:
            recent_combined = ""

        semantic = self._get_semantic_context(query)
        if semantic:
            semantic_normalized = semantic.lower().strip()
            recent_normalized = recent_combined.lower().strip()

            if semantic_normalized and semantic_normalized not in recent_normalized:
                parts.append("Relevant past context:\n" + semantic)

        return "\n\n".join(parts) if parts else ""

    def add_message(self, role: str, content: str):
        """
        Add a message to conversation history.
        Stores in in-memory cache (recent buffer) and ChromaDB (semantic search).
        """
        if not self._initialized:
            self.initialize()

        self._message_order += 1

        msg = {"role": role, "content": content, "order": self._message_order}
        self._recent_cache.append(msg)
        self._recent_cache = self._recent_cache[-config.MEMORY_RECENT_COUNT * 2:]

        try:
            index = VectorStoreIndex.from_vector_store(self.vector_store)
            node = TextNode(
                text=f"{role}: {content}",
                metadata={"role": role, "order": self._message_order}
            )
            index.insert(node)
        except Exception as e:
            print(f"[memory] ChromaDB insert error: {e}")


_conversation_memory = None


def get_memory():
    """Get or create the global conversation memory instance."""
    global _conversation_memory
    if _conversation_memory is None:
        _conversation_memory = ConversationMemory()
    return _conversation_memory