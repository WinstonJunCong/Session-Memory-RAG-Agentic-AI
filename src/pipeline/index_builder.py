# pipeline/index_builder.py
# Uses Unstructured.io for semantic document parsing.
# Automatically handles markdown, tables, lists, and preserves heading hierarchy.

import chromadb
from llama_index.core import VectorStoreIndex, StorageContext, Settings, Document
from llama_index.core.schema import TextNode
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from unstructured.partition.md import partition_md
from unstructured.partition.text import partition_text
from unstructured.chunking.title import chunk_by_title
from pathlib import Path
import shutil

import config


def configure_settings():
    """Set global LlamaIndex settings."""
    if not config.GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY environment variable not set")
    
    Settings.embed_model = GoogleGenAIEmbedding(
        model_name=config.EMBED_MODEL,
        api_key=config.GOOGLE_API_KEY,
    )
    Settings.llm = GoogleGenAI(
        model=config.GEMINI_MODEL,
        api_key=config.GOOGLE_API_KEY,
    )
    print(f"[settings] Embed: {config.EMBED_MODEL} | LLM: {config.GEMINI_MODEL}")


def get_index_store():
    """Returns ChromaDB vector store for knowledge base."""
    db = chromadb.PersistentClient(path=config.CHROMA_INDEX_PATH)
    collection = db.get_or_create_collection(config.CHROMA_INDEX_COLLECTION)
    return ChromaVectorStore(chroma_collection=collection)


def partition_document(filepath: str):
    """
    Partition a document into elements using Unstructured.
    Automatically handles .md, .txt (transcripts), .pdf, .html
    """
    path = Path(filepath)
    ext = path.suffix.lower()

    if ext == ".md":
        return partition_md(filename=filepath)
    elif ext in (".txt", ".vtt", ".srt"):
        return partition_text(filename=filepath)
    else:
        from unstructured.partition.auto import partition
        return partition(filename=filepath)


def elements_to_nodes(chunks, source_file: str) -> list[TextNode]:
    """
    Convert Unstructured chunks into LlamaIndex TextNodes.
    Preserves all metadata: heading hierarchy, element type, source.
    """
    nodes = []
    for chunk in chunks:
        text = chunk.text.strip()
        if not text:
            continue

        meta = {
            "filename": Path(source_file).name,
            "source": source_file,
            "element_type": type(chunk).__name__,
        }

        if hasattr(chunk, "metadata"):
            um = chunk.metadata
            if hasattr(um, "page_number") and um.page_number:
                meta["page"] = um.page_number
            if hasattr(um, "parent_id") and um.parent_id:
                meta["parent_id"] = str(um.parent_id)
            if hasattr(um, "filename") and um.filename:
                meta["filename"] = Path(um.filename).name

        nodes.append(TextNode(text=text, metadata=meta))

    return nodes


def _delete_collection_safe(client, collection_name: str):
    """Delete ChromaDB collection and clean up orphaned physical folder."""
    collection_id = None
    
    try:
        collection = client.get_collection(collection_name)
        collection_id = collection.id
    except Exception:
        pass
    
    try:
        client.delete_collection(collection_name)
        print(f"[index_builder] Deleted collection: {collection_name}")
    except Exception as e:
        print(f"[index_builder] WARNING: delete_collection failed: {e}")
    
    if collection_id:
        folder_path = Path(config.CHROMA_INDEX_PATH) / str(collection_id)
        if folder_path.exists():
            shutil.rmtree(folder_path)
            print(f"[index_builder] Cleaned up folder: {collection_id}")
    
    _cleanup_orphaned_folders(client)


def _cleanup_orphaned_folders(client):
    """Remove any folders not referenced by active collections."""
    active_ids = {col.id for col in client.list_collections()}
    
    chroma_path = Path(config.CHROMA_INDEX_PATH)
    for entry in chroma_path.iterdir():
        if entry.is_dir() and entry.name not in active_ids and entry.name != ".chroma":
            shutil.rmtree(entry)
            print(f"[index_builder] Removed orphaned folder: {entry.name}")


def build_index(documents: list[Document]) -> VectorStoreIndex:
    """
    Full ingestion pipeline:
    1. Partition with Unstructured (structure-aware)
    2. Chunk by title sections (keeps Q&A pairs together)
    3. Convert to LlamaIndex nodes
    4. Embed and store in ChromaDB
    """
    configure_settings()

    chroma_client = chromadb.PersistentClient(path=config.CHROMA_INDEX_PATH)
    _delete_collection_safe(chroma_client, config.CHROMA_INDEX_COLLECTION)

    all_nodes = []
    for doc in documents:
        filepath = doc.metadata.get("file_path", "")
        if not filepath or not Path(filepath).exists():
            continue

        print(f"   Parsing: {Path(filepath).name}")

        elements = partition_document(filepath)

        chunks = chunk_by_title(
            elements,
            max_characters=config.CHUNK_MAX_CHARS,
            new_after_n_chars=config.CHUNK_SOFT_LIMIT,
            combine_text_under_n_chars=config.CHUNK_MIN_CHARS,
            multipage_sections=True,
        )

        nodes = elements_to_nodes(chunks, filepath)
        all_nodes.extend(nodes)
        print(f"      -> {len(nodes)} chunks")

    print(f"[index_builder] Total chunks: {len(all_nodes)}")


    index_store = get_index_store()
    storage_context = StorageContext.from_defaults(vector_store=index_store)

    print(f"[index_builder] Embedding {len(all_nodes)} chunks into ChromaDB...")
    index = VectorStoreIndex(
        all_nodes,
        storage_context=storage_context,
        show_progress=True,
    )

    print(f"[index_builder] Index built — {len(all_nodes)} chunks in ChromaDB")
    return index


def load_index() -> VectorStoreIndex:
    """Load existing index from ChromaDB."""
    index_store = get_index_store()
    index = VectorStoreIndex.from_vector_store(index_store)
    print("[index_builder] Index loaded from ChromaDB")
    return index
