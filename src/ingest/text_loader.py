# ingest/text_loader.py
# Loads .txt and .md files from a folder recursively.

from pathlib import Path
from llama_index.core import Document


def load_text_files(folder: str) -> list[Document]:
    """
    Recursively loads all .txt and .md files from a folder.
    Returns a list of LlamaIndex Document objects.
    """
    docs = []
    folder_path = Path(folder)

    if not folder_path.exists():
        print(f"[text_loader] Folder not found: {folder}")
        return docs

    for path in folder_path.rglob("*"):
        if path.suffix.lower() in {".txt", ".md"}:
            try:
                text = path.read_text(encoding="utf-8")
                if text.strip():
                    docs.append(Document(
                        text=text,
                        metadata={
                            "source": str(path),
                            "type": "text",
                            "filename": path.name,
                            "file_path": str(path.absolute()),
                        }
                    ))
                    print(f"[text_loader] Loaded: {path.name}")
            except Exception as e:
                print(f"[text_loader] Error reading {path}: {e}")

    return docs
