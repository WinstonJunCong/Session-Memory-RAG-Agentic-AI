#!/usr/bin/env python3
# ingest.py — Run this once to load all your documents into ChromaDB.
# Edit the SOURCES section below to point to your files/URLs/videos.

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.ingest.text_loader import load_text_files
from src.pipeline.index_builder import build_index

# ============================================================
#  👇 EDIT THIS SECTION — point to your actual sources
# ============================================================

TEXT_FOLDERS = [
    "./data/input",           # all .txt and .md files in this folder
]

HTML_URLS = [
    # "https://en.wikipedia.org/wiki/Retrieval-augmented_generation",
    # "https://docs.llamaindex.ai/en/stable/",
]

VIDEO_FILES = [
    # "./videos/lecture.mp4",
    # "./videos/interview.mp3",
]

# Whisper model size: "tiny" | "base" | "small" | "medium" | "large"
WHISPER_MODEL = "base"

# ============================================================

def main():
    all_docs = []

    print("\n>> Loading text/markdown files...")
    for folder in TEXT_FOLDERS:
        docs = load_text_files(folder)
        all_docs.extend(docs)
        print(f"   {len(docs)} docs from {folder}")

    # print("\n>> Loading HTML pages...")
    # if HTML_URLS:
    #     docs = load_html_urls(HTML_URLS)
    #     all_docs.extend(docs)
    #     print(f"   {len(docs)} pages loaded")
    # else:
    #     print("   (no URLs configured)")

    # print("\n>> Transcribing videos...")
    # if VIDEO_FILES:
    #     docs = load_videos(VIDEO_FILES, model_size=WHISPER_MODEL)
    #     all_docs.extend(docs)
    #     print(f"   {len(docs)} segments transcribed")
    # else:
    #     print("   (no video files configured)")

    if not all_docs:
        print("\nWARNING: No documents loaded! Check your TEXT_FOLDERS / HTML_URLS / VIDEO_FILES above.")
        return

    print(f"\n>> Building index from {len(all_docs)} total documents...")
    build_index(all_docs)

    print("\nDone! Run `python query.py` to start asking questions.\n")


if __name__ == "__main__":
    main()
