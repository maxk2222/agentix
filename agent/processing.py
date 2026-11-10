"""Cleaning and chunking helpers for the ingestion pipeline."""

import logging
import re
import time
import unicodedata
from typing import List, Optional

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.logging_utils import colorize_process_time

logger = logging.getLogger(__name__)

# Precompile common boilerplate patterns for performance and safety
_BOILERPLATE_PATTERNS = [
    re.compile(r'all rights reserved.*', flags=re.IGNORECASE),
    re.compile(r'this document is.*', flags=re.IGNORECASE),
    re.compile(r'version as of: \d{2}\.\d{2}\.\d{4}', flags=re.IGNORECASE),
]

_LIGATURE_TRANSLATION = str.maketrans(
    {
        "ﬀ": "ff",
        "ﬁ": "fi",
        "ﬂ": "fl",
        "ﬃ": "ffi",
        "ﬄ": "ffl",
        "ﬅ": "st",
        "ﬆ": "st",
    }
)

_ALLOWED_CONTROL_CHARACTERS = {"\n", "\r", "\t"}


def _sanitize_text_for_storage(text: str) -> str:
    text = unicodedata.normalize("NFKC", text.translate(_LIGATURE_TRANSLATION))
    return "".join(
        char
        for char in text
        if char in _ALLOWED_CONTROL_CHARACTERS or unicodedata.category(char)[0] != "C"
    )


def clean_documents(docs: List[Document]) -> List[Document]:
    """Remove boilerplate, page headers, and whitespace noise from loaded documents, while preserving structural newlines."""
    logger.info("Cleaning %s documents...", len(docs))
    cleaned_docs = []
    start_time = time.perf_counter()
    
    for doc in docs:
        text = _sanitize_text_for_storage(doc.page_content)
        # Remove page headers/numbers
        text = re.sub(r'(Seite|Page)\s*\d+(\s*von\s*\d+)?', '', text, flags=re.IGNORECASE)
        # Remove boilerplate patterns
        for pattern in _BOILERPLATE_PATTERNS:
            text = pattern.sub('', text)
        
        # Split by newlines, normalize spaces on each line, and remove empty lines
        lines = []
        for line in text.split('\n'):
            cleaned_line = re.sub(r'\s+', ' ', line).strip()
            if cleaned_line:
                lines.append(cleaned_line)
        text = '\n'.join(lines)
        
        if text.strip():
            cleaned_docs.append(Document(page_content=text.strip(), metadata=doc.metadata))
            
    duration_ms = (time.perf_counter() - start_time) * 1000
    logger.info(
        "Cleaning complete. %s documents remain. | Process Time: %s",
        len(cleaned_docs),
        colorize_process_time(duration_ms),
        extra={"process_time": duration_ms},
    )
    return cleaned_docs

def chunk_documents(docs: List[Document], chunk_size: Optional[int] = None, chunk_overlap: Optional[int] = None) -> List[Document]:
    """Split cleaned documents into overlap-aware chunks for downstream embeddings."""
    if not docs:
        logger.warning("No documents to chunk. Skipping.")
        return []

    logger.info("Chunking documents...")
    start_time = time.perf_counter()
    chunk_size = chunk_size or 750
    chunk_overlap = chunk_overlap or 100
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    doc_splits = text_splitter.split_documents(docs)
    duration_ms = (time.perf_counter() - start_time) * 1000
    logger.info(
        "Created %s document chunks. | Process Time: %s",
        len(doc_splits),
        colorize_process_time(duration_ms),
        extra={"process_time": duration_ms},
    )
    return doc_splits