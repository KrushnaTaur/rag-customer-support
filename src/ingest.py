"""
ingest.py — PDF Ingestion Pipeline
====================================
Responsibilities:
  1. Load PDF documents using LangChain's PyPDFLoader
  2. Split into overlapping chunks via RecursiveCharacterTextSplitter
  3. Generate embeddings using sentence-transformers/all-MiniLM-L6-v2
  4. Persist chunks + embeddings into ChromaDB

Usage:
    python -m src.ingest --pdf path/to/document.pdf
    python -m src.ingest --pdf path/to/doc1.pdf path/to/doc2.pdf
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import List

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from src.config import settings
from src.logger import logger


# ─────────────────────────────────────────────────────────────────────────────
# Module-level singleton for embeddings (expensive to reload)
# ─────────────────────────────────────────────────────────────────────────────
_embedding_model: HuggingFaceEmbeddings | None = None


def get_embedding_model() -> HuggingFaceEmbeddings:
    """Return a cached HuggingFace embedding model instance."""
    global _embedding_model
    if _embedding_model is None:
        logger.info("Loading embedding model: {}", settings.embedding_model)
        _embedding_model = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        logger.success("Embedding model loaded.")
    return _embedding_model


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Document Loading
# ─────────────────────────────────────────────────────────────────────────────
def load_pdf(pdf_path: str) -> List[Document]:
    """
    Load a PDF file and return a list of LangChain Document objects.
    Each Document represents one page, with metadata: source, page.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file, got: {path.suffix}")

    logger.info("Loading PDF: {}", pdf_path)
    loader = PyPDFLoader(str(path))
    documents = loader.load()

    # Enrich metadata
    for doc in documents:
        doc.metadata["source"] = path.name
        doc.metadata["file_path"] = str(path)

    logger.success("Loaded {} pages from '{}'", len(documents), path.name)
    return documents


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Chunking
# ─────────────────────────────────────────────────────────────────────────────
def chunk_documents(documents: List[Document]) -> List[Document]:
    """
    Split documents into overlapping chunks.
    Preserves original metadata and adds chunk_index to each chunk.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunks = splitter.split_documents(documents)

    # Add chunk index to metadata
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
        chunk.metadata["chunk_total"] = len(chunks)

    logger.info(
        "Chunked {} pages → {} chunks (size={}, overlap={})",
        len(documents),
        len(chunks),
        settings.chunk_size,
        settings.chunk_overlap,
    )
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Vector Store Persistence
# ─────────────────────────────────────────────────────────────────────────────
def store_chunks(chunks: List[Document]) -> Chroma:
    """
    Embed chunks and persist them into ChromaDB.
    Returns the Chroma vector store instance.
    """
    os.makedirs(settings.chroma_persist_dir, exist_ok=True)
    embeddings = get_embedding_model()

    logger.info(
        "Storing {} chunks into ChromaDB collection '{}'",
        len(chunks),
        settings.chroma_collection_name,
    )

    start = time.time()
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=settings.chroma_collection_name,
        persist_directory=settings.chroma_persist_dir,
    )
    elapsed = time.time() - start

    logger.success(
        "Stored {} chunks in ChromaDB in {:.2f}s. Path: {}",
        len(chunks),
        elapsed,
        settings.chroma_persist_dir,
    )
    return vector_store


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Load Existing Vector Store
# ─────────────────────────────────────────────────────────────────────────────
def load_vector_store() -> Chroma:
    """
    Load an existing ChromaDB vector store from disk.
    Raises RuntimeError if the store has not been initialised yet.
    """
    if not os.path.exists(settings.chroma_persist_dir):
        raise RuntimeError(
            f"ChromaDB not found at '{settings.chroma_persist_dir}'. "
            "Run ingestion first: python -m src.ingest --pdf <file.pdf>"
        )

    embeddings = get_embedding_model()
    vector_store = Chroma(
        collection_name=settings.chroma_collection_name,
        embedding_function=embeddings,
        persist_directory=settings.chroma_persist_dir,
    )
    count = vector_store._collection.count()
    logger.info(
        "Loaded ChromaDB from '{}'. Collection '{}' has {} chunks.",
        settings.chroma_persist_dir,
        settings.chroma_collection_name,
        count,
    )
    return vector_store


# ─────────────────────────────────────────────────────────────────────────────
# Main Pipeline
# ─────────────────────────────────────────────────────────────────────────────
def run_ingestion(pdf_paths: List[str]) -> None:
    """Full ingestion pipeline for one or more PDF files."""
    all_chunks: List[Document] = []

    for pdf_path in pdf_paths:
        docs = load_pdf(pdf_path)
        chunks = chunk_documents(docs)
        all_chunks.extend(chunks)

    store_chunks(all_chunks)
    logger.success(
        "Ingestion complete. Total chunks indexed: {}", len(all_chunks)
    )


# ─────────────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingest PDF documents into the RAG knowledge base."
    )
    parser.add_argument(
        "--pdf",
        nargs="+",
        required=True,
        help="Path(s) to PDF file(s) to ingest.",
    )
    args = parser.parse_args()
    run_ingestion(args.pdf)
