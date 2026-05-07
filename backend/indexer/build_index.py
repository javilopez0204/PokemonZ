"""Construye el índice FAISS a partir del PDF y lo sube a GCS.

Uso:
    python -m indexer.build_index

Variables de entorno requeridas:
    GCP_PROJECT, GCS_BUCKET, GOOGLE_API_KEY
Opcionales: PDF_BLOB, INDEX_BLOB, CHUNKS_BLOB, EMBEDDING_MODEL, EMBEDDING_DIMENSIONS
"""
from __future__ import annotations

import logging
import os
import pickle
import re
import sys
import tempfile
from pathlib import Path
from typing import List

import faiss  # type: ignore
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pypdf import PdfReader

# Permite ejecutar tanto `python -m indexer.build_index` (paquete) como
# `python build_index.py` desde la carpeta indexer/.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.settings import get_settings  # noqa: E402
from app.storage import download_blob, upload_file  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("indexer")


def normalize_text(text: str) -> str:
    lines: List[str] = []
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped and stripped == stripped.upper() and len(stripped) > 4:
            stripped = stripped.title()
        stripped = re.sub(r"^[¿¡]+", "", stripped)
        lines.append(stripped)
    text = "\n".join(lines)
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def extract_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    pages = []
    for i, page in enumerate(reader.pages):
        content = page.extract_text() or ""
        if content.strip():
            pages.append(f"[Página {i+1}]\n{normalize_text(content)}")
    return "\n\n".join(pages)


def split_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(text)


def embed_chunks(chunks: List[str], model: str, api_key: str, batch: int = 100) -> np.ndarray:
    embedder = GoogleGenerativeAIEmbeddings(
        model=f"models/{model}", google_api_key=api_key
    )
    vectors: List[List[float]] = []
    for i in range(0, len(chunks), batch):
        slice_ = chunks[i : i + batch]
        logger.info("Embed batch %d-%d / %d", i, i + len(slice_), len(chunks))
        vectors.extend(embedder.embed_documents(slice_))
    return np.array(vectors, dtype="float32")


def build_faiss(vectors: np.ndarray) -> faiss.Index:
    dim = vectors.shape[1]
    # Producto interno con vectores normalizados ≈ similitud coseno
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    return index


def main() -> int:
    settings = get_settings()
    api_key = settings.google_api_key or os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        logger.error("Falta GOOGLE_API_KEY")
        return 1
    if not settings.gcs_bucket:
        logger.error("Falta GCS_BUCKET")
        return 1

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        pdf_local = tmpdir / "guia.pdf"
        download_blob(settings.gcs_bucket, settings.pdf_blob, pdf_local)

        logger.info("Extrayendo texto del PDF…")
        text = extract_text(pdf_local)
        chunks = split_text(text, settings.chunk_size, settings.chunk_overlap)
        logger.info("Texto: %d caracteres → %d chunks", len(text), len(chunks))

        vectors = embed_chunks(chunks, settings.embedding_model, api_key)
        logger.info("Vectores: shape=%s", vectors.shape)

        index = build_faiss(vectors)

        faiss_path = tmpdir / "index.faiss"
        chunks_path = tmpdir / "chunks.pkl"
        faiss.write_index(index, str(faiss_path))
        with open(chunks_path, "wb") as fh:
            pickle.dump({"chunks": chunks, "model": settings.embedding_model}, fh)

        upload_file(settings.gcs_bucket, settings.index_blob, faiss_path)
        upload_file(settings.gcs_bucket, settings.chunks_blob, chunks_path)
        logger.info("Índice publicado en gs://%s/%s", settings.gcs_bucket, settings.index_blob)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
