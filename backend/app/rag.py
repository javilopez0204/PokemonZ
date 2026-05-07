"""Carga del índice FAISS desde GCS y construcción del retriever híbrido."""
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.retrievers import EnsembleRetriever

from .settings import Settings
from .storage import download_blob

logger = logging.getLogger(__name__)


def _embeddings(settings: Settings) -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(
        model=f"models/{settings.embedding_model}",
        google_api_key=settings.google_api_key,
    )


def load_index(settings: Settings) -> Tuple[VectorStore, List[str]]:
    """Descarga el índice prefabricado del bucket y lo carga en memoria."""
    index_dir = Path(settings.local_index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    faiss_path = index_dir / "index.faiss"
    chunks_path = index_dir / "chunks.pkl"

    if not faiss_path.exists():
        download_blob(settings.gcs_bucket, settings.index_blob, faiss_path)
    if not chunks_path.exists():
        download_blob(settings.gcs_bucket, settings.chunks_blob, chunks_path)

    embeddings = _embeddings(settings)
    # FAISS.load_local usa dos ficheros: index.faiss e index.pkl
    # Persistimos el .pkl con sólo la lista de chunks (texto), así que
    # reconstruimos el FAISS desde texts+embeddings de forma cacheada.
    with open(chunks_path, "rb") as fh:
        payload = pickle.load(fh)

    chunks: List[str] = payload["chunks"]
    # FAISS index binario serializado por faiss
    import faiss  # type: ignore

    raw_index = faiss.read_index(str(faiss_path))
    # Reconstruimos el FAISS de langchain manualmente
    docstore_dict = {str(i): _doc(text) for i, text in enumerate(chunks)}
    from langchain_community.docstore.in_memory import InMemoryDocstore

    vectorstore = FAISS(
        embedding_function=embeddings,
        index=raw_index,
        docstore=InMemoryDocstore(docstore_dict),
        index_to_docstore_id={i: str(i) for i in range(len(chunks))},
    )
    logger.info("Índice cargado: %d chunks", len(chunks))
    return vectorstore, chunks


def _doc(text: str):
    from langchain_core.documents import Document

    return Document(page_content=text)


def build_hybrid_retriever(
    vectorstore: VectorStore, chunks: List[str], settings: Settings
):
    semantic = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": settings.top_k}
    )
    bm25 = BM25Retriever.from_texts(chunks)
    bm25.k = settings.bm25_k
    return EnsembleRetriever(retrievers=[semantic, bm25], weights=[0.6, 0.4])
