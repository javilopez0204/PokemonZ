"""Helpers de Cloud Storage para descargar/subir el PDF y el índice FAISS."""
from __future__ import annotations

import logging
from pathlib import Path

from google.cloud import storage

logger = logging.getLogger(__name__)


def _client() -> storage.Client:
    return storage.Client()


def download_blob(bucket_name: str, blob_name: str, dest: Path) -> Path:
    """Descarga `blob_name` del bucket a la ruta local `dest`."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    bucket = _client().bucket(bucket_name)
    blob = bucket.blob(blob_name)
    if not blob.exists():
        raise FileNotFoundError(f"gs://{bucket_name}/{blob_name} no existe")
    logger.info("Descargando gs://%s/%s -> %s", bucket_name, blob_name, dest)
    blob.download_to_filename(str(dest))
    return dest


def upload_file(bucket_name: str, blob_name: str, src: Path) -> None:
    """Sube un fichero local a gs://bucket/blob."""
    bucket = _client().bucket(bucket_name)
    blob = bucket.blob(blob_name)
    logger.info("Subiendo %s -> gs://%s/%s", src, bucket_name, blob_name)
    blob.upload_from_filename(str(src))


def blob_exists(bucket_name: str, blob_name: str) -> bool:
    return _client().bucket(bucket_name).blob(blob_name).exists()
