"""Configuración centralizada cargada desde variables de entorno."""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # GCP
    gcp_project: str = ""
    gcp_region: str = "us-central1"
    gcs_bucket: str = ""

    # Modelos Gemini
    llm_model: str = "gemini-2.5-flash-lite"
    embedding_model: str = "gemini-embedding-001"
    embedding_dimensions: int = 768

    # API key (en local viene de .env, en Cloud Run de Secret Manager)
    google_api_key: str = ""

    # Rutas dentro del bucket
    pdf_blob: str = "GuiaPokemonZ.pdf"
    index_blob: str = "index/index.faiss"
    chunks_blob: str = "index/chunks.pkl"

    # Rutas locales en el contenedor (después de descargar de GCS)
    local_index_dir: str = "/tmp/pokez_index"

    # Retrieval
    chunk_size: int = 500
    chunk_overlap: int = 100
    top_k: int = 15
    bm25_k: int = 15

    # Servir estáticos del frontend
    static_dir: str = "/app/static"


@lru_cache
def get_settings() -> Settings:
    return Settings()
