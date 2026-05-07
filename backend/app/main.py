"""Aplicación FastAPI: API de chat + estáticos del frontend."""
from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .llm import hydrate_memory, make_chain
from .rag import build_hybrid_retriever, load_index
from .schemas import ChatRequest, ChatResponse, HealthResponse
from .settings import get_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("pokez")

state: dict = {"chain": None, "chunks": [], "ready": False}


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    try:
        logger.info("Cargando índice desde gs://%s/%s", settings.gcs_bucket, settings.index_blob)
        vectorstore, chunks = load_index(settings)
        retriever = build_hybrid_retriever(vectorstore, chunks, settings)
        state["chain"] = make_chain(retriever, settings)
        state["chunks"] = chunks
        state["ready"] = True
        logger.info("Sistema RAG listo (%d chunks)", len(chunks))
    except Exception as exc:  # pragma: no cover - log y arranca igualmente
        logger.exception("Fallo cargando índice: %s", exc)
        state["ready"] = False
    yield


app = FastAPI(title="Pokédex Z API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok" if state["ready"] else "starting",
        index_loaded=state["ready"],
        chunks=len(state["chunks"]),
    )


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    if not state["ready"] or state["chain"] is None:
        raise HTTPException(status_code=503, detail="Índice todavía no cargado")

    chain = state["chain"]
    hydrate_memory(chain, req.history)

    t0 = time.perf_counter()
    try:
        result = chain.invoke({"question": req.message})
    except Exception as exc:
        logger.exception("Error en cadena LLM: %s", exc)
        raise HTTPException(status_code=500, detail="Error generando respuesta") from exc

    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    sources = [d.page_content for d in result.get("source_documents", [])]
    return ChatResponse(answer=result["answer"], sources=sources, elapsed_ms=elapsed_ms)


# ── Estáticos del frontend (build de Vite copiado a /app/static en el Dockerfile)
settings = get_settings()
static_dir = Path(settings.static_dir)
if static_dir.exists():
    app.mount("/assets", StaticFiles(directory=static_dir / "assets"), name="assets")

    @app.get("/")
    async def index():
        return FileResponse(static_dir / "index.html")

    @app.get("/{path:path}")
    async def spa_fallback(path: str):
        # Para rutas SPA que no son /api/*, devolver index.html
        if path.startswith("api/"):
            raise HTTPException(status_code=404)
        candidate = static_dir / path
        if candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(static_dir / "index.html")
