"""Smoke test del endpoint de salud sin cargar índice real."""
from fastapi.testclient import TestClient

from app import main


def test_health_endpoint_responds(monkeypatch):
    # No queremos descargar nada de GCS en CI: simulamos lifespan vacío.
    monkeypatch.setitem(main.state, "ready", False)
    monkeypatch.setitem(main.state, "chunks", [])
    with TestClient(main.app) as client:
        # Saltamos el lifespan real: el TestClient lo ejecuta, pero como no hay
        # bucket configurado simplemente cae al `except` y deja ready=False.
        resp = client.get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] in ("ok", "starting")
    assert "chunks" in body
