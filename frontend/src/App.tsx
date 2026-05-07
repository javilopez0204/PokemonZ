import { useEffect, useState } from "react";
import Chat from "./components/Chat";
import { fetchHealth } from "./api";

export default function App() {
  const [status, setStatus] = useState<"loading" | "ready" | "starting" | "error">("loading");
  const [chunks, setChunks] = useState(0);

  useEffect(() => {
    let cancelled = false;
    const tick = async () => {
      try {
        const h = await fetchHealth();
        if (cancelled) return;
        setChunks(h.chunks);
        setStatus(h.index_loaded ? "ready" : "starting");
      } catch {
        if (!cancelled) setStatus("error");
      }
    };
    tick();
    const id = setInterval(tick, 5000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, []);

  return (
    <div className="app">
      <header className="topbar">
        <span className="logo">⚡</span>
        <div>
          <h1>Pokédex Z</h1>
          <p className="subtitle">Profesor Z · RAG sobre Gemini</p>
        </div>
        <div className={`status status-${status}`}>
          {status === "ready" && `🟢 ${chunks} fragmentos`}
          {status === "starting" && "🟡 cargando índice…"}
          {status === "loading" && "⏳ conectando…"}
          {status === "error" && "🔴 error"}
        </div>
      </header>
      <main>
        <Chat disabled={status !== "ready"} />
      </main>
      <footer>
        <small>FastAPI · Cloud Run · Gemini Embedding · FAISS</small>
      </footer>
    </div>
  );
}
