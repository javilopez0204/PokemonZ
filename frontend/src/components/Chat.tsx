import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import { ChatMessage, sendChat } from "../api";

interface Props {
  disabled: boolean;
}

const WELCOME: ChatMessage = {
  role: "assistant",
  content:
    "¡Hola, entrenador! Soy el **Profesor Z** ⚡  \nHe memorizado la guía completa de Pokémon Z. ¡Pregúntame lo que quieras!",
};

export default function Chat({ disabled }: Props) {
  const [messages, setMessages] = useState<ChatMessage[]>([WELCOME]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    const trimmed = input.trim();
    if (!trimmed || loading || disabled) return;

    const newUserMsg: ChatMessage = { role: "user", content: trimmed };
    const history = messages.filter((m) => m !== WELCOME);
    setMessages((m) => [...m, newUserMsg]);
    setInput("");
    setLoading(true);
    setError(null);

    try {
      const res = await sendChat(trimmed, history);
      setMessages((m) => [...m, { role: "assistant", content: res.answer }]);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Error desconocido");
    } finally {
      setLoading(false);
    }
  };

  const reset = () => {
    setMessages([WELCOME]);
    setError(null);
  };

  return (
    <div className="chat">
      <div className="messages">
        {messages.map((msg, i) => (
          <div key={i} className={`bubble bubble-${msg.role}`}>
            <ReactMarkdown>{msg.content}</ReactMarkdown>
          </div>
        ))}
        {loading && (
          <div className="bubble bubble-assistant typing">
            ⚡ <em>El Profesor Z está consultando la guía…</em>
          </div>
        )}
        {error && <div className="bubble bubble-error">❌ {error}</div>}
        <div ref={endRef} />
      </div>

      <form className="input-row" onSubmit={submit}>
        <input
          type="text"
          placeholder={disabled ? "Esperando al sistema RAG…" : "Pregunta al Profesor Z…"}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          disabled={disabled || loading}
        />
        <button type="submit" disabled={disabled || loading || !input.trim()}>
          Enviar
        </button>
        <button type="button" className="ghost" onClick={reset} disabled={loading}>
          Reiniciar
        </button>
      </form>
    </div>
  );
}
