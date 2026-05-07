export type Role = "user" | "assistant";

export interface ChatMessage {
  role: Role;
  content: string;
}

export interface ChatResponse {
  answer: string;
  sources: string[];
  elapsed_ms: number;
}

const API_BASE = import.meta.env.VITE_API_BASE ?? "";

export async function sendChat(
  message: string,
  history: ChatMessage[]
): Promise<ChatResponse> {
  const res = await fetch(`${API_BASE}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, history }),
  });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`HTTP ${res.status}: ${detail}`);
  }
  return res.json();
}

export async function fetchHealth(): Promise<{
  status: string;
  index_loaded: boolean;
  chunks: number;
}> {
  const res = await fetch(`${API_BASE}/api/health`);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}
