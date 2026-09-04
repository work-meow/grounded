/**
 * Backend client.
 *
 * The session lives in an HttpOnly cookie, so it is never readable from here —
 * `credentials: "include"` is what carries it, and a 401 is the only way this
 * code learns the user is signed out.
 */

const BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export class ApiError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message);
  }
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const response = await fetch(`${BASE}${path}`, {
    ...init,
    credentials: "include",
    headers: { ...(init.body instanceof FormData ? {} : { "Content-Type": "application/json" }), ...init.headers },
  });

  if (!response.ok) {
    throw new ApiError(response.status, await errorMessage(response));
  }
  return response.status === 204 ? (undefined as T) : ((await response.json()) as T);
}

async function errorMessage(response: Response): Promise<string> {
  try {
    const body = await response.json();
    const detail = body?.detail;
    if (typeof detail === "string") return detail;
    // FastAPI validation errors arrive as a list of objects.
    if (Array.isArray(detail) && detail[0]?.msg) return detail[0].msg;
  } catch {
    /* fall through to the status text */
  }
  return response.statusText || `Ошибка ${response.status}`;
}

// --- types (mirror the FastAPI response models) -----------------------------

export type Citation = {
  n: number;
  document_id: string | null;
  filename: string | null;
  page: number | null;
  snippet: string;
};

export type DocumentOut = {
  id: string;
  filename: string;
  mime_type: string;
  size_bytes: number;
  created_at: string;
  status: "processing" | "ready";
  chunks: number;
};

export type ChatOut = { id: string; title: string; created_at: string };

export type MessageOut = {
  id: string;
  role: "user" | "assistant";
  content: string;
  citations: Citation[];
  created_at: string;
};

// --- endpoints ---------------------------------------------------------------

export const api = {
  me: () => request<{ user_id: string }>("/api/auth/me"),
  login: (token: string) =>
    request<{ user_id: string }>("/api/auth/login", {
      method: "POST",
      body: JSON.stringify({ token }),
    }),
  logout: () => request<{ ok: boolean }>("/api/auth/logout", { method: "POST" }),

  documents: () => request<DocumentOut[]>("/api/sources"),
  upload: (file: File) => {
    const form = new FormData();
    form.append("file", file);
    return request<DocumentOut>("/api/sources", { method: "POST", body: form });
  },
  deleteDocument: (id: string) =>
    request<void>(`/api/sources/${id}`, { method: "DELETE" }),
  documentLink: (id: string) => request<{ url: string }>(`/api/sources/${id}/link`),

  chats: () => request<ChatOut[]>("/api/chats"),
  createChat: () => request<ChatOut>("/api/chats", { method: "POST" }),
  deleteChat: (id: string) => request<void>(`/api/chats/${id}`, { method: "DELETE" }),
  messages: (chatId: string) => request<MessageOut[]>(`/api/chats/${chatId}/messages`),
};

// --- streaming ---------------------------------------------------------------

type StreamHandlers = {
  onToken: (text: string) => void;
  onCitations: (citations: Citation[]) => void;
  onError: (message: string) => void;
};

/**
 * POST a question and consume the SSE response.
 *
 * EventSource cannot POST, so we read the body stream directly and split on the
 * SSE record separator ourselves.
 */
export async function ask(
  chatId: string,
  question: string,
  handlers: StreamHandlers,
  signal?: AbortSignal,
): Promise<void> {
  const response = await fetch(`${BASE}/api/chats/${chatId}/messages`, {
    method: "POST",
    credentials: "include",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
    signal,
  });

  if (!response.ok || !response.body) {
    throw new ApiError(response.status, await errorMessage(response));
  }

  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += value;

    // A record ends at a blank line; anything after it is an incomplete record.
    let separator = buffer.indexOf("\n\n");
    while (separator !== -1) {
      dispatch(buffer.slice(0, separator), handlers);
      buffer = buffer.slice(separator + 2);
      separator = buffer.indexOf("\n\n");
    }
  }
}

function dispatch(record: string, handlers: StreamHandlers): void {
  let event = "message";
  const dataLines: string[] = [];

  for (const line of record.split("\n")) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
  }
  if (dataLines.length === 0) return;

  let payload: unknown;
  try {
    payload = JSON.parse(dataLines.join("\n"));
  } catch {
    return;
  }

  if (event === "token") handlers.onToken(payload as string);
  else if (event === "citations") handlers.onCitations(payload as Citation[]);
  else if (event === "error") handlers.onError(String(payload));
}
