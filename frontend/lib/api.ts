/**
 * Backend client.
 *
 * The session lives in an HttpOnly cookie, so it is never readable from here —
 * `credentials: "include"` is what carries it, and a 401 is the only way this
 * code learns the user is signed out.
 */

const BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

// Without a deadline a stalled request leaves the UI spinning forever. The
// streaming endpoint is deliberately exempt: an answer may legitimately take
// minutes, and it carries its own abort signal.
const TIMEOUT_MS = 30_000;
const UPLOAD_TIMEOUT_MS = 5 * 60_000;

export class ApiError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message);
  }
}

async function request<T>(
  path: string,
  init: RequestInit = {},
  timeoutMs: number = TIMEOUT_MS,
): Promise<T> {
  const response = await fetch(`${BASE}${path}`, {
    ...init,
    credentials: "include",
    // Both, when a caller brings its own: a request that can be cancelled
    // should still have a deadline. Passing the caller's signal straight
    // through would silently drop the one every other call gets.
    signal: init.signal
      ? AbortSignal.any([init.signal, AbortSignal.timeout(timeoutMs)])
      : AbortSignal.timeout(timeoutMs),
    headers: {
      // FormData must set its own Content-Type: the boundary is part of it.
      ...(init.body instanceof FormData ? {} : { "Content-Type": "application/json" }),
      ...init.headers,
    },
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
  /** Unknown for a Notion page, which is not a file anywhere. */
  size_bytes: number | null;
  created_at: string;
  status: "processing" | "ready";
  source_name: string;
  /** False for a document from a connected source: it is removed where it lives. */
  removable: boolean;
};

export type ConnectorKind = "gdrive" | "notion" | "yandex" | "dropbox" | "onedrive";

/** What the indexer last managed to say about a source. */
export type SourceStatus = "ok" | "error" | "unknown";

export type ConnectorOut = {
  id: string;
  kind: ConnectorKind;
  name: string;
  created_at: string;
  /** "unknown" until the indexer has looked once — never "ok" by default. */
  status: SourceStatus;
  /** Why it is not working, in words to show as they are. Empty otherwise. */
  problem: string;
  /** When the indexer last looked, and how many entries it saw then. */
  checked_at: string | null;
  documents: number | null;
};

export type ConnectorsOut = {
  sources: ConnectorOut[];
  /** False when the server has no SECRETS_KEY and cannot hold a credential. */
  enabled: boolean;
  gdrive_service_account_email: string;
  /** The API decides which fields a kind needs, so the form cannot disagree. */
  required_fields: Record<ConnectorKind, string[]>;
};

/** One run of a search snippet. `hit` marks the words the query matched. */
export type SearchPiece = { text: string; hit: boolean };

export type SearchHit = {
  document_id: string | null;
  filename: string | null;
  page: number | null;
  snippet: SearchPiece[];
};

export type SearchOut = {
  hits: SearchHit[];
  /** The whole lookup, embedding included. Worth showing: it is the point. */
  took_ms: number;
};

export type ChatOut = { id: string; title: string; created_at: string };

/** One page of a chat, oldest first. */
export type MessagesPage = {
  messages: MessageOut[];
  /** Pass back as `before` for the page above this one. Null at the start. */
  next_cursor: string | null;
};

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
    // Uploads run to 64 MB, which the default deadline would cut short.
    return request<DocumentOut>(
      "/api/sources",
      { method: "POST", body: form },
      UPLOAD_TIMEOUT_MS,
    );
  },
  deleteDocument: (id: string) =>
    request<void>(`/api/sources/${id}`, { method: "DELETE" }),
  documentLink: (id: string) => request<{ url: string }>(`/api/sources/${id}/link`),

  connectors: () => request<ConnectorsOut>("/api/connectors"),
  addConnector: (kind: ConnectorKind, name: string, config: Record<string, string>) =>
    request<ConnectorOut>("/api/connectors", {
      method: "POST",
      body: JSON.stringify({ kind, name, config }),
    }),
  deleteConnector: (id: string) =>
    request<void>(`/api/connectors/${id}`, { method: "DELETE" }),

  /** Fragments straight from the index — no model, no tokens, ~100 ms. */
  search: (q: string, signal?: AbortSignal) =>
    request<SearchOut>(`/api/search?q=${encodeURIComponent(q)}`, { signal }),

  chats: () => request<ChatOut[]>("/api/chats"),
  createChat: () => request<ChatOut>("/api/chats", { method: "POST" }),
  deleteChat: (id: string) => request<void>(`/api/chats/${id}`, { method: "DELETE" }),
  /** The end of the chat, or — given a cursor — the page just before it. */
  messages: (chatId: string, before?: string) =>
    request<MessagesPage>(
      `/api/chats/${chatId}/messages${before ? `?before=${encodeURIComponent(before)}` : ""}`,
    ),
};

// --- streaming ---------------------------------------------------------------

type StreamHandlers = {
  /** The name of a tool the agent has just decided to call. */
  onStep: (tool: string) => void;
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

  try {
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
  } finally {
    // A handler that throws would otherwise leave the body locked and the
    // connection open for as long as the tab lives.
    await reader.cancel().catch(() => undefined);
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
  else if (event === "step") handlers.onStep(payload as string);
  else if (event === "citations") handlers.onCitations(payload as Citation[]);
  else if (event === "error") handlers.onError(String(payload));
}
