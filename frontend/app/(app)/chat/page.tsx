"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Loader2, Plus, SendHorizontal, Trash2 } from "lucide-react";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { Textarea } from "@/components/ui/textarea";
import { ApiError, api, ask, type ChatOut, type Citation, type MessageOut } from "@/lib/api";
import { cn } from "@/lib/utils";

export default function ChatPage() {
  const [chats, setChats] = useState<ChatOut[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [messages, setMessages] = useState<MessageOut[]>([]);
  const [draft, setDraft] = useState("");
  const [streaming, setStreaming] = useState<{ text: string; citations: Citation[] } | null>(
    null,
  );
  const [loading, setLoading] = useState(true);

  const bottomRef = useRef<HTMLDivElement>(null);

  // Load the chat list once, opening the newest chat or creating the first one.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        let list = await api.chats();
        if (list.length === 0) list = [await api.createChat()];
        if (cancelled) return;
        setChats(list);
        setActiveId(list[0].id);
      } catch (cause) {
        if (!cancelled) toast.error(describe(cause));
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!activeId) return;
    let cancelled = false;
    api
      .messages(activeId)
      .then((loaded) => !cancelled && setMessages(loaded))
      .catch((cause) => !cancelled && toast.error(describe(cause)));
    return () => {
      cancelled = true;
    };
  }, [activeId]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, streaming?.text]);

  const send = useCallback(async () => {
    const question = draft.trim();
    if (!question || !activeId || streaming) return;

    setDraft("");
    setMessages((current) => [...current, localMessage("user", question)]);
    setStreaming({ text: "", citations: [] });

    let failed = false;
    try {
      await ask(activeId, question, {
        onToken: (text) =>
          setStreaming((current) => ({ ...(current ?? { citations: [] }), text: (current?.text ?? "") + text })),
        onCitations: (citations) =>
          setStreaming((current) => ({ text: current?.text ?? "", citations })),
        onError: (message) => {
          failed = true;
          toast.error(message);
        },
      });
    } catch (cause) {
      failed = true;
      toast.error(describe(cause));
    }

    // Move whatever arrived into the transcript so a partial answer is kept.
    setStreaming((current) => {
      if (current?.text) {
        setMessages((existing) => [
          ...existing,
          localMessage("assistant", current.text, current.citations),
        ]);
      } else if (!failed) {
        toast.error("Пустой ответ от модели");
      }
      return null;
    });

    // The first question becomes the chat title on the server.
    api.chats().then(setChats).catch(() => undefined);
  }, [activeId, draft, streaming]);

  async function newChat() {
    try {
      const chat = await api.createChat();
      setChats((current) => [chat, ...current]);
      setActiveId(chat.id);
      setMessages([]);
    } catch (cause) {
      toast.error(describe(cause));
    }
  }

  async function removeChat(id: string) {
    try {
      await api.deleteChat(id);
      const remaining = chats.filter((chat) => chat.id !== id);
      setChats(remaining);
      if (activeId === id) {
        setActiveId(remaining[0]?.id ?? null);
        setMessages([]);
      }
    } catch (cause) {
      toast.error(describe(cause));
    }
  }

  return (
    <div className="flex flex-1 overflow-hidden">
      <aside className="hidden w-60 shrink-0 flex-col border-r md:flex">
        <div className="p-3">
          <Button onClick={newChat} className="w-full" variant="outline" size="sm">
            <Plus className="size-4" />
            Новый чат
          </Button>
        </div>
        <div className="flex-1 space-y-1 overflow-y-auto px-2 pb-3">
          {chats.map((chat) => (
            <div
              key={chat.id}
              className={cn(
                "group flex items-center gap-1 rounded-md pr-1 text-sm",
                chat.id === activeId ? "bg-accent" : "hover:bg-accent/50",
              )}
            >
              <button
                onClick={() => setActiveId(chat.id)}
                className="flex-1 truncate px-2 py-2 text-left"
                title={chat.title}
              >
                {chat.title}
              </button>
              <Button
                variant="ghost"
                size="icon"
                className="size-7 opacity-0 group-hover:opacity-100"
                onClick={() => removeChat(chat.id)}
                aria-label={`Удалить чат «${chat.title}»`}
              >
                <Trash2 className="size-3.5" />
              </Button>
            </div>
          ))}
        </div>
      </aside>

      <section className="flex flex-1 flex-col overflow-hidden">
        <div className="flex-1 overflow-y-auto">
          <div className="mx-auto flex w-full max-w-3xl flex-col gap-6 p-4 sm:p-6">
            {loading ? (
              <Skeleton className="h-20 w-full" />
            ) : messages.length === 0 && !streaming ? (
              <EmptyState />
            ) : null}

            {messages.map((message) => (
              <Bubble key={message.id} message={message} />
            ))}

            {streaming && (
              <Bubble
                message={localMessage("assistant", streaming.text, streaming.citations)}
                pending={streaming.text === ""}
              />
            )}
            <div ref={bottomRef} />
          </div>
        </div>

        <div className="border-t p-3 sm:p-4">
          <div className="mx-auto flex w-full max-w-3xl items-end gap-2">
            <Textarea
              value={draft}
              onChange={(event) => setDraft(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter" && !event.shiftKey) {
                  event.preventDefault();
                  void send();
                }
              }}
              placeholder="Спросите что-нибудь о ваших документах…"
              className="max-h-40 min-h-11 resize-none"
              aria-label="Вопрос"
              disabled={!activeId}
            />
            <Button
              onClick={() => void send()}
              disabled={!draft.trim() || !activeId || streaming !== null}
              size="icon"
              className="size-11 shrink-0"
              aria-label="Отправить"
            >
              {streaming ? (
                <Loader2 className="size-4 animate-spin" />
              ) : (
                <SendHorizontal className="size-4" />
              )}
            </Button>
          </div>
        </div>
      </section>
    </div>
  );
}

function EmptyState() {
  return (
    <div className="py-16 text-center">
      <p className="font-medium">Задайте вопрос по вашим документам</p>
      <p className="mt-1 text-sm text-muted-foreground">
        Ответы формируются только по загруженным файлам, со ссылками на источники.
      </p>
    </div>
  );
}

function Bubble({ message, pending = false }: { message: MessageOut; pending?: boolean }) {
  const isUser = message.role === "user";
  return (
    <div className={cn("flex", isUser && "justify-end")}>
      <div className={cn("max-w-full space-y-2", isUser && "max-w-[85%]")}>
        <div
          className={cn(
            "rounded-2xl px-4 py-2.5 text-sm whitespace-pre-wrap",
            isUser ? "bg-primary text-primary-foreground" : "bg-muted",
          )}
        >
          {pending ? <Loader2 className="size-4 animate-spin" /> : message.content}
        </div>
        {message.citations.length > 0 && <Citations citations={message.citations} />}
      </div>
    </div>
  );
}

function Citations({ citations }: { citations: Citation[] }) {
  async function open(citation: Citation) {
    if (!citation.document_id) return;
    try {
      const { url } = await api.documentLink(citation.document_id);
      window.open(url, "_blank", "noopener,noreferrer");
    } catch (cause) {
      toast.error(describe(cause));
    }
  }

  return (
    <div className="flex flex-wrap gap-1.5">
      {citations.map((citation) => (
        <button
          key={`${citation.document_id}-${citation.page}-${citation.n}`}
          onClick={() => void open(citation)}
          disabled={!citation.document_id}
          title={citation.snippet}
          className="inline-flex items-center gap-1 rounded-full border bg-background px-2.5 py-1 text-xs text-muted-foreground transition-colors hover:bg-accent hover:text-foreground disabled:pointer-events-none disabled:opacity-50"
        >
          <span className="font-medium text-foreground">[{citation.n}]</span>
          <span className="max-w-52 truncate">{citation.filename ?? "документ"}</span>
          {citation.page !== null && <span>· стр. {citation.page}</span>}
        </button>
      ))}
    </div>
  );
}

/** A message rendered before the server has assigned it an id. */
function localMessage(
  role: "user" | "assistant",
  content: string,
  citations: Citation[] = [],
): MessageOut {
  return {
    id: `local-${role}-${crypto.randomUUID()}`,
    role,
    content,
    citations,
    created_at: new Date().toISOString(),
  };
}

function describe(cause: unknown): string {
  return cause instanceof ApiError ? cause.message : "Что-то пошло не так";
}
