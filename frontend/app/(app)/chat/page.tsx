"use client";

import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import {
  Check,
  Copy,
  Globe,
  Loader2,
  PanelLeft,
  Plus,
  SendHorizontal,
  Square,
  ThumbsDown,
  ThumbsUp,
  Trash2,
} from "lucide-react";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { Sheet, SheetContent, SheetTitle, SheetTrigger } from "@/components/ui/sheet";
import { Skeleton } from "@/components/ui/skeleton";
import { Textarea } from "@/components/ui/textarea";
import {
  ApiError,
  api,
  ask,
  type ChatOut,
  type Citation,
  type MessageOut,
  type Step,
} from "@/lib/api";
import { cn } from "@/lib/utils";

/**
 * How close to the top counts as asking for the page above.
 *
 * A screen height would fetch before anyone had read anything; zero would only
 * fetch once the reader had already hit the wall and seen nothing happen.
 */
const LOAD_OLDER_PX = 300;

/**
 * What the agent is doing, in words, while there is nothing to read yet.
 *
 * A turn spends two to seven seconds in retrieval before its first token, and
 * a bare spinner for that long is indistinguishable from a hung request. The
 * server sends the tool's name the moment the model picks it — before the
 * arguments have even finished streaming — so this costs a round trip of
 * nothing and is the earliest honest thing to show.
 */
const STEP_LABEL: Record<string, string> = {
  search_knowledge: "Ищу в базе знаний",
  search_web: "Ищу в сети",
  list_sources: "Смотрю, какие есть документы",
  read_document: "Читаю документ",
};

export default function ChatPage() {
  const [chats, setChats] = useState<ChatOut[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [messages, setMessages] = useState<MessageOut[]>([]);
  const [draft, setDraft] = useState("");
  // Tagged with the chat it belongs to, so switching chats simply stops
  // matching instead of needing the effect to reach in and clear it.
  const [streaming, setStreaming] = useState<{
    chatId: string;
    text: string;
    citations: Citation[];
    /** The tool the agent reached for last. Null before it has reached. */
    step: Step | null;
  } | null>(null);
  const [loading, setLoading] = useState(true);
  // Where the chat continues above what is on screen, and whether that page is
  // on its way. Null means the beginning of the conversation is already here.
  const [olderCursor, setOlderCursor] = useState<string | null>(null);
  const [loadingOlder, setLoadingOlder] = useState(false);
  // The chat list, on a screen too narrow for it to live beside the messages.
  const [drawer, setDrawer] = useState(false);
  // Whether the agent may search the web. Sticky once turned on, like every
  // other chat program: it is a mode somebody chooses, not a per-message
  // decision they want to make twice.
  const [web, setWeb] = useState(false);

  // The stream only counts while its chat is the open one.
  const active = streaming?.chatId === activeId ? streaming : null;

  const bottomRef = useRef<HTMLDivElement>(null);
  const viewportRef = useRef<HTMLDivElement>(null);
  // The scroll height measured just before a page of older messages goes in.
  // Set means "keep the reader where they were" rather than "follow the end".
  const restoreFrom = useRef<number | null>(null);
  // A chat is opened at its end without animating there from wherever the last
  // one was; only new messages arriving are worth sliding to.
  const jumpToEnd = useRef(false);
  // Guards the fetch against being started twice by consecutive scroll events,
  // which fire far faster than a request comes back.
  const fetchingOlder = useRef(false);
  // Read after an await to tell whether the chat is still the one that asked.
  const activeIdRef = useRef<string | null>(null);
  // Aborts the answer in flight when the chat is switched or the page unmounts,
  // so a stream cannot outlive the view that asked for it.
  const inFlight = useRef<AbortController | null>(null);

  useEffect(() => () => inFlight.current?.abort(), []);
  useEffect(() => {
    activeIdRef.current = activeId;
  }, [activeId]);

  // Load the chat list once, opening the newest chat or creating the first one.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const existing = await api.chats();
        const list = existing.length > 0 ? existing : [await api.createChat()];
        if (cancelled) return;
        setChats(list);
        // `?? null` rather than an assertion: the list is non-empty by
        // construction two lines up, and saying so with `!` would be the one
        // place in this file where a type is asserted rather than proved. An
        // empty list leaves no chat open, which the sidebar already handles.
        setActiveId(list[0]?.id ?? null);
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
    // Whatever was streaming belonged to the previous chat.
    inFlight.current?.abort();

    let cancelled = false;
    api
      .messages(activeId)
      .then((page) => {
        if (cancelled) return;
        // A chat opens on its last page, at the bottom, the way every other
        // chat program opens: the newest thing said is the thing you came back
        // for. Everything before it is fetched when it is scrolled towards.
        jumpToEnd.current = true;
        setMessages(page.messages);
        setOlderCursor(page.next_cursor);
      })
      .catch((cause) => !cancelled && toast.error(describe(cause)));
    return () => {
      cancelled = true;
    };
  }, [activeId]);

  const loadOlder = useCallback(async () => {
    const chatId = activeId;
    if (!chatId || !olderCursor || fetchingOlder.current) return;

    fetchingOlder.current = true;
    setLoadingOlder(true);
    try {
      const page = await api.messages(chatId, olderCursor);
      // Prepending a page fetched for a chat the reader has since left would
      // splice one conversation into another.
      if (activeIdRef.current !== chatId) return;
      // Measured now, used once the taller list has been laid out.
      restoreFrom.current = viewportRef.current?.scrollHeight ?? null;
      setMessages((current) => [...page.messages, ...current]);
      setOlderCursor(page.next_cursor);
    } catch (cause) {
      toast.error(describe(cause));
    } finally {
      fetchingOlder.current = false;
      setLoadingOlder(false);
    }
  }, [activeId, olderCursor]);

  // Layout, not effect: both branches move the scroll position, and doing that
  // after the browser has painted the new list is a visible jump.
  useLayoutEffect(() => {
    const viewport = viewportRef.current;
    if (restoreFrom.current !== null && viewport) {
      // Older messages went in above. Growing the list upward would otherwise
      // carry the message being read off the bottom of the screen; this keeps
      // it exactly where it was.
      viewport.scrollTop += viewport.scrollHeight - restoreFrom.current;
      restoreFrom.current = null;
      return;
    }
    bottomRef.current?.scrollIntoView({ behavior: jumpToEnd.current ? "auto" : "smooth" });
    jumpToEnd.current = false;
  }, [messages, active?.text]);

  const send = useCallback(async () => {
    const question = draft.trim();
    // `active`, not `streaming`: an aborted stream from another chat is left
    // behind deliberately and must not block this one.
    if (!question || !activeId || active) return;

    setDraft("");
    setMessages((current) => [...current, localMessage("user", question)]);
    setStreaming({ chatId: activeId, text: "", citations: [], step: null });

    // The answer is accumulated here rather than read back out of state: a
    // setState updater must stay pure, and React calls it twice in StrictMode.
    let text = "";
    let citations: Citation[] = [];
    let step: Step | null = null;
    let failed = false;

    const controller = new AbortController();
    inFlight.current = controller;

    try {
      await ask(
        activeId,
        question,
        {
          onStep: (reached) => {
            step = reached;
            setStreaming({ chatId: activeId, text, citations, step });
          },
          onToken: (chunk) => {
            text += chunk;
            setStreaming({ chatId: activeId, text, citations, step });
          },
          onCitations: (received) => {
            citations = received;
            setStreaming({ chatId: activeId, text, citations, step });
          },
          onError: (message) => {
            failed = true;
            toast.error(message);
          },
        },
        controller.signal,
        web,
      );
    } catch (cause) {
      // An abort is a stop, a chat switch or an unmount — not a failure.
      if (!controller.signal.aborted) {
        failed = true;
        toast.error(describe(cause));
      }
    } finally {
      if (inFlight.current === controller) inFlight.current = null;
    }

    if (controller.signal.aborted) {
      // Leaving the half-finished stream in state would strand this chat: on
      // returning to it the phantom answer reappears and, because `active` is
      // what disables the composer, the send button never comes back.
      setStreaming((current) => (current?.chatId === activeId ? null : current));
      if (text) {
        setMessages((current) => [...current, localMessage("assistant", text, citations)]);
      }
      return;
    }

    setStreaming(null);
    if (text) {
      // Keep a partial answer rather than losing the turn.
      setMessages((current) => [...current, localMessage("assistant", text, citations)]);
    } else if (!failed) {
      toast.error("Пустой ответ от модели");
    }

    // The first question becomes the chat title on the server.
    api.chats().then(setChats).catch(() => undefined);
  }, [activeId, draft, active, web]);

  const onScroll = useCallback(() => {
    const viewport = viewportRef.current;
    if (viewport && viewport.scrollTop <= LOAD_OLDER_PX) void loadOlder();
  }, [loadOlder]);

  // The answer keeps streaming on the server; this stops waiting for it and
  // keeps what has already arrived.
  const stop = useCallback(() => inFlight.current?.abort(), []);

  async function newChat() {
    try {
      const chat = await api.createChat();
      setChats((current) => [chat, ...current]);
      setActiveId(chat.id);
      setMessages([]);
      setOlderCursor(null);
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
        setOlderCursor(null);
      }
    } catch (cause) {
      toast.error(describe(cause));
    }
  }

  return (
    <div className="flex flex-1 overflow-hidden">
      <aside className="hidden w-60 shrink-0 flex-col border-r md:flex">
        <ChatList
          chats={chats}
          activeId={activeId}
          onSelect={setActiveId}
          onCreate={newChat}
          onRemove={removeChat}
        />
      </aside>

      <section className="flex flex-1 flex-col overflow-hidden">
        {/* Below md the sidebar is gone, and with it the only way to switch or
            start a chat. This bar is what puts it back. */}
        <div className="flex h-12 shrink-0 items-center gap-1 border-b px-2 md:hidden">
          <Sheet open={drawer} onOpenChange={setDrawer}>
            <SheetTrigger asChild>
              <Button variant="ghost" size="icon" className="size-9" aria-label="Список чатов">
                <PanelLeft className="size-4" />
              </Button>
            </SheetTrigger>
            <SheetContent>
              <SheetTitle className="px-4 pt-4">Чаты</SheetTitle>
              <ChatList
                chats={chats}
                activeId={activeId}
                onSelect={(id) => {
                  setActiveId(id);
                  setDrawer(false);
                }}
                onCreate={async () => {
                  await newChat();
                  setDrawer(false);
                }}
                // Deleting does not close it: removing several in a row is the
                // reason somebody opens this list in the first place.
                onRemove={removeChat}
              />
            </SheetContent>
          </Sheet>
          <p className="min-w-0 flex-1 truncate text-sm font-medium">
            {chats.find((chat) => chat.id === activeId)?.title ?? "Чат"}
          </p>
        </div>

        <div className="relative flex flex-1 flex-col overflow-hidden">
          {loadingOlder && (
            // Out of the flow on purpose: anything that took up space here would
            // change the scroll height between measuring it and restoring it.
            <div className="pointer-events-none absolute inset-x-0 top-2 z-10 flex justify-center">
              <span className="rounded-full border bg-background px-2 py-1 shadow-sm">
                <Loader2 className="size-4 animate-spin text-muted-foreground" />
              </span>
            </div>
          )}
          <div
            ref={viewportRef}
            onScroll={onScroll}
            className="flex-1 overflow-y-auto"
            // The browser's own scroll anchoring would move the same scroll
            // position this component adjusts by hand, and the two together
            // overshoot.
            style={{ overflowAnchor: "none" }}
          >
            <div className="mx-auto flex w-full max-w-3xl flex-col gap-6 p-4 sm:p-6">
              {loading ? (
                <Skeleton className="h-20 w-full" />
              ) : messages.length === 0 && !active ? (
                <EmptyState />
              ) : null}

              {messages.map((message) => (
                <Bubble key={message.id} message={message} chatId={activeId} />
              ))}

              {active && (
                <Bubble
                  // A fixed id: localMessage() would mint a fresh UUID on every
                  // token as the answer streams in.
                  message={{
                    id: "streaming",
                    role: "assistant",
                    content: active.text,
                    citations: active.citations,
                    created_at: "",
                  }}
                  pending={active.text === ""}
                  step={active.step}
                />
              )}
              <div ref={bottomRef} />
            </div>
          </div>
        </div>

        <div className="shrink-0 border-t p-3 sm:p-4">
          <div className="mx-auto flex w-full max-w-3xl items-end gap-2">
            <Tools web={web} onToggle={setWeb} />
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
            {active ? (
              <Button
                onClick={stop}
                size="icon"
                variant="secondary"
                className="size-11 shrink-0"
                aria-label="Остановить"
              >
                <Square className="size-3.5 fill-current" />
              </Button>
            ) : (
              <Button
                onClick={() => void send()}
                disabled={!draft.trim() || !activeId}
                size="icon"
                className="size-11 shrink-0"
                aria-label="Отправить"
              >
                <SendHorizontal className="size-4" />
              </Button>
            )}
          </div>
        </div>
      </section>
    </div>
  );
}

function ChatList({
  chats,
  activeId,
  onSelect,
  onCreate,
  onRemove,
}: {
  chats: ChatOut[];
  activeId: string | null;
  onSelect: (id: string) => void;
  onCreate: () => void | Promise<void>;
  onRemove: (id: string) => void;
}) {
  return (
    <>
      <div className="shrink-0 p-3">
        <Button onClick={() => void onCreate()} className="w-full" variant="outline" size="sm">
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
              onClick={() => onSelect(chat.id)}
              className="flex-1 truncate px-2 py-2 text-left"
              title={chat.title}
            >
              {chat.title}
            </button>
            <Button
              variant="ghost"
              size="icon"
              // Always reachable on a touch screen, where there is no hover to
              // reveal it with.
              className="size-7 opacity-100 transition-opacity md:opacity-0 md:group-hover:opacity-100 md:focus-visible:opacity-100"
              onClick={() => onRemove(chat.id)}
              aria-label={`Удалить чат «${chat.title}»`}
            >
              <Trash2 className="size-3.5" />
            </Button>
          </div>
        ))}
      </div>
    </>
  );
}

/**
 * What the agent is allowed to reach for, beyond the documents.
 *
 * One entry today, which is why it is a list rather than a switch: the shape
 * that holds two is the same one, and the shape that holds one is the one
 * people already know from every other chat.
 *
 * The button doubles as the indicator — when the web is on it says so, rather
 * than a second chip somewhere else saying it. There is one control and one
 * place to look.
 */
function Tools({ web, onToggle }: { web: boolean; onToggle: (on: boolean) => void }) {
  const [open, setOpen] = useState(false);

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <Button
          variant={web ? "secondary" : "outline"}
          size="icon"
          className={cn("h-11 shrink-0", web ? "w-auto gap-1.5 px-3" : "w-11")}
          aria-label={web ? "Инструменты: поиск в сети включён" : "Инструменты"}
        >
          {web ? (
            <>
              <Globe className="size-4" />
              <span className="text-xs">Сеть</span>
            </>
          ) : (
            <Plus className="size-4" />
          )}
        </Button>
      </PopoverTrigger>
      <PopoverContent side="top">
        <p className="px-2 pt-1.5 pb-1 text-xs text-muted-foreground">Инструменты</p>
        <button
          type="button"
          onClick={() => {
            onToggle(!web);
            setOpen(false);
          }}
          aria-pressed={web}
          className="flex w-full items-start gap-2.5 rounded-md p-2 text-left transition-colors hover:bg-accent focus-visible:bg-accent focus-visible:outline-none"
        >
          <Globe className="mt-0.5 size-4 shrink-0 text-muted-foreground" />
          <span className="min-w-0 flex-1">
            <span className="block text-sm font-medium">Поиск в сети</span>
            <span className="block text-xs text-muted-foreground">
              Агент сможет искать в интернете то, чего нет в документах
            </span>
          </span>
          {web && <Check className="mt-0.5 size-4 shrink-0" />}
        </button>
      </PopoverContent>
    </Popover>
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

function Bubble({
  message,
  pending = false,
  step = null,
  chatId = null,
}: {
  message: MessageOut;
  pending?: boolean;
  step?: Step | null;
  /** Null while the answer is still streaming: there is nothing to rate yet. */
  chatId?: string | null;
}) {
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
          {pending ? (
            <span className="flex items-center gap-2 text-muted-foreground">
              <Loader2 className="size-4 shrink-0 animate-spin" />
              {/* Truncated by the box rather than by character count: a wide
                  screen shows the whole query, a narrow one shows what fits. */}
              <span className="min-w-0 truncate">
                {step ? (STEP_LABEL[step.tool] ?? "Работаю") : "Думаю"}
                {step?.query ? `: ${step.query}` : ""}…
              </span>
            </span>
          ) : (
            message.content
          )}
        </div>
        {/* One row for everything the finished answer offers: where it came
            from, and a way to take it with you. */}
        {!isUser && !pending && message.content && (
          <div className="flex flex-wrap items-center gap-1.5">
            <Citations citations={message.citations} />
            <CopyButton text={message.content} />
            {chatId && <Rating chatId={chatId} messageId={message.id} was={message.rating} />}
          </div>
        )}
      </div>
    </div>
  );
}

/**
 * Take the answer with you.
 *
 * Kept visible rather than revealed on hover: there is no hover on a phone,
 * and a control that only exists for a mouse is a control half the time.
 */
function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(
    () => () => {
      if (timer.current) clearTimeout(timer.current);
    },
    [],
  );

  async function copy() {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      if (timer.current) clearTimeout(timer.current);
      timer.current = setTimeout(() => setCopied(false), 1500);
    } catch {
      // Refused rather than broken: the clipboard needs a secure context and a
      // real user gesture, and some browsers refuse it in an iframe regardless.
      toast.error("Браузер не дал скопировать");
    }
  }

  return (
    <button
      type="button"
      onClick={() => void copy()}
      aria-label={copied ? "Скопировано" : "Скопировать ответ"}
      className="inline-flex items-center gap-1 rounded-full border bg-background px-2.5 py-1 text-xs text-muted-foreground transition-colors hover:bg-accent hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none"
    >
      {copied ? <Check className="size-3" /> : <Copy className="size-3" />}
      {copied ? "Скопировано" : "Копировать"}
    </button>
  );
}

/**
 * Was this answer any good.
 *
 * Not a score to average — there is nothing here to average. It is that a
 * question somebody marked wrong is a question worth adding to the evaluation
 * set, and those are otherwise remembered by nobody.
 */
function Rating({
  chatId,
  messageId,
  was,
}: {
  chatId: string;
  messageId: string;
  was?: number | null;
}) {
  const [rating, setRating] = useState(was ?? 0);

  async function set(next: -1 | 1) {
    // Pressing the same thumb again takes it back, which is what somebody
    // expects from a control that shows its own state.
    const value = rating === next ? 0 : next;
    setRating(value);
    try {
      await api.rate(chatId, messageId, value);
    } catch (cause) {
      setRating(rating);
      toast.error(describe(cause));
    }
  }

  return (
    <>
      {([1, -1] as const).map((value) => {
        const Icon = value === 1 ? ThumbsUp : ThumbsDown;
        const active = rating === value;
        return (
          <button
            key={value}
            type="button"
            onClick={() => void set(value)}
            aria-pressed={active}
            aria-label={value === 1 ? "Хороший ответ" : "Плохой ответ"}
            className={cn(
              "inline-flex items-center rounded-full border px-2 py-1 text-xs transition-colors",
              active
                ? "border-transparent bg-primary text-primary-foreground"
                : "bg-background text-muted-foreground hover:bg-accent hover:text-foreground",
            )}
          >
            <Icon className="size-3" />
          </button>
        );
      })}
    </>
  );
}

function Citations({ citations }: { citations: Citation[] }) {
  async function open(citation: Citation) {
    // A page from the web is already a link; a document is one only after the
    // server signs it. The server has already checked that either is a scheme
    // a browser may be handed.
    if (citation.url) {
      window.open(citation.url, "_blank", "noopener,noreferrer");
      return;
    }
    if (!citation.document_id) return;
    try {
      const { url } = await api.documentLink(citation.document_id);
      window.open(url, "_blank", "noopener,noreferrer");
    } catch (cause) {
      toast.error(describe(cause));
    }
  }

  return (
    <>
      {citations.map((citation) => (
        <button
          key={`${citation.document_id ?? citation.url}-${citation.page}-${citation.n}`}
          onClick={() => void open(citation)}
          disabled={!citation.document_id && !citation.url}
          title={citation.snippet}
          className="inline-flex items-center gap-1 rounded-full border bg-background px-2.5 py-1 text-xs text-muted-foreground transition-colors hover:bg-accent hover:text-foreground disabled:pointer-events-none disabled:opacity-50"
        >
          <span className="font-medium text-foreground">[{citation.n}]</span>
          {citation.url && <Globe className="size-3 shrink-0" />}
          <span className="max-w-52 truncate">{citation.filename ?? "документ"}</span>
          {citation.page !== null && <span>· стр. {citation.page}</span>}
        </button>
      ))}
    </>
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
