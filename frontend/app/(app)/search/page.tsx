"use client";

/**
 * Search, without the model.
 *
 * The chat is the good way to get an answer and an expensive way to get a
 * location: two model calls and several seconds to be told where something is
 * written. This is the same index, the same ranking and the same tenant
 * filter, straight — about a tenth of a second, and no tokens.
 *
 * So the page is built around that being fast: it searches as you type, keeps
 * the previous results on screen while the next ones arrive, and says how long
 * the index took, because that number is the whole argument for this page
 * existing next to the chat.
 */

import { useEffect, useState } from "react";
import { FileText, Search, SquareArrowOutUpRight } from "lucide-react";
import { toast } from "sonner";

import { Input } from "@/components/ui/input";
import { Skeleton } from "@/components/ui/skeleton";
import { ApiError, api, type SearchHit, type SearchOut } from "@/lib/api";
import { plural } from "@/lib/plural";

/**
 * Long enough that typing a word is one request rather than six, short enough
 * that the results feel like they are keeping up. The request in flight is
 * aborted when the next keystroke lands, so a slow one never overwrites a
 * newer one.
 */
const DEBOUNCE_MS = 220;

export default function SearchPage() {
  const [query, setQuery] = useState("");
  const [result, setResult] = useState<SearchOut | null>(null);
  const [failure, setFailure] = useState("");
  const [busy, setBusy] = useState(false);

  const text = query.trim();

  useEffect(() => {
    if (!text) return;

    const controller = new AbortController();
    const timer = setTimeout(async () => {
      setBusy(true);
      try {
        const found = await api.search(text, controller.signal);
        setResult(found);
        setFailure("");
      } catch (cause) {
        // An abort is the next keystroke, not a failure.
        if (!controller.signal.aborted) setFailure(describe(cause));
      } finally {
        setBusy(false);
      }
    }, DEBOUNCE_MS);

    return () => {
      clearTimeout(timer);
      controller.abort();
    };
  }, [text]);

  return (
    <div className="flex-1 overflow-y-auto">
      <div className="mx-auto w-full max-w-4xl space-y-5 p-4 sm:p-6">
        <div>
          <h1 className="text-lg font-semibold">Поиск</h1>
          <p className="text-sm text-muted-foreground">
            По фрагментам документов, без модели — мгновенно и бесплатно
          </p>
        </div>

        <div className="relative">
          <Search className="pointer-events-none absolute top-1/2 left-3 size-4 -translate-y-1/2 text-muted-foreground" />
          <Input
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            onKeyDown={(event) => event.key === "Escape" && setQuery("")}
            placeholder="Что найти в документах?"
            aria-label="Поисковый запрос"
            autoFocus
            className="h-11 pl-9"
          />
        </div>

        {/* Deliberately keyed off the trimmed query rather than off `result`:
            the previous results stay on screen while the next ones are on
            their way, so the list does not blink on every keystroke. */}
        {!text ? (
          <Hint />
        ) : failure ? (
          <p className="text-sm text-destructive">{failure}</p>
        ) : result ? (
          <Results result={result} busy={busy} />
        ) : (
          <div className="space-y-3">
            <Skeleton className="h-20" />
            <Skeleton className="h-20" />
          </div>
        )}
      </div>
    </div>
  );
}

function Hint() {
  return (
    <div className="rounded-lg border border-dashed p-6 text-center">
      <p className="text-sm font-medium">Начните печатать</p>
      <p className="mt-1 text-sm text-muted-foreground">
        Найдётся и по смыслу, и по точному слову — во всём, что загружено и подключено.
        <br />
        За ответом своими словами идите в чат.
      </p>
    </div>
  );
}

function Results({ result, busy }: { result: SearchOut; busy: boolean }) {
  if (result.hits.length === 0) {
    return (
      <div className="rounded-lg border border-dashed p-6 text-center">
        <p className="text-sm font-medium">Ничего не нашлось</p>
        <p className="mt-1 text-sm text-muted-foreground">
          Попробуйте другое слово — или спросите в чате, он умеет переформулировать.
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      <p
        aria-live="polite"
        className={`text-xs text-muted-foreground transition-opacity ${busy ? "opacity-50" : ""}`}
      >
        {plural(result.hits.length, "фрагмент", "фрагмента", "фрагментов")} · {result.took_ms} мс
      </p>
      {result.hits.map((hit, index) => (
        // The index is part of the key because the same page of the same
        // document can legitimately match twice, and rank is what orders them.
        <Row key={`${hit.document_id}-${hit.page}-${index}`} hit={hit} />
      ))}
    </div>
  );
}

function Row({ hit }: { hit: SearchHit }) {
  async function open() {
    if (!hit.document_id) return;
    try {
      const { url } = await api.documentLink(hit.document_id);
      window.open(url, "_blank", "noopener,noreferrer");
    } catch (cause) {
      toast.error(describe(cause));
    }
  }

  return (
    <button
      type="button"
      onClick={() => void open()}
      disabled={!hit.document_id}
      className="group block w-full rounded-lg border bg-card p-3 text-left transition-colors hover:bg-accent/40 focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none disabled:pointer-events-none disabled:opacity-70"
    >
      <div className="flex items-center gap-2 text-xs text-muted-foreground">
        <FileText className="size-3.5 shrink-0" />
        <span className="truncate font-medium text-foreground">{hit.filename ?? "документ"}</span>
        {hit.page !== null && <span className="shrink-0">· стр. {hit.page}</span>}
        <SquareArrowOutUpRight className="ml-auto size-3.5 shrink-0 opacity-0 transition-opacity group-hover:opacity-100 group-focus-visible:opacity-100" />
      </div>
      <p className="mt-1.5 text-sm leading-relaxed text-muted-foreground">
        {hit.snippet.map((piece, index) =>
          piece.hit ? (
            // Positional runs of one string, never reordered — the one place
            // an index really is the identity.
            <mark key={index} className="rounded-sm bg-primary/15 px-0.5 text-foreground">
              {piece.text}
            </mark>
          ) : (
            <span key={index}>{piece.text}</span>
          ),
        )}
      </p>
    </button>
  );
}

function describe(cause: unknown): string {
  return cause instanceof ApiError ? cause.message : "Что-то пошло не так";
}
