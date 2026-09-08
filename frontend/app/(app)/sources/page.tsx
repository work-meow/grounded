"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  CheckCircle2,
  FileText,
  Link as LinkIcon,
  Loader2,
  TriangleAlert,
  Trash2,
  Upload,
} from "lucide-react";
import { toast } from "sonner";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Skeleton } from "@/components/ui/skeleton";
import { ApiError, api, type DocumentOut } from "@/lib/api";
import { cn } from "@/lib/utils";

import { ConnectedSources } from "./connected";

const ACCEPT = ".pdf,.docx,.pptx,.xlsx,.txt,.md";
const POLL_MS = 3000;

export default function SourcesPage() {
  const [documents, setDocuments] = useState<DocumentOut[]>([]);
  const [loadedAt, setLoadedAt] = useState(0);
  const [linking, setLinking] = useState(false);
  const [url, setUrl] = useState("");
  const [fetchingPage, setFetchingPage] = useState(false);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);

  // Bumped to ask for a fresh load once an upload has landed. A delete just
  // drops the row locally — there is nothing to wait for.
  const [reload, setReload] = useState(0);
  const refresh = useCallback(() => setReload((n) => n + 1), []);

  // One effect owns fetching. Indexing happens out of band in Pathway, so the
  // list polls itself — but only while something is still being indexed, and
  // never in parallel: the next poll is scheduled by the one that finished,
  // where setInterval would stack requests whenever a call ran long.
  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout>;

    const load = async (announceFailure: boolean) => {
      try {
        const loaded = await api.documents();
        if (cancelled) return;
        setDocuments(loaded);
        // When this list is current as of. Read here, in a callback, rather
        // than during render: "изменилось за 7 дней" needs a clock, and a
        // clock read while rendering makes the render depend on when it ran.
        setLoadedAt(Date.now());
        setLoading(false);
        if (loaded.some((document) => document.status === "processing")) {
          timer = setTimeout(() => void load(false), POLL_MS);
        }
      } catch (cause) {
        if (cancelled) return;
        setLoading(false);
        // A dropped poll is not worth interrupting the user over; a failed
        // first load is the difference between an empty list and a broken one.
        if (announceFailure) toast.error(describe(cause));
        else timer = setTimeout(() => void load(false), POLL_MS);
      }
    };

    void load(true);
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [reload]);

  async function upload(files: FileList | null) {
    if (!files?.length) return;
    setUploading(true);
    for (const file of Array.from(files)) {
      try {
        const created = await api.upload(file);
        if (created.text_layer === false) {
          // Said now, not discovered later: the file is stored and indexed
          // either way, but it will never come back from a search, and this is
          // the one moment somebody is standing here able to do anything.
          toast.warning(`${file.name}: похоже на скан — текста в файле нет, поиск его не найдёт`);
        } else {
          toast.success(`${file.name} загружен, идёт индексация`);
        }
      } catch (cause) {
        toast.error(`${file.name}: ${describe(cause)}`);
      }
    }
    setUploading(false);
    if (fileInput.current) fileInput.current.value = "";
    refresh();
  }

  async function addUrl() {
    const address = url.trim();
    if (!address || fetchingPage) return;
    setFetchingPage(true);
    try {
      const added = await api.addUrl(address);
      // The server's refusals are sentences meant for whoever pasted the link
      // — an unreachable host, a login wall, an address inside the network —
      // so they are shown as they came.
      toast.success(`Добавлено: ${added.filename}`);
      setUrl("");
      setLinking(false);
      refresh();
    } catch (cause) {
      toast.error(describe(cause));
    } finally {
      setFetchingPage(false);
    }
  }

  async function remove(document: DocumentOut) {
    // Guarded here as well as in the UI: a document from a connected source is
    // removed where it lives, and the API answers 404 for anything else.
    if (!document.removable) return;
    try {
      await api.deleteDocument(document.id);
      setDocuments((current) => current.filter((item) => item.id !== document.id));
      toast.success(`${document.filename} удалён`);
    } catch (cause) {
      toast.error(describe(cause));
    }
  }

  return (
    <div className="flex-1 overflow-y-auto">
      <div className="mx-auto w-full max-w-4xl space-y-8 p-4 sm:p-6">
        <div className="flex items-center justify-between gap-4">
          <div>
            <h1 className="text-lg font-semibold">Источники</h1>
            <p className="text-sm text-muted-foreground">
              PDF, DOCX, PPTX, XLSX, TXT, MD — до 64 МБ
            </p>
          </div>
          <input
            ref={fileInput}
            type="file"
            accept={ACCEPT}
            multiple
            className="sr-only"
            onChange={(event) => void upload(event.target.files)}
          />
          <Button variant="outline" onClick={() => setLinking((was) => !was)}>
            <LinkIcon className="size-4" />
            По ссылке
          </Button>
          <Button onClick={() => fileInput.current?.click()} disabled={uploading}>
            {uploading ? <Loader2 className="size-4 animate-spin" /> : <Upload className="size-4" />}
            Загрузить
          </Button>
        </div>

        {linking && (
          <form
            className="flex flex-wrap gap-2"
            onSubmit={(event) => {
              event.preventDefault();
              void addUrl();
            }}
          >
            <Input
              value={url}
              onChange={(event) => setUrl(event.target.value)}
              placeholder="https://example.com/статья"
              type="url"
              aria-label="Адрес страницы"
              className="min-w-64 flex-1"
              autoFocus
            />
            <Button type="submit" disabled={!url.trim() || fetchingPage}>
              {fetchingPage && <Loader2 className="size-4 animate-spin" />}
              {fetchingPage ? "Читаю…" : "Добавить"}
            </Button>
          </form>
        )}

        <ConnectedSources onChanged={refresh} />

        {loadedAt > 0 && <Changes documents={documents} asOf={loadedAt} />}

        <section className="grid gap-3 sm:grid-cols-2">
          {loading ? (
            <>
              <Skeleton className="h-24" />
              <Skeleton className="h-24" />
            </>
          ) : documents.length === 0 ? (
            <p className="text-sm text-muted-foreground sm:col-span-2">
              Пока ничего нет: загрузите файлы или подключите источник.
            </p>
          ) : (
            documents.map((document) => (
              <DocumentCard key={document.id} document={document} onRemove={remove} />
            ))
          )}
        </section>

      </div>
    </div>
  );
}

/**
 * What moved lately, by source.
 *
 * A base fed by Notion, Drive and a shared folder changes without anybody
 * watching, and the answer to "что я пропустил" was to scroll everything by
 * date and remember where you stopped.
 *
 * Computed from the list already on screen rather than fetched: the server has
 * an endpoint for callers who have no list, but this page has one, and a
 * second request for a summary of what is already here would be a request for
 * nothing.
 */
function Changes({ documents, asOf }: { documents: DocumentOut[]; asOf: number }) {
  const [days, setDays] = useState(7);
  const since = asOf - days * 24 * 3600 * 1000;

  const grouped = new Map<string, DocumentOut[]>();
  for (const document of documents) {
    if (new Date(document.created_at).getTime() < since) continue;
    const listed = grouped.get(document.source_name) ?? [];
    listed.push(document);
    grouped.set(document.source_name, listed);
  }
  // The source that moved most recently first, not whichever is alphabetically
  // lucky.
  const sources = [...grouped.entries()].sort(
    (a, b) => newest(b[1]) - newest(a[1]),
  );
  const total = [...grouped.values()].reduce((sum, listed) => sum + listed.length, 0);

  return (
    <section className="space-y-2 rounded-lg border p-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h2 className="text-sm font-medium">Что изменилось</h2>
        <div className="flex gap-1">
          {[7, 30, 90].map((window) => (
            <button
              key={window}
              type="button"
              onClick={() => setDays(window)}
              className={cn(
                "rounded-md px-2 py-0.5 text-xs transition-colors",
                days === window
                  ? "bg-accent font-medium text-accent-foreground"
                  : "text-muted-foreground hover:bg-accent/50 hover:text-foreground",
              )}
            >
              {window} дн.
            </button>
          ))}
        </div>
      </div>
      {total === 0 ? (
        <p className="text-sm text-muted-foreground">За это время ничего не менялось.</p>
      ) : (
        <ul className="space-y-1 text-sm">
          {sources.map(([name, listed]) => (
            <li key={name} className="flex flex-wrap items-baseline gap-x-2">
              <span className="text-muted-foreground">{name}:</span>
              <span className="min-w-0 truncate">
                {listed
                  .slice(0, 3)
                  .map((document) => document.filename)
                  .join(", ")}
                {listed.length > 3 && ` и ещё ${listed.length - 3}`}
              </span>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}

function newest(documents: DocumentOut[]): number {
  return Math.max(...documents.map((document) => new Date(document.created_at).getTime()));
}

function DocumentCard({
  document,
  onRemove,
}: {
  document: DocumentOut;
  onRemove: (document: DocumentOut) => void;
}) {
  const ready = document.status === "ready";
  return (
    <Card className="group">
      <CardContent className="flex items-start gap-3">
        <FileText className="mt-0.5 size-5 shrink-0 text-muted-foreground" />
        <div className="min-w-0 flex-1">
          <p className="truncate text-sm font-medium" title={document.filename}>
            {document.filename}
          </p>
          <div className="mt-2 flex flex-wrap items-center gap-2">
            <Badge variant={ready ? "secondary" : "outline"} className="gap-1">
              {ready ? (
                <CheckCircle2 className="size-3" />
              ) : (
                <Loader2 className="size-3 animate-spin" />
              )}
              {ready ? "Проиндексирован" : "Индексация"}
            </Badge>
            {document.text_layer === false && (
              <Badge
                variant="outline"
                className="gap-1 border-amber-500/40 bg-amber-500/10 text-amber-700 dark:text-amber-400"
                title="Похоже на скан: в файле нет текстового слоя, поэтому поиск по его содержимому ничего не находит"
              >
                <TriangleAlert className="size-3" />
                Без текста
              </Badge>
            )}
            <span className="text-xs text-muted-foreground">{size(document.size_bytes)}</span>
            <span className="text-xs text-muted-foreground">{document.source_name}</span>
          </div>
        </div>
        {/* A document from a connected source is deleted where it lives; a
            button here would either lie or bring it back on the next poll. */}
        {document.removable && (
          <Button
            variant="ghost"
            size="icon"
            className="size-8 opacity-0 transition-opacity group-hover:opacity-100 focus-visible:opacity-100"
            onClick={() => onRemove(document)}
            aria-label={`Удалить ${document.filename}`}
          >
            <Trash2 className="size-4" />
          </Button>
        )}
      </CardContent>
    </Card>
  );
}

function size(bytes: number | null): string {
  // A Notion page is not a file anywhere, so it has no size to report.
  if (bytes === null) return "";
  if (bytes < 1024) return `${bytes} Б`;
  if (bytes < 1024 * 1024) return `${Math.round(bytes / 1024)} КБ`;
  return `${(bytes / 1024 / 1024).toFixed(1)} МБ`;
}

function describe(cause: unknown): string {
  return cause instanceof ApiError ? cause.message : "Что-то пошло не так";
}
