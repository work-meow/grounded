"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  CheckCircle2,
  FileText,
  GitBranch,
  HardDrive,
  Loader2,
  NotebookPen,
  Trash2,
  Upload,
} from "lucide-react";
import { toast } from "sonner";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { ApiError, api, type DocumentOut } from "@/lib/api";

const ACCEPT = ".pdf,.docx,.pptx,.xlsx,.txt,.md";
const POLL_MS = 3000;

const PLANNED = [
  { name: "Google Drive", icon: HardDrive },
  { name: "GitHub", icon: GitBranch },
  { name: "Notion", icon: NotebookPen },
] as const;

export default function SourcesPage() {
  const [documents, setDocuments] = useState<DocumentOut[]>([]);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);

  const refresh = useCallback(async () => {
    try {
      setDocuments(await api.documents());
    } catch (cause) {
      toast.error(describe(cause));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const loaded = await api.documents();
        if (!cancelled) setDocuments(loaded);
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

  // Indexing happens out of band in Pathway, so poll — but only while something
  // is actually still being indexed.
  const anyProcessing = documents.some((document) => document.status === "processing");
  useEffect(() => {
    if (!anyProcessing) return;
    const timer = setInterval(() => void refresh(), POLL_MS);
    return () => clearInterval(timer);
  }, [anyProcessing, refresh]);

  async function upload(files: FileList | null) {
    if (!files?.length) return;
    setUploading(true);
    for (const file of Array.from(files)) {
      try {
        await api.upload(file);
        toast.success(`${file.name} загружен, идёт индексация`);
      } catch (cause) {
        toast.error(`${file.name}: ${describe(cause)}`);
      }
    }
    setUploading(false);
    if (fileInput.current) fileInput.current.value = "";
    await refresh();
  }

  async function remove(document: DocumentOut) {
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
          <Button onClick={() => fileInput.current?.click()} disabled={uploading}>
            {uploading ? <Loader2 className="size-4 animate-spin" /> : <Upload className="size-4" />}
            Добавить
          </Button>
        </div>

        <section className="grid gap-3 sm:grid-cols-2">
          {loading ? (
            <>
              <Skeleton className="h-24" />
              <Skeleton className="h-24" />
            </>
          ) : documents.length === 0 ? (
            <p className="text-sm text-muted-foreground sm:col-span-2">
              Пока ничего не загружено.
            </p>
          ) : (
            documents.map((document) => (
              <DocumentCard key={document.id} document={document} onRemove={remove} />
            ))
          )}
        </section>

        <section>
          <h2 className="mb-3 text-sm font-medium text-muted-foreground">Коннекторы</h2>
          <div className="grid gap-3 sm:grid-cols-3">
            {PLANNED.map(({ name, icon: Icon }) => (
              <Card key={name} className="opacity-60">
                <CardContent className="flex items-center gap-3">
                  <Icon className="size-5 shrink-0 text-muted-foreground" />
                  <div className="min-w-0">
                    <p className="truncate text-sm font-medium">{name}</p>
                    <p className="text-xs text-muted-foreground">Скоро</p>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </section>
      </div>
    </div>
  );
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
            <span className="text-xs text-muted-foreground">{size(document.size_bytes)}</span>
          </div>
        </div>
        <Button
          variant="ghost"
          size="icon"
          className="size-8 opacity-0 transition-opacity group-hover:opacity-100 focus-visible:opacity-100"
          onClick={() => onRemove(document)}
          aria-label={`Удалить ${document.filename}`}
        >
          <Trash2 className="size-4" />
        </Button>
      </CardContent>
    </Card>
  );
}

function size(bytes: number): string {
  if (bytes < 1024) return `${bytes} Б`;
  if (bytes < 1024 * 1024) return `${Math.round(bytes / 1024)} КБ`;
  return `${(bytes / 1024 / 1024).toFixed(1)} МБ`;
}

function describe(cause: unknown): string {
  return cause instanceof ApiError ? cause.message : "Что-то пошло не так";
}
