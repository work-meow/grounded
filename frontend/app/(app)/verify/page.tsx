"use client";

import { useRef, useState } from "react";
import { AlertTriangle, CheckCircle2, CircleSlash, HelpCircle, Loader2 } from "lucide-react";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { ApiError, api, type ClaimOut, type VerifyOut } from "@/lib/api";
import { cn } from "@/lib/utils";

/**
 * How each verdict reads, and what the reader should do about it.
 *
 * Four and not three: "не удалось проверить" is not "в базе этого нет". The
 * second sends somebody off to write a document, and doing that when the
 * judge merely failed is the one mistake here that costs real work.
 */
const VERDICT = {
  supported: {
    label: "подтверждено",
    icon: CheckCircle2,
    tone: "text-emerald-600 dark:text-emerald-400",
    ring: "border-emerald-500/30",
  },
  contradicted: {
    label: "противоречит документам",
    icon: AlertTriangle,
    tone: "text-rose-600 dark:text-rose-400",
    ring: "border-rose-500/40",
  },
  absent: {
    label: "в базе этого нет",
    icon: CircleSlash,
    tone: "text-muted-foreground",
    ring: "border-border",
  },
  unknown: {
    label: "не удалось проверить",
    icon: HelpCircle,
    tone: "text-amber-600 dark:text-amber-400",
    ring: "border-amber-500/30",
  },
} as const;

const MAX_CHARS = 20_000;

export default function VerifyPage() {
  const [text, setText] = useState("");
  const [checking, setChecking] = useState(false);
  const [result, setResult] = useState<VerifyOut | null>(null);
  const running = useRef<AbortController | null>(null);

  async function check() {
    const body = text.trim();
    if (!body || checking) return;
    running.current?.abort();
    const controller = new AbortController();
    running.current = controller;
    setChecking(true);
    setResult(null);
    try {
      setResult(await api.verify(body, controller.signal));
    } catch (cause) {
      if (!controller.signal.aborted) toast.error(describe(cause));
    } finally {
      if (running.current === controller) setChecking(false);
    }
  }

  const counted = result ? tally(result.claims) : null;

  return (
    <div className="flex-1 overflow-y-auto">
      <div className="mx-auto max-w-3xl space-y-6 px-5 py-8 sm:px-8">
        <header>
          <h1 className="text-xl font-semibold tracking-tight">Проверка по базе</h1>
          <p className="mt-2 text-sm text-muted-foreground">
            Вставьте текст — черновик письма, пересказ политики, чужое утверждение. Каждая мысль
            будет проверена по вашим документам: что они подтверждают, чему противоречат и о чём
            вообще не знают.
          </p>
        </header>

        <div className="space-y-2">
          <Textarea
            value={text}
            onChange={(event) => setText(event.target.value.slice(0, MAX_CHARS))}
            placeholder="Основной отпуск — 28 дней, заявление за две недели. Премия выплачивается раз в квартал…"
            className="min-h-40 resize-y"
            aria-label="Текст для проверки"
          />
          <div className="flex items-center justify-between gap-3">
            <span className="text-xs text-muted-foreground">
              {text.length > 0 && `${text.length} из ${MAX_CHARS} символов`}
            </span>
            <Button onClick={check} disabled={!text.trim() || checking}>
              {checking && <Loader2 className="size-4 animate-spin" />}
              {checking ? "Проверяю…" : "Проверить"}
            </Button>
          </div>
        </div>

        {counted && (
          <div className="flex flex-wrap items-center gap-x-4 gap-y-1 border-t pt-4 text-sm">
            {(Object.keys(VERDICT) as (keyof typeof VERDICT)[]).map((verdict) =>
              counted[verdict] ? (
                <span key={verdict} className={cn("flex items-center gap-1.5", VERDICT[verdict].tone)}>
                  {counted[verdict]} {VERDICT[verdict].label}
                </span>
              ) : null,
            )}
            <span className="ml-auto font-mono text-xs text-muted-foreground">
              ${result?.usage.cost_usd.toFixed(5)} · {Math.round((result?.took_ms ?? 0) / 100) / 10}с
            </span>
          </div>
        )}

        <div className="space-y-3">
          {result?.claims.map((claim, index) => <Claim key={index} checked={claim} />)}
        </div>
      </div>
    </div>
  );
}

function Claim({ checked }: { checked: ClaimOut }) {
  const { label, icon: Icon, tone, ring } = VERDICT[checked.verdict] ?? VERDICT.unknown;
  return (
    <div className={cn("space-y-2 rounded-lg border border-l-2 p-4", ring)}>
      <p className="text-sm">{checked.claim}</p>
      <div className={cn("flex items-center gap-1.5 text-xs", tone)}>
        <Icon className="size-3.5 shrink-0" />
        {label}
      </div>
      {checked.why && (
        <p className="border-l-2 border-border pl-3 text-xs text-muted-foreground">{checked.why}</p>
      )}
      {checked.citations.length > 0 && (
        <div className="flex flex-wrap gap-1.5 pt-0.5">
          {checked.citations.map((citation) => (
            <Source key={citation.n} citation={citation} />
          ))}
        </div>
      )}
    </div>
  );
}

/** A fragment the verdict was based on — the reader's own way to check it. */
function Source({ citation }: { citation: ClaimOut["citations"][number] }) {
  async function open() {
    if (!citation.document_id) return;
    try {
      const { url } = await api.documentLink(citation.document_id, {
        page: citation.page,
        quote: citation.snippet,
      });
      window.open(url, "_blank", "noopener,noreferrer");
    } catch (cause) {
      toast.error(describe(cause));
    }
  }

  return (
    <button
      type="button"
      onClick={open}
      title={citation.snippet}
      className="max-w-full truncate rounded-md border px-2 py-0.5 text-xs text-muted-foreground transition-colors hover:bg-accent/50 hover:text-foreground"
    >
      {citation.filename ?? "документ"}
      {citation.page !== null && ` · стр. ${citation.page}`}
    </button>
  );
}

function tally(claims: ClaimOut[]): Record<keyof typeof VERDICT, number> {
  const counted = { supported: 0, contradicted: 0, absent: 0, unknown: 0 };
  for (const claim of claims) counted[claim.verdict] = (counted[claim.verdict] ?? 0) + 1;
  return counted;
}

function describe(cause: unknown): string {
  if (cause instanceof ApiError) return cause.message;
  return "Не удалось проверить текст";
}
