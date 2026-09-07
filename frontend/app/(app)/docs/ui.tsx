"use client";

import { useState } from "react";
import { Check, Copy } from "lucide-react";

import { cn } from "@/lib/utils";

/** An HTTP method, coloured the way every API reference colours them. */
export function Method({ children }: { children: "GET" | "POST" | "PUT" | "DELETE" }) {
  const tone = {
    GET: "text-sky-600 dark:text-sky-400",
    POST: "text-emerald-600 dark:text-emerald-400",
    PUT: "text-amber-600 dark:text-amber-400",
    DELETE: "text-rose-600 dark:text-rose-400",
  }[children];
  return (
    <span className={cn("shrink-0 font-mono text-[11px] font-semibold tracking-wide", tone)}>
      {children}
    </span>
  );
}

/**
 * A code sample with a copy button.
 *
 * Copying is the whole point of a sample, and the alternative — selecting
 * seven lines of shell in a scrolling container — is the part people give up
 * on. The button says what happened rather than toasting: it is next to the
 * thing it acted on.
 */
export function Code({ children, className }: { children: string; className?: string }) {
  const [copied, setCopied] = useState(false);

  async function copy() {
    try {
      await navigator.clipboard.writeText(children);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      // A denied clipboard permission is not worth an error message: the text
      // is right there and selectable.
    }
  }

  return (
    <div className={cn("group relative", className)}>
      <pre className="overflow-x-auto rounded-lg border bg-muted/40 p-4 pr-12 text-[13px] leading-relaxed">
        <code className="font-mono">{children}</code>
      </pre>
      <button
        type="button"
        onClick={copy}
        aria-label={copied ? "Скопировано" : "Скопировать"}
        className="absolute top-2.5 right-2.5 rounded-md border bg-background/80 p-1.5 text-muted-foreground opacity-0 transition group-hover:opacity-100 hover:text-foreground focus-visible:opacity-100"
      >
        {copied ? <Check className="size-3.5 text-emerald-500" /> : <Copy className="size-3.5" />}
      </button>
    </div>
  );
}

/** One row of a two-column reference table. */
export type Row = { name: string; type?: string; text: React.ReactNode };

export function Fields({ rows, head }: { rows: Row[]; head?: [string, string] }) {
  return (
    <div className="overflow-hidden rounded-lg border">
      {head && (
        <div className="grid grid-cols-[minmax(9rem,14rem)_1fr] gap-4 border-b bg-muted/40 px-4 py-2 text-[11px] font-medium tracking-wide text-muted-foreground uppercase">
          <div>{head[0]}</div>
          <div>{head[1]}</div>
        </div>
      )}
      <dl className="divide-y">
        {rows.map((row) => (
          <div
            key={row.name}
            className="grid gap-1 px-4 py-3 sm:grid-cols-[minmax(9rem,14rem)_1fr] sm:gap-4"
          >
            <dt className="flex flex-wrap items-baseline gap-2">
              <span className="font-mono text-[13px] text-foreground">{row.name}</span>
              {row.type && (
                <span className="font-mono text-[11px] text-muted-foreground">{row.type}</span>
              )}
            </dt>
            <dd className="text-sm text-muted-foreground">{row.text}</dd>
          </div>
        ))}
      </dl>
    </div>
  );
}

/** A section with the id its table-of-contents entry points at. */
export function Section({
  id,
  title,
  lead,
  children,
}: {
  id: string;
  title: string;
  lead?: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <section id={id} className="scroll-mt-6 border-t pt-10 first:border-t-0 first:pt-0">
      <h2 className="text-xl font-semibold tracking-tight">{title}</h2>
      {lead && <p className="mt-2 max-w-2xl text-sm text-muted-foreground">{lead}</p>}
      <div className="mt-5 space-y-5">{children}</div>
    </section>
  );
}

/** A short aside: a measurement, a caveat, a thing that costs money. */
export function Note({ children }: { children: React.ReactNode }) {
  return (
    <p className="border-l-2 border-border pl-4 text-sm text-muted-foreground">{children}</p>
  );
}
