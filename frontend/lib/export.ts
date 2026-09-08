/**
 * A conversation as markdown, with its sources.
 *
 * A good answer wants to leave the app — into a ticket, a document, a message
 * to somebody who asked. Copying it from the screen loses the citations, which
 * are most of why it was worth keeping: without them it is an assertion, and
 * the whole point of this system is that its answers are checkable.
 *
 * Built here rather than on the server because everything needed is already on
 * screen, and a download that has to wait for a round trip is one that
 * sometimes does not happen.
 */
import type { Citation, MessageOut } from "@/lib/api";

export function asMarkdown(title: string, messages: MessageOut[]): string {
  const lines: string[] = [`# ${title}`, ""];
  for (const message of messages) {
    if (message.role === "user") {
      lines.push(`## ${message.content.trim()}`, "");
      continue;
    }
    lines.push(message.content.trim(), "");
    if (message.citations.length) {
      lines.push("Источники:");
      // Numbered as the answer numbered them, gaps and all: the markers in the
      // text point at these, and renumbering would break that tie silently.
      for (const citation of message.citations) lines.push(`- ${source(citation)}`);
      lines.push("");
    }
  }
  return lines.join("\n");
}

function source(citation: Citation): string {
  const name = citation.filename ?? "документ";
  const where = citation.page !== null && citation.page !== undefined ? `, стр. ${citation.page}` : "";
  // A web page keeps its address; a document has none that would still work
  // tomorrow — the signed link expires in fifteen minutes, so putting it in a
  // file somebody keeps would be a dead link by the time they opened it.
  const link = citation.url ? ` — ${citation.url}` : "";
  return `[${citation.n}] ${name}${where}${link}`;
}

/** Hand the markdown to the browser as a file. */
export function download(filename: string, body: string): void {
  const url = URL.createObjectURL(new Blob([body], { type: "text/markdown;charset=utf-8" }));
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  // Revoked on the next tick rather than immediately: the click is
  // asynchronous, and freeing the blob first gives some browsers an empty file.
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

/** A filename from the chat's title: no slashes, no surprises, dated. */
export function filenameFor(title: string): string {
  const stem = title.replace(/[^\p{L}\p{N} _-]+/gu, "").trim().slice(0, 60) || "разговор";
  return `${stem} — ${new Date().toISOString().slice(0, 10)}.md`;
}
