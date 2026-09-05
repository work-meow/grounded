"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";
import { KeyRound, Loader2 } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import { ApiError, api } from "@/lib/api";

export default function LoginPage() {
  const router = useRouter();
  const [token, setToken] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [pending, setPending] = useState(false);

  async function submit(event: React.FormEvent) {
    event.preventDefault();
    if (!token.trim() || pending) return;

    setPending(true);
    setError(null);
    try {
      await api.login(token.trim());
      // The server has set an HttpOnly cookie; nothing to store on this side.
      router.replace("/chat");
    } catch (cause) {
      setError(cause instanceof ApiError ? cause.message : "Не удалось войти");
      setPending(false);
    }
  }

  return (
    // Its own scroll: the shell around it no longer scrolls, and this card is
    // taller than a phone in landscape with the keyboard up.
    <main className="flex flex-1 items-center justify-center overflow-y-auto p-6">
      <Card className="w-full max-w-md">
        <CardHeader>
          <div className="flex size-10 items-center justify-center rounded-lg bg-primary/10 text-primary">
            <KeyRound className="size-5" />
          </div>
          <CardTitle className="mt-3">Вход</CardTitle>
          <CardDescription>
            Вставьте токен доступа. Он будет сохранён в защищённой cookie и не попадёт в
            localStorage.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <form onSubmit={submit} className="space-y-4">
            <Textarea
              value={token}
              onChange={(event) => setToken(event.target.value)}
              placeholder="eyJhbGciOiJIUzI1NiIs..."
              autoFocus
              spellCheck={false}
              className="min-h-28 font-mono text-xs break-all"
              aria-label="Токен доступа"
              aria-invalid={error !== null}
            />
            {error && (
              <p role="alert" className="text-sm text-destructive">
                {error}
              </p>
            )}
            <Button type="submit" className="w-full" disabled={!token.trim() || pending}>
              {pending && <Loader2 className="size-4 animate-spin" />}
              Войти
            </Button>
          </form>
          <p className="mt-4 text-xs text-muted-foreground">
            Токен выпускается командой{" "}
            <code className="font-mono">uv run rag-token --email вы@example.com</code>
          </p>
        </CardContent>
      </Card>
    </main>
  );
}
