"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { FolderOpen, LogOut, MessagesSquare, Search, Sparkles } from "lucide-react";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { ApiError, api } from "@/lib/api";
import { cn } from "@/lib/utils";

const NAV = [
  { href: "/chat", label: "Чат", icon: MessagesSquare },
  { href: "/search", label: "Поиск", icon: Search },
  { href: "/sources", label: "Источники", icon: FolderOpen },
] as const;

export default function AppLayout({ children }: { children: React.ReactNode }) {
  const router = useRouter();
  const pathname = usePathname();
  const [userId, setUserId] = useState<string | null>(null);

  // The session cookie is HttpOnly, so asking the server is the only way to
  // know whether we are signed in.
  useEffect(() => {
    let cancelled = false;
    api
      .me()
      .then(({ user_id }) => !cancelled && setUserId(user_id))
      .catch((cause) => {
        if (cancelled) return;
        if (cause instanceof ApiError && cause.status === 401) {
          router.replace("/login");
          return;
        }
        // Anything else — the API is down, the network is gone — used to be
        // swallowed, leaving a skeleton spinning in the header with no reason
        // given and no reason to expect it to end.
        toast.error("Сервер недоступен. Обновите страницу.");
      });
    return () => {
      cancelled = true;
    };
  }, [router]);

  async function logout() {
    await api.logout().catch(() => undefined);
    router.replace("/login");
  }

  return (
    <div className="flex flex-1 flex-col">
      <header className="flex h-14 shrink-0 items-center justify-between border-b px-4">
        <Link href="/chat" className="flex items-center gap-2 font-semibold">
          <Sparkles className="size-5 text-primary" />
          База знаний
        </Link>
        <div className="flex items-center gap-3">
          {userId ? (
            <span className="hidden font-mono text-xs text-muted-foreground sm:inline">
              {userId.slice(0, 8)}
            </span>
          ) : (
            <Skeleton className="hidden h-4 w-16 sm:block" />
          )}
          <Button variant="ghost" size="icon" onClick={logout} aria-label="Выйти">
            <LogOut className="size-4" />
          </Button>
        </div>
      </header>

      <div className="flex flex-1 overflow-hidden">
        <nav className="flex w-14 shrink-0 flex-col gap-1 border-r p-2 sm:w-44 sm:p-3">
          {NAV.map(({ href, label, icon: Icon }) => {
            const active = pathname.startsWith(href);
            return (
              <Link
                key={href}
                href={href}
                aria-current={active ? "page" : undefined}
                className={cn(
                  "flex items-center gap-2 rounded-md px-2 py-2 text-sm transition-colors sm:px-3",
                  active
                    ? "bg-accent font-medium text-accent-foreground"
                    : "text-muted-foreground hover:bg-accent/50 hover:text-foreground",
                )}
              >
                <Icon className="size-4 shrink-0" />
                <span className="hidden sm:inline">{label}</span>
              </Link>
            );
          })}
        </nav>

        <main className="flex flex-1 flex-col overflow-hidden">{children}</main>
      </div>
    </div>
  );
}
