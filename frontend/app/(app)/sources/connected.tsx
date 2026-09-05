"use client";

/**
 * Connecting a source.
 *
 * The form is driven by `required_fields` from the API rather than by a list
 * kept here: the server decides what a kind needs, and a form that disagreed
 * would fail validation with no way for the user to tell why. Everything in
 * CATALOGUE below is presentation — a name, an icon, and the instructions for
 * getting the credential, which is the part people actually get stuck on.
 */

import { useCallback, useEffect, useState } from "react";
import {
  Cloud,
  CloudUpload,
  HardDrive,
  Loader2,
  type LucideIcon,
  NotebookPen,
  Package,
  Plus,
  Trash2,
} from "lucide-react";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Skeleton } from "@/components/ui/skeleton";
import {
  ApiError,
  api,
  type ConnectorKind,
  type ConnectorOut,
  type ConnectorsOut,
  type SourceStatus,
} from "@/lib/api";
import { cn } from "@/lib/utils";

type Presentation = {
  label: string;
  icon: LucideIcon;
  /**
   * How to obtain the credential, one step per line. `email` is the service
   * account a Drive folder is shared with.
   *
   * Written out in full rather than summarised. Three of these five services
   * hand out a refresh token exactly once, in a browser redirect, and a hint
   * that stops at "create an app" leaves someone holding a client id and no
   * way to turn it into anything.
   */
  steps: (email: string) => string[];
  labels: Record<string, string>;
  placeholders?: Record<string, string>;
};

const CATALOGUE: Record<ConnectorKind, Presentation> = {
  gdrive: {
    label: "Google Drive",
    icon: HardDrive,
    steps: (email) => [
      "Создайте папку в своём Google Drive и положите в неё документы.",
      `Правой кнопкой по папке → «Открыть доступ» → добавьте ${email || "адрес сервисного аккаунта"} с правом «Читатель».`,
      "Откройте папку и скопируйте ID из адресной строки — это часть после /folders/.",
    ],
    labels: { folder_id: "ID папки" },
    placeholders: { folder_id: "1AbCdEfGhIjKlMnOpQrStUvWxYz" },
  },
  notion: {
    label: "Notion",
    icon: NotebookPen,
    steps: () => [
      "notion.so/my-integrations → «New integration» → выберите workspace.",
      "Скопируйте Internal Integration Secret.",
      "На каждой нужной странице: «⋯» → Connections → выберите интеграцию. Читается только то, что вы подключили явно.",
    ],
    labels: { token: "Integration Secret" },
    placeholders: { token: "ntn_..." },
  },
  yandex: {
    label: "Яндекс.Диск",
    icon: Cloud,
    steps: () => [
      "oauth.yandex.ru/client/new → создайте приложение с правом «Чтение всего Диска» (cloud_api:disk.read).",
      "Скопируйте ClientID приложения.",
      "Откройте https://oauth.yandex.ru/authorize?response_type=token&client_id=ВАШ_CLIENTID и подтвердите доступ.",
      "Вас перебросит на страницу, где будет сам токен — он же виден в адресной строке после access_token=. Именно его сюда, не ClientID.",
    ],
    labels: { token: "OAuth-токен", path: "Папка" },
    placeholders: { token: "y0_...", path: "/Документы" },
  },
  dropbox: {
    label: "Dropbox",
    icon: Package,
    steps: () => [
      "dropbox.com/developers/apps → Create app → Scoped access → выберите доступ к папке или ко всему Dropbox.",
      "Вкладка Permissions → отметьте files.metadata.read и files.content.read → Submit.",
      "Вкладка Settings → скопируйте App key и App secret.",
      "Откройте https://www.dropbox.com/oauth2/authorize?client_id=APP_KEY&response_type=code&token_access_type=offline и скопируйте выданный код.",
      "Обменяйте код на постоянный refresh token: curl -u APP_KEY:APP_SECRET -d grant_type=authorization_code -d code=КОД https://api.dropbox.com/oauth2/token",
    ],
    labels: {
      app_key: "App key",
      app_secret: "App secret",
      refresh_token: "Refresh token",
      path: "Папка",
    },
    placeholders: { path: "/Документы" },
  },
  onedrive: {
    label: "OneDrive",
    icon: CloudUpload,
    steps: () => [
      "portal.azure.com → App registrations → New registration → разрешите личные аккаунты Microsoft, Redirect URI типа Web: http://localhost",
      "API permissions → Microsoft Graph → Delegated → Files.Read.All и offline_access.",
      "Certificates & secrets → New client secret → скопируйте значение (Value, не Secret ID).",
      "Откройте https://login.microsoftonline.com/common/oauth2/v2.0/authorize?client_id=CLIENT_ID&response_type=code&redirect_uri=http://localhost&scope=Files.Read.All%20offline_access и скопируйте code из адресной строки.",
      "Обменяйте код на refresh token: curl -d client_id=... -d client_secret=... -d grant_type=authorization_code -d redirect_uri=http://localhost -d code=КОД https://login.microsoftonline.com/common/oauth2/v2.0/token",
    ],
    labels: {
      client_id: "Client ID",
      client_secret: "Client secret",
      refresh_token: "Refresh token",
      path: "Папка",
    },
    placeholders: { path: "Документы" },
  },
};

const KINDS = Object.keys(CATALOGUE) as ConnectorKind[];

/**
 * How long to wait before asking again.
 *
 * Quickly while a source is still unchecked: adding one restarts the indexer,
 * which then has to poll before it can say anything, and up to a minute of "not
 * looked at yet" is the honest answer to a question somebody just asked.
 *
 * Slowly the rest of the time, but not never — the line says how long ago a
 * source was checked, and one rendered once would go on claiming three minutes
 * for as long as the tab stayed open.
 */
const POLL_MS = 10_000;
const IDLE_POLL_MS = 60_000;

const DOT: Record<SourceStatus, string> = {
  ok: "bg-emerald-500",
  error: "bg-destructive",
  unknown: "bg-muted-foreground/40",
};

/**
 * Drive is the one kind that needs setting up on the server: the service
 * account key lives in the indexer, and its address is what a user shares a
 * folder with. Without that address there is nothing to tell them to share
 * with, so the option is not offered — the API refuses it for the same reason.
 */
function offered(serviceAccount: string): ConnectorKind[] {
  return serviceAccount ? KINDS : KINDS.filter((kind) => kind !== "gdrive");
}

/** A field whose value is a credential is not typed in plain sight. */
function isSecret(field: string): boolean {
  return /secret|token|key/.test(field);
}

export function ConnectedSources({ onChanged }: { onChanged: () => void }) {
  const [state, setState] = useState<ConnectorsOut | null>(null);
  const [failed, setFailed] = useState(false);
  const [adding, setAdding] = useState<ConnectorKind | null>(null);

  // Same shape as the document list: one effect owns fetching, a counter asks
  // for a fresh one. Setting state from inside a callback an effect started is
  // what react-hooks/set-state-in-effect exists to prevent.
  const [reload, setReload] = useState(0);
  const refresh = useCallback(() => setReload((n) => n + 1), []);

  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout>;

    const load = async (first: boolean) => {
      try {
        const loaded = await api.connectors();
        if (cancelled) return;
        setState(loaded);
        if (loaded.sources.length > 0) {
          const waiting = loaded.sources.some((source) => source.status === "unknown");
          timer = setTimeout(() => void load(false), waiting ? POLL_MS : IDLE_POLL_MS);
        }
      } catch {
        // A dropped poll is not worth replacing the list with an error; a
        // failed first load is the difference between empty and broken.
        if (!cancelled && first) setFailed(true);
      }
    };

    void load(true);
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [reload]);

  async function remove(source: ConnectorOut) {
    try {
      await api.deleteConnector(source.id);
      toast.success(`${source.name} отключён`);
    } catch (cause) {
      toast.error(cause instanceof ApiError ? cause.message : "Что-то пошло не так");
    } finally {
      // Either way. Removing a source deletes the row and then republishes the
      // manifest, so a failure can still have changed the list — and a list
      // left showing what is no longer there invites the user to try again on
      // something that is already gone.
      refresh();
      onChanged();
    }
  }

  if (failed) {
    return (
      <Section>
        <p className="text-sm text-muted-foreground">Не удалось загрузить список источников.</p>
      </Section>
    );
  }
  if (!state) {
    return (
      <Section>
        <Skeleton className="h-20" />
      </Section>
    );
  }
  if (!state.enabled) {
    return (
      <Section>
        <p className="text-sm text-muted-foreground">
          Подключение источников не настроено на этом сервере: не задан SECRETS_KEY.
        </p>
      </Section>
    );
  }

  return (
    <Section>
      <div className="space-y-3">
        {state.sources.map((source) => (
          <ConnectedCard key={source.id} source={source} onRemove={remove} />
        ))}

        {adding ? (
          <AddForm
            kind={adding}
            fields={state.required_fields[adding] ?? []}
            serviceAccount={state.gdrive_service_account_email}
            onCancel={() => setAdding(null)}
            onAdded={() => {
              setAdding(null);
              refresh();
              onChanged();
            }}
          />
        ) : (
          <div className="flex flex-wrap gap-2">
            {offered(state.gdrive_service_account_email).map((kind) => {
              const { label, icon: Icon } = CATALOGUE[kind];
              return (
                <Button key={kind} variant="outline" size="sm" onClick={() => setAdding(kind)}>
                  <Icon className="size-4" />
                  {label}
                  <Plus className="size-3 opacity-60" />
                </Button>
              );
            })}
          </div>
        )}
      </div>
    </Section>
  );
}

function Section({ children }: { children: React.ReactNode }) {
  return (
    <section>
      <h2 className="mb-3 text-sm font-medium text-muted-foreground">Подключённые источники</h2>
      {children}
    </section>
  );
}

function ConnectedCard({
  source,
  onRemove,
}: {
  source: ConnectorOut;
  onRemove: (source: ConnectorOut) => void;
}) {
  const { label, icon: Icon } = CATALOGUE[source.kind];
  return (
    <Card className="group">
      <CardContent className="flex items-center gap-3">
        <Icon className="size-5 shrink-0 text-muted-foreground" />
        <div className="min-w-0 flex-1">
          <p className="truncate text-sm font-medium">{source.name}</p>
          <p className="text-xs text-muted-foreground">{label}</p>
          <p
            className={cn(
              "mt-1 flex items-center gap-1.5 text-xs",
              source.status === "error" ? "text-destructive" : "text-muted-foreground",
            )}
          >
            <span className={cn("size-1.5 shrink-0 rounded-full", DOT[source.status])} />
            <span className="min-w-0 flex-1">{report(source)}</span>
          </p>
        </div>
        <Button
          variant="ghost"
          size="icon"
          className="size-8 opacity-0 transition-opacity group-hover:opacity-100 focus-visible:opacity-100"
          onClick={() => onRemove(source)}
          aria-label={`Отключить ${source.name}`}
        >
          <Trash2 className="size-4" />
        </Button>
      </CardContent>
    </Card>
  );
}

/**
 * What to say about a source, in one line.
 *
 * The problem text comes from the indexer, which is the only process that ever
 * talks to the service. It has already been reduced there to a sentence about
 * the cause, because the exception behind it carries urls, internal hostnames
 * and on some clients the credential itself.
 */
function report(source: ConnectorOut): string {
  if (source.status === "error") return source.problem || "источник недоступен";
  if (source.status === "unknown") return "ожидает первой проверки";
  const seen = source.documents === null ? "" : ` · ${files(source.documents)}`;
  return `проверен ${ago(source.checked_at)}${seen}`;
}

function ago(iso: string | null): string {
  if (!iso) return "только что";
  const minutes = Math.round((Date.now() - Date.parse(iso)) / 60_000);
  if (minutes < 1) return "только что";
  if (minutes < 60) return `${minutes} мин назад`;
  const hours = Math.round(minutes / 60);
  return hours < 24 ? `${hours} ч назад` : `${Math.round(hours / 24)} дн назад`;
}

/** What the last listing saw. Zero is the useful number here, not a blank. */
function files(count: number): string {
  const tail = count % 10;
  const teens = count % 100;
  if (tail === 1 && teens !== 11) return `${count} файл`;
  if (tail >= 2 && tail <= 4 && (teens < 12 || teens > 14)) return `${count} файла`;
  return `${count} файлов`;
}

function AddForm({
  kind,
  fields,
  serviceAccount,
  onCancel,
  onAdded,
}: {
  kind: ConnectorKind;
  fields: string[];
  serviceAccount: string;
  onCancel: () => void;
  onAdded: () => void;
}) {
  const presentation = CATALOGUE[kind];
  const [name, setName] = useState(presentation.label);
  const [config, setConfig] = useState<Record<string, string>>({});
  const [saving, setSaving] = useState(false);

  async function submit(event: React.FormEvent) {
    event.preventDefault();
    setSaving(true);
    try {
      await api.addConnector(kind, name, config);
      toast.success("Источник подключён — индекс перестраивается, это занимает до полминуты");
    } catch (cause) {
      toast.error(cause instanceof ApiError ? cause.message : "Что-то пошло не так");
      setSaving(false);
      return;
    }
    onAdded();
  }

  return (
    <Card>
      <CardContent>
        <form className="space-y-3" onSubmit={submit}>
          <div className="flex items-center gap-2">
            <presentation.icon className="size-5 text-muted-foreground" />
            <span className="text-sm font-medium">{presentation.label}</span>
          </div>
          <ol className="list-decimal space-y-1 pl-4 text-xs leading-relaxed text-muted-foreground">
            {presentation.steps(serviceAccount).map((step) => (
              // The step text is the key: these lists are static and never
              // reordered, and an index would be a worse identity than the
              // sentence itself.
              <li key={step} className="break-words">
                {step}
              </li>
            ))}
          </ol>

          <label className="block space-y-1">
            <span className="text-xs text-muted-foreground">Название</span>
            <Input value={name} onChange={(event) => setName(event.target.value)} required />
          </label>

          {fields.map((field) => (
            <label key={field} className="block space-y-1">
              <span className="text-xs text-muted-foreground">
                {presentation.labels[field] ?? field}
              </span>
              <Input
                type={isSecret(field) ? "password" : "text"}
                autoComplete="off"
                placeholder={presentation.placeholders?.[field]}
                value={config[field] ?? ""}
                onChange={(event) =>
                  setConfig((current) => ({ ...current, [field]: event.target.value }))
                }
                required
              />
            </label>
          ))}

          <div className="flex gap-2">
            <Button type="submit" size="sm" disabled={saving}>
              {saving && <Loader2 className="size-4 animate-spin" />}
              Подключить
            </Button>
            <Button type="button" variant="ghost" size="sm" onClick={onCancel} disabled={saving}>
              Отмена
            </Button>
          </div>
        </form>
      </CardContent>
    </Card>
  );
}
