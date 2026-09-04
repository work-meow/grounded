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
} from "@/lib/api";

type Presentation = {
  label: string;
  icon: LucideIcon;
  /** How to obtain the credential. `email` is the service account to share with. */
  help: (email: string) => string;
  labels: Record<string, string>;
  placeholders?: Record<string, string>;
};

const CATALOGUE: Record<ConnectorKind, Presentation> = {
  gdrive: {
    label: "Google Drive",
    icon: HardDrive,
    help: (email) =>
      `Откройте папку в Drive → «Поделиться» → добавьте ${email || "адрес сервисного аккаунта"} с правом «Читатель». ID папки — это часть ссылки после /folders/.`,
    labels: { folder_id: "ID папки" },
    placeholders: { folder_id: "1AbCdEfGhIjKlMnOpQrStUvWxYz" },
  },
  notion: {
    label: "Notion",
    icon: NotebookPen,
    help: () =>
      "notion.so/my-integrations → «New integration» → скопируйте Internal Integration Secret. Затем на каждой нужной странице: «⋯» → Connections → выберите интеграцию. Читаются только те страницы, которые вы подключили.",
    labels: { token: "Integration Secret" },
    placeholders: { token: "ntn_..." },
  },
  yandex: {
    label: "Яндекс.Диск",
    icon: Cloud,
    help: () =>
      "oauth.yandex.ru → создайте приложение с правом «Чтение всего Диска» → получите OAuth-токен.",
    labels: { token: "OAuth-токен", path: "Папка" },
    placeholders: { path: "/Документы" },
  },
  dropbox: {
    label: "Dropbox",
    icon: Package,
    help: () =>
      "dropbox.com/developers/apps → приложение с правом files.content.read и files.metadata.read → App key и App secret оттуда же, refresh token выдаётся один раз при первой авторизации с параметром token_access_type=offline.",
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
    help: () =>
      "portal.azure.com → App registrations → приложение с правами Files.Read.All и offline_access → Client ID и Client secret оттуда же, refresh token выдаётся при первой авторизации.",
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
    api
      .connectors()
      .then((loaded) => !cancelled && setState(loaded))
      .catch(() => !cancelled && setFailed(true));
    return () => {
      cancelled = true;
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
          <p className="text-xs leading-relaxed text-muted-foreground">
            {presentation.help(serviceAccount)}
          </p>

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
