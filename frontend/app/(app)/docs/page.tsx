"use client";

import { useEffect, useRef, useState, useSyncExternalStore } from "react";
import { ExternalLink } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";

import { Code, Fields, Method, Note, Section } from "./ui";

const CONTENTS = [
  { id: "start", label: "Быстрый старт" },
  { id: "auth", label: "Аутентификация" },
  { id: "endpoints", label: "Все методы" },
  { id: "answer", label: "Спросить" },
  { id: "stream", label: "Поток" },
  { id: "verify", label: "Проверка текста" },
  { id: "search", label: "Поиск без модели" },
  { id: "documents", label: "Документы" },
  { id: "chats", label: "Чаты" },
  { id: "cost", label: "Стоимость" },
  { id: "openai", label: "Формат OpenAI" },
  { id: "errors", label: "Ошибки" },
  { id: "limits", label: "Ограничения" },
] as const;

const ENDPOINTS = [
  ["POST", "/api/v1/answer", "Ответ по документам: одним JSON или потоком"],
  ["POST", "/api/v1/verify", "Проверить готовый текст по документам, утверждение за утверждением"],
  ["GET", "/api/v1/search", "Фрагменты без модели — десятая доля секунды, ноль токенов"],
  ["GET", "/api/v1/documents", "Всё, что доступно для поиска"],
  ["POST", "/api/v1/documents", "Загрузить файл (multipart/form-data)"],
  ["GET", "/api/v1/documents/{id}/link", "Где открыть оригинал, с ?page= и ?quote= — на нужном месте"],
  ["DELETE", "/api/v1/documents/{id}", "Удалить загруженный файл"],
  ["GET", "/api/v1/sources", "Подключённые источники и их состояние"],
  ["POST", "/api/v1/chats", "Создать сохранённый диалог"],
  ["GET", "/api/v1/chats", "Список диалогов"],
  ["GET", "/api/v1/chats/{id}/messages", "Конец чата страницами, в порядке чтения"],
  ["DELETE", "/api/v1/chats/{id}", "Удалить диалог с сообщениями"],
  ["POST", "/api/v1/chat/completions", "То же самое в формате OpenAI"],
  ["GET", "/api/v1/models", "grounded и grounded-web"],
  ["GET", "/api/v1/me", "Чей это токен — самая дешёвая проверка"],
] as const;

const LANGUAGES = ["curl", "Python", "JavaScript", "OpenAI SDK"] as const;

function samples(host: string): Record<(typeof LANGUAGES)[number], string> {
  return {
    curl: `curl -X POST ${host}/api/v1/answer \\
  -H "Authorization: Bearer $TOKEN" \\
  -H "Content-Type: application/json" \\
  -d '{"question": "сколько дней отпуска?"}'`,
    Python: `import httpx

response = httpx.post(
    "${host}/api/v1/answer",
    headers={"Authorization": f"Bearer {TOKEN}"},
    json={"question": "сколько дней отпуска?"},
    timeout=180,
)
answer = response.json()
print(answer["answer"])
print(answer["citations"], answer["usage"]["cost_usd"])`,
    JavaScript: `const response = await fetch("${host}/api/v1/answer", {
  method: "POST",
  headers: {
    Authorization: \`Bearer \${TOKEN}\`,
    "Content-Type": "application/json",
  },
  body: JSON.stringify({ question: "сколько дней отпуска?" }),
});

const { answer, citations, usage } = await response.json();
console.log(answer, citations, usage.cost_usd);`,
    "OpenAI SDK": `from openai import OpenAI

client = OpenAI(base_url="${host}/api/v1", api_key=TOKEN)

done = client.chat.completions.create(
    model="grounded",  # grounded-web — с поиском в интернете
    messages=[{"role": "user", "content": "сколько дней отпуска?"}],
)
print(done.choices[0].message.content)
print(done.usage.model_extra["cost"], done.model_extra["citations"])`,
  };
}

const ANSWER_EXAMPLE = `{
  "answer": "Основной отпуск — 28 календарных дней [1].",
  "citations": [
    {
      "n": 1,
      "document_id": "0f0f9d3c-9c4e-4f4b-9d61-0b1d1c1c1c1c",
      "filename": "Политика.pdf",
      "page": 3,
      "url": null,
      "snippet": "28 календарных дней…"
    }
  ],
  "steps": [{ "tool": "search_knowledge", "query": "отпуск дней политика" }],
  "usage": {
    "prompt_tokens": 5000,
    "completion_tokens": 60,
    "total_tokens": 5060,
    "cost_usd": 0.00133215,
    "cost_complete": true,
    "calls": 3,
    "stages": [
      { "stage": "rerank", "model": "google/gemini-2.5-flash-lite", "calls": 2,
        "prompt_tokens": 1900, "completion_tokens": 22, "cost_usd": 0.0002269 },
      { "stage": "answer", "model": "openai/gpt-5-mini", "calls": 1,
        "prompt_tokens": 3100, "completion_tokens": 38, "cost_usd": 0.00110525 }
    ]
  },
  "took_ms": 6227,
  "chat_id": null,
  "message_id": null
}`;

/** The origin never changes while the page is open, so there is nothing to
 * subscribe to — the store is read once on the client and once on the server. */
const subscribeToNothing = () => () => {};
const readOrigin = () => window.location.origin;
const serverOrigin = () => "https://ваш-хост";

export default function DocsPage() {
  const scroller = useRef<HTMLDivElement>(null);
  const [active, setActive] = useState<string>(CONTENTS[0].id);
  const [language, setLanguage] = useState<(typeof LANGUAGES)[number]>("curl");
  // The real host, so that every sample on the page is worth copying. Read
  // through useSyncExternalStore rather than set from an effect: this page is
  // prerendered, `window` does not exist then, and this is the one API that
  // has a server snapshot for exactly that — no cascading render, no
  // hydration mismatch.
  const host = useSyncExternalStore(subscribeToNothing, readOrigin, serverOrigin);

  // Which section the reader is in. Observed rather than computed on scroll:
  // one callback per crossing instead of one per frame.
  useEffect(() => {
    const root = scroller.current;
    if (!root) return;
    const observer = new IntersectionObserver(
      (entries) => {
        const visible = entries
          .filter((entry) => entry.isIntersecting)
          .sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top)[0];
        if (visible) setActive(visible.target.id);
      },
      { root, rootMargin: "0px 0px -70% 0px", threshold: 0 },
    );
    for (const { id } of CONTENTS) {
      const element = document.getElementById(id);
      if (element) observer.observe(element);
    }
    return () => observer.disconnect();
  }, []);

  const code = samples(host);

  return (
    <div ref={scroller} className="flex-1 overflow-y-auto">
      <div className="mx-auto flex max-w-6xl gap-10 px-5 py-8 sm:px-8">
        <article className="min-w-0 flex-1 space-y-10 pb-24">
          <header>
            <p className="text-[11px] font-medium tracking-widest text-muted-foreground uppercase">
              Документация
            </p>
            <h1 className="mt-2 text-3xl font-semibold tracking-tight">API</h1>
            <p className="mt-3 max-w-2xl text-[15px] text-muted-foreground">
              Спросить по своим документам, найти фрагмент без модели, загрузить файл — из своего
              кода. Один и тот же движок отвечает в собственном формате и в формате OpenAI, так что
              готовый SDK работает по одному <code className="font-mono text-[13px]">base_url</code>.
            </p>
            <div className="mt-4 flex flex-wrap items-center gap-2">
              <Badge variant="secondary" className="font-mono text-[11px]">
                {host}/api/v1
              </Badge>
              <a
                href="/api/docs"
                target="_blank"
                rel="noreferrer"
                className="inline-flex items-center gap-1 text-xs text-muted-foreground underline-offset-4 hover:text-foreground hover:underline"
              >
                OpenAPI и живые схемы
                <ExternalLink className="size-3" />
              </a>
            </div>
          </header>

          <Section
            id="start"
            title="Быстрый старт"
            lead="Токен выпускается на сервере одной командой, дальше — обычный HTTP."
          >
            <Code>{`cd backend && uv run rag-token --email вы@example.com --days 365`}</Code>
            <div className="flex flex-wrap gap-1">
              {LANGUAGES.map((name) => (
                <button
                  key={name}
                  type="button"
                  onClick={() => setLanguage(name)}
                  className={cn(
                    "rounded-md px-2.5 py-1 text-xs transition-colors",
                    language === name
                      ? "bg-accent font-medium text-accent-foreground"
                      : "text-muted-foreground hover:bg-accent/50 hover:text-foreground",
                  )}
                >
                  {name}
                </button>
              ))}
            </div>
            <Code>{code[language]}</Code>
          </Section>

          <Section
            id="auth"
            title="Аутентификация"
            lead="Тот же JWT, которым входят в интерфейс, в заголовке Authorization."
          >
            <Code>{`curl -H "Authorization: Bearer $TOKEN" ${host}/api/v1/me
# {"user_id":"db8122ef-4ce8-4dce-a3d4-d6b0ab53cf08"}`}</Code>
            <Note>
              Токен — это и есть пользователь: всё, что видно по нему, принадлежит одному человеку,
              а фильтр по <code className="font-mono">user_id</code> уходит в сам индекс. Отозвать
              токен досрочно нельзя — проверка подписи не ходит в базу, — поэтому для интеграции
              стоит выпустить отдельный, с нужным сроком.
            </Note>
          </Section>

          <Section id="endpoints" title="Все методы">
            <div className="overflow-hidden rounded-lg border">
              <ul className="divide-y">
                {ENDPOINTS.map(([method, path, text]) => (
                  <li
                    key={`${method} ${path}`}
                    className="flex flex-col gap-1 px-4 py-2.5 sm:flex-row sm:items-baseline sm:gap-3"
                  >
                    <span className="flex items-baseline gap-3 sm:w-[19rem] sm:shrink-0">
                      <Method>{method}</Method>
                      <code className="font-mono text-[13px]">{path}</code>
                    </span>
                    <span className="text-sm text-muted-foreground">{text}</span>
                  </li>
                ))}
              </ul>
            </div>
          </Section>

          <Section
            id="answer"
            title="Спросить"
            lead="POST /api/v1/answer — вопрос, ответ со сносками, шаги агента и стоимость запроса."
          >
            <Fields
              head={["поле запроса", "что делает"]}
              rows={[
                { name: "question", type: "string", text: "Вопрос, 1–8000 символов" },
                {
                  name: "stream",
                  type: "bool · false",
                  text: "true — server-sent events вместо одного JSON",
                },
                {
                  name: "web",
                  type: "bool · false",
                  text: "Разрешить поиск в интернете, когда в документах ответа нет. Выключено по умолчанию: один поиск стоит около $0.00125",
                },
                {
                  name: "chat_id",
                  type: "uuid · null",
                  text: "Продолжить сохранённый диалог: история подгрузится, вопрос и ответ запишутся",
                },
                {
                  name: "history",
                  type: "array · []",
                  text: "Или ведите историю сами. Вместе с chat_id — 422: это два диалога, называющих себя одним",
                },
              ]}
            />
            <Fields
              head={["поле ответа", "что это"]}
              rows={[
                { name: "answer", type: "string", text: "Текст со сносками [1], [2]" },
                {
                  name: "citations",
                  type: "array",
                  text: "Только те источники, на которые ответ действительно ссылается. Пустой список — ответа в базе нет, и это факт о выдаче, а не мнение модели",
                },
                {
                  name: "steps",
                  type: "array",
                  text: "Что агент делал, по порядку. Два поиска подряд означают, что первый ничего не дал",
                },
                { name: "usage", type: "object", text: "Токены и деньги — ниже" },
                { name: "took_ms", type: "int", text: "Сколько ждал вызывающий" },
                {
                  name: "chat_id, message_id",
                  type: "uuid · null",
                  text: "Заполнены, только если вопрос задан в сохранённый чат",
                },
              ]}
            />
            <Code>{ANSWER_EXAMPLE}</Code>
            <Note>
              Сноска бывает и на страницу из интернета — тогда{" "}
              <code className="font-mono">document_id</code> пустой, а{" "}
              <code className="font-mono">url</code> заполнен. У фрагмента из базы наоборот: ссылку
              на оригинал даёт <code className="font-mono">/documents/&#123;id&#125;/link</code>,
              который умеет её подписать.
            </Note>
          </Section>

          <Section
            id="stream"
            title="Поток"
            lead="stream: true — тот же ответ как text/event-stream. Данные каждого события — JSON."
          >
            <Fields
              head={["событие", "данные"]}
              rows={[
                {
                  name: "step",
                  text: "{ tool, query } — приходит дважды: сначала имя инструмента, затем запрос, как только он разобрался",
                },
                { name: "token", text: "Строка с куском ответа" },
                { name: "citations", text: "Итоговый список источников" },
                { name: "usage", text: "Объект usage целиком" },
                {
                  name: "done",
                  text: "Весь ответ одним объектом — тот же JSON, что вернул бы вызов без потока",
                },
                { name: "error", text: "Ход не удался; done после него не будет" },
              ]}
            />
            <Note>
              Последнее событие — это и есть весь ответ, поэтому собирать токены вручную не нужно:
              показывайте их для скорости, а читайте <code className="font-mono">done</code>.
            </Note>
            <Code>{`const response = await fetch("${host}/api/v1/answer", {
  method: "POST",
  headers: { Authorization: \`Bearer \${TOKEN}\`, "Content-Type": "application/json" },
  body: JSON.stringify({ question: "что у меня по проекту?", stream: true }),
});

const reader = response.body.getReader();
const decoder = new TextDecoder();
let buffer = "";

while (true) {
  const { done, value } = await reader.read();
  if (done) break;
  buffer += decoder.decode(value, { stream: true });

  const frames = buffer.split("\\n\\n");
  buffer = frames.pop() ?? "";
  for (const frame of frames) {
    const [head, body] = frame.split("\\n");
    const event = head.replace("event: ", "");
    const data = JSON.parse(body.replace("data: ", ""));
    if (event === "token") process.stdout.write(data);
    if (event === "done") console.log(data.citations, data.usage.cost_usd);
  }
}`}</Code>
          </Section>

          <Section
            id="verify"
            title="Проверка текста"
            lead="POST /api/v1/verify — обратная сторона обещания «только по документам»: даёте абзац, получаете разбор по утверждениям."
          >
            <Code>{`curl -X POST ${host}/api/v1/verify \\
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \\
  -d '{"text": "Основной отпуск — 28 дней, заявление за три месяца."}'`}</Code>
            <Fields
              head={["вердикт", "что значит"]}
              rows={[
                { name: "supported", text: "Фрагменты подтверждают утверждение" },
                {
                  name: "contradicted",
                  text: "Во фрагментах сказано иное — самый дорогой случай: кто-то собирается сказать то, что записано иначе",
                },
                { name: "absent", text: "В документах об этом ничего нет — повод написать документ" },
                {
                  name: "unknown",
                  text: "Судья не ответил. Не то же самое, что absent: про документы мы ничего не узнали, и принять это за «в базе нет» значит отправить человека писать то, что уже написано",
                },
              ]}
            />
            <Note>
              До 20 000 символов и не больше 20 утверждений: каждое стоит поиска и вердикта, около
              $0.002 за страницу прозы. Каждый вердикт приходит с фрагментами, на которых основан,
              — чтобы с ним можно было не согласиться.
            </Note>
          </Section>

          <Section
            id="search"
            title="Поиск без модели"
            lead="GET /api/v1/search — говорит, где написано; никогда не объясняет и не пересказывает."
          >
            <Fields
              head={["параметр", "что делает"]}
              rows={[
                { name: "q", type: "string", text: "Запрос, 1–500 символов" },
                { name: "limit", type: "int · 20", text: "1–50" },
                { name: "source", type: "uuid", text: "Искать только в одном источнике — сужение уходит в сам индекс" },
                { name: "days", type: "int", text: "1–3650: только документы, изменённые за последние N дней" },
              ]}
            />
            <Code>{`curl -G ${host}/api/v1/search \\
  -H "Authorization: Bearer $TOKEN" \\
  --data-urlencode "q=вишлист" --data-urlencode "limit=5"`}</Code>
            <Code>{`{
  "hits": [
    {
      "document_id": "0f0f…",
      "filename": "Заметки.md",
      "page": null,
      "snippet": [
        { "text": "добавил в ", "hit": false },
        { "text": "вишлисте", "hit": true },
        { "text": " наушники", "hit": false }
      ]
    }
  ],
  "found": 20,
  "took_ms": 96
}`}</Code>
            <Note>
              <code className="font-mono">snippet</code> приходит уже разрезанным на куски с флагом{" "}
              <code className="font-mono">hit</code> — правило, что считать совпадением, живёт на
              сервере, а клиенту не нужно класть в DOM ничего, кроме текста.{" "}
              <code className="font-mono">found</code> — сколько индекс предложил до отбора: «3 из
              20» говорит, что поиск работал. Ответ <code className="font-mono">503</code> означает,
              что индекс перестраивается, а не что запрос неверный.
            </Note>
          </Section>

          <Section
            id="documents"
            title="Документы"
            lead="Форматы: .pdf, .docx, .pptx, .xlsx, .txt, .md — до 64 МБ."
          >
            <Code>{`curl -X POST ${host}/api/v1/documents \\
  -H "Authorization: Bearer $TOKEN" \\
  -F "file=@Политика.pdf"`}</Code>
            <Fields
              head={["поле", "что это"]}
              rows={[
                {
                  name: "status",
                  text: "processing или ready. Читается прямо из индекса: ready значит, что документ действительно находится поиском, а не что его приняли",
                },
                {
                  name: "text_layer",
                  text: "false — это PDF без текстового слоя, то есть скан: он загрузится и будет считаться готовым, но не найдётся никогда",
                },
                {
                  name: "removable",
                  text: "false — документ из подключённого источника: удалять его нужно там, где он лежит, а DELETE вернёт 404",
                },
              ]}
            />
            <Note>
              Формат, который мы не читаем, — это <code className="font-mono">415</code> сразу, а не
              документ, навсегда застрявший в статусе «индексируется».
            </Note>
          </Section>

          <Section
            id="chats"
            title="Чаты"
            lead="Нужны, только если вы хотите, чтобы историю хранили мы. Без chat_id не сохраняется ничего."
          >
            <Code>{`CHAT=$(curl -sX POST ${host}/api/v1/chats \\
  -H "Authorization: Bearer $TOKEN" | jq -r .id)

curl -X POST ${host}/api/v1/answer \\
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \\
  -d "{\\"question\\": \\"а что там по срокам?\\", \\"chat_id\\": \\"$CHAT\\"}"`}</Code>
            <Note>
              История отдаётся страницами по 50 сообщений:{" "}
              <code className="font-mono">next_cursor</code> из ответа передаётся обратно в{" "}
              <code className="font-mono">before</code>. Заголовок чата ставится из первого вопроса.
            </Note>
          </Section>

          <Section
            id="cost"
            title="Стоимость"
            lead="Не оценка по таблице цен, которая устареет, а сумма того, что провайдер выставил за каждый вызов."
          >
            <Fields
              head={["поле", "что это"]}
              rows={[
                { name: "cost_usd", text: "Доллары, сумма по вызовам" },
                {
                  name: "cost_complete",
                  text: "false, если какой-то вызов цену не сообщил — тогда cost_usd это нижняя граница, а не счёт",
                },
                { name: "calls", text: "Сколько платных вызовов сделал ход" },
                {
                  name: "stages",
                  text: "Те же числа по этапам: answer (сам агент), rerank (отбор релевантного за каждым поиском), web_search",
                },
              ]}
            />
            <Note>
              Разбивка обычно и есть самое интересное. Замерено на живых вызовах: отбор релевантного
              — $0.000021 из четырёх кандидатов и около $0.00017 из двадцати; один поиск в сети —
              $0.0012607, то есть поход в интернет дороже, чем всё остальное в коротком ответе.
              Реальный ответ про ключевую ставку с включённой сетью стоил $0.0024171: ответ
              $0.001068 плюс поиск $0.0013491.
            </Note>
            <Note>
              Один запрос ограничен сверху: не больше 3 поисков по базе, 2 в сети, 6 вызовов
              инструментов и 8 вызовов модели за ход, плюс общий таймаут. Сбежать со счётом один
              вызов не может.
            </Note>
          </Section>

          <Section
            id="openai"
            title="Формат OpenAI"
            lead="Чтобы работал уже написанный клиент: официальные SDK, n8n, LangChain, LibreChat, плагины редакторов."
          >
            <Code>{code["OpenAI SDK"]}</Code>
            <Note>
              <b className="text-foreground">Имя модели — это переключатель.</b> Фиксированный
              протокол не даёт передать наши флаги, но имя модели даёт всегда:{" "}
              <code className="font-mono">grounded</code> отвечает по документам,{" "}
              <code className="font-mono">grounded-web</code> может ещё и поискать в интернете.
              Любое другое имя — <code className="font-mono">404</code>, а не молчаливая подмена.
            </Note>
            <Fields
              head={["что теряется в переводе", "почему"]}
              rows={[
                {
                  name: "системное сообщение",
                  text: "Игнорируется: наш промпт — это то, что держит ответ на документах и нумерует сноски",
                },
                {
                  name: "temperature, top_p, max_tokens",
                  text: "Принимаются и игнорируются. Один ход — это несколько вызовов модели плюс поиск; выполнить такой параметр наполовину хуже, чем не выполнять вовсе",
                },
                {
                  name: "вопрос",
                  text: "Это последнее сообщение от пользователя, а не последнее сообщение: клиент, дославший реплику ассистента, продолжает диалог, чей вопрос по-прежнему выше",
                },
                {
                  name: "сноски",
                  text: "Приходят в нестандартном поле citations рядом с choices — так же делают Perplexity и OpenRouter. Клиент, который его не читает, всё равно покажет связный ответ: [1] стоит в самом тексте",
                },
                {
                  name: "стоимость",
                  text: "В usage.cost — расширение OpenRouter к тому же объекту. Разбивка по этапам есть только в нативном /answer",
                },
                {
                  name: "обрыв потока",
                  text: "finish_reason: length и [DONE]: своего кадра для ошибки протокол не предусматривает, и «ответ не закончен» — честное его прочтение",
                },
              ]}
            />
          </Section>

          <Section id="errors" title="Ошибки">
            <Note>
              Везде <code className="font-mono">{`{"detail": "…"}`}</code>, кроме двух путей OpenAI,
              где <code className="font-mono">{`{"error": {…}}`}</code> — так читает ошибку клиент
              OpenAI.
            </Note>
            <Fields
              head={["код", "что значит"]}
              rows={[
                { name: "400", text: "Пустой вопрос, пустой файл, имя файла без имени" },
                {
                  name: "401",
                  text: "Нет токена, не наша подпись, истёк. В ответе заголовок WWW-Authenticate: Bearer",
                },
                {
                  name: "404",
                  text: "Нет такого чата, документа или модели — или он не ваш: какие id существуют, знает только владелец",
                },
                { name: "413", text: "Файл больше 64 МБ или тело запроса больше 4 МБ" },
                { name: "415", text: "Формат, который мы не читаем" },
                { name: "422", text: "Тело не проходит валидацию — detail это список полей" },
                {
                  name: "502",
                  text: "Ход не удался. Причина в логах сервера, не в ответе: в тексте таких ошибок бывают внутренние адреса и ключи",
                },
                { name: "503", text: "Индекс перестраивается — только у /search" },
                { name: "504", text: "Модель не ответила за отведённое время" },
              ]}
            />
          </Section>

          <Section id="limits" title="Ограничения">
            <Fields
              rows={[
                {
                  name: "дневной лимит выключен",
                  text: "DAILY_COST_LIMIT_USD задаёт потолок в долларах на сутки UTC; по умолчанию 0 — без потолка, потому что у личной установки один вызывающий. Ставить стоит, как только токен отдан чему-то ещё: досрочно он не истекает",
                },
                {
                  name: "тело запроса — 4 МБ",
                  text: "У двух эндпоинтов загрузки — 64 МБ плюс запас. Ограничение стоит перед всем остальным, потому что тело разбирается до проверки токена",
                },
                {
                  name: "/answer без фильтров",
                  text: "Агент сам решает, когда сузить поиск по дате, по формулировке вопроса. Явные фильтры есть у /search",
                },
                {
                  name: "источники — через интерфейс",
                  text: "Подключение и отключение только на странице «Источники»: там пошаговые инструкции под каждым видом",
                },
                {
                  name: "CORS закрыт",
                  text: "Кроме CORS_ORIGINS из настроек. API рассчитан на вызов с сервера: браузерная интеграция потребует добавить свой origin и решить, где будет лежать токен",
                },
              ]}
            />
          </Section>
        </article>

        <nav className="sticky top-8 hidden h-fit w-52 shrink-0 lg:block">
          <p className="mb-3 text-[11px] font-medium tracking-widest text-muted-foreground uppercase">
            Содержание
          </p>
          <ul className="space-y-0.5 border-l">
            {CONTENTS.map(({ id, label }) => (
              <li key={id}>
                <a
                  href={`#${id}`}
                  className={cn(
                    "-ml-px block border-l py-1 pl-3 text-sm transition-colors",
                    active === id
                      ? "border-primary font-medium text-foreground"
                      : "border-transparent text-muted-foreground hover:text-foreground",
                  )}
                >
                  {label}
                </a>
              </li>
            ))}
          </ul>
        </nav>
      </div>
    </div>
  );
}
