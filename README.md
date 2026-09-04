# База знаний

Личная RAG-система: загружаете документы — задаёте вопросы — получаете ответы
со ссылками на конкретные страницы.

```
                        браузер
                           │
                    Next.js + shadcn/ui
                           │  JWT в HttpOnly cookie
                           ▼
                   FastAPI  ──────────────►  PostgreSQL
                      │                      users / sources / documents
                LangChain agent              chats / messages
                      │
             search_knowledge
             list_sources
             read_document
                      │
                      ▼
              Pathway  /v1/retrieve
        usearch KNN + tantivy BM25 → RRF
                      │
                      ▼
                     S3
              оригиналы файлов
```

Два процесса, три внешних сервиса. Индексация живая: файл, попавший в S3,
становится доступным для поиска сам, без перезапуска и без очередей.

```
backend/    FastAPI, агент, PostgreSQL          собственный venv
indexer/    Pathway: парсинг, чанки, индекс     собственный venv
shared/     раскладка ключей в S3               ставится в оба
frontend/   Next.js
```

## Почему так

**Статус индексации нигде не дублируется.** В PostgreSQL нет колонки `status` —
она читается прямо из Pathway (`/v1/inputs`) на каждый запрос списка файлов.
Двум источникам правды нечему разъезжаться. Если индексер лежит, список файлов
всё равно отдаётся.

**Изоляция пользователей — на точном равенстве, а не на совпадении строк.**
Ключ в S3 (`users/<user>/sources/<source>/<document>/<file>`) разбирается
индексером обратно в метаданные каждого чанка, и поиск фильтруется выражением
`user_id == '<uuid>'`. В фильтр подставляются объекты `UUID`, а не строки —
именно типизация здесь и есть защита от инъекции.

**Гибридный поиск, а не только векторный.** BM25 находит точные термины —
фамилии, номера, ссылки на статьи — которые эмбеддинги регулярно теряют.
`HybridIndexFactory` объединяет обе выдачи через reciprocal rank fusion.

**Два virtualenv, а не один.** `pathway[xpack-llm]` пинит `langchain<0.4` ради
адаптеров, которыми мы не пользуемся, и это несовместимо с `langchain>=1.0`,
нужным агенту. Процессы общаются по HTTP, поэтому ни одному не нужно дерево
зависимостей другого. `shared/` подключён к обоим как path-зависимость — не
через uv workspace, потому что workspace сводится к одному lock-файлу и одному
окружению, то есть ровно к тому, чего эти два проекта разделить не могут.

## Что нужно

- Python 3.13 (ставится через `uv`; `unstructured` пока не поддерживает 3.14)
- Node.js 20.9+
- PostgreSQL 14+
- S3-совместимое хранилище (AWS, Cloudflare R2, Backblaze, MinIO)
- Ключ [OpenRouter](https://openrouter.ai) — и для модели агента, и для эмбеддингов

## Установка

```bash
# 1. база
createdb rag
psql rag -c "CREATE ROLE rag LOGIN PASSWORD 'rag'; GRANT ALL ON DATABASE rag TO rag;"

# 2. конфигурация — три файла, у каждого есть шаблон рядом
cp backend/.env.example         backend/.env
cp indexer/.env.example         indexer/.env
cp frontend/.env.example        frontend/.env.local

# секрет для подписи токенов
openssl rand -hex 32   # → JWT_SECRET в backend/.env
```

Заполните в обоих `.env` одинаковые доступы к S3 и `OPENROUTER_API_KEY`.

```bash
# 3. зависимости
cd backend         && uv sync && uv run alembic upgrade head
cd indexer         && uv sync
cd frontend        && npm install
```

## Запуск

Три процесса, каждый в своём терминале:

```bash
# индексер — поднимать первым, он проверяет ключ эмбеддингов на старте
cd indexer && uv run python -m rag_indexer.pipeline

# API
cd backend && uv run uvicorn app.main:app --reload --port 8000

# интерфейс
cd frontend && npm run dev
```

Выпустите себе токен и войдите с ним на <http://localhost:3000>:

```bash
cd backend && uv run rag-token --email вы@example.com
```

> При работе по обычному `http` на не-localhost хосте поставьте
> `COOKIE_SECURE=false` в `backend/.env`, иначе браузер отклонит cookie.

## Проверки

```bash
cd backend && uv run pytest && uv run ruff check .
cd frontend && npx tsc --noEmit && npx eslint . && npm run build
```

Тесты бьют по тому, что действительно может сломаться незаметно: реальное
JMESPath-выражение проверяется на метаданных, собранных реальным
post-processor'ом, — включая перекрёстные запросы, ключи не нашего формата и
имена файлов с кавычками и кириллицей.

## Как добавить

**Инструмент агенту** — функция с `@tool` в `_build_tools`
(`backend/app/agent.py`). Она автоматически замкнётся на текущего пользователя.

**Коннектор** — новый вход в `build_store` (`indexer/rag_indexer/pipeline.py`).
`DocumentStore` принимает список таблиц, так что Google Drive или SharePoint
добавляются рядом с S3, а не вместо него. Метаданные должны нести `user_id`,
иначе чанки будут недоступны никому — это и задумано.

**Поле в БД** — правьте `app/models.py`, затем
`uv run alembic revision --autogenerate -m "..."`.

## Ограничения

Осознанные упрощения помечены в коде комментарием `ponytail:` — там же назван
потолок и путь наверх.

- **Сканированные PDF не читаются.** Парсинг идёт через `strategy="fast"`
  (pdfminer), потому что `unstructured[pdf]` тянет torch и CUDA (~2 ГБ) ради
  layout-моделей. Для сканов нужен extra `pdf` и `strategy="hi_res"`.
- **Агент компилируется на каждый запрос**, чтобы инструменты замыкались на
  пользователе. Против одного вызова LLM это ничего не стоит.
- **История чата отдаётся модели целиком** (последние 20 сообщений), без
  суммаризации. При длинных диалогах стоит подключить `SummarizationMiddleware`.
- **Pathway распространяется под BSL 1.1.** Для личного использования это не
  ограничение, но перед коммерческим запуском проверьте Additional Use Grant.
