# frontend

Интерфейс базы знаний: Next.js 16 (App Router) + shadcn/ui.

Установка, переменные окружения и запуск всех трёх процессов описаны в
[README в корне репозитория](../README.md).

```bash
npm run dev     # разработка
npm run build   # продакшн-сборка
npx tsc --noEmit && npx eslint .
```

Требует запущенный API — адрес задаётся в `NEXT_PUBLIC_API_URL` (см. `.env.example`).
