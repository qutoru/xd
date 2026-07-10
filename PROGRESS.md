# PROGRESS

MVP крипто-сигнального бота для Bybit (LightGBM → сигналы long/short/flat →
Telegram). **Без автоисполнения ордеров.**

## Общий план фаз
- [x] **Фаза 1** — Каркас проекта + сбор данных с Bybit
- [ ] Фаза 2 — Feature engineering + разметка
- [ ] Фаза 3 — Обучение модели + базовые метрики
- [ ] Фаза 4 — Бэктест без риск-менеджмента
- [ ] Фаза 5 — Риск-менеджмент (SL/TP, position sizing) + бэктест v2
- [ ] Фаза 6 — Telegram-уведомления
- [ ] Фаза 7 — Live-режим (планировщик, постоянная работа)
- [ ] Фаза 8 — Деплой (systemd/Docker)

---

## Фаза 1 — Каркас проекта + сбор данных ✅

### Что сделано
- Структура пакета `crypto_signal_bot/` с `__init__.py`.
- `requirements.txt`: pybit, pandas, numpy, python-dotenv, loguru (+ pyarrow
  для parquet).
- `.env.example` — `BYBIT_API_KEY`, `BYBIT_API_SECRET` (пустые).
- `.gitignore` — venv, `.env`, `__pycache__`, `*.csv`, `models/*.txt`.
- `crypto_signal_bot/config.py` — пути, ключи из `.env`, параметры
  `SYMBOL="BTCUSDT"`, `INTERVAL="15"`, `HISTORY_DAYS=180`,
  `CATEGORY="linear"`, `MAX_LIMIT=1000`.
- `crypto_signal_bot/data/fetcher.py` — `fetch_ohlcv()` с постраничной
  подгрузкой истории (backwards pagination) и retry/exponential-backoff.
- `crypto_signal_bot/data/storage.py` — `save_parquet()` / `load_parquet()`.
- `main.py` — CLI-команда `fetch`.

### Как запустить/проверить
```bash
python -m venv venv
venv\Scripts\activate        # Windows PowerShell: venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py fetch
```
Ожидается: лог с числом скачанных свечей и файл
`data/raw/BTCUSDT_15.parquet` (~17k свечей за 180 дней на 15m).

### Что осталось / заметки на будущее
- Ключи API для kline не нужны (публичный эндпоинт), но `.env` уже
  подхватывается — пригодится позже.
- Параметры символа/таймфрейма/горизонта предсказания — обсудить в Фазе 2.
