# PROGRESS

MVP крипто-сигнального бота для Bybit (LightGBM → сигналы long/short/flat →
Telegram). **Без автоисполнения ордеров.**

## Общий план фаз
- [x] **Фаза 1** — Каркас проекта + сбор данных с Bybit
- [x] **Фаза 2** — Feature engineering + разметка
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
- Параметры горизонта предсказания зафиксированы в Фазе 2 (см. ниже).
- Запускать интерпретатором `py`, не `python` (в PATH — заглушка MS Store).

---

## Фаза 2 — Feature engineering + разметка ✅

### Решения по параметрам
- **Разметка:** triple-barrier (Lopez de Prado). Барьеры: `close ± ATR_MULT*ATR`
  (горизонтальные) + временной барьер = горизонт. Класс = какой барьер задет
  первым: вверх → long (1), вниз → short (-1), время/ничего → flat (0).
  Одновременное касание обоих барьеров в одном баре → flat (внутрибарный
  порядок по OHLC неизвестен, воздерживаемся).
- **Горизонт:** `HORIZON = 8` баров (2 часа на 15m).
- **Барьеры:** `ATR_PERIOD = 14`, `ATR_MULT = 1.5` (ATR берётся на баре-якоре).

### Что сделано
- `crypto_signal_bot/features/indicators.py` — `build_features()`: лог-доходности
  (1/4/8/16/24), реализованная волатильность, ATR (абс. и % от цены),
  price/EMA (9/21/50/200), MACD, RSI, позиция в Bollinger, форма свечи
  (range/body/wicks), объём (z-score, изменение), циклическое время суток
  (sin/cos). Всё — только трейлинг-окна, без заглядывания вперёд.
- `crypto_signal_bot/features/labeling.py` — `triple_barrier_labels()`,
  векторизованная разметка; ATR переиспользуется из фич.
- `crypto_signal_bot/features/dataset.py` — `build_and_save()`: load raw →
  features → labels → dropna (warm-up/неразрешённые) → save в
  `data/processed/`. Логирует баланс классов.
- `crypto_signal_bot/data/storage.py` — `save_processed()` / `load_processed()`.
- `config.py` — `PROCESSED_DIR`, `HORIZON`, `ATR_PERIOD`, `ATR_MULT`, метки.
- `main.py` — команда `build`.

### Как запустить/проверить
```bash
py main.py build
```
Результат (на данных Фазы 1): 17 280 сырых свечей → отброшено 207
(warm-up EMA200 + горизонт) → **17 073** строки в
`data/processed/BTCUSDT_15.parquet`. Баланс классов: short 35.4% /
flat 32.0% / long 32.5%.

### Что осталось / заметки на будущее
- Фичи и `ATR_MULT`/горизонт можно тюнить в Фазе 3 по метрикам.
- При обучении важен временной сплит (no shuffle) — данные автокоррелированы,
  плюс метки перекрываются на горизонте (учесть при валидации/purging).
