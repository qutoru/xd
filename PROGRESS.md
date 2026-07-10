# PROGRESS

MVP крипто-сигнального бота для Bybit (LightGBM → сигналы long/short/flat →
Telegram). **Без автоисполнения ордеров.**

## Общий план фаз
- [x] **Фаза 1** — Каркас проекта + сбор данных с Bybit
- [x] **Фаза 2** — Feature engineering + разметка
- [x] **Фаза 3** — Обучение модели + базовые метрики
- [x] **Фаза 4** — Бэктест без риск-менеджмента
- [x] **Фаза 5** — Риск-менеджмент (SL/TP, position sizing) + бэктест v2
- [x] **Фаза 6** — Telegram-уведомления
- [x] **Фаза 7** — Live-режим (планировщик, постоянная работа)
- [x] **Фаза 8** — Запуск/деплой (лёгкий: batch-launcher под Windows)
- [x] **Мультивалютность** — топ-50 ликвидных пар Bybit (REST-поллинг)

Идеи по прибыльности вынесены в отдельный файл — см. `RESEARCH.md`.

---

## Мультивалютность (топ-50 пар) ✅

### Решения
- Торгуемая «вселенная» = **топ-50 linear USDT-перпетуалов по 24ч обороту**,
  автоподхват с Bybit (`get_tickers`) и кэш в `data/universe.json`.
- Все стадии (`fetch/build/train/signal/live`) работают по всей вселенной;
  `--symbol S` — принудительно одна пара. Диагностика (`backtest*`) — одна пара.
- **Отдельная модель на каждую пару** (одна на всех — плохо). Пары с малой
  историей (< `MIN_ROWS_FOR_TRAINING=3000`) пропускаются.
- Данные — REST-поллинг (WebSocket рассмотрен и отложен: на 15m выигрыш мал,
  сложность высокая; см. заметку ниже).
- Антиспам в live: no-trade по-прежнему не шлём; на каждом баре — sweep по 50
  парам, изоляция ошибок по символу (одна плохая пара не рушит проход).

### Что сделано
- `crypto_signal_bot/data/universe.py` — `discover_top_symbols()`,
  `save/load/refresh/get_universe()`.
- `crypto_signal_bot/pipeline.py` — `fetch_all/build_all/train_all/bootstrap_all`
  с изоляцией ошибок и отчётом (ok/skipped/failed).
- `live/scheduler.py` — мультисимвольный sweep, пер-символьный дедуп, кэш моделей.
- `live/signal.py` — `generate_signal(..., model=...)` (кэш модели).
- `config.py` — `TOP_N_SYMBOLS=50`, `UNIVERSE_PATH`, `MIN_ROWS_FOR_TRAINING`.
- `main.py` — команда `symbols`; `--symbol` у fetch/build/train/signal/live.

### Как запустить/проверить
```bash
py main.py symbols     # обновить список топ-50 -> data/universe.json
py main.py fetch       # скачать историю по всем 50
py main.py build       # фичи+метки по всем
py main.py train       # обучить модель на каждую пару
py main.py live        # live-цикл по всем 50, сигналы в Telegram
```
Проверено: обнаружение вселенной (50 из 618 USDT-перпов), пайплайн на одной
паре (ETHUSDT: 17073 строки, обучение ok), полный прогон по 50 парам.

### Заметка про WebSocket (на будущее)
Можно перейти на push вместо REST-поллинга: подписка `kline.15.<symbol>` по
всем парам, обработка `confirm=true`. Нужен единоразовый REST-seed для разогрева
фич (EMA200 ⇒ ~200 баров), затем скользящий буфер и обработка пушей без запросов.
На 15m таймфрейме выигрыш невелик — отложено.

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
- Фичи и `ATR_MULT`/горизонт можно тюнить по метрикам.
- При обучении важен временной сплит (no shuffle) — данные автокоррелированы,
  плюс метки перекрываются на горизонте (реализовано в Фазе 3: purging).

---

## Фаза 3 — Обучение модели + базовые метрики ✅

### Решения
- Модель: **LightGBM multiclass** (3 класса short/flat/long), `class_weight=balanced`.
- Сплит: **хронологический** 70/15/15 без shuffle. Между сегментами — **purging**:
  из хвоста train и valid убирается `EMBARGO=HORIZON` баров, чтобы forward-метка
  не заглядывала в следующий сегмент.
- Early stopping по valid `multi_logloss` (`EARLY_STOPPING_ROUNDS=100`).
- Метрики: accuracy, macro-F1, per-class P/R/F1, confusion matrix + majority-baseline.

### Что сделано
- `crypto_signal_bot/model/splits.py` — `time_split()` (purged chronological).
- `crypto_signal_bot/model/metrics.py` — `evaluate()` / `log_report()`.
- `crypto_signal_bot/model/train.py` — `train()`: load processed → split → fit
  (early stopping) → оценка valid/test → сохранение booster (`models/SYMBOL_INT.txt`)
  + метаданные (`*_meta.json`: фичи, params, метрики, best_iteration).
- `config.py` — `RANDOM_SEED`, `TRAIN_FRAC`/`VALID_FRAC`, `EMBARGO`, `LGBM_PARAMS`,
  `CLASS_ORDER`/`CLASS_NAMES`.
- `main.py` — команда `train`. `requirements.txt` — lightgbm, scikit-learn.
- `.gitignore` — `models/*.txt`, `models/*_meta.json` (генерируемые артефакты).

### Как запустить/проверить
```bash
py main.py train
```
Результат (26 фич, split train=11943 / valid=2553 / test=2561, best_iter=68):
- **test:** accuracy **0.401** (baseline 0.357 = always short), macro-F1 **0.402**.
- Модель предсказывает все 3 класса (не вырождается). Преимущество над baseline
  скромное (~+4 п.п.) — ожидаемо для сырого direction-прогноза; это базовая точка.

### Что осталось / заметки на будущее
- Улучшения: доп. фичи, тюнинг гиперпараметров, порог по вероятности (торговать
  только уверенные сигналы), walk-forward валидация вместо одного сплита.
- Ключевой вопрос — не accuracy, а прибыльность после комиссий: проверено в
  Фазе 4 (бэктест).

---

## Фаза 4 — Бэктест без риск-менеджмента ✅

### Решения
- Модель: `position[t] = signal[t] ∈ {-1,0,1}`; доход реализуется на баре
  `t → t+1` (без look-ahead). Всегда ±1 unit нотионала — **без SL/TP и без
  position sizing** (это Фаза 5).
- Комиссии: `FEE_RATE = 0.00055` (Bybit linear taker) на каждую единицу
  turnover (|Δposition|). Оценка на **test-сегменте** (честный out-of-sample).
- Метрики: total/annual return, Sharpe, max drawdown, exposure, trades,
  win-rate, суммарные комиссии + benchmark buy&hold.

### Что сделано
- `crypto_signal_bot/model/predict.py` — `SignalModel.load()` /
  `predict_signals()` (загрузка booster + meta, argmax → сигнал).
- `crypto_signal_bot/backtest/engine.py` — `run_backtest()` (векторный PnL).
- `crypto_signal_bot/backtest/metrics.py` — `performance()` / `log_performance()`.
- `crypto_signal_bot/backtest/runner.py` — `run()`: load processed → split →
  предсказание на сегменте → бэктест → сохранение кривой в `data/backtests/`.
- `config.py` — `FEE_RATE`, `BARS_PER_YEAR`, `BACKTEST_DIR`.
- `main.py` — команда `backtest [--segment {train,valid,test}]`.

### Как запустить/проверить
```bash
py main.py backtest            # test-сегмент по умолчанию
```
Результат (test, 2560 баров ≈ 27 дней):
- total_return **−29.2%**, Sharpe −12.4, max_dd −29.8%, exposure 66.8%,
  **524 сделки**, win-rate 46.9%, **суммарные комиссии 40.4%**; buy&hold +0.47%.

### Ключевой вывод
**Комиссии убивают сырой per-bar сигнал.** Модель переворачивает позицию почти
каждый бар → овертрейдинг, накопленная комиссия ~40% за месяц. До комиссий
стратегия около нуля. Движок корректен — он честно показывает стоимость
частой торговли. Это и есть мотивация Фазы 5.

### Что осталось / заметки на будущее (→ Фаза 5)
- Снизить оборот: удержание позиции на горизонт (не переворачивать каждый бар),
  порог по вероятности (сигнал только при уверенности), гистерезис.
- SL/TP, position sizing по волатильности; повторный бэктест v2 после этого.

---

## Фаза 5 — Риск-менеджмент + бэктест v2 ✅

### Решения
- **Event-driven** бэктест, одна позиция за раз. Вход по сигналу на close бара;
  выход по тому, что раньше: **SL / TP / тайм-барьер** (горизонт 8 баров).
- SL/TP = `±ATR_MULT·ATR` (те же барьеры, что в разметке; `SL_ATR_MULT=TP_ATR_MULT=1.5`).
- **Sizing — risk-based:** нотионал так, что удар по стопу теряет `RISK_PER_TRADE=1%`
  капитала; кап `MAX_LEVERAGE=5x`.
- **Порог уверенности** входа подбирается на **valid** (grid `0.34..0.60`, по
  Sharpe, ≥5 сделок), затем применяется на test (без подгонки под test).
- Комиссии Bybit taker на вход и выход; equity mark-to-market по барам.

### Что сделано
- `crypto_signal_bot/backtest/event_engine.py` — `run_event_backtest()`:
  state-машина позиции, intrabar-проверка SL/TP, тайм-выход, risk-based sizing,
  лог сделок. Без look-ahead (позиция под риском с t+1).
- `crypto_signal_bot/backtest/metrics.py` — `event_performance()` /
  `log_event_performance()` (+ per-trade статистика: win-rate, avg trade, exits).
- `crypto_signal_bot/backtest/runner.py` — `run_risk_managed()`: подбор порога
  на valid → бэктест на test → сохранение кривой и сделок.
- `crypto_signal_bot/model/predict.py` — `predict_with_conf()` (сигнал + top-proba).
- `config.py` — `SL/TP_ATR_MULT`, `RISK_PER_TRADE`, `MAX_LEVERAGE`,
  `PROB_THRESHOLD_GRID`. `main.py` — команда `backtest-rm`.

### Как запустить/проверить
```bash
py main.py backtest-rm
```
Результат (порог 0.55, test 2561 баров ≈ 27 дней, 11 сделок):
| Метрика | Phase 4 (без RM) | **Phase 5 (RM)** |
|---|---|---|
| total_return | −29.2% | **−2.56%** |
| max_drawdown | −29.8% | **−4.17%** |
| Sharpe | −12.4 | **−3.13** |
| сделок | 524 | **11** |
| комиссии | 40.4% | **3.54%** |
| win-rate | 46.9% | 54.5% |

### Ключевой вывод
Риск-менеджмент **сработал как задумано**: овертрейдинг устранён (524→11),
комиссии 40%→3.5%, просадка 30%→4%, убыток ↓ в ~11 раз. **Но стратегия ещё не
прибыльна** — edge модели слишком слаб, чтобы покрыть издержки (avg-сделка
−0.23%). Оговорка: test ≈ 27 дней / 11 сделок — статистически это шум.

### Что осталось / заметки на будущее
- Усилить edge (это важнее любой доработки RM): больше истории, доп. фичи,
  тюнинг модели, асимметричные TP/TP, meta-labeling, walk-forward.
- Прежде чем идти в live (Фазы 6–8), стоит получить **устойчиво положительный
  Sharpe на test** — сейчас его нет. Telegram/live можно строить параллельно,
  но торговать вживую по текущей модели нельзя.

---

## Фаза 6 — Telegram-уведомления ✅

### Решения
- Сигнал считается по **последней закрытой** свече (незакрытый форминг-бар
  отбрасывается, чтобы не реагировать на неполные данные).
- Направление гейтится по уверенности (`LIVE_PROB_THRESHOLD=0.55`, из тюнинга
  Фазы 5): ниже порога → «no trade», иначе long/short + ATR-based SL/TP.
- Telegram Bot API через `requests`; при отсутствии токена или `--dry-run`
  сообщение логируется, а не отправляется (можно проверять без бота).
- **Без автоисполнения** — только уведомление; дисклеймер в тексте.

### Что сделано
- `crypto_signal_bot/live/signal.py` — `generate_signal()` → `Signal` (dataclass
  с `.format()`); fetch → drop forming bar → features → predict_with_conf →
  gate → SL/TP. Обе ветки (trade / no-trade) форматируются.
- `crypto_signal_bot/live/telegram.py` — `send_telegram(text, dry_run)`.
- `config.py` — `TELEGRAM_BOT_TOKEN/CHAT_ID`, `LIVE_PROB_THRESHOLD`,
  `LIVE_LOOKBACK_DAYS`. `.env.example` — Telegram-переменные.
- `main.py` — команда `signal [--dry-run]`. `requirements.txt` — requests.

### Как запустить/проверить
```bash
py main.py signal --dry-run     # без токена: печатает сообщение
py main.py signal               # шлёт в Telegram, если .env заполнен
```
Проверено: dry-run отрабатывает; с реальными `.env`-креденшелами сообщение
доставлено в чат (`SUCCESS Sent Telegram message`). Свежий бар часто даёт
NO-TRADE (модель ниже порога) — уведомление всё равно приходит.

### Что осталось / заметки на будущее
- Порог `LIVE_PROB_THRESHOLD` стоит переподбирать после каждого переобучения
  модели (сейчас захардкожен из Фазы 5).
- Фаза 7 сделает вызов `signal` по расписанию (планировщик, постоянная работа).

---

## Фаза 7 — Live-режим (планировщик) ✅

### Решения
- Цикл просыпается через `LIVE_BAR_BUFFER_SEC=20`с после каждой границы 15m
  (чтобы биржа успела опубликовать закрытую свечу), затем считает сигнал.
- **Дедупликация** по времени бара: если данные ещё не обновились и бар тот же —
  итерация пропускается (один бар не уходит дважды).
- Антиспам: `no-trade` бары по умолчанию только логируются
  (`LIVE_NOTIFY_NO_TRADE=False`); флаг `--notify-no-trade` включает их отправку.
- Устойчивость: ошибка внутри итерации логируется, но цикл не падает; Ctrl+C —
  чистая остановка. `--once` — одна прогонка (для крона / Фазы 8).
- Позиций не держит, ордера не шлёт — только рассылка сигналов.

### Что сделано
- `crypto_signal_bot/live/scheduler.py` — `run_live()` (цикл + сон до бара),
  `run_once()` (одна итерация с дедупом/отправкой), `_seconds_to_next_bar()`.
- `config.py` — `LIVE_BAR_BUFFER_SEC`, `LIVE_NOTIFY_NO_TRADE`.
- `main.py` — команда `live [--once] [--notify-no-trade] [--dry-run]`.

### Как запустить/проверить
```bash
py main.py live --once --dry-run   # одна прогонка, без отправки
py main.py live                    # бесконечный цикл, шлёт в Telegram
```
Проверено: `--once --dry-run` считает сигнал и по умолчанию логирует no-trade
без отправки; расчёт сна до следующей свечи корректен (напр. 681с до 19:00+20с).

### Что осталось / заметки на будущее
- Постоянная работа/автозапуск и переживание перезагрузок — Фаза 8.
- Опционально: персист `last_bar_time` между рестартами (сейчас in-memory).

---

## Фаза 8 — Запуск/деплой ✅

### Решение
- Проект крутится локально на Windows, Linux-VPS пока не нужен → вместо
  Docker/systemd выбран **лёгкий launcher**: `start_bot.bat` в корне.
- Двойной клик → `cd` в папку проекта → `py main.py live` → окно остаётся
  открытым с логами; остановка Ctrl+C или закрытием окна; на выходе `pause`
  показывает код выхода (видно причину падения).

### Как запустить
- Двойной клик по `start_bot.bat` (или из терминала `py main.py live`).
- Перед первым запуском: заполнить `.env` (Telegram-токен/chat_id) и один раз
  собрать модель: `py main.py fetch && py main.py build && py main.py train`.

### Проверено
- Launcher стартует live-цикл: «Live loop started … Sleeping ~168s until next
  bar close» (сон до границы 15m + буфер — корректно).

### Апгрейд-пути (если понадобится «настоящий» деплой)
- **Windows Task Scheduler** + `py main.py live --once` каждые 15 мин —
  автозапуск после ребута, самовосстановление (дедуп по бару уже есть).
- **Linux-VPS**: Dockerfile + docker-compose (`restart: unless-stopped`,
  volume на `models/`+`data/`, `.env` через env_file) и/или systemd-unit —
  entrypoint бутстрапит модель (fetch→build→train), затем `live`.
