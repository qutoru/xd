"""Localization catalog for the Telegram bot (RU + EN).

A single, dependency-free string table. Everything the user sees — the trade
message labels, the ``/start`` greeting, the language menu — is keyed here by
language code so the layout stays in one place and adding a language later means
adding one dict, not touching the formatter or the listener.

English is the default: an unknown or missing language code falls back to it, so
the existing English rendering (and its tests) is preserved byte-for-byte.
"""

from __future__ import annotations

# Supported language codes, default first.
LANGUAGES: tuple[str, ...] = ("en", "ru")
DEFAULT_LANGUAGE = "en"

# Flag + display name shown on the language-selection buttons.
FLAG = {"en": "🇺🇸", "ru": "🇷🇺"}
LANG_NAME = {"en": "English", "ru": "Русский"}

# Bilingual prompt for the /language menu (shown regardless of current language).
MENU_PROMPT = "🌐 Select language / Выберите язык:"

# Subscription-plan text for /subscribe. Kept as module constants (built from a
# line list) because it is long and multi-section; referenced from the catalog.
_SEP = "━━━━━━━━━━━━━━"

_SUBSCRIBE_RU = "\n".join([
    "💎 Тарифы Bybit Smart Signals",
    "",
    "Выберите подходящий план 👇",
    "",
    _SEP,
    "",
    "🟢 START — $20 / месяц",
    "",
    "- Сигналы Long / Short",
    "",
    "- Уровни TP / SL",
    "",
    "- До 3 торговых пар",
    "",
    "- Уведомления в реальном времени",
    "",
    _SEP,
    "",
    "🔵 PRO — $50 / 3 месяца",
    "",
    "- Всё из START",
    "",
    "- Все торговые пары",
    "",
    "- Приоритетная отправка сигналов",
    "",
    "- Статистика по сделкам",
    "",
    "💰 Экономия ~17%",
    "",
    _SEP,
    "",
    "🟣 VIP — $120 / год",
    "",
    "- Всё из PRO",
    "",
    "- Ранний доступ к новым стратегиям",
    "",
    "- Персональные настройки риска",
    "",
    "- Поддержка в приоритете",
    "",
    "💰 Экономия ~50%",
    "",
    _SEP,
    "",
    "📩 По вопросам и оплате: @Daxakson",
])

_SUBSCRIBE_EN = "\n".join([
    "💎 Bybit Smart Signals Plans",
    "",
    "Choose the plan that suits you 👇",
    "",
    _SEP,
    "",
    "🟢 START — $20 / month",
    "",
    "- Long / Short signals",
    "",
    "- TP / SL levels",
    "",
    "- Up to 3 trading pairs",
    "",
    "- Real-time notifications",
    "",
    _SEP,
    "",
    "🔵 PRO — $50 / 3 months",
    "",
    "- Everything in START",
    "",
    "- All trading pairs",
    "",
    "- Priority signal delivery",
    "",
    "- Trade statistics",
    "",
    "💰 Save ~17%",
    "",
    _SEP,
    "",
    "🟣 VIP — $120 / year",
    "",
    "- Everything in PRO",
    "",
    "- Early access to new strategies",
    "",
    "- Personal risk settings",
    "",
    "- Priority support",
    "",
    "💰 Save ~50%",
    "",
    _SEP,
    "",
    "📩 Questions & payment: @Daxakson",
])

_CATALOG: dict[str, dict[str, str]] = {
    "en": {
        "side_long": "🟢 LONG",
        "side_short": "🔴 SHORT",
        "entry": "Entry",
        "take_profit": "Take Profit",
        "stop_loss": "Stop Loss",
        "confidence": "Confidence",
        "welcome": (
            "Hi, this is Bybit Smart Signals!\n"
            "\n"
            "What I can do:\n"
            "\n"
            "📊 Analyze the market and generate signals (Long / Short / Flat)\n"
            "\n"
            "🎯 Calculate TP/SL levels based on ATR\n"
            "\n"
            "🔔 Send alerts as soon as an entry point appears\n"
            "\n"
            "Commands:\n"
            "\n"
            "/subscribe — get a subscription\n"
            "\n"
            "/signals — latest signals\n"
            "\n"
            "/help — help"
        ),
        "disclaimer": (
            "⚠️ Signals are for informational purposes only. Trading "
            "cryptocurrency carries a risk of losing funds."
        ),
        "subscribe": _SUBSCRIBE_EN,
        "back_button": "⬅️ Back",
        "admin_grant_usage": "Usage: /grant <user_id|@username> <days> [plan]",
        "admin_grant_ok": "✅ Granted to {id} — active until {until} (plan: {plan}).",
        "admin_revoke_usage": "Usage: /revoke <user_id|@username>",
        "admin_revoke_ok": "✅ Subscription revoked for {id}.",
        "admin_revoke_none": "ℹ️ {id} had no subscription.",
        "admin_user_not_found": "⚠️ {name} not found — they must message the bot first.",
        "admin_subs_empty": "No active subscriptions.",
        "admin_subs_header": "Active subscriptions:",
        "admin_users_empty": "No known users yet.",
        "admin_users_header": "Known users:",
        "stats_locked": "📊 Trade statistics are a PRO/VIP feature. Use /subscribe to upgrade.",
        "stats_empty": "📊 No closed trades yet — statistics appear here once trades close.",
        "stats_header": "📊 Trade statistics",
        "stats_trades": "Closed trades",
        "stats_winrate": "Win rate",
        "stats_net": "Net PnL",
        "stats_fees": "fees",
        "stats_best": "Best",
        "stats_worst": "Worst",
        "risk_locked": "⚙️ Personal risk settings are a VIP feature. Use /subscribe to upgrade.",
        "risk_current": "⚙️ Your risk level: {level}.\nChange it with /risk <low|medium|high>.",
        "risk_set": "✅ Risk level set to {level}.",
        "risk_usage": "Usage: /risk <low|medium|high>",
        "risk_low": "Low",
        "risk_medium": "Medium",
        "risk_high": "High",
        "cmd_start": "Restart / choose language",
        "cmd_language": "Change language",
        "cmd_subscribe": "View subscription plans",
        "cmd_stats": "Trade statistics (PRO/VIP)",
        "cmd_risk": "Personal risk settings (VIP)",
        "switched": (
            "✅ Language switched to English\n"
            "\n"
            "👋 Welcome to the Crypto Signal Bot\n"
            "\n"
            "I send daily market-neutral trade signals\n"
            "(entry, take-profit, stop-loss, confidence)\n"
            "from the ALX strategy.\n"
            "\n"
            "Done — all messages will now be in English."
        ),
        "toast": "Language: English",
    },
    "ru": {
        "side_long": "🟢 ЛОНГ",
        "side_short": "🔴 ШОРТ",
        "entry": "Вход",
        "take_profit": "Тейк-профит",
        "stop_loss": "Стоп-лосс",
        "confidence": "Уверенность",
        "welcome": (
            "Привет, это Bybit Smart Signals!\n"
            "\n"
            "Что я умею:\n"
            "\n"
            "📊 Анализирую рынок и формирую сигналы (Long / Short / Flat)\n"
            "\n"
            "🎯 Рассчитываю уровни TP/SL на основе ATR\n"
            "\n"
            "🔔 Присылаю уведомления, как только появляется точка входа\n"
            "\n"
            "Команды:\n"
            "\n"
            "/subscribe — оформить подписку\n"
            "\n"
            "/signals — последние сигналы\n"
            "\n"
            "/help — помощь"
        ),
        "disclaimer": (
            "⚠️ Сигналы носят информационный характер. Торговля "
            "криптовалютой сопряжена с риском потери средств."
        ),
        "subscribe": _SUBSCRIBE_RU,
        "back_button": "⬅️ Назад",
        "admin_grant_usage": "Использование: /grant <user_id|@username> <дней> [план]",
        "admin_grant_ok": "✅ Выдано {id} — активно до {until} (план: {plan}).",
        "admin_revoke_usage": "Использование: /revoke <user_id|@username>",
        "admin_revoke_ok": "✅ Подписка отозвана у {id}.",
        "admin_revoke_none": "ℹ️ У {id} не было подписки.",
        "admin_user_not_found": "⚠️ {name} не найден — сначала он должен написать боту.",
        "admin_subs_empty": "Активных подписок нет.",
        "admin_subs_header": "Активные подписки:",
        "admin_users_empty": "Пока нет известных пользователей.",
        "admin_users_header": "Известные пользователи:",
        "stats_locked": "📊 Статистика по сделкам — функция PRO/VIP. Оформить: /subscribe.",
        "stats_empty": "📊 Пока нет закрытых сделок — статистика появится после первых закрытий.",
        "stats_header": "📊 Статистика по сделкам",
        "stats_trades": "Закрытых сделок",
        "stats_winrate": "Винрейт",
        "stats_net": "Чистый PnL",
        "stats_fees": "комиссии",
        "stats_best": "Лучшая",
        "stats_worst": "Худшая",
        "risk_locked": "⚙️ Персональные настройки риска — функция VIP. Оформить: /subscribe.",
        "risk_current": "⚙️ Ваш уровень риска: {level}.\nИзменить: /risk <low|medium|high>.",
        "risk_set": "✅ Уровень риска установлен: {level}.",
        "risk_usage": "Использование: /risk <low|medium|high>",
        "risk_low": "Низкий",
        "risk_medium": "Средний",
        "risk_high": "Высокий",
        "cmd_start": "Перезапуск / выбор языка",
        "cmd_language": "Сменить язык",
        "cmd_subscribe": "Тарифы подписки",
        "cmd_stats": "Статистика по сделкам (PRO/VIP)",
        "cmd_risk": "Настройки риска (VIP)",
        "switched": (
            "✅ Язык переключён на русский\n"
            "\n"
            "👋 Добро пожаловать в Crypto Signal Bot\n"
            "\n"
            "Я присылаю ежедневные рыночно-нейтральные\n"
            "торговые сигналы (вход, тейк-профит, стоп-лосс,\n"
            "уверенность) по стратегии ALX.\n"
            "\n"
            "Готово — теперь все сообщения будут на русском."
        ),
        "toast": "Язык: Русский",
    },
}


def normalize(lang: str | None) -> str:
    """Return a supported language code, falling back to the default."""
    return lang if lang in _CATALOG else DEFAULT_LANGUAGE


def t(lang: str | None, key: str) -> str:
    """Translate ``key`` into ``lang`` (default-language fallback)."""
    return _CATALOG[normalize(lang)][key]
