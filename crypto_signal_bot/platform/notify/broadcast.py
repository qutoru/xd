"""SubscriberBroadcaster — fan-out delivery of signals to paying subscribers.

The :class:`~crypto_signal_bot.platform.notify.notifier.TelegramNotifier` sends
to a single configured chat; this is its subscription-service counterpart. It
renders each :class:`TradeIntent` and delivers it to *every active subscriber*,
shaped by that subscriber's plan (see
:mod:`crypto_signal_bot.platform.notify.plans`):

    * START sees only the top ``max_pairs`` signals (by confidence); PRO/VIP all.
    * early-access ("beta") strategies are delivered to VIP only.
    * richer tiers are served first ("priority delivery").

Transport- and store-injected, so the whole fan-out is unit-tested offline with a
fake client — no network, no token. Delivery is best-effort: a failed send to one
subscriber is logged and recorded, never raised, so it cannot abort the fan-out.

NOTE: this is intentionally **not** wired into the live pipeline yet — it ships as
a standalone, tested unit pending platform Stage 18. A future Stage 21 will add a
semi-automatic, admin-only accept/decline gate in front of this broadcast.
"""

from __future__ import annotations

from dataclasses import dataclass

from loguru import logger

from crypto_signal_bot.platform.execution.intent import TradeIntent
from crypto_signal_bot.platform.notify.i18n import DEFAULT_LANGUAGE
from crypto_signal_bot.platform.notify.notifier import TelegramClient
from crypto_signal_bot.platform.notify.plans import (
    ENTITLEMENTS,
    Entitlements,
    Tier,
    tier_of,
)
from crypto_signal_bot.platform.notify.prefs import LanguagePrefsStore
from crypto_signal_bot.platform.notify.subscriptions import (
    Subscription,
    SubscriptionStore,
)
from crypto_signal_bot.platform.notify.telegram import TelegramFormatter


@dataclass(frozen=True)
class BroadcastResult:
    """Outcome of one :meth:`SubscriberBroadcaster.broadcast` call."""

    recipients: int = 0  # active subscribers considered
    sent: int = 0        # messages delivered
    failed: int = 0
    errors: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return self.failed == 0


class SubscriberBroadcaster:
    """Fans a batch of TradeIntents out to active subscribers, gated by tier."""

    def __init__(
        self,
        client: TelegramClient,
        subscriptions: SubscriptionStore,
        *,
        formatter: TelegramFormatter | None = None,
        prefs: LanguagePrefsStore | None = None,
        risk_prefs=None,
        beta_strategies: frozenset[str] = frozenset(),
        default_tier: Tier = Tier.START,
    ) -> None:
        self._client = client
        self._subs = subscriptions
        self._formatter = formatter or TelegramFormatter()
        self._prefs = prefs
        # Personal risk level store (VIP): drives the recommended risk-per-trade
        # line, shown only to subscribers whose tier has custom_risk (VIP).
        self._risk_prefs = risk_prefs
        # Strategy names treated as early-access; withheld from non-VIP tiers.
        self._beta = frozenset(beta_strategies)
        # Tier applied to an active subscriber whose plan label is unrecognised
        # (e.g. granted without a plan): they paid, so give base access.
        self._default_tier = default_tier

    def _entitlements(self, sub: Subscription) -> Entitlements:
        return ENTITLEMENTS[tier_of(sub.plan) or self._default_tier]

    def _language(self, chat_id: str) -> str:
        return self._prefs.get(chat_id) if self._prefs is not None else DEFAULT_LANGUAGE

    def _visible(
        self, intents: list[TradeIntent], ent: Entitlements
    ) -> list[TradeIntent]:
        """The subset of ``intents`` this tier may receive, in delivery order."""
        items = [
            i for i in intents if ent.beta_strategies or i.strategy not in self._beta
        ]
        if ent.max_pairs is not None:
            # Give the capped tier its best signals: top-N by confidence, with a
            # symbol tie-break so every capped subscriber sees the same, stable set.
            items = sorted(
                items, key=lambda i: (-(i.confidence or 0.0), i.symbol)
            )[: ent.max_pairs]
        return items

    def broadcast(
        self, intents: list[TradeIntent], *, exclude: set[str] | None = None
    ) -> BroadcastResult:
        """Deliver ``intents`` to every active subscriber; never raises on send.

        ``exclude`` drops chat_ids that must not receive the plain broadcast — used
        to keep the owner out of the subscriber fan-out when they already got the
        owner-only buttoned copy, so no one gets a duplicate.
        """
        active = self._subs.active()
        if exclude:
            active = [s for s in active if s.chat_id not in exclude]
        if not active or not intents:
            return BroadcastResult(recipients=len(active))

        # Richer tiers first ("priority delivery"); chat_id keeps it deterministic.
        active.sort(key=lambda s: (self._entitlements(s).priority, s.chat_id))

        sent = failed = 0
        errors: list[str] = []
        for sub in active:
            ent = self._entitlements(sub)
            lang = self._language(sub.chat_id)
            # VIP-only recommended risk-per-trade, from the subscriber's /risk level.
            risk_level = (
                self._risk_prefs.get(sub.chat_id)
                if ent.custom_risk and self._risk_prefs is not None else None
            )
            for intent in self._visible(intents, ent):
                text = self._formatter.format(intent, lang, risk_level=risk_level)
                try:
                    self._client.send_message(chat_id=sub.chat_id, text=text)
                    sent += 1
                except Exception as exc:  # best-effort: one failure can't stop the fan-out
                    failed += 1
                    errors.append(f"{sub.chat_id}/{intent.symbol}: {exc}")
                    logger.error(
                        "Broadcast to {} failed for {}: {}", sub.chat_id, intent.symbol, exc
                    )
        if sent:
            logger.success(
                "Broadcast {} message(s) to {} active subscriber(s)", sent, len(active)
            )
        return BroadcastResult(
            recipients=len(active), sent=sent, failed=failed, errors=tuple(errors)
        )
