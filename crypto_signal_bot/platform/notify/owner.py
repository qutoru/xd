"""Owner-only semi-automatic approval: Accept / Ignore buttons on a signal.

This is the interactive half of semi-auto mode. The pipeline pushes a proposed
trade to the owner as a Telegram message carrying two inline buttons; this module
builds that keyboard and handles the tap:

    own:acc:<token>  -> risk-gate the pending signal, then place the order
    own:ign:<token>  -> discard it

Everything here is owner-gated on the sender's id, so the surface is invisible and
inert for every other user: the buttons are only ever attached to messages sent to
the owner, and a tap from anyone else is refused before any order is considered.

The broker, risk control and pending store are injected, so the whole flow is
unit-tested offline with a fake broker — no network, no real token, no Bybit.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from loguru import logger

from crypto_signal_bot.platform.execution.domain import Side
from crypto_signal_bot.platform.execution.engine import ExecutionEngine
from crypto_signal_bot.platform.execution.intent import TradeIntent
from crypto_signal_bot.platform.notify.i18n import DEFAULT_LANGUAGE, t
from crypto_signal_bot.platform.notify.pending import (
    STATUS_ACCEPTED,
    STATUS_BLOCKED,
    STATUS_EXPIRED,
    STATUS_IGNORED,
    PendingSignalStore,
)
from crypto_signal_bot.platform.notify.prefs import LanguagePrefsStore
from crypto_signal_bot.platform.risk_control.control import ProductionRiskControl
from crypto_signal_bot.platform.risk_control.state import RiskStateStore

OWNER_PREFIX = "own:"  # callback_data namespace for the approval buttons
_ACCEPT = "acc"
_IGNORE = "ign"
_DEFAULT_TTL_S = 24 * 3600  # a pending signal older than this is refused as stale


def approval_keyboard(token: str, lang: str) -> dict[str, Any]:
    """Inline keyboard with Accept / Ignore for one pending signal ``token``."""
    return {
        "inline_keyboard": [[
            {"text": t(lang, "owner_accept"), "callback_data": f"{OWNER_PREFIX}{_ACCEPT}:{token}"},
            {"text": t(lang, "owner_ignore"), "callback_data": f"{OWNER_PREFIX}{_IGNORE}:{token}"},
        ]]
    }


def owner_only_keyboard(
    token: str, lang: str, *, recipient: str | None, owner_id: str | None
) -> dict[str, Any] | None:
    """Approval keyboard for the owner only; ``None`` for any other recipient.

    The single choke point that enforces the invariant "Accept/Ignore buttons are
    never attached to a message shown to a non-owner user". Callers build every
    approval message through this, so the buttons cannot leak to subscribers or any
    other chat even if the push is later pointed at more recipients.
    """
    if not owner_id or recipient is None or str(recipient) != str(owner_id):
        return None
    return approval_keyboard(token, lang)


def _intent_from_dict(d: dict) -> TradeIntent:
    """Rebuild a TradeIntent from its persisted ``to_dict`` form."""
    ts = d.get("timestamp")
    return TradeIntent(
        symbol=d["symbol"],
        side=Side(d["side"]),
        target_notional=float(d["target_notional"]),
        entry=d.get("entry"),
        stop_loss=d.get("stop_loss"),
        take_profit=d.get("take_profit"),
        confidence=d.get("confidence"),
        strategy=d.get("strategy", ""),
        timestamp=None if ts is None else pd.Timestamp(ts),
    )


class OwnerApprovalHandler:
    """Handles ``own:`` approval callbacks: owner-gate, risk-gate, then place."""

    def __init__(
        self,
        *,
        owner_id: str | None,
        pending: PendingSignalStore,
        engine: ExecutionEngine,
        risk_control: ProductionRiskControl,
        risk_state: RiskStateStore | None = None,
        ledger_store=None,
        prefs: LanguagePrefsStore | None = None,
        ttl_seconds: float = _DEFAULT_TTL_S,
    ) -> None:
        self._owner_id = str(owner_id) if owner_id else None
        self._pending = pending
        self._engine = engine
        self._risk_control = risk_control
        self._risk_state = risk_state
        self._ledger_store = ledger_store
        self._prefs = prefs
        self._ttl = ttl_seconds

    def handles(self, data: str) -> bool:
        """True for callback data this handler owns (the ``own:`` namespace)."""
        return data.startswith(OWNER_PREFIX)

    def _lang(self, chat_id: str) -> str:
        return self._prefs.get(chat_id) if self._prefs is not None else DEFAULT_LANGUAGE

    def handle(
        self,
        *,
        data: str,
        from_id: str,
        chat_id: str,
        message_id: int | None,
        callback_id: str,
        client,
    ) -> None:
        """Dispatch one ``own:`` callback. Never raises to the polling loop."""
        lang = self._lang(chat_id)

        # Owner gate: the buttons are only ever sent to the owner, but a leaked
        # callback_data must still be refused for anyone else — no order, no state
        # change, and the surface stays invisible (a generic denial toast only).
        if self._owner_id is None or from_id != self._owner_id:
            if callback_id:
                client.answer_callback_query(
                    callback_query_id=callback_id, text=t(lang, "owner_denied")
                )
            logger.warning("Owner approval refused for non-owner {}", from_id)
            return

        parts = data[len(OWNER_PREFIX):].split(":")
        action = parts[0] if parts else ""
        token = parts[1] if len(parts) > 1 else ""
        signal = self._pending.get(token)

        # Already resolved, unknown, or a restart lost it: report and stop.
        if signal is None or not signal.is_pending():
            self._finish(client, chat_id, message_id, callback_id, t(lang, "owner_gone"))
            return

        age = signal.age_seconds()
        if age is not None and age > self._ttl:
            self._pending.resolve(token, STATUS_EXPIRED, "ttl")
            self._finish(client, chat_id, message_id, callback_id, t(lang, "owner_expired"))
            return

        if action == _IGNORE:
            self._pending.resolve(token, STATUS_IGNORED)
            self._finish(client, chat_id, message_id, callback_id, t(lang, "owner_ignored"))
            return

        if action == _ACCEPT:
            self._accept(signal, token, lang, client, chat_id, message_id, callback_id)
            return

        # Unknown action verb: answer so the client stops spinning, do nothing else.
        if callback_id:
            client.answer_callback_query(callback_query_id=callback_id)

    # -- accept path ---------------------------------------------------------

    def _accept(self, signal, token, lang, client, chat_id, message_id, callback_id) -> None:
        intent = _intent_from_dict(signal.intent)
        allowed, reason = self._risk_gate(intent)
        if not allowed:
            self._pending.resolve(token, STATUS_BLOCKED, reason)
            text = t(lang, "owner_blocked").format(reason=reason)
            self._finish(client, chat_id, message_id, callback_id, text)
            logger.warning("Owner accepted {} but risk gate blocked it: {}", intent.symbol, reason)
            return
        try:
            report = self._engine.execute_bracket_intents([intent])
        except Exception as exc:  # placing must never crash the polling loop
            self._pending.resolve(token, STATUS_BLOCKED, f"error: {exc}")
            self._finish(client, chat_id, message_id, callback_id, t(lang, "owner_error"))
            logger.error("Owner order placement failed for {}: {}", intent.symbol, exc)
            return
        self._book_fills()
        summary = t(lang, "owner_accepted").format(
            symbol=intent.symbol,
            side=t(lang, "side_long" if intent.side is Side.BUY else "side_short"),
            filled=report.n_filled,
            rejected=report.n_rejected,
        )
        self._pending.resolve(token, STATUS_ACCEPTED, summary)
        self._finish(client, chat_id, message_id, callback_id, summary)
        logger.info("Owner accepted {} — filled={} rejected={}",
                    intent.symbol, report.n_filled, report.n_rejected)

    def _risk_gate(self, intent: TradeIntent) -> tuple[bool, str]:
        """Run the accepted intent through ProductionRiskControl. (ok, reason)."""
        broker = self._engine.broker
        state = broker.get_portfolio_state()
        nav = float(state.value)
        positions = pd.Series(
            {s: p.quantity for s, p in state.positions.items()}, dtype="float64"
        )
        signed = intent.target_notional if intent.side is Side.BUY else -intent.target_notional
        proposed = pd.Series({intent.symbol: signed}, dtype="float64")

        # The emergency-stop latch and (when present) the day-start NAV are owned by
        # the pipeline's persisted risk state; honour the latch here so a tripped stop
        # also refuses manual approvals. The realized daily-loss anchor stays a
        # pipeline concern — this gate enforces the entry-level limits.
        day_start_nav = None
        if self._risk_state is not None:
            rs = self._risk_state.load()
            if rs.emergency_stopped:
                self._risk_control.trip_emergency_stop()
            day_start_nav = rs.day_start_nav

        decision = self._risk_control.evaluate(
            proposed, positions, nav, day_start_nav=day_start_nav
        )
        if decision.halted:
            return False, decision.halt_reason.value if decision.halt_reason else "halted"
        if not decision.is_allowed(intent.symbol):
            reason = decision.reason_for(intent.symbol)
            return False, reason.value if reason is not None else "blocked"
        return True, ""

    def _book_fills(self) -> None:
        """Poll the broker's new fills into the ledger (best-effort, PAPER/LIVE)."""
        if self._ledger_store is None:
            return
        try:
            ledger = self._ledger_store.load()
            fills = self._engine.broker.get_fills(since=ledger.watermark())
            if ledger.record(fills):
                self._ledger_store.save(ledger)
        except Exception as exc:  # accounting must never break the approval flow
            logger.warning("Owner-approval accounting update failed: {}", exc)

    @staticmethod
    def _finish(client, chat_id, message_id, callback_id, text: str) -> None:
        """Edit the original signal message to its outcome and stop the spinner."""
        if chat_id and message_id is not None:
            try:
                client.edit_message_text(
                    chat_id=chat_id, message_id=int(message_id), text=text
                )
            except Exception as exc:  # a failed edit must not lose the outcome
                logger.warning("Failed to edit owner approval message: {}", exc)
        if callback_id:
            client.answer_callback_query(callback_query_id=callback_id)
