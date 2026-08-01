"""Trade-outcome reporter — WIN/LOSE Telegram messages when a position closes.

The semi-auto owner path opens each accepted position with a reduce-only TP/SL
bracket. When a close happens on the venue (take-profit, stop-loss, or a
rebalance flatten) the venue's fill carries the realized PnL. This reporter,
driven by the long-poll listener on each loop tick, polls the broker's fills into
the durable ledger and then messages the owner the result of every newly-closed
trade — WIN / LOSE + net PnL — de-duplicated by execution id across restarts via a
small JSON state file.

It reads the same :class:`AccountingLedger`/store the pipeline already books into,
so it manufactures no PnL of its own: a close's WIN/LOSE is purely the sign of the
venue-reported realized PnL (net of fees). Everything is injected (broker, store,
client), so the whole flow is unit-tested offline with fakes.
"""

from __future__ import annotations

import json
from pathlib import Path

from loguru import logger

from crypto_signal_bot.platform.accounting.attribution import Attribution
from crypto_signal_bot.platform.notify.i18n import DEFAULT_LANGUAGE, t

# Ledger attribution -> localized "how it closed" line.
_CLOSE_LABELS: dict[str, str] = {
    Attribution.TAKE_PROFIT.value: "outcome_tp",
    Attribution.STOP_LOSS.value: "outcome_sl",
    Attribution.REBALANCE.value: "outcome_rebalance",
    Attribution.MANUAL.value: "outcome_manual",
}


class NotifiedOutcomeStore:
    """Persist the set of already-announced execution ids (idempotent messaging)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> set[str]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, ValueError, OSError):
            return set()
        return set(data.get("notified", []))

    def save(self, notified: set[str]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps({"notified": sorted(notified)}), encoding="utf-8"
        )


def _is_close(attribution: str) -> bool:
    """True for a closing fill (anything that is not a pure entry)."""
    return attribution != Attribution.ENTRY.value


class TradeOutcomeReporter:
    """Announce each newly-closed trade to the owner as WIN / LOSE + net PnL."""

    def __init__(
        self,
        *,
        broker,
        ledger_store,
        client,
        chat_id,
        state: NotifiedOutcomeStore,
        prefs=None,
    ) -> None:
        self._broker = broker
        self._ledger_store = ledger_store
        self._client = client
        self._chat_id = str(chat_id) if chat_id else None
        self._state = state
        self._prefs = prefs

    def _lang(self) -> str:
        if self._prefs is not None and self._chat_id:
            return self._prefs.get(self._chat_id)
        return DEFAULT_LANGUAGE

    def seed_if_new(self) -> int:
        """Baseline pre-existing closes on first activation so they are not announced.

        Only closes that happen *after* the reporter starts should be messaged;
        without this, activating the reporter over a ledger full of historical
        closes would spam one WIN/LOSE per past trade. When the state file already
        exists this is a no-op. Returns the number of closes baselined.
        """
        if self._state.path.exists() or self._ledger_store is None:
            return 0
        try:
            ledger = self._ledger_store.load()
        except Exception as exc:  # never fail startup over a baseline read
            logger.warning("Outcome reporter: baseline read failed: {}", exc)
            return 0
        seen = {
            e.exec_id
            for e in ledger.entries
            if e.exec_id and _is_close(e.attribution)
        }
        self._state.save(seen)
        return len(seen)

    def poll(self) -> int:
        """Book new venue fills, then message each not-yet-announced close.

        Best-effort end to end: a broker/network failure or a single failed send is
        logged and swallowed so the listener loop is never broken. A fill whose send
        fails is left unmarked so a later tick retries it. Returns the number of
        outcome messages sent this tick.
        """
        if self._broker is None or self._ledger_store is None or not self._chat_id:
            return 0
        try:
            ledger = self._ledger_store.load()
            fills = self._broker.get_fills(since=ledger.watermark())
            if ledger.record(fills):
                self._ledger_store.save(ledger)
        except Exception as exc:  # venue read / ledger write must not break the loop
            logger.warning("Outcome reporter: fill/ledger poll failed: {}", exc)
            return 0

        notified = self._state.load()
        lang = self._lang()

        # One venue close (stop-loss / take-profit) can settle in many partial
        # executions, each a distinct execId with its own prorated PnL. Announcing
        # per execId would spam one WIN/LOSE per partial fill, so aggregate the
        # not-yet-announced closes by (symbol, attribution) into a single message
        # carrying the summed net PnL of the whole close.
        groups: dict[tuple[str, str], dict] = {}
        order: list[tuple[str, str]] = []
        for entry in ledger.entries:
            if not _is_close(entry.attribution) or entry.exec_id is None:
                continue
            if entry.exec_id in notified:
                continue
            key = (entry.symbol, entry.attribution)
            g = groups.get(key)
            if g is None:
                g = {"net": 0.0, "ids": []}
                groups[key] = g
                order.append(key)
            g["net"] += entry.realized_pnl - entry.fee
            g["ids"].append(entry.exec_id)

        sent = 0
        changed = False
        for key in order:
            symbol, attribution = key
            g = groups[key]
            try:
                self._client.send_message(
                    chat_id=self._chat_id,
                    text=self._format(symbol, attribution, g["net"], lang),
                )
                sent += 1
            except Exception as exc:  # retry on a later tick; leave the group unmarked
                logger.error("Outcome reporter: send failed for {}: {}", symbol, exc)
                continue
            notified.update(g["ids"])
            changed = True
        if changed:
            self._state.save(notified)
        return sent

    @staticmethod
    def _format(symbol: str, attribution: str, net: float, lang: str) -> str:
        """Render an aggregated close (summed net PnL) as a WIN/LOSE message."""
        if net > 0:
            emoji, head = "🏆", t(lang, "outcome_win")
        elif net < 0:
            emoji, head = "❌", t(lang, "outcome_lose")
        else:
            emoji, head = "➖", t(lang, "outcome_breakeven")
        reason = t(lang, _CLOSE_LABELS.get(attribution, "outcome_manual"))
        return "\n".join([
            f"{emoji} {symbol} — {head}",
            reason,
            f"{t(lang, 'outcome_pnl')}: {net:+.2f} USDT",
        ])
