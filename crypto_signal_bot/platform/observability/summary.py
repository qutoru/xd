"""CycleSummary — a single structured view of one completed pipeline cycle.

Pure projection over an existing ``PipelineResult`` (and the reports it already
holds): it copies fields and derives an overall status from flags that are already
set. It never recomputes PnL, never touches a broker/ledger/risk control, and
imports the pipeline only under ``TYPE_CHECKING`` (fields are read duck-typed), so
there is no runtime dependency on the pipeline and no import cycle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # annotation only — no runtime import (keeps observability acyclic)
    from crypto_signal_bot.platform.pipeline import PipelineResult


class CycleStatus(str, Enum):
    """Overall health of a cycle, derived from flags already set on the result."""

    OK = "ok"              # traded/observed cleanly
    DEGRADED = "degraded"  # completed but with rejects / blocks / discrepancies
    HALTED = "halted"      # risk control halted new entries
    FAILED = "failed"      # execution/reconciliation failed this cycle


@dataclass(frozen=True)
class CycleSummary:
    """Flat, log-friendly snapshot of one cycle. All values are read, not computed."""

    asof: str
    status: CycleStatus
    duration_s: float | None
    # execution
    filled: int
    rejected: int
    traded_notional: float
    # risk control
    halted: bool
    halt_reason: str | None
    blocked: int
    # reconciliation (message doubles as the emergency-stop cause when it failed)
    reconciliation_ok: bool
    reconciliation_failed: bool
    reconciliation_message: str | None
    discrepancies: tuple[str, ...]
    # accounting / performance (cumulative, copied from PerformanceReport)
    realized_pnl: float | None
    fees: float | None
    net_pnl: float | None
    realized_by_reason: dict[str, float] = field(default_factory=dict)
    # shadow (virtual) economics
    shadow_daily_pnl: float = 0.0
    shadow_cum_pnl: float = 0.0
    # notification delivery
    notify_failed: int = 0

    @classmethod
    def from_result(cls, result: "PipelineResult", *, duration_s: float | None = None) -> "CycleSummary":
        """Project a finished ``PipelineResult`` into a summary (no computation)."""
        ex = result.execution
        rc = result.risk_control
        rec = result.reconciliation
        perf = result.performance
        notify = result.notify

        halted = bool(rc and rc.halted)
        halt_reason = rc.halt_reason.value if (rc and rc.halt_reason) else None
        blocked = rc.n_blocked if rc else 0
        rejected = ex.n_rejected if ex else 0
        rec_failed = bool(rec and rec.failed)
        discrepancies = (
            tuple(sorted(k.value for k in rec.kinds())) if rec is not None else ()
        )
        notify_failed = notify.failed if notify else 0

        status = cls._status(rec_failed, halted, rejected, blocked, discrepancies, notify_failed)

        return cls(
            asof=str(getattr(result.asof, "date", lambda: result.asof)()),
            status=status,
            duration_s=duration_s,
            filled=ex.n_filled if ex else 0,
            rejected=rejected,
            traded_notional=ex.traded_notional if ex else 0.0,
            halted=halted,
            halt_reason=halt_reason,
            blocked=blocked,
            reconciliation_ok=bool(rec and rec.ok),
            reconciliation_failed=rec_failed,
            reconciliation_message=(rec.message or None) if rec_failed else None,
            discrepancies=discrepancies,
            realized_pnl=perf.realized_pnl if perf else None,
            fees=perf.fees if perf else None,
            net_pnl=perf.net_pnl if perf else None,
            realized_by_reason=dict(perf.realized_by_reason) if perf else {},
            shadow_daily_pnl=result.shadow.daily_pnl,
            shadow_cum_pnl=result.shadow.cum_pnl,
            notify_failed=notify_failed,
        )

    @staticmethod
    def _status(
        rec_failed: bool, halted: bool, rejected: int, blocked: int,
        discrepancies: tuple[str, ...], notify_failed: int,
    ) -> CycleStatus:
        if rec_failed:
            return CycleStatus.FAILED
        if halted:
            return CycleStatus.HALTED
        if rejected or blocked or discrepancies or notify_failed:
            return CycleStatus.DEGRADED
        return CycleStatus.OK

    def as_dict(self) -> dict:
        """Flat dict for structured logging / persistence (status as its value)."""
        return {
            "asof": self.asof,
            "status": self.status.value,
            "duration_s": self.duration_s,
            "filled": self.filled,
            "rejected": self.rejected,
            "traded_notional": self.traded_notional,
            "halted": self.halted,
            "halt_reason": self.halt_reason,
            "blocked": self.blocked,
            "reconciliation_ok": self.reconciliation_ok,
            "reconciliation_failed": self.reconciliation_failed,
            "reconciliation_message": self.reconciliation_message,
            "discrepancies": list(self.discrepancies),
            "realized_pnl": self.realized_pnl,
            "fees": self.fees,
            "net_pnl": self.net_pnl,
            "realized_by_reason": dict(self.realized_by_reason),
            "shadow_daily_pnl": self.shadow_daily_pnl,
            "shadow_cum_pnl": self.shadow_cum_pnl,
            "notify_failed": self.notify_failed,
        }

    def format_lines(self) -> list[str]:
        """A few human-readable lines for the operator log."""
        dur = "n/a" if self.duration_s is None else f"{self.duration_s:.2f}s"
        lines = [
            f"cycle {self.asof} status={self.status.value.upper()} duration={dur}",
            f"  execution: filled={self.filled} rejected={self.rejected} "
            f"traded_notional={self.traded_notional:.2f}",
            f"  risk: halted={self.halted} reason={self.halt_reason or '-'} "
            f"blocked={self.blocked}",
        ]
        if self.reconciliation_failed:
            lines.append(f"  reconciliation: FAILED ({self.reconciliation_message or 'unknown'})")
        elif self.discrepancies:
            lines.append(f"  reconciliation: discrepancies={list(self.discrepancies)}")
        else:
            lines.append("  reconciliation: ok")
        if self.realized_pnl is not None:
            lines.append(
                f"  accounting: realized={self.realized_pnl:+.4f} fees={self.fees:.4f} "
                f"net={self.net_pnl:+.4f} by_reason={self.realized_by_reason}"
            )
        else:
            lines.append("  accounting: disabled")
        lines.append(
            f"  shadow: daily_pnl={self.shadow_daily_pnl:+.5f} cum_pnl={self.shadow_cum_pnl:+.5f}"
        )
        return lines
