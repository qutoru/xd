"""ProductionRiskControl — the final gate deciding which new entries may open.

Given the proposed new-entry notionals (already sized by RiskManager), the current
holdings and the account NAV, it returns the set of entries that are allowed and
whether trading is halted. It is a pure decision layer: it never sends, sizes or
closes anything — the pipeline drops the blocked intents before they reach the
broker, so existing positions are always left untouched.

Independent by construction: imports only pandas + stdlib. It knows nothing about
brokers, exchanges (Bybit), execution, research, portfolio construction, alphas,
strategy or Telegram.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum

import pandas as pd

_TOL = 1e-9


class BlockReason(str, Enum):
    """Why a new entry was refused (or why trading is globally halted)."""

    KILL_SWITCH = "kill_switch"
    EMERGENCY_STOP = "emergency_stop"
    DAILY_LOSS_LIMIT = "daily_loss_limit"
    MAX_OPEN_POSITIONS = "max_open_positions"
    MAX_EXPOSURE = "max_exposure"
    MAX_POSITION_SIZE = "max_position_size"


@dataclass(frozen=True)
class RiskControlConfig:
    """Production risk limits. The all-None/False default is a no-op gate.

    Fractions are of NAV. A limit left as ``None`` (or ``kill_switch=False``) is
    simply not enforced, so wiring the control with the default config changes no
    behaviour while still providing the emergency-stop machinery.
    """

    kill_switch: bool = False               # block ALL new entries when True
    daily_loss_limit: float | None = None   # halt when day loss >= this fraction of day-start NAV
    # When True (and a realized daily PnL is supplied), the daily-loss halt measures
    # loss from *realized* PnL booked today instead of the NAV delta. Off by default
    # and it falls back to the NAV delta whenever no realized value is provided, so
    # existing behaviour is unchanged.
    use_realized_daily_loss: bool = False
    max_open_positions: int | None = None   # cap on simultaneously open positions
    max_exposure: float | None = None       # cap on total gross exposure, as a fraction of NAV
    max_position_size: float | None = None  # cap per order, as a fraction of NAV
    # When True, a reconciliation mismatch/failure (platform view != exchange) trips
    # the latched emergency stop, so no new entries open until the owner resets. Off
    # by default (a reconciliation discrepancy stays a logged diagnostic).
    halt_on_recon_mismatch: bool = False

    @classmethod
    def from_env(cls) -> RiskControlConfig:
        """Build limits from ``RISK_*`` env vars (unset -> the no-op default).

        This is the single production configuration path for the risk controls,
        mirroring ``BybitConfig.from_env``. With no ``RISK_*`` set it returns the
        same all-off default, so behaviour is unchanged unless explicitly tuned.
        """
        def _f(name: str) -> float | None:
            raw = os.getenv(name)
            return float(raw) if raw not in (None, "") else None

        def _i(name: str) -> int | None:
            raw = os.getenv(name)
            return int(raw) if raw not in (None, "") else None

        def _b(name: str) -> bool:
            raw = os.getenv(name)
            return raw.strip().lower() in {"1", "true", "yes", "on"} if raw is not None else False

        return cls(
            kill_switch=_b("RISK_KILL_SWITCH"),
            daily_loss_limit=_f("RISK_DAILY_LOSS_LIMIT"),
            use_realized_daily_loss=_b("RISK_USE_REALIZED_DAILY_LOSS"),
            max_open_positions=_i("RISK_MAX_OPEN_POSITIONS"),
            max_exposure=_f("RISK_MAX_EXPOSURE"),
            max_position_size=_f("RISK_MAX_POSITION_SIZE"),
            halt_on_recon_mismatch=_b("RISK_HALT_ON_RECON_MISMATCH"),
        )


@dataclass(frozen=True)
class RiskControlDecision:
    """Outcome of one gate evaluation."""

    allowed: tuple[str, ...]
    blocked: tuple[tuple[str, BlockReason], ...] = ()
    halted: bool = False
    halt_reason: BlockReason | None = None

    @property
    def ok(self) -> bool:
        """True when nothing was halted or blocked."""
        return not self.halted and not self.blocked

    @property
    def n_blocked(self) -> int:
        return len(self.blocked)

    def is_allowed(self, symbol: str) -> bool:
        return symbol in set(self.allowed)

    def reason_for(self, symbol: str) -> BlockReason | None:
        for sym, reason in self.blocked:
            if sym == symbol:
                return reason
        return None


class ProductionRiskControl:
    """The final gate: decide which new entries may open (never closes positions)."""

    def __init__(self, config: RiskControlConfig | None = None) -> None:
        self.config = config or RiskControlConfig()
        self._emergency_stopped = False

    # --- emergency stop (latched) ------------------------------------------
    @property
    def emergency_stopped(self) -> bool:
        return self._emergency_stopped

    def trip_emergency_stop(self) -> None:
        """Latch the emergency stop: block all new entries until explicitly reset.

        Does not stop the application — the caller keeps running; only *new*
        entries are refused on subsequent evaluations.
        """
        self._emergency_stopped = True

    def reset_emergency_stop(self) -> None:
        self._emergency_stopped = False

    # --- the gate -----------------------------------------------------------
    def evaluate(
        self,
        proposed: pd.Series,
        positions: pd.Series | None = None,
        nav: float = 0.0,
        *,
        day_start_nav: float | None = None,
        realized_daily_pnl: float | None = None,
    ) -> RiskControlDecision:
        """Decide which proposed new entries may open.

        ``proposed`` — signed target notional per symbol (from RiskManager).
        ``positions`` — signed notional per currently-held symbol.
        ``nav`` — current account equity; ``day_start_nav`` — equity at the start
        of the trading day, used only for the daily-loss halt.
        ``realized_daily_pnl`` — PnL actually realized today (a plain number, from
        the caller's accounting); consulted for the daily-loss halt only when
        ``use_realized_daily_loss`` is set. The control never sees a ledger/broker.

        A global halt (kill switch / emergency stop / daily-loss) blocks every
        proposed entry. Otherwise each entry is checked against max_position_size,
        and each *brand-new* symbol additionally against max_open_positions and
        max_exposure (entries re-affirming an already-open symbol never add a new
        position and are allowed once they pass the per-order size cap).
        """
        cfg = self.config
        proposed = self._clean(proposed)
        symbols = list(proposed.index)

        halt = self._halt_reason(nav, day_start_nav, realized_daily_pnl)
        if halt is not None:
            return RiskControlDecision(
                allowed=(),
                blocked=tuple((s, halt) for s in symbols),
                halted=True,
                halt_reason=halt,
            )

        held = self._clean(positions)
        held_symbols = set(held.index)
        running_gross = float(held.abs().sum())
        running_open = len(held_symbols)

        allowed: list[str] = []
        blocked: list[tuple[str, BlockReason]] = []
        for s in sorted(symbols):
            size = abs(float(proposed[s]))

            # Per-order size cap — applies to every entry, even a re-affirm, so an
            # oversized order slips through even if RiskManager mis-sizes it.
            if cfg.max_position_size is not None and size > cfg.max_position_size * nav + _TOL:
                blocked.append((s, BlockReason.MAX_POSITION_SIZE))
                continue

            if s not in held_symbols:  # a brand-new position
                if cfg.max_open_positions is not None and running_open + 1 > cfg.max_open_positions:
                    blocked.append((s, BlockReason.MAX_OPEN_POSITIONS))
                    continue
                if cfg.max_exposure is not None and running_gross + size > cfg.max_exposure * nav + _TOL:
                    blocked.append((s, BlockReason.MAX_EXPOSURE))
                    continue
                running_open += 1
                running_gross += size

            allowed.append(s)

        return RiskControlDecision(allowed=tuple(allowed), blocked=tuple(blocked))

    # --- helpers ------------------------------------------------------------
    def _halt_reason(
        self, nav: float, day_start_nav: float | None,
        realized_daily_pnl: float | None = None,
    ) -> BlockReason | None:
        cfg = self.config
        if cfg.kill_switch:
            return BlockReason.KILL_SWITCH
        if self._emergency_stopped:
            return BlockReason.EMERGENCY_STOP
        if cfg.daily_loss_limit is not None and day_start_nav is not None and day_start_nav > 0:
            # realized-PnL mode: today's loss is the negative of realized PnL booked
            # today; otherwise the legacy NAV delta. Fall back to NAV when no realized
            # value is supplied (accounting off) so behaviour is unchanged.
            if cfg.use_realized_daily_loss and realized_daily_pnl is not None:
                loss = -realized_daily_pnl
            else:
                loss = day_start_nav - nav
            if loss >= cfg.daily_loss_limit * day_start_nav - _TOL:
                return BlockReason.DAILY_LOSS_LIMIT
        return None

    @staticmethod
    def _clean(series: pd.Series | None) -> pd.Series:
        """Drop NaNs and (near-)zero entries; return a float64 Series."""
        if series is None or len(series) == 0:
            return pd.Series(dtype="float64")
        s = series.astype("float64").dropna()
        return s[s.abs() > _TOL]
