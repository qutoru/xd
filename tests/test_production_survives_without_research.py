"""M2 — Deletion-safety: production works with research fully removed.

Proves the project's core invariant: the trading bot imports and runs even if the
entire research archive is deleted from the repo. We build a temporary copy of
the ``crypto_signal_bot`` package with the ``research`` subpackage OMITTED, and
with NO ``alpha_library`` / ``experiments`` present, then import every production
module in a clean subprocess. Success ⇒ production has zero research dependency.

The real repository (including all frozen research) is never modified — we only
copy out of it into a temp dir.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Leaf production modules whose import pulls in the whole production graph.
PRODUCTION_MODULES = [
    "crypto_signal_bot.config",
    "crypto_signal_bot.data.fetcher",
    "crypto_signal_bot.data.storage",
    "crypto_signal_bot.data.universe",
    "crypto_signal_bot.data.derivatives",
    "crypto_signal_bot.features.dataset",
    "crypto_signal_bot.model.train",
    "crypto_signal_bot.model.predict",
    "crypto_signal_bot.backtest.runner",
    "crypto_signal_bot.backtest.walkforward",
    "crypto_signal_bot.backtest.portfolio",
    "crypto_signal_bot.live.signal",
    "crypto_signal_bot.live.scheduler",
    "crypto_signal_bot.live.telegram",
    "crypto_signal_bot.pipeline",
    "crypto_signal_bot.platform.config",
    "crypto_signal_bot.platform.interfaces",
    "crypto_signal_bot.platform.signal",
    "crypto_signal_bot.platform.sleeve",
    "crypto_signal_bot.platform.core.portfolio",
    "crypto_signal_bot.platform.core.metrics",
    "crypto_signal_bot.platform.data.interfaces",
    "crypto_signal_bot.platform.data.cache",
    "crypto_signal_bot.platform.data.bybit_source",
    "crypto_signal_bot.platform.data.snapshot",
    "crypto_signal_bot.platform.signals.base",
    "crypto_signal_bot.platform.signals.registry",
    "crypto_signal_bot.platform.signals.alx.signal",
    "crypto_signal_bot.platform.portfolio.book",
    "crypto_signal_bot.platform.portfolio.normalization",
    "crypto_signal_bot.platform.portfolio.combiner",
    "crypto_signal_bot.platform.portfolio.allocator",
    "crypto_signal_bot.platform.portfolio.constraints",
    "crypto_signal_bot.platform.portfolio.turnover",
    "crypto_signal_bot.platform.portfolio.builder",
    "crypto_signal_bot.platform.shadow.report",
    "crypto_signal_bot.platform.shadow.pnl",
    "crypto_signal_bot.platform.shadow.funding",
    "crypto_signal_bot.platform.shadow.costs",
    "crypto_signal_bot.platform.shadow.replay",
    "crypto_signal_bot.platform.shadow.runner",
    "crypto_signal_bot.platform.execution.domain",
    "crypto_signal_bot.platform.execution.broker",
    "crypto_signal_bot.platform.execution.engine",
    "crypto_signal_bot.platform.execution.fake_broker",
    "crypto_signal_bot.platform.pipeline",
    "crypto_signal_bot.app.cli",
    "main",
]


def _build_research_free_copy(dst: Path) -> None:
    """Copy the production package into ``dst`` WITHOUT any research code."""
    # crypto_signal_bot package, excluding the research subpackage + caches.
    shutil.copytree(
        REPO / "crypto_signal_bot",
        dst / "crypto_signal_bot",
        ignore=shutil.ignore_patterns("__pycache__", "research"),
    )
    shutil.copy2(REPO / "main.py", dst / "main.py")
    # Deliberately do NOT copy alpha_library/ or experiments/ at all.
    assert not (dst / "crypto_signal_bot" / "research").exists()
    assert not (dst / "alpha_library").exists()
    assert not (dst / "experiments").exists()


def test_production_imports_without_research(tmp_path: Path):
    _build_research_free_copy(tmp_path)

    script = textwrap.dedent(
        f"""
        import importlib
        mods = {PRODUCTION_MODULES!r}
        for m in mods:
            importlib.import_module(m)
        print("OK", len(mods))
        """
    )
    # Clean subprocess rooted at the research-free copy so `crypto_signal_bot`
    # resolves there; third-party deps still come from site-packages. Keep the
    # real environment (PATH/SystemRoot for DLL loading) but point PYTHONPATH at
    # the temp copy only.
    env = dict(os.environ)
    env["PYTHONPATH"] = str(tmp_path)
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(tmp_path),
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, (
        "production failed to import without research:\n"
        f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )
    assert "OK" in proc.stdout
