"""M1 — Production import-boundary guard.

Static (AST) guarantee that NO production module imports frozen research:
``alpha_library`` / ``experiments`` / ``crypto_signal_bot.research`` (which also
covers ``validate.py`` / ``run.py``, since those live under ``alpha_library``).

This locks the architectural seam: production and research can now evolve
independently, and an accidental ``from crypto_signal_bot.research...`` added to
any production file will fail this test in CI.
"""

from __future__ import annotations

import ast
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[1]
PKG = REPO / "crypto_signal_bot"

# The full production surface: the ML bot + the autonomous platform + the CLI.
# Explicitly EXCLUDES crypto_signal_bot/research (frozen research infra).
PRODUCTION_PATHS = [
    PKG / "config.py",
    PKG / "data",
    PKG / "features",
    PKG / "model",
    PKG / "backtest",
    PKG / "live",
    PKG / "pipeline.py",
    PKG / "platform",
    PKG / "app",
    REPO / "main.py",
]

FORBIDDEN = ("alpha_library", "experiments", "crypto_signal_bot.research")


def _iter_production_files():
    for p in PRODUCTION_PATHS:
        if p.is_dir():
            yield from (f for f in p.rglob("*.py") if "__pycache__" not in f.parts)
        elif p.is_file():
            yield p


def _imported_modules(path: pathlib.Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    mods: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            mods.append(node.module or "")
    return mods


def test_production_never_imports_research():
    offenders: dict[str, list[str]] = {}
    for py in _iter_production_files():
        bad = [m for m in _imported_modules(py) if any(f in m for f in FORBIDDEN)]
        if bad:
            offenders[str(py.relative_to(REPO))] = bad
    assert not offenders, (
        "production imports frozen research code (must be empty): " + repr(offenders)
    )


def test_production_surface_is_non_trivial():
    # Guard the guard: make sure we are actually scanning files (path drift).
    files = list(_iter_production_files())
    assert len(files) >= 20, f"expected the production surface, scanned only {len(files)}"
