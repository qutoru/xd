"""Import-boundary guard: the production platform must be fully autonomous.

Fails if any module under ``crypto_signal_bot/platform`` imports frozen research
alpha or the research package. This is the seam that keeps production independent
of research/alpha_library so the two can evolve separately.
"""

from __future__ import annotations

import ast
import pathlib

PLATFORM = (
    pathlib.Path(__file__).resolve().parents[1]
    / "crypto_signal_bot"
    / "platform"
)

FORBIDDEN = ("alpha_library", "experiments", "crypto_signal_bot.research")


def _imported_modules(path: pathlib.Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    mods: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            mods.append(node.module or "")
    return mods


def test_platform_has_no_research_or_alpha_imports():
    offenders: dict[str, list[str]] = {}
    for py in PLATFORM.rglob("*.py"):
        bad = [m for m in _imported_modules(py) if any(f in m for f in FORBIDDEN)]
        if bad:
            offenders[str(py.relative_to(PLATFORM))] = bad
    assert not offenders, f"production platform imports frozen/research code: {offenders}"
