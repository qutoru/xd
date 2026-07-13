"""Signal registry — plug strategies in without changing the core.

A new strategy is added by dropping ``signals/<name>/signal.py`` that decorates
its provider with ``@register_signal("<name>")``. Discovery imports each
subpackage's ``signal`` module lazily, so ``list_signals`` / ``load_signal`` see
it with zero edits to the platform. This is the Open/Closed seam.
"""

from __future__ import annotations

import importlib
import pkgutil
from typing import Any, Callable

from crypto_signal_bot.platform import signals as _pkg
from crypto_signal_bot.platform.signals.base import SignalProvider

_REGISTRY: dict[str, type[SignalProvider]] = {}
_discovered = False


def register_signal(name: str) -> Callable[[type[SignalProvider]], type[SignalProvider]]:
    """Class decorator that registers a :class:`SignalProvider` under ``name``."""

    def deco(cls: type[SignalProvider]) -> type[SignalProvider]:
        if not (isinstance(cls, type) and issubclass(cls, SignalProvider)):
            raise TypeError(f"{cls!r} is not a SignalProvider subclass")
        if name in _REGISTRY and _REGISTRY[name] is not cls:
            raise ValueError(f"signal '{name}' already registered")
        cls.name = name
        _REGISTRY[name] = cls
        return cls

    return deco


def _discover() -> None:
    """Import every ``signals/<name>/signal.py`` so plugins self-register."""
    global _discovered
    if _discovered:
        return
    for mod in pkgutil.iter_modules(_pkg.__path__):
        if not mod.ispkg:
            continue
        target = f"{_pkg.__name__}.{mod.name}.signal"
        try:
            importlib.import_module(target)
        except ModuleNotFoundError as exc:
            # A subpackage without its own signal.py is simply skipped; a
            # genuinely missing dependency inside a plugin still propagates.
            if exc.name != target:
                raise
    _discovered = True


def list_signals() -> list[str]:
    """Names of all registered signals (triggers discovery)."""
    _discover()
    return sorted(_REGISTRY)


def load_signal(name: str, **kwargs: Any) -> SignalProvider:
    """Instantiate a registered signal by name."""
    _discover()
    if name not in _REGISTRY:
        raise KeyError(f"unknown signal '{name}'; available: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)
