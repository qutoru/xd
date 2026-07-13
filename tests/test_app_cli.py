"""Stage 6 — app layer: thin main bootstrap + CLI wiring (offline)."""

from __future__ import annotations

import importlib

from crypto_signal_bot.app import cli


def test_parser_has_all_commands_including_shadow():
    parser = cli.build_parser()
    actions = [a for a in parser._actions if getattr(a, "choices", None) and hasattr(a, "_name_parser_map")]
    commands = set()
    for a in actions:
        commands |= set(a.choices)
    for expected in {"symbols", "fetch", "build", "train", "backtest", "backtest-rm",
                     "walkforward", "portfolio", "signal", "live", "shadow"}:
        assert expected in commands


def test_shadow_command_parses_and_binds():
    parser = cli.build_parser()
    args = parser.parse_args(["shadow", "--signals", "alx", "--asof", "2023-05-01"])
    assert args.func is cli.cmd_shadow
    assert args.signals == "alx" and args.asof == "2023-05-01"


def test_main_is_thin_bootstrap_delegating_to_cli():
    main_module = importlib.import_module("main")
    # main.py must expose exactly the app-layer main (no own orchestration).
    assert main_module.main is cli.main
