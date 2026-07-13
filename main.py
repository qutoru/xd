"""Thin bootstrap entry point for crypto_signal_bot.

All command orchestration lives in :mod:`crypto_signal_bot.app.cli`; this file
only forwards the process entry point to it. No business logic here.

    py main.py <command> [options]      # see `py main.py --help`
"""

from __future__ import annotations

import sys

from crypto_signal_bot.app.cli import main

if __name__ == "__main__":
    sys.exit(main())
