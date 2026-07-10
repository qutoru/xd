"""Chronological train/validation/test splitting with label purging.

The data is time-ordered and must never be shuffled: bars are autocorrelated
and, crucially, every triple-barrier label looks ``HORIZON`` bars into the
future. A training sample located right before a split boundary would therefore
have its outcome partly determined by bars belonging to the next split — a
look-ahead leak. We remove (``purge``) the last ``embargo`` samples from the
tail of each earlier split to prevent this.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from crypto_signal_bot.config import EMBARGO, TRAIN_FRAC, VALID_FRAC


@dataclass(frozen=True)
class SplitIndices:
    """Positional row indices for each chronological split."""

    train: np.ndarray
    valid: np.ndarray
    test: np.ndarray


def time_split(
    n_rows: int,
    *,
    train_frac: float = TRAIN_FRAC,
    valid_frac: float = VALID_FRAC,
    embargo: int = EMBARGO,
) -> SplitIndices:
    """Split ``n_rows`` chronologically into train/valid/test with purging.

    Args:
        n_rows: Total number of (time-sorted) rows.
        train_frac: Fraction of rows for training.
        valid_frac: Fraction of rows for validation (test gets the remainder).
        embargo: Number of tail rows purged from train and valid to avoid
            forward label leakage across the boundary.

    Returns:
        A :class:`SplitIndices` with disjoint, time-ordered index arrays.
    """
    if train_frac + valid_frac >= 1.0:
        raise ValueError("train_frac + valid_frac must be < 1.0 (test is the remainder)")

    train_end = int(n_rows * train_frac)
    valid_end = int(n_rows * (train_frac + valid_frac))

    idx = np.arange(n_rows)
    # Purge the tail of train and valid so their forward-looking labels do not
    # overlap the following split.
    train = idx[: max(0, train_end - embargo)]
    valid = idx[train_end : max(train_end, valid_end - embargo)]
    test = idx[valid_end:]

    return SplitIndices(train=train, valid=valid, test=test)
