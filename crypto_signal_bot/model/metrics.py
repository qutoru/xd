"""Classification metrics and human-readable reporting for the signal model.

Accuracy alone is misleading for a 3-class trading signal, so we report macro
metrics and per-class precision/recall/F1, plus a majority-class baseline for
context. Labels are kept in the ``-1/0/1`` signal space throughout.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from crypto_signal_bot.config import CLASS_NAMES, CLASS_ORDER


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute a metrics dict for signal-space (-1/0/1) predictions.

    Returns:
        Dict with ``accuracy``, ``macro_f1``, ``baseline_accuracy`` and a
        ``per_class`` mapping of class name -> precision/recall/f1/support.
    """
    labels = list(CLASS_ORDER)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )

    # Majority-class baseline: always predict the most frequent true class.
    majority = Counter(y_true).most_common(1)[0][0]
    baseline_acc = float(np.mean(np.asarray(y_true) == majority))

    per_class = {
        CLASS_NAMES[label]: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i, label in enumerate(labels)
    }

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "baseline_accuracy": baseline_acc,
        "baseline_class": CLASS_NAMES[int(majority)],
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def log_report(name: str, metrics: dict) -> None:
    """Pretty-log a metrics dict produced by :func:`evaluate`."""
    logger.info(
        "[{}] accuracy={:.3f} (baseline {:.3f} = always '{}') | macro_f1={:.3f}",
        name,
        metrics["accuracy"],
        metrics["baseline_accuracy"],
        metrics["baseline_class"],
        metrics["macro_f1"],
    )
    for cls, m in metrics["per_class"].items():
        logger.info(
            "  {:>5}: precision={:.3f} recall={:.3f} f1={:.3f} (support={})",
            cls,
            m["precision"],
            m["recall"],
            m["f1"],
            m["support"],
        )
    # Confusion matrix rows = true class, cols = predicted class.
    header = " ".join(f"{CLASS_NAMES[c]:>6}" for c in CLASS_ORDER)
    logger.info("  confusion (rows=true, cols=pred): {}", header)
    for c, row in zip(CLASS_ORDER, metrics["confusion_matrix"]):
        logger.info("    {:>5} {}", CLASS_NAMES[c], " ".join(f"{v:>6}" for v in row))
