from __future__ import annotations

import numpy as np

try:
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )
except ImportError:
    roc_auc_score = None
    f1_score = None
    accuracy_score = None
    precision_score = None
    recall_score = None


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error: average gap between confidence and accuracy per bin.

    Args:
        y_true: Ground truth binary labels.
        y_prob: Predicted probabilities (or array with positive-class column).
        n_bins: Number of probability bins for ECE.

    Returns:
        Scalar ECE in [0, 1]; 0 if no samples.
    """
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()
    if len(y_prob.shape) > 1:
        y_prob = y_prob[:, 1]  # positive class
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = 0
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if mask.sum() == 0:
            continue
        acc = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += np.abs(acc - conf) * mask.sum()
        total += mask.sum()
    return float(ece / total) if total else 0.0


def compute_all_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute classification metrics and calibration.

    Args:
        y_true: Ground truth binary labels.
        y_prob: Predicted probabilities (n_samples,) or (n_samples, 2); positive class used.
        threshold: Decision threshold for F1, accuracy, precision, recall.

    Returns:
        Dict with auroc, f1, accuracy, precision, recall, ece, n_samples, positive_rate.
    """
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob)
    if y_prob.ndim > 1:
        y_prob = y_prob[:, 1]
    y_pred = (y_prob >= threshold).astype(np.int64)

    out: dict[str, float] = {
        "n_samples": float(len(y_true)),
        "positive_rate": float(np.mean(y_true)),
    }
    # AUROC undefined when only one class in y_true (avoids UndefinedMetricWarning)
    n_classes = len(np.unique(y_true))
    if roc_auc_score is not None and n_classes >= 2:
        try:
            out["auroc"] = float(roc_auc_score(y_true, y_prob))
        except Exception:
            out["auroc"] = 0.5
    else:
        out["auroc"] = float("nan") if n_classes < 2 else 0.5
    if f1_score is not None:
        out["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    else:
        out["f1"] = 0.0
    if accuracy_score is not None:
        out["accuracy"] = float(accuracy_score(y_true, y_pred))
    else:
        out["accuracy"] = float(np.mean(y_true == y_pred))
    if precision_score is not None:
        out["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    else:
        out["precision"] = 0.0
    if recall_score is not None:
        out["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    else:
        out["recall"] = 0.0
    out["ece"] = expected_calibration_error(y_true, y_prob)
    return out
