from __future__ import annotations

import logging
from collections import Counter

from .base import HallucinationSample

logger = logging.getLogger(__name__)


def validate_schema(
    samples: list[HallucinationSample],
    dataset_name: str = "dataset",
    warn_imbalance_threshold: float = 0.8,
) -> dict[str, bool]:
    """Check loaded data for common schema and quality issues.

    Logs warnings for label imbalance, empty question/answer, duplicate (question, answer),
    and non-binary labels.

    Args:
        samples: List of HallucinationSample to validate.
        dataset_name: Name used in log messages.
        warn_imbalance_threshold: Warn if positive rate > this or < (1 - this).

    Returns:
        Dict of check names to bool (True if passed): no_empty_question, no_empty_answer,
        binary_labels, label_balance_ok, no_duplicate_qa.
    """
    results: dict[str, bool] = {
        "no_empty_question": True,
        "no_empty_answer": True,
        "binary_labels": True,
        "label_balance_ok": True,
        "no_duplicate_qa": True,
    }
    if not samples:
        logger.warning(f"[{dataset_name}] Empty sample list")
        return results

    # Empty strings
    for s in samples:
        if not (s.question or "").strip():
            results["no_empty_question"] = False
            break
    for s in samples:
        if not (s.answer or "").strip():
            results["no_empty_answer"] = False
            break

    # Labels
    labels = [s.label for s in samples]
    unique = set(labels)
    if unique != {0, 1} and unique != {0} and unique != {1}:
        results["binary_labels"] = False
        logger.warning(f"[{dataset_name}] Labels are not binary: {unique}")

    # Balance
    n_pos = sum(1 for s in samples if s.label == 1)
    ratio = n_pos / len(samples)
    if ratio >= warn_imbalance_threshold or ratio <= (1 - warn_imbalance_threshold):
        results["label_balance_ok"] = False
        logger.warning(
            f"[{dataset_name}] Label imbalance: {n_pos}/{len(samples)} positive ({ratio:.2%}); "
            f"consider balancing or documenting."
        )

    # Duplicate (question, answer)
    pairs = [(s.question.strip(), s.answer.strip()) for s in samples]
    counts = Counter(pairs)
    dups = sum(1 for c in counts.values() if c > 1)
    if dups > 0:
        results["no_duplicate_qa"] = False
        logger.warning(f"[{dataset_name}] Found duplicate (question, answer) pairs: {dups}")

    return results
