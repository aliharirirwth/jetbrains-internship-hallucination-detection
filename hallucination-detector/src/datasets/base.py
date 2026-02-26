from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class HallucinationSample:
    """Single sample for hallucination detection.

    Attributes:
        question: Input question or prompt.
        answer: Model answer (faithful or hallucinated).
        label: 1 = hallucination, 0 = faithful.
        dataset: Source dataset name.
        metadata: Optional extra fields.
    """
    question: str
    answer: str
    label: int  # 1 = hallucination, 0 = faithful
    dataset: str
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.metadata is None:
            self.metadata = {}


class BaseDataset(ABC):
    """Abstract base for loading and normalizing a hallucination dataset to HallucinationSample list."""

    name: str = "base"

    @abstractmethod
    def load(self) -> list[HallucinationSample]:
        """Load from source (e.g. HuggingFace) and return list of HallucinationSample.

        Returns:
            List of HallucinationSample with question, answer, label, dataset.
        """
        ...

    def get_dataframe(self) -> pd.DataFrame:
        """Return a DataFrame with columns question, answer, label, dataset.

        Returns:
            One row per sample.
        """
        samples = self.load()
        return pd.DataFrame(
            [
                {
                    "question": s.question,
                    "answer": s.answer,
                    "label": s.label,
                    "dataset": s.dataset,
                }
                for s in samples
            ]
        )

    def label_distribution(self) -> dict[str, int]:
        """Return counts per label.

        Returns:
            Dict mapping label string (e.g. '0', '1') to count.
        """
        samples = self.load()
        counts: dict[str, int] = {}
        for s in samples:
            k = str(s.label)
            counts[k] = counts.get(k, 0) + 1
        return counts
