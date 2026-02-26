"""Abstract base class and schema for hallucination datasets."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class HallucinationSample:
    """Single sample: question, model answer, binary label, source dataset, optional metadata."""
    question: str
    answer: str
    label: int  # 1 = hallucination, 0 = faithful
    dataset: str
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.metadata is None:
            self.metadata = {}


class BaseDataset(ABC):
    """Load and normalize a hallucination dataset to HallucinationSample list."""

    name: str = "base"

    @abstractmethod
    def load(self) -> list[HallucinationSample]:
        """Load from source (e.g. HuggingFace) and return list of HallucinationSample."""
        ...

    def get_dataframe(self) -> pd.DataFrame:
        """Return a DataFrame with columns question, answer, label, dataset."""
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
        """Return counts per label (e.g. {'0': n_faithful, '1': n_hallucination})."""
        samples = self.load()
        counts: dict[str, int] = {}
        for s in samples:
            k = str(s.label)
            counts[k] = counts.get(k, 0) + 1
        return counts
