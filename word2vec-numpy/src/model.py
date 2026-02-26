"""Skip-gram model: W_in (center), W_out (context); forward pass only (no gradients here)."""

from __future__ import annotations

from typing import List

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )


class SkipGram:
    """Two weight matrices (no weight tying). W_in for center, W_out for context."""

    def __init__(self, vocab_size: int, embedding_dim: int = 100, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.W_in = rng.uniform(
            -0.5 / embedding_dim,
            0.5 / embedding_dim,
            (vocab_size, embedding_dim),
        ).astype(np.float64)
        self.W_out = np.zeros((vocab_size, embedding_dim), dtype=np.float64)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

    def forward(
        self,
        center: int,
        context: int,
        negatives: list[int],
    ) -> tuple[float, np.ndarray]:
        """Compute positive score (scalar) and negative scores (array of K scalars)."""
        v_c = self.W_in[center]   # (D,)
        v_o = self.W_out[context]  # (D,)
        pos_score = float(np.dot(v_c, v_o))

        neg_scores = np.array(
            [float(np.dot(v_c, self.W_out[k])) for k in negatives],
            dtype=np.float64,
        )
        return pos_score, neg_scores
