from __future__ import annotations

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid to avoid overflow for large |x|.

    Args:
        x: Input array (any shape).

    Returns:
        Array same shape as x with values in (0, 1).
    """
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )


class SkipGram:
    """Skip-gram model with two weight matrices (no weight tying).

    W_in: center word embeddings; W_out: context/negative word embeddings.
    """

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
        """Compute positive and negative dot-product scores.

        Args:
            center: Center word index.
            context: Context word index.
            negatives: List of K negative word indices.

        Returns:
            Tuple of (pos_score, neg_scores): pos_score is v_c·v_o; neg_scores
            is array of v_c·v_k for each negative k.
        """
        v_c = self.W_in[center]   # (D,)
        v_o = self.W_out[context]  # (D,)
        pos_score = float(np.dot(v_c, v_o))

        neg_scores = np.array(
            [float(np.dot(v_c, self.W_out[k])) for k in negatives],
            dtype=np.float64,
        )
        return pos_score, neg_scores
