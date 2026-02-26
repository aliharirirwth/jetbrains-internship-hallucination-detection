"""Skip-gram training pair generator with subsampling and negative sampling via noise table."""

from __future__ import annotations

from typing import Iterator

import numpy as np

from .vocab import Vocabulary


class SkipGramDataset:
    """Yields (center_idx, context_idx, [neg_1, ..., neg_k]) with subsampling and dynamic window."""

    def __init__(
        self,
        tokens: list[str],
        vocab: Vocabulary,
        window_size: int = 5,
        neg_samples: int = 5,
        subsample_t: float = 1e-5,
        seed: int = 42,
    ):
        self.tokens = tokens
        self.vocab = vocab
        self.window_size = window_size
        self.neg_samples = neg_samples
        self.subsample_t = subsample_t
        self.rng = np.random.default_rng(seed)
        self._table = vocab.noise_table
        self._table_len = len(self._table)

    def __iter__(self) -> Iterator[tuple[int, int, list[int]]]:
        """Apply subsampling, then for each kept position yield (center_idx, context_idx, neg_indices)."""
        # Subsample: keep each token with prob subsample_prob(word)
        kept: list[tuple[int, int]] = []  # (position, word_idx)
        for i, w in enumerate(self.tokens):
            if w not in self.vocab.word2idx:
                continue
            if self.rng.random() < self.vocab.subsample_prob(w, self.subsample_t):
                kept.append((i, self.vocab.word2idx[w]))

        V = self.vocab.size
        for pos, center_idx in kept:
            # Dynamic window: sample actual window in [1, window_size]
            win = self.rng.integers(1, self.window_size + 1)
            start = max(0, pos - win)
            end = min(len(self.tokens), pos + win + 1)
            for j in range(start, end):
                if j == pos:
                    continue
                w = self.tokens[j]
                if w not in self.vocab.word2idx:
                    continue
                context_idx = self.vocab.word2idx[w]
                # Negative samples from noise table; exclude center and context
                neg_list: list[int] = []
                while len(neg_list) < self.neg_samples:
                    idx = self.rng.integers(0, self._table_len)
                    neg_idx = int(self._table[idx])
                    if neg_idx != center_idx and neg_idx != context_idx and neg_idx not in neg_list:
                        neg_list.append(neg_idx)
                yield center_idx, context_idx, neg_list
