from __future__ import annotations

from collections import Counter

import numpy as np


class Vocabulary:
    """Vocabulary from raw text with subsampling and O(1) negative sampling.

    Attributes:
        word2idx: Map word -> index.
        idx2word: Map index -> word.
        word_counts: Raw counts per word.
        word_freqs: Normalized frequencies (length V).
        noise_table: Unigram^(3/4) table for negative sampling.
    """

    def __init__(self, min_count: int = 5, noise_table_size: int = 100_000_000):
        self.min_count = min_count
        self.noise_table_size = noise_table_size
        self.word2idx: dict[str, int] = {}
        self.idx2word: dict[int, str] = {}
        self.word_counts: dict[str, int] = {}
        self.word_freqs: np.ndarray = np.array([])  # normalized frequencies, length V
        self.noise_table: np.ndarray = np.array([], dtype=np.int32)  # size table_size

    def build(self, tokens: list[str]) -> None:
        """Build vocabulary from token list.

        Words with count < min_count are excluded. Populates word2idx, idx2word,
        word_counts, word_freqs, and noise_table.

        Args:
            tokens: List of token strings (e.g. from tokenized corpus).
        """
        counts = Counter(tokens)
        vocab_words = [w for w, c in counts.items() if c >= self.min_count]
        vocab_words.sort(key=lambda w: -counts[w])  # by frequency descending

        self.word2idx = {w: i for i, w in enumerate(vocab_words)}
        self.idx2word = {i: w for i, w in enumerate(vocab_words)}
        self.word_counts = {w: counts[w] for w in vocab_words}

        total = sum(self.word_counts[w] for w in vocab_words)
        self.word_freqs = np.array(
            [self.word_counts[self.idx2word[i]] / total for i in range(len(vocab_words))],
            dtype=np.float64,
        )
        self.noise_table = self.get_noise_table(table_size=self.noise_table_size)

    def subsample_prob(self, word: str, t: float = 1e-5) -> float:
        """Probability of keeping the word (Mikolov subsampling).

        P_keep such that P_discard = 1 - sqrt(t / freq(w)); high-freq words are
        downsampled.

        Args:
            word: Word string.
            t: Subsampling threshold (default 1e-5).

        Returns:
            Float in [0, 1]; 0.0 if word not in vocabulary.
        """
        if word not in self.word_counts:
            return 0.0
        freq = self.word_counts[word] / sum(self.word_counts.values())
        keep = np.sqrt(t / (freq + 1e-12))
        return float(np.minimum(keep, 1.0))

    def get_noise_table(self, table_size: int = 100_000_000) -> np.ndarray:
        """Build unigram^(3/4) noise table for O(1) negative sampling.

        Each word fills slots proportional to freq(w)^(3/4). Uniform index into
        the table yields a negative sample with the correct distribution.

        Args:
            table_size: Length of the noise table.

        Returns:
            int32 array of shape (table_size,) of word indices; empty if vocab not built.
        """
        if not self.word_freqs.size:
            return np.zeros(0, dtype=np.int32)

        # Unigram^(3/4) distribution, normalized
        pow_freqs = np.power(self.word_freqs, 0.75)
        pow_freqs /= pow_freqs.sum()
        # How many slots each word gets (approximate; we fill exactly table_size)
        counts_float = pow_freqs * table_size
        counts_int = np.floor(counts_float).astype(np.int64)
        remainder = table_size - counts_int.sum()
        # Give extra slots to words with largest fractional part
        if remainder > 0:
            extra = np.argsort(counts_float - counts_int)[-remainder:]
            counts_int[extra] += 1

        table = np.zeros(table_size, dtype=np.int32)
        idx = 0
        for i in range(len(self.idx2word)):
            n = int(counts_int[i])
            table[idx : idx + n] = i
            idx += n
        # Shuffle so uniform random index gives correct distribution
        rng = np.random.default_rng(42)
        rng.shuffle(table)
        return table

    @property
    def size(self) -> int:
        return len(self.word2idx)
