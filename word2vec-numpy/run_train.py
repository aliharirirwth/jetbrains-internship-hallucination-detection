#!/usr/bin/env python3
"""Build vocab, dataset, train Skip-gram, save checkpoints. Run from word2vec-numpy/."""
import re
import sys
from pathlib import Path

# Add src to path when running from repo root or word2vec-numpy/
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.vocab import Vocabulary
from src.dataset import SkipGramDataset
from src.model import SkipGram
from src.train import train

DATA_DIR = Path(__file__).resolve().parent / "data"
TEXT_PATH = DATA_DIR / "text8"
SAVE_DIR = Path(__file__).resolve().parent / "checkpoints"


def tokenize(text: str) -> list[str]:
    return re.findall(r"\b[a-z]+\b", text.lower())


def main():
    if not TEXT_PATH.exists():
        print(f"Run: python data/fetch_text8.py  (or place text8 at {TEXT_PATH})")
        sys.exit(1)
    text = TEXT_PATH.read_text(encoding="utf-8", errors="ignore")
    tokens = tokenize(text)
    print(f"Corpus: {len(tokens):,} tokens")

    # Use smaller noise table for low-memory; spec uses 100M for O(1) sampling
    vocab = Vocabulary(min_count=5, noise_table_size=100_000_000)
    vocab.build(tokens)
    print(f"Vocab size: {vocab.size:,}")

    dataset = SkipGramDataset(
        tokens, vocab, window_size=5, neg_samples=5, subsample_t=1e-5, seed=42
    )
    model = SkipGram(vocab_size=vocab.size, embedding_dim=100, seed=42)
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    train(
        dataset, model, vocab,
        learning_rate=0.025, min_lr=0.0001, epochs=5,
        lr_schedule=True, log_every=10_000, save_dir=str(SAVE_DIR),
    )
    print(f"Checkpoints saved to {SAVE_DIR}")


if __name__ == "__main__":
    main()
