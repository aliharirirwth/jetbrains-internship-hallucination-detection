"""Word analogy evaluation using Google questions-words.txt; semantic/syntactic accuracy."""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

import numpy as np

ANALOGY_URL = "http://download.tensorflow.org/data/questions-words.txt"


def download_analogies(path: str | Path = "questions-words.txt") -> Path:
    path = Path(path)
    if path.exists():
        return path
    urllib.request.urlretrieve(ANALOGY_URL, path)
    return path


def load_analogies(path: str | Path) -> tuple[list[str], list[tuple[str, str, str, str]]]:
    """Parse questions-words.txt. Returns (categories, list of (a, b, c, d) tuples)."""
    path = Path(path)
    categories: list[str] = []
    quadruples: list[tuple[str, str, str, str]] = []
    current_section = ""
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line.startswith(":"):
                current_section = line[1:].strip()
                categories.append(current_section)
                continue
            parts = line.split()
            if len(parts) == 4:
                quadruples.append((parts[0], parts[1], parts[2], parts[3]))
    return categories, quadruples


def load_analogies_by_section(path: str | Path) -> dict[str, list[tuple[str, str, str, str]]]:
    """Map each section header to the list of (a, b, c, d) analogies in that section."""
    path = Path(path)
    by_section: dict[str, list[tuple[str, str, str, str]]] = {}
    current = ""
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line.startswith(":"):
                current = line[1:].strip()
                by_section[current] = []
                continue
            parts = line.split()
            if len(parts) == 4 and current:
                by_section[current].append((parts[0], parts[1], parts[2], parts[3]))
    return by_section


def analogy_accuracy(
    W: np.ndarray,
    word2idx: dict[str, int],
    idx2word: dict[int, str],
    analogies: list[tuple[str, str, str, str]],
    exclude_query: bool = True,
) -> float:
    """For each a:b::c:? find argmax cosine(W[a]-W[b]+W[c]). Return accuracy."""
    W_norm = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-10)
    correct = 0
    total = 0
    for a, b, c, d in analogies:
        if a not in word2idx or b not in word2idx or c not in word2idx or d not in word2idx:
            continue
        total += 1
        ia, ib, ic, id_ = word2idx[a], word2idx[b], word2idx[c], word2idx[d]
        v = W_norm[ia] - W_norm[ib] + W_norm[ic]
        v /= np.linalg.norm(v) + 1e-10
        sims = W_norm @ v
        if exclude_query:
            sims[ia] = -np.inf
            sims[ib] = -np.inf
            sims[ic] = -np.inf
        pred = int(np.argmax(sims))
        if pred == id_:
            correct += 1
    return correct / total if total else 0.0


def run_evaluation(
    W_in: np.ndarray,
    word2idx: dict[str, int],
    idx2word: dict[int, str],
    analogies_path: str | Path | None = None,
) -> dict[str, float]:
    """Download questions-words if needed, run analogy accuracy by category. Return dict category -> accuracy."""
    path = Path(analogies_path) if analogies_path else download_analogies()
    if not path.exists():
        download_analogies(path)
    by_section = load_analogies_by_section(path)

    results: dict[str, float] = {}
    semantic: list[float] = []
    syntactic: list[float] = []
    gram_cats = {"gram1-adjective-to-adverb", "gram2-opposite", "gram3-comparative", "gram4-superlative", "gram5-present-participle", "gram6-nationality-adjective", "gram7-past-tense", "gram8-plural", "gram9-plural-verbs"}
    for cat, quads_cat in by_section.items():
        acc = analogy_accuracy(W_in, word2idx, idx2word, quads_cat)
        results[cat] = acc
        if cat in gram_cats or "gram" in cat.lower():
            syntactic.append(acc)
        else:
            semantic.append(acc)
    results["semantic_avg"] = float(np.mean(semantic)) if semantic else 0.0
    results["syntactic_avg"] = float(np.mean(syntactic)) if syntactic else 0.0
    cat_accs = [results[cat] for cat in by_section]
    results["overall"] = float(np.mean(cat_accs)) if cat_accs else 0.0
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate word2vec on Google word analogies")
    parser.add_argument("--W_in", default="W_in_epoch5.npy", help="Path to W_in matrix")
    parser.add_argument("--vocab", default="vocab_word2idx.npy", help="Path to word2idx keys .npy")
    parser.add_argument("--analogies", default="questions-words.txt", help="Path to questions-words.txt")
    args = parser.parse_args()

    W_in = np.load(args.W_in)
    words = np.load(args.vocab, allow_pickle=True).tolist()
    word2idx = {w: i for i, w in enumerate(words)}
    idx2word = {i: w for i, w in enumerate(words)}

    path = args.analogies
    if not Path(path).exists():
        download_analogies(path)
    results = run_evaluation(W_in, word2idx, idx2word, path)
    print("Category accuracies:")
    for k, v in sorted(results.items(), key=lambda x: x[0]):
        print(f"  {k}: {v:.4f}")
    print(f"Semantic (avg): {results.get('semantic_avg', 0):.4f}")
    print(f"Syntactic (avg): {results.get('syntactic_avg', 0):.4f}")
