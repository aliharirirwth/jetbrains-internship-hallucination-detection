from __future__ import annotations

from pathlib import Path

import numpy as np
from tqdm import tqdm

from .dataset import SkipGramDataset
from .gradients import gradients
from .loss import loss
from .model import SkipGram
from .vocab import Vocabulary


def train(
    dataset: SkipGramDataset,
    model: SkipGram,
    vocab: Vocabulary,
    learning_rate: float = 0.025,
    min_lr: float = 0.0001,
    epochs: int = 5,
    lr_schedule: bool = True,
    log_every: int = 10_000,
    save_dir: str | None = None,
) -> list[tuple[int, float]]:
    """Run SGD training on SkipGramDataset.

    Updates model one (center, context, negatives) at a time. Optionally applies
    linear LR decay to min_lr over epochs. Saves checkpoints every epoch if save_dir set.

    Args:
        dataset: SkipGramDataset iterator.
        model: SkipGram model to train.
        vocab: Vocabulary (for checkpointing).
        learning_rate: Initial learning rate.
        min_lr: Minimum learning rate when using schedule.
        epochs: Number of passes over the dataset.
        lr_schedule: If True, decay LR linearly to min_lr.
        log_every: Log loss every this many steps.
        save_dir: If set, save W_in, W_out, vocab per epoch.

    Returns:
        List of (step, loss) for logging.
    """
    history: list[tuple[int, float]] = []
    total_steps = 0
    for epoch in range(epochs):
        epoch_dataset = SkipGramDataset(
            dataset.tokens,
            dataset.vocab,
            window_size=dataset.window_size,
            neg_samples=dataset.neg_samples,
            subsample_t=dataset.subsample_t,
            seed=epoch + 42,
        )
        epoch_loss = 0.0
        n = 0
        # Epoch-based linear decay to min_lr
        if lr_schedule and epochs > 0:
            lr = max(min_lr, learning_rate * (1.0 - (epoch / epochs)))
        else:
            lr = learning_rate
        pbar = tqdm(epoch_dataset, desc=f"Epoch {epoch+1}/{epochs}", leave=True)
        for center, context, negs in pbar:

            pos_score, neg_scores = model.forward(center, context, negs)
            L = loss(pos_score, neg_scores)
            history.append((total_steps, L))
            epoch_loss += L
            n += 1

            dW_in, dW_out = gradients(
                center, context, negs,
                model.W_in, model.W_out,
                pos_score, neg_scores,
            )
            # Ascend on J: W += lr * grad (grad is ∂J/∂W)
            model.W_in += lr * dW_in
            model.W_out += lr * dW_out

            total_steps += 1
            if total_steps % log_every == 0:
                recent = [h[1] for h in history[-log_every:]]
                pbar.set_postfix(loss=f"{np.mean(recent):.4f}", lr=f"{lr:.5f}")

        avg_epoch = epoch_loss / n if n else 0.0
        tqdm.write(f"Epoch {epoch+1} avg loss: {avg_epoch:.4f}")

        if save_dir:
            path = Path(save_dir)
            path.mkdir(parents=True, exist_ok=True)
            np.save(path / f"W_in_epoch{epoch+1}.npy", model.W_in)
            np.save(path / f"W_out_epoch{epoch+1}.npy", model.W_out)
            np.save(path / "vocab_word2idx.npy", np.array(list(vocab.word2idx.keys()), dtype=object))
            np.save(path / "vocab_idx2word.npy", np.array(list(vocab.idx2word.values()), dtype=object))

    return history
