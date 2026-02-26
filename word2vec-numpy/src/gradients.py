from __future__ import annotations

from typing import Tuple

import numpy as np

from .model import sigmoid


def gradients(
    center: int,
    context: int,
    negatives: list[int],
    W_in: np.ndarray,
    W_out: np.ndarray,
    pos_score: float,
    neg_scores: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute gradients of the SGNS objective J w.r.t. W_in and W_out.

    For SGD we ascend on J: W += lr * grad.

    Args:
        center: Center word index.
        context: Context (positive) word index.
        negatives: List of negative word indices.
        W_in: Center embedding matrix (V, D).
        W_out: Context embedding matrix (V, D).
        pos_score: Scalar v_c·v_o.
        neg_scores: Array of v_c·v_k for each negative k.

    Returns:
        Tuple (dW_in, dW_out): dW_in is (V, D), only row center non-zero;
        dW_out is (V, D), rows for context and negatives non-zero.
    """
    v_c = W_in[center]
    v_o = W_out[context]
    sig_pos = sigmoid(np.array([pos_score]))[0]
    sig_neg = sigmoid(neg_scores)

    # ∂J/∂v_c = (1 - σ(v_c·v_o))·v_o - Σ_k σ(v_c·v_k)·v_k
    d_vc = (1.0 - sig_pos) * v_o - np.sum(sig_neg[:, np.newaxis] * W_out[negatives], axis=0)

    # ∂J/∂v_o = (1 - σ(v_c·v_o))·v_c
    d_vo = (1.0 - sig_pos) * v_c

    # ∂J/∂v_k = -σ(v_c·v_k)·v_c for each k
    d_vk_list = [-sig_neg[i] * v_c for i in range(len(negatives))]

    V, D = W_in.shape
    dW_in = np.zeros_like(W_in)
    dW_out = np.zeros_like(W_out)
    dW_in[center] = d_vc
    dW_out[context] = d_vo
    for i, k in enumerate(negatives):
        dW_out[k] += d_vk_list[i]

    return dW_in, dW_out
