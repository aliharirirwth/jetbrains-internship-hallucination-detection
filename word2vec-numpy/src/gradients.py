"""Exact analytic gradients for SGNS (no autograd). See README for derivation."""

from __future__ import annotations

from typing import List, Tuple

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
    """
    Compute ∂J/∂W_in and ∂J/∂W_out (gradient of objective J we maximize).
    So for SGD we do: W -= lr * (-grad) = W + lr * grad (ascend on J).

    Returns:
        dW_in: (V, D) array, only row center is non-zero.
        dW_out: (V, D) array, rows context and neg indices are non-zero.
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
