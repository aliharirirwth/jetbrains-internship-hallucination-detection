# Word2Vec in Pure NumPy (Skip-gram with Negative Sampling)

Skip-gram with negative sampling (SGNS) implemented in NumPy only. No PyTorch/TensorFlow/JAX; scikit-learn is used only for evaluation (t-SNE, analogy accuracy).

## Quick start

```bash
cd word2vec-numpy
python data/fetch_text8.py
python run_train.py
# Evaluate (from checkpoints/)
python -m src.evaluate --W_in checkpoints/W_in_epoch5.npy --vocab checkpoints/vocab_word2idx.npy
```

---

## 1. Gradient derivation (negative sampling objective)

We **maximize** the objective:

\[
J = \log \sigma(\mathbf{v}_c^\top \mathbf{v}_o) + \sum_{k=1}^{K} \log \sigma(-\mathbf{v}_c^\top \mathbf{v}_k)
\]

where \(\mathbf{v}_c = W_{\text{in}}[\text{center}]\), \(\mathbf{v}_o = W_{\text{out}}[\text{context}]\), \(\mathbf{v}_k = W_{\text{out}}[\text{neg}_k]\), and \(\sigma(x) = 1/(1+e^{-x})\).

Let \(s_o = \mathbf{v}_c^\top \mathbf{v}_o\) and \(s_k = \mathbf{v}_c^\top \mathbf{v}_k\). Then:

- \(\frac{\partial}{\partial s_o} \log \sigma(s_o) = 1 - \sigma(s_o)\)
- \(\frac{\partial}{\partial s_k} \log \sigma(-s_k) = -\sigma(s_k)\)

By chain rule:

\[
\frac{\partial J}{\partial \mathbf{v}_c}
= (1 - \sigma(s_o))\,\mathbf{v}_o - \sum_{k=1}^{K} \sigma(s_k)\,\mathbf{v}_k
\]

\[
\frac{\partial J}{\partial \mathbf{v}_o} = (1 - \sigma(s_o))\,\mathbf{v}_c
\]

\[
\frac{\partial J}{\partial \mathbf{v}_k} = -\sigma(s_k)\,\mathbf{v}_c \quad \text{for each } k
\]

We perform **gradient ascent** on \(J\): \(W \leftarrow W + \eta \cdot \frac{\partial J}{\partial W}\).

---

## 2. Why \(\text{freq}^{3/4}\) for negative sampling?

Raw unigram frequency would oversample very common words (e.g. "the", "a") as negatives, so the model would mostly learn to separate center from these few words. Raising to the power **3/4** (Mikolov et al.) flattens the distribution: rare words get a relatively higher probability of being chosen as negatives. This improves representation quality for rare words while still favoring more frequent words than uniform sampling. So we use \(p(w) \propto \text{count}(w)^{3/4}\) for drawing negative samples.

---

## 3. Why two weight matrices (\(W_{\text{in}}\), \(W_{\text{out}}\))?

- **\(W_{\text{in}}\)**: center-word embeddings (one vector per word when it is the center).
- **\(W_{\text{out}}\)**: context-word embeddings (when the word appears in the context window).

Using two matrices (no weight tying) gives more capacity: the same word can have different representations as center vs context. In practice, after training we typically **use only \(W_{\text{in}}\)** for downstream tasks (e.g. word similarity, analogy); \(W_{\text{out}}\) is discarded or averaged with \(W_{\text{in}}\) in some variants.

---

## 4. Subsampling and why it helps

Subsampling randomly **drops** frequent words with probability \(P_{\text{discard}} = 1 - \sqrt{t/f(w)}\) (with \(t \approx 10^{-5}\)). So we **keep** a word with probability \(\sqrt{t/f(w)}\) (capped at 1).

- Reduces training time and dominance of very frequent words.
- Balances the signal: rare words appear more often relative to "the", "a", etc.
- Often improves accuracy on rare words and overall embedding quality.

---

## 5. Alternatives

| Choice | Alternative | Trade-off |
|--------|-------------|-----------|
| **Skip-gram** | CBOW | Skip-gram: predict context from center; usually better for small corpora and rare words. CBOW: predict center from context; faster, sometimes better for frequent words. |
| **Negative sampling** | Hierarchical softmax | Negative sampling: simple, works well with 5–20 negatives. Hierarchical softmax: exact softmax over vocab via a tree; no negatives, but more complex and tree-dependent. |
| **Two matrices** | Weight tying (\(W_{\text{out}} = W_{\text{in}}\)) | Tying reduces parameters and can regularize; two matrices give more flexibility and are standard in word2vec. |

---

## File layout

- `src/vocab.py` — Vocabulary, word counts, frequencies, noise table (unigram^3/4).
- `src/dataset.py` — SkipGramDataset: (center, context, negatives) with subsampling and dynamic window.
- `src/model.py` — SkipGram: \(W_{\text{in}}\), \(W_{\text{out}}\), forward pass (scores only).
- `src/loss.py` — Negative sampling loss (negative of \(J\)).
- `src/gradients.py` — Analytic gradients (no autograd).
- `src/train.py` — SGD training loop, optional LR decay, checkpoints per epoch.
- `src/evaluate.py` — Google word analogy evaluation (semantic/syntactic accuracy).
- `data/text8` — Raw corpus (run `data/fetch_text8.py` to download).
- `notebooks/analysis.ipynb` — Optional t-SNE visualization of embeddings.
