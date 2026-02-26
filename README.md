# Hallucination Detection Project

Cross-dataset hallucination detector for LLM outputs using geometric methods on hidden states. The detector is trained (probed) on one dataset and evaluated on others to measure cross-domain transferability.

## Repository layout

- **word2vec-numpy/** — Skip-gram word2vec in pure NumPy (Task #1; no ML frameworks except sklearn for evaluation).
- **hallucination-detector/** — Full geometric hallucination detection pipeline (datasets, extractor, probe, transfer evaluation).

All implementation lives under this repo; code from `jetbrains-rl` and `jetbrains-reason-slm` was used as reference where applicable.

## Quick start

### Word2Vec (NumPy)

```bash
cd word2vec-numpy
pip install -r requirements.txt
python data/fetch_text8.py
python run_train.py
python -m src.evaluate --W_in checkpoints/W_in_epoch5.npy --vocab checkpoints/vocab_word2idx.npy
```

### Hallucination detector

```bash
cd hallucination-detector
pip install -r requirements.txt
python scripts/01_download_datasets.py
python scripts/02_preprocess.py
# GPU (Colab): extract features, then train and evaluate (see hallucination-detector/README.md)
```

## Hardware

- **H100 (80GB)** — Recommended for full pipeline (Llama 8B, 4-bit).
- **A100 / L4** — Suitable for extraction and training.
- **T4 (16GB)** — Use small model (e.g. `facebook/opt-125m`) and 4-bit if needed.
- Runs on **CUDA/GPU** (e.g. Google Colab).

## Task summary

1. **Word2Vec**: Implement SGNS in NumPy (forward, loss, gradients, SGD); evaluate with Google word analogies; document gradient derivation and design choices.
2. **Hallucination detector**: Implement geometric probe (Mahalanobis, cosine, norm, layer diff), train on one dataset, evaluate transfer to others; produce transfer matrix (AUROC) and ablations.
