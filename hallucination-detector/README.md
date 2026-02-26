# Hallucination Detector — Geometric Probe on LLM Hidden States

Cross-dataset hallucination detection: train a probe on one dataset, evaluate on others to measure transferability.

## Setup

```bash
cd hallucination-detector
pip install -r requirements.txt
# For GPU (Colab): pip install bitsandbytes
```

## Evaluation notebook

**[evaluation.ipynb](evaluation.ipynb)** — Paper-ready evaluation: download/preprocess data, extract features (2k samples per dataset), run full transfer matrix with multiple seeds, and ablations. Designed for Google Colab (GPU); set `PROJECT_DIR` and `HF_TOKEN`.

## Pipeline (scripts run from `hallucination-detector/`)

Scripts show **progress (tqdm)** and **elapsed time** by default so reviewers can see what is running. Use `--quiet` or `--no-progress` where supported to reduce output.

```bash
# 1. Download datasets (progress + time; first run downloads from HuggingFace)
python scripts/01_download_datasets.py

# 2. Preprocess to common schema
python scripts/02_preprocess.py --out data/processed

# 3. Extract hidden states (GPU; use facebook/opt-125m for quick tests)
python scripts/03_extract_features.py --model facebook/opt-125m --datasets halueval --max_samples 500

# 4. Train probe on reference dataset
python scripts/04_train_probe.py --train halueval --output results/probe_halueval.pkl --model_short opt-125m

# 5. Cross-dataset evaluation
python scripts/05_evaluate_transfer.py --probe results/probe_halueval.pkl --eval medhallu --model_short opt-125m

# 6. Ablations
python scripts/06_ablations.py --model_short opt-125m
```

## Config

Edit `configs/config.yaml`: model name, layers, pooling, feature flags, probe type. Use `facebook/opt-125m` and `load_in_4bit: false` for CPU/small GPU; use `meta-llama/Llama-3.1-8B` and `load_in_4bit: true` on H100/A100.

## Design decisions (documented)

1. **Linear probes** — Interpretable; weights show which directions in hidden space correlate with hallucination.
2. **Mahalanobis over Euclidean** — Accounts for covariance of LLM representations (anisotropic).
3. **Layer selection** — Later layers capture semantics; empirically tune which layers help most (see ablations).
4. **Transfer failure modes** — When AUROC drops on transfer: different hallucination types (factual vs logical), domain vocabulary shifting geometry, or subtle label definition differences across datasets.

## Results template

| Train → | HaluEval | MedHal | MedHallu | Med-HALT |
|---------|----------|--------|----------|----------|
| **HaluEval** | (in-domain) | ? | ? | ? |
| **MedHal** | ? | (in-domain) | ? | ? |
| **MedHallu** | ? | ? | (in-domain) | ? |

Fill with AUROC from `results/transfer_matrix.csv` after running the full transfer matrix.

## Tests

Run from the **hallucination-detector/** directory (where this README lives):

```bash
cd hallucination-detector

# Fast tests only (no network/downloads) — returns to prompt in a few seconds
pytest -m "not slow" -v

# Full suite (dataset loaders + extractor download from HuggingFace; can take 5–10 min)
# Use -s to see "Loading..." progress so the terminal doesn’t look stuck
pytest tests/ -v -s
```
