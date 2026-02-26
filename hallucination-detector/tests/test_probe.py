import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.metrics import compute_all_metrics
from src.models import HallucinationProbe


def test_probe_trains_and_predicts():
    rng = np.random.default_rng(42)
    n = 200
    X = rng.standard_normal((n, 10))
    y = (rng.random(n) > 0.5).astype(np.int64)
    probe = HallucinationProbe(probe_type="logistic", C=1.0, max_iter=1000)
    probe.fit(X, y)
    proba = probe.predict_proba(X)
    assert proba.shape == (n, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    metrics = compute_all_metrics(y, proba)
    assert metrics["auroc"] >= 0.45
    assert 0 <= metrics["f1"] <= 1
    assert 0 <= metrics["accuracy"] <= 1


def test_probe_save_load():
    rng = np.random.default_rng(43)
    X = rng.standard_normal((50, 5))
    y = (rng.random(50) > 0.5).astype(np.int64)
    probe = HallucinationProbe(probe_type="logistic", C=1.0)
    probe.fit(X, y)
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        path = f.name
    try:
        probe.save(path)
        loaded = HallucinationProbe.load(path)
        np.testing.assert_allclose(loaded.predict_proba(X), probe.predict_proba(X))
    finally:
        Path(path).unlink(missing_ok=True)
