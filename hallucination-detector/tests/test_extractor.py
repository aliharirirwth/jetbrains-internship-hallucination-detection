"""Test hidden state extractor with a small model (e.g. facebook/opt-125m)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.datasets.base import HallucinationSample
from src.models import HiddenStateExtractor


@pytest.fixture
def small_config():
    return {
        "device": "cpu",
        "load_in_4bit": False,
        "layers_to_extract": [-1, -2],
        "pooling": "mean",
    }


def test_extractor_with_small_model(small_config):
    try:
        extractor = HiddenStateExtractor("facebook/opt-125m", small_config)
    except Exception as e:
        pytest.skip(f"Could not load model: {e}")
    layers = [-1, -2]
    out = extractor.extract(
        question="What is the capital of France?",
        answer="Paris.",
        layers=layers,
        pooling="mean",
    )
    assert isinstance(out, dict)
    for li in layers:
        assert li in out
        vec = out[li]
        assert isinstance(vec, np.ndarray)
        assert vec.ndim == 1
        assert vec.dtype in (np.float32, np.float64)
    # Different inputs -> different outputs
    out2 = extractor.extract("Different question?", "Different answer.", layers=layers, pooling="mean")
    assert not np.allclose(out[-1], out2[-1])


def test_extractor_layer_indexing(small_config):
    try:
        extractor = HiddenStateExtractor("facebook/opt-125m", small_config)
    except Exception as e:
        pytest.skip(f"Could not load model: {e}")
    out = extractor.extract("Hello.", "World.", layers=[-1], pooling="last_token")
    assert -1 in out
    assert out[-1].shape == extractor.extract("Hi.", "Bye.", layers=[-1], pooling="mean")[-1].shape
