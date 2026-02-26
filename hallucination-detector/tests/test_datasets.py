import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.datasets import HaluEvalDataset, MedHalluDataset
from src.datasets.base import HallucinationSample
from src.datasets.utils import validate_schema


@pytest.mark.slow
@pytest.mark.parametrize("loader_cls", [HaluEvalDataset, MedHalluDataset])
def test_loader_returns_samples(loader_cls):
    print(f"\n  Loading {loader_cls.__name__} from HuggingFace (may take 1-5 min)...", flush=True)
    loader = loader_cls()
    try:
        samples = loader.load()
    except Exception as e:
        pytest.skip(f"Dataset load failed (network/auth): {e}")
    assert isinstance(samples, list)
    for s in samples[:10]:
        assert isinstance(s, HallucinationSample)
        assert hasattr(s, "question") and hasattr(s, "answer") and hasattr(s, "label") and hasattr(s, "dataset")


@pytest.mark.slow
def test_halueval_schema():
    print("\n  Loading HaluEval from HuggingFace (may take 1-5 min)...", flush=True)
    loader = HaluEvalDataset()
    try:
        samples = loader.load()
    except Exception as e:
        pytest.skip(f"HaluEval load failed: {e}")
    if not samples:
        pytest.skip("No samples")
    for s in samples:
        assert isinstance(s.question, str)
        assert isinstance(s.answer, str)
        assert s.label in (0, 1)
        assert s.dataset == "halueval"
    results = validate_schema(samples, dataset_name="halueval")
    assert results["binary_labels"] is True
    assert results["no_empty_question"] is True
    assert results["no_empty_answer"] is True
