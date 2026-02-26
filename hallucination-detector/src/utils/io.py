"""Save/load features and models."""
from pathlib import Path
from typing import Any

import numpy as np


def save_features(features: np.ndarray, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, features)


def load_features(path: str | Path) -> np.ndarray:
    return np.load(path)
