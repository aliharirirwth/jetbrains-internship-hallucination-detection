from pathlib import Path

import numpy as np


def save_features(features: np.ndarray, path: str | Path) -> None:
    """Save feature array to .npy file.

    Args:
        features: Array to save.
        path: Output path (.npy); parent directories created if needed.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, features)


def load_features(path: str | Path) -> np.ndarray:
    """Load feature array from .npy file.

    Args:
        path: Path to .npy file.

    Returns:
        Loaded numpy array.
    """
    return np.load(path)
