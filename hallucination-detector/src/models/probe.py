"""Geometric probe: logistic regression or MLP on geometric features."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
except ImportError:
    LogisticRegression = None
    MLPClassifier = None
    StandardScaler = None


class HallucinationProbe:
    """Classifier on geometric feature vectors. Standardize features; fit probe; save/load with scaler."""

    def __init__(self, probe_type: str = "logistic", **kwargs: Any):
        if StandardScaler is None or LogisticRegression is None:
            raise ImportError("Install scikit-learn for HallucinationProbe")
        self.probe_type = probe_type
        self.kwargs = kwargs
        self.scaler = StandardScaler()
        self.clf: Any = None
        self.feature_importances_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        X_scaled = self.scaler.fit_transform(X)
        rng = self.kwargs.get("random_state", 42)
        if self.probe_type == "logistic":
            C = self.kwargs.get("C", 1.0)
            max_iter = self.kwargs.get("max_iter", 1000)
            self.clf = LogisticRegression(C=C, max_iter=max_iter, random_state=rng)
            self.clf.fit(X_scaled, y)
            self.feature_importances_ = np.abs(self.clf.coef_).ravel()
        elif self.probe_type == "mlp":
            self.clf = MLPClassifier(
                hidden_layer_sizes=(64, 32),
                max_iter=self.kwargs.get("max_iter", 500),
                random_state=rng,
            )
            self.clf.fit(X_scaled, y)
            self.feature_importances_ = None
        else:
            self.clf = LogisticRegression(C=1.0, max_iter=1000, random_state=rng)
            self.clf.fit(X_scaled, y)
            self.feature_importances_ = np.abs(self.clf.coef_).ravel()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        X_scaled = self.scaler.transform(X)
        return self.clf.predict_proba(X_scaled)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(
                {"scaler": self.scaler, "clf": self.clf, "probe_type": self.probe_type, "kwargs": self.kwargs},
                f,
            )

    @classmethod
    def load(cls, path: str | Path) -> "HallucinationProbe":
        path = Path(path)
        with path.open("rb") as f:
            data = pickle.load(f)
        obj = cls(probe_type=data["probe_type"], **data.get("kwargs", {}))
        obj.scaler = data["scaler"]
        obj.clf = data["clf"]
        return obj
