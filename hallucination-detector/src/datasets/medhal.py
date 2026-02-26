"""MedHal loader. Medical hallucination dataset (2025). Normalize to HallucinationSample."""

from __future__ import annotations

from .base import BaseDataset, HallucinationSample

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None


# MedHal (arXiv 2504.08596) — if not on HF yet, load() returns [] and the pipeline continues.
MEDHAL_HF_PATHS = [
    "GayaMehenni/MedHal",
    "GM07/MedHal",
    "medhal/MedHal",
]


class MedHalDataset(BaseDataset):
    """
    MedHal: medical domain hallucination detection.
    Label definition: 1 = hallucination (factual inconsistency), 0 = faithful.
    If the dataset uses different column names, we map them here.
    """

    name = "medhal"

    def load(self) -> list[HallucinationSample]:
        if load_dataset is None:
            raise ImportError("Install 'datasets' to load MedHal: pip install datasets")
        ds = None
        for path in MEDHAL_HF_PATHS:
            try:
                ds = load_dataset(path)
                break
            except Exception:
                continue
        if ds is None:
            # MedHal (arXiv 2504.08596) may not be published on HuggingFace yet; return empty so pipeline continues
            return []
        samples: list[HallucinationSample] = []
        for split in list(ds.keys()):
            for row in ds[split]:
                # MedHal schema (arxiv 2504.08596): Statement, Context (optional), Factual label (Yes/No), Explanation
                statement = self._str(
                    row,
                    "Statement", "statement", "claim", "text", "output", "response",
                    "answer", "Answer",
                )
                context = self._str(row, "Context", "context", "input", "question", "Question", "query", "source")
                if not statement:
                    continue
                question = context or statement[:500]  # use context as question, or truncate statement if no context
                answer = statement
                # Factual label: Yes = factual = 0, No = non-factual = hallucination = 1
                label_raw = row.get("Factual label") or row.get("factual_label") or row.get("label") or row.get("is_hallucination") or row.get("hallucination") or "Yes"
                if isinstance(label_raw, str):
                    label = 0 if label_raw.strip().lower() in ("yes", "true", "1", "factual") else 1
                else:
                    label = 1 if int(label_raw) == 1 else 0
                samples.append(
                    HallucinationSample(
                        question=question,
                        answer=answer,
                        label=label,
                        dataset=self.name,
                        metadata={"split": split},
                    )
                )
        return samples

    @staticmethod
    def _str(row: dict, *keys: str) -> str:
        for k in keys:
            v = row.get(k)
            if v is None:
                continue
            if isinstance(v, (list, tuple)):
                v = v[0] if v else ""
            s = str(v).strip()
            if s:
                return s
        return ""
