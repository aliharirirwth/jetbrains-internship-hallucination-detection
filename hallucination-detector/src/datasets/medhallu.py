from __future__ import annotations

from .base import BaseDataset, HallucinationSample

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None


class MedHalluDataset(BaseDataset):
    """MedHallu: medical QA hallucination benchmark (2025).

    Loads splits pqa_labeled and pqa_artificial. Each row has question, correct
    answer, and hallucinated answer; we emit two samples (label 0 and 1) per row.
    """

    name = "medhallu"

    def load(self) -> list[HallucinationSample]:
        if load_dataset is None:
            raise ImportError("Install 'datasets' to load MedHallu: pip install datasets")
        samples: list[HallucinationSample] = []
        for split_name in ("pqa_labeled", "pqa_artificial"):
            try:
                ds = load_dataset("UTAustin-AIHealth/MedHallu", split_name)
            except Exception:
                continue
            # Can be DatasetDict (splits) or single Dataset; Dataset has .column_names, DatasetDict doesn't
            try:
                from datasets import DatasetDict
                if isinstance(ds, DatasetDict):
                    for split in ds.keys():
                        for row in ds[split]:
                            self._add_row(self._row_to_dict(row), split, split_name, samples)
                else:
                    for row in ds:
                        self._add_row(self._row_to_dict(row), "train", split_name, samples)
            except Exception:
                continue
        return samples

    @staticmethod
    def _row_to_dict(row) -> dict:
        if hasattr(row, "keys") and callable(getattr(row, "get", None)):
            return row
        if hasattr(row, "__iter__") and not isinstance(row, (str, bytes)):
            return dict(row) if hasattr(row, "__len__") and len(row) else {}
        return {}

    def _add_row(self, row: dict, split: str, split_name: str, samples: list) -> None:
        if not row:
            return
        # MedHallu actual columns: Question, Ground Truth, Hallucinated Answer (with spaces)
        question = self._str(row, "Question", "question")
        gt = self._str(row, "Ground Truth", "GroundTruth", "ground_truth_answer", "correct_answer", "answer")
        hallu = self._str(row, "Hallucinated Answer", "HallucinatedAnswer", "hallucinated_answer", "hallucination_answer", "hallucination")
        if not question:
            return
        if gt and hallu:
            samples.append(
                HallucinationSample(
                    question=question,
                    answer=gt,
                    label=0,
                    dataset=self.name,
                    metadata={"split": split, "subset": split_name},
                )
            )
            samples.append(
                HallucinationSample(
                    question=question,
                    answer=hallu,
                    label=1,
                    dataset=self.name,
                    metadata={"split": split, "subset": split_name},
                )
            )
        else:
            answer = gt or hallu or self._str(row, "Answer", "answer")
            if not answer:
                return
            label = int(row.get("label", row.get("is_hallucination", 0)))
            samples.append(
                HallucinationSample(
                    question=question,
                    answer=answer,
                    label=label,
                    dataset=self.name,
                    metadata={"split": split, "subset": split_name},
                )
            )

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
