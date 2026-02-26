from __future__ import annotations

from .base import BaseDataset, HallucinationSample

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None


class HaluEvalDataset(BaseDataset):
    """HaluEval QA: each row has question, right_answer, hallucinated_answer → two samples (0, 1)."""

    name = "halueval"

    def load(self) -> list[HallucinationSample]:
        if load_dataset is None:
            raise ImportError("Install 'datasets' to load HaluEval: pip install datasets")
        ds = load_dataset("pminervini/HaluEval", "qa_samples")
        samples: list[HallucinationSample] = []
        # qa_samples split is named "data"; columns: knowledge, question, answer, hallucination ("yes"/"no")
        splits = list(ds.keys()) if hasattr(ds, "keys") else ["data"]
        for split in splits:
            data = ds[split] if hasattr(ds, "__getitem__") and split in ds else ds
            for row in data:
                question = (row.get("question") or "").strip()
                if not question:
                    continue
                # Schema 1: two answers per row
                right = row.get("right_answer") or row.get("answer") or row.get("correct_answer") or ""
                hallu = row.get("hallucinated_answer") or row.get("hallucination_answer") or ""
                if right and hallu:
                    samples.append(
                        HallucinationSample(
                            question=question,
                            answer=right.strip(),
                            label=0,
                            dataset=self.name,
                            metadata={"split": split},
                        )
                    )
                    samples.append(
                        HallucinationSample(
                            question=question,
                            answer=hallu.strip(),
                            label=1,
                            dataset=self.name,
                            metadata={"split": split},
                        )
                    )
                    continue
                # Schema 2: single answer + binary hallucination (qa_samples: question, answer, hallucination)
                answer = (row.get("answer") or "").strip()
                if not answer:
                    continue
                hallu_flag = row.get("hallucination", row.get("label", "no"))
                if isinstance(hallu_flag, str):
                    label = 1 if hallu_flag.lower() in ("yes", "true", "1") else 0
                else:
                    label = int(hallu_flag)
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
