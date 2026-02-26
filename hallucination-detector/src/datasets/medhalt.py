"""Med-HALT loader (ACL/EMNLP 2023). openlifescienceai/Med-HALT, multiple subsets."""

from __future__ import annotations

from .base import BaseDataset, HallucinationSample

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None


# Med-HALT has multiple configs; we try reasoning-style ones that have Q/A and label
MEDHALT_CONFIGS = [
    "reasoning_FCT",
    "reasoning_fake",
    "reasoning_nota",
    "IR_abstract2pubmedlink",
]
# Cap rows per config so preprocessing finishes in reasonable time (~10k total)
MEDHALT_MAX_ROWS_PER_CONFIG = 2500


class MedHALTDataset(BaseDataset):
    """
    Med-HALT: medical benchmark with reasoning and IR tasks.
    We map configs that have question/answer and a hallucination indicator to HallucinationSample.
    """

    name = "medhalt"

    def load(self) -> list[HallucinationSample]:
        if load_dataset is None:
            raise ImportError("Install 'datasets' to load Med-HALT: pip install datasets")
        samples: list[HallucinationSample] = []
        for config in MEDHALT_CONFIGS:
            try:
                ds = load_dataset("openlifescienceai/Med-HALT", config)
            except Exception:
                continue
            splits = list(ds.keys()) if hasattr(ds, "keys") else []
            n_in_config = 0
            for split in splits:
                for row in ds[split]:
                    if n_in_config >= MEDHALT_MAX_ROWS_PER_CONFIG:
                        break
                    question = (row.get("question") or row.get("Question") or row.get("query") or row.get("input") or row.get("prompt") or "").strip()
                    # Med-HALT reasoning_*: has student_answer (model response) and correct_answer; no "answer" column
                    answer = (row.get("student_answer") or row.get("answer") or row.get("Answer") or row.get("output") or row.get("response") or row.get("completion") or "").strip()
                    if not question and row.get("context"):
                        question = str(row.get("context", ""))[:500]
                    if not question or not answer:
                        continue
                    # Label: 1 = hallucination (wrong answer), 0 = faithful (correct). Use correct_index vs student_index when present.
                    ci, si = row.get("correct_index"), row.get("student_index")
                    if ci is not None and si is not None:
                        label = 0 if (int(ci) == int(si)) else 1
                    else:
                        label_raw = row.get("label") or row.get("is_hallucination") or row.get("hallucination") or row.get("correct")
                        if label_raw is None:
                            label = 0
                        else:
                            label = 0 if (int(label_raw) == 0 or str(label_raw).lower() in ("false", "no", "correct")) else 1
                    samples.append(
                        HallucinationSample(
                            question=question,
                            answer=answer,
                            label=label,
                            dataset=self.name,
                            metadata={"config": config, "split": split},
                        )
                    )
                    n_in_config += 1
                if n_in_config >= MEDHALT_MAX_ROWS_PER_CONFIG:
                    break
        return samples
