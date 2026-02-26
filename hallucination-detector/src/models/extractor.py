from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from ..datasets.base import HallucinationSample

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
except ImportError:
    torch = None
    AutoModelForCausalLM = None
    AutoTokenizer = None
    BitsAndBytesConfig = None


def _short_name(model_name: str) -> str:
    """Produce a short name for file naming (e.g. meta-llama/Llama-3.1-8B -> Llama-3.1-8B).

    Args:
        model_name: Full model name, possibly with org/ prefix.

    Returns:
        Sanitized string with slashes and spaces replaced by underscores.
    """
    return re.sub(r"[/\s]", "_", model_name.split("/")[-1] if "/" in model_name else model_name)


class HiddenStateExtractor:
    """Extract hidden states from a causal LM for (question, answer) pairs.

    Loads the model with output_hidden_states=True and returns selected layer
    representations with configurable pooling (mean, last_token, answer_mean).
    """

    def __init__(self, model_name: str, config: dict[str, Any]):
        if torch is None or AutoModelForCausalLM is None:
            raise ImportError("Install torch and transformers for HiddenStateExtractor")
        self.model_name = model_name
        self.config = config
        device = config.get("device", "cuda")
        load_in_4bit = config.get("load_in_4bit", False)
        self.pooling = config.get("pooling", "mean")

        bnb_config = None
        if load_in_4bit and BitsAndBytesConfig is not None:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto" if device == "cuda" else None,
            output_hidden_states=True,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if device != "cuda" and hasattr(self.model, "to"):
            self.model = self.model.to(device)
        self.model.eval()
        self.device = next(self.model.parameters()).device

    def _tokenize_qa(self, question: str, answer: str) -> tuple[Any, int, int]:
        """Tokenize "Question: {q}\nAnswer: {a}" and locate answer span.

        Args:
            question: Question text.
            answer: Answer text.

        Returns:
            Tuple (encoding_dict, answer_start_ix, answer_end_ix) with token indices for the answer span.
        """
        # Format: "Question: {q}\nAnswer: {a}" so we can find answer token range
        text = f"Question: {question}\nAnswer: {answer}"
        enc = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
            padding=False,
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        answer_part = f"\nAnswer: {answer}"
        answer_enc = self.tokenizer(
            answer_part,
            add_special_tokens=False,
            return_tensors="pt",
        )
        answer_len = answer_enc["input_ids"].shape[1]
        input_len = enc["input_ids"].shape[1]
        answer_start = max(0, input_len - answer_len - 1)
        answer_end = input_len
        return enc, answer_start, answer_end

    def extract(
        self,
        question: str,
        answer: str,
        layers: list[int],
        pooling: str | None = None,
    ) -> dict[int, np.ndarray]:
        """Extract hidden states for one (question, answer) and pool per layer.

        Args:
            question: Question text.
            answer: Answer text.
            layers: Layer indices to extract (0-based or negative from end).
            pooling: One of "mean", "last_token", "answer_mean"; overrides config if set.

        Returns:
            Dict mapping layer_index -> pooled feature vector (hidden_dim,).
        """
        pool = pooling or self.pooling
        enc, answer_start_ix, answer_end_ix = self._tokenize_qa(question, answer)
        with torch.no_grad():
            out = self.model(**enc)
        hidden_states = out.hidden_states  # tuple of (1, seq, dim)

        result: dict[int, np.ndarray] = {}
        n_layers = len(hidden_states)
        for li in layers:
            idx = n_layers + li if li < 0 else li
            if idx < 0 or idx >= n_layers:
                continue
            h = hidden_states[idx][0].cpu().float().numpy()  # (seq, dim)
            if pool == "mean":
                vec = np.mean(h, axis=0)
            elif pool == "last_token":
                vec = h[-1]
            elif pool == "answer_mean":
                if answer_end_ix > answer_start_ix:
                    vec = np.mean(h[answer_start_ix:answer_end_ix], axis=0)
                else:
                    vec = np.mean(h, axis=0)
            else:
                vec = np.mean(h, axis=0)
            result[li] = vec.astype(np.float32)
        return result

    def extract_batch(
        self,
        samples: list[HallucinationSample],
        layers: list[int] | None = None,
        batch_size: int = 16,
        save_path: str | Path | None = None,
        dataset_name: str | None = None,
        show_progress: bool = True,
    ) -> np.ndarray:
        """Process samples in batches and optionally save per-layer features to disk.

        Args:
            samples: List of HallucinationSample (question, answer, label).
            layers: Layer indices to extract; defaults from config.
            batch_size: Number of samples per forward pass.
            save_path: If set, write per-layer .npy arrays and labels incrementally.
            dataset_name: Used in saved filenames when save_path is set.
            show_progress: If True, show tqdm progress bar with ETA and elapsed time.

        Returns:
            Stacked hidden states for the last layer only (for compatibility);
            full per-layer arrays are written to save_path when provided.
        """
        layers = layers or self.config.get("layers_to_extract", [-1])
        save_path = Path(save_path) if save_path else None
        model_short = _short_name(self.model_name)
        ds_name = dataset_name or "dataset"

        all_layer_arrays: dict[int, list[np.ndarray]] = {li: [] for li in layers}
        labels_list: list[int] = []

        if not samples:
            raise ValueError("extract_batch called with 0 samples; load datasets first (01/02) and ensure they return samples.")

        n_batches = (len(samples) + batch_size - 1) // batch_size
        batch_range = range(0, len(samples), batch_size)
        if show_progress:
            batch_range = tqdm(
                batch_range,
                desc="extract",
                unit=" batch",
                total=n_batches,
                dynamic_ncols=True,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
            )
        for start in batch_range:
            batch = samples[start : start + batch_size]
            for s in batch:
                try:
                    layer_vecs = self.extract(s.question, s.answer, layers)
                    for li, vec in layer_vecs.items():
                        all_layer_arrays[li].append(vec)
                    labels_list.append(s.label)
                except Exception as e:
                    tqdm.write(f"Skip sample: {e}")
                    continue

            if save_path and (start + batch_size) % (batch_size * 10) == 0 and labels_list:
                save_path.mkdir(parents=True, exist_ok=True)
                for li in layers:
                    if all_layer_arrays[li]:
                        arr = np.stack(all_layer_arrays[li], axis=0)
                        np.save(save_path / f"{ds_name}_{model_short}_layer{li}_{self.pooling}.npy", arr)
                np.save(save_path / f"{ds_name}_labels.npy", np.array(labels_list))

        if not labels_list:
            raise ValueError("No samples were successfully extracted; check model and data.")

        # Final save
        if save_path:
            save_path.mkdir(parents=True, exist_ok=True)
            for li in layers:
                arr = np.stack(all_layer_arrays[li], axis=0)
                np.save(save_path / f"{ds_name}_{model_short}_layer{li}_{self.pooling}.npy", arr)
            np.save(save_path / f"{ds_name}_labels.npy", np.array(labels_list))

        # Return last layer stack for API compatibility
        last_li = layers[-1]
        return np.stack(all_layer_arrays[last_li], axis=0)
