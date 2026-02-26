#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None, help="Path to transfer_matrix_seeded.csv")
    ap.add_argument("--results-dir", default=None, help="Directory to save PNGs (default: script dir/../results)")
    ap.add_argument("--dpi", type=int, default=150)
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    results_dir = Path(args.results_dir) if args.results_dir else root / "results"
    csv_path = Path(args.csv) if args.csv else results_dir / "transfer_matrix_seeded.csv"

    if not csv_path.exists():
        print(f"CSV not found: {csv_path}. Run the evaluation notebook or add example CSV.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(csv_path)
    if df.empty or "auroc_mean" not in df.columns:
        print("No transfer matrix data to plot.", file=sys.stderr)
        sys.exit(1)

    # Coerce numeric; empty string -> nan
    df["auroc_mean"] = pd.to_numeric(df["auroc_mean"], errors="coerce")
    df["f1_mean"] = pd.to_numeric(df["f1_mean"], errors="coerce")

    auroc = df.pivot(index="train_dataset", columns="eval_dataset", values="auroc_mean")
    f1 = df.pivot(index="train_dataset", columns="eval_dataset", values="f1_mean")

    results_dir.mkdir(parents=True, exist_ok=True)

    # 1) Heatmaps (AUROC + F1 side by side)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    im = ax.imshow(auroc.values, cmap="RdYlGn", vmin=0.45, vmax=0.85, aspect="auto")
    ax.set_xticks(range(len(auroc.columns)))
    ax.set_yticks(range(len(auroc.index)))
    ax.set_xticklabels(auroc.columns, rotation=45, ha="right")
    ax.set_yticklabels(auroc.index)
    ax.set_title("AUROC (train → eval)")
    for i in range(auroc.shape[0]):
        for j in range(auroc.shape[1]):
            val = auroc.values[i, j]
            label = f"{val:.2f}" if pd.notna(val) else "—"
            ax.text(j, i, label, ha="center", va="center", fontsize=9)
    plt.colorbar(im, ax=ax, label="AUROC")

    ax = axes[1]
    im = ax.imshow(f1.values, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(f1.columns)))
    ax.set_yticks(range(len(f1.index)))
    ax.set_xticklabels(f1.columns, rotation=45, ha="right")
    ax.set_yticklabels(f1.index)
    ax.set_title("F1 (train → eval)")
    for i in range(f1.shape[0]):
        for j in range(f1.shape[1]):
            val = f1.values[i, j]
            label = f"{val:.2f}" if pd.notna(val) else "—"
            ax.text(j, i, label, ha="center", va="center", fontsize=9)
    plt.colorbar(im, ax=ax, label="F1")
    plt.tight_layout()
    out = results_dir / "transfer_heatmaps.png"
    plt.savefig(out, dpi=args.dpi, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")

    # 2) In-domain vs transfer bar chart
    in_domain, transfer, labels = [], [], []
    for train in auroc.index:
        if train not in auroc.columns:
            continue
        in_domain.append(auroc.loc[train, train])
        off_diag = auroc.loc[train].drop(train, errors="ignore").dropna()
        transfer.append(off_diag.mean() if len(off_diag) else np.nan)
        labels.append(train)
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(labels))
    w = 0.35
    ax.bar(x - w / 2, in_domain, w, label="In-domain AUROC", color="C0")
    ax.bar(x + w / 2, transfer, w, label="Avg transfer AUROC", color="C1")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("AUROC")
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.7)
    ax.legend()
    ax.set_title("In-domain vs transfer performance by train dataset")
    plt.tight_layout()
    out = results_dir / "transfer_in_domain_vs_transfer.png"
    plt.savefig(out, dpi=args.dpi, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
