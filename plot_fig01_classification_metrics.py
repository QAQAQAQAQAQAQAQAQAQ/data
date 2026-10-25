# -*- coding: utf-8 -*-
"""
Figure 01: Classification performance comparison

Input:
    origin_data/01_classification_metrics_long.csv

Output:
    python_figures/fig01_classification_metrics_bar.svg
    python_figures/fig01_classification_metrics_bar.pdf
    python_figures/fig01_classification_metrics_bar.png
    python_figures/fig01_classification_metrics_bar.tiff

Run:
    python plot_fig01_classification_metrics.py

Optional:
    python plot_fig01_classification_metrics.py --data origin_data/01_classification_metrics_long.csv --out-dir python_figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def configure_style(font_size: float = 8.5) -> None:
    # Keep text editable in SVG/PDF.
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": font_size,
        "axes.labelsize": font_size,
        "axes.titlesize": font_size + 0.5,
        "xtick.labelsize": font_size - 0.5,
        "ytick.labelsize": font_size - 0.5,
        "legend.fontsize": font_size - 0.5,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.dpi": 600,
    })


def cm_to_inch(cm: float) -> float:
    return cm / 2.54


def save_all(fig: plt.Figure, out_base: Path) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    for suffix in [".svg", ".pdf", ".png", ".tiff"]:
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.04}
        if suffix in [".png", ".tiff"]:
            kwargs["dpi"] = 600
        fig.savefig(out_base.with_suffix(suffix), **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="origin_data/01_classification_metrics_long.csv")
    parser.add_argument("--out-dir", default="python_figures")
    parser.add_argument("--font-size", type=float, default=8.5)
    parser.add_argument("--ymin", type=float, default=0.70)
    parser.add_argument("--ymax", type=float, default=1.00)
    parser.add_argument("--show-values", action="store_true")
    args = parser.parse_args()

    configure_style(args.font_size)

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Cannot find data file: {data_path}")

    df = pd.read_csv(data_path)

    # Expected columns: Experiment, Metric, Score
    required = {"Experiment", "Metric", "Score"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns: {missing}. Existing columns: {list(df.columns)}")

    # Keep a clear order for paper.
    metric_order = [
        "Accuracy",
        "Macro-F1",
        "QWK",
        "High-risk Recall",
        "High-risk Precision",
        "High-risk F1",
        "High-risk AUC",
    ]
    exp_order = ["Strict", "Update-12h"]

    df["Metric"] = pd.Categorical(df["Metric"], categories=metric_order, ordered=True)
    df["Experiment"] = pd.Categorical(df["Experiment"], categories=exp_order, ordered=True)
    df = df.sort_values(["Metric", "Experiment"]).copy()

    wide = df.pivot(index="Metric", columns="Experiment", values="Score").loc[metric_order, exp_order]
    wide.to_csv(Path(args.out_dir) / "fig01_classification_metrics_wide.csv", encoding="utf-8-sig")

    labels = [
        "Accuracy",
        "Macro-F1",
        "QWK",
        "HR recall",
        "HR precision",
        "HR F1",
        "HR AUC",
    ]

    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(cm_to_inch(13.2), cm_to_inch(7.2)))

    bars1 = ax.bar(x - width / 2, wide["Strict"].to_numpy(float), width, label="Strict prediction")
    bars2 = ax.bar(x + width / 2, wide["Update-12h"].to_numpy(float), width, label="Update-12h")

    ax.set_ylabel("Score")
    ax.set_ylim(args.ymin, args.ymax)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_title("Classification performance comparison")

    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if args.show_values:
        for bars in [bars1, bars2]:
            for b in bars:
                h = b.get_height()
                ax.text(
                    b.get_x() + b.get_width() / 2,
                    h + 0.006,
                    f"{h:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=args.font_size - 1.5,
                    rotation=90,
                )

    fig.tight_layout()
    out_base = Path(args.out_dir) / "fig01_classification_metrics_bar"
    save_all(fig, out_base)
    plt.close(fig)

    print("\n=== Figure 01 saved ===")
    print("Data:", data_path)
    print("Wide table:", Path(args.out_dir) / "fig01_classification_metrics_wide.csv")
    print("SVG:", out_base.with_suffix(".svg"))
    print("PDF:", out_base.with_suffix(".pdf"))
    print("PNG:", out_base.with_suffix(".png"))
    print("TIFF:", out_base.with_suffix(".tiff"))


if __name__ == "__main__":
    main()
