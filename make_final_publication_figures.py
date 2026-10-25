# -*- coding: utf-8 -*-
"""
Make publication-ready editable figures for the strict prediction and update-12h experiments.

Outputs editable vector figures:
    SVG: editable in Inkscape / Adobe Illustrator / PowerPoint
    PDF: suitable for LaTeX / journal submission
    PNG: high-resolution preview
    TIFF: optional high-resolution raster backup

The script reconstructs:
    1) Strict prediction: validation-selected top-k ensemble
    2) Update-12h: 5-seed soft ensemble

Recommended run:

python -m py_compile make_final_publication_figures.py

python make_final_publication_figures.py ^
  --strict-dirs outputs_v3_probe_sumfocus_s42 outputs_v3_probe_sumfocus_s7 outputs_v3_probe_sumfocus_s123 outputs_v3_probe_sumfocus_s2026 outputs_v3_probe_sumfocus_s3407 outputs_v3_probe_clsboost_s42 outputs_v3_probe_clsboost_s7 outputs_v3_probe_clsboost_s123 outputs_v3_probe_clsboost_s2026 outputs_v3_probe_clsboost_s3407 outputs_v3_dualhead_sumfocus_s42 outputs_v3_dualhead_sumfocus_s7 outputs_v3_dualhead_sumfocus_s123 outputs_v3_dualhead_sumfocus_s2026 outputs_v3_dualhead_sumfocus_s3407 ^
  --strict-top-k 7 ^
  --strict-weighting softmax ^
  --strict-temp 0.05 ^
  --update12-dirs outputs_v3_dualhead_update12h_s42 outputs_v3_dualhead_update12h_s7 outputs_v3_dualhead_update12h_s123 outputs_v3_dualhead_update12h_s2026 outputs_v3_dualhead_update12h_s3407 ^
  --update12-weighting equal ^
  --split stratified_random_9_1 ^
  --mode ral_val ^
  --out-dir outputs_publication_figures ^
  --formats svg pdf png

Send the printed sections:
    === Publication comparison metrics ===
    === Saved publication figures ===
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from sklearn.metrics import (
    accuracy_score,
    auc,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


TARGET_ORDER = ["peak_log", "sum_log", "mean_log", "count_high_log"]
TARGET_LABELS = {
    "peak_log": "Peak energy (log)",
    "sum_log": "Cumulative energy (log)",
    "mean_log": "Mean energy (log)",
    "count_high_log": "High-energy count (log)",
}


def cm_to_inch(cm: float) -> float:
    return cm / 2.54


def configure_matplotlib(font_size: float = 8.5) -> None:
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
        "figure.titlesize": font_size + 1,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "lines.linewidth": 1.2,
        "lines.markersize": 4.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.dpi": 600,
    })


def save_figure(fig: plt.Figure, out_base: Path, formats: List[str]) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fmt = fmt.lower().lstrip(".")
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.03}
        if fmt in {"png", "tif", "tiff"}:
            kwargs["dpi"] = 600
        fig.savefig(out_base.with_suffix("." + fmt), **kwargs)


def find_pair(d: Path, split: str, mode: str) -> Tuple[Path, Path]:
    pred_candidates = sorted(d.rglob(f"*{split}*{mode}*predictions*.csv"))
    if not pred_candidates:
        pred_candidates = sorted(d.rglob("*predictions*.csv"))
    if not pred_candidates:
        raise FileNotFoundError(f"No prediction CSV found in {d}")

    metric_candidates = sorted(d.rglob(f"*{split}*{mode}*metrics*.json"))
    if not metric_candidates:
        metric_candidates = sorted(d.rglob("*metrics*.json"))
    if not metric_candidates:
        raise FileNotFoundError(f"No metrics JSON found in {d}")

    return pred_candidates[0], metric_candidates[0]


def load_metrics(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_pred(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "sample_index" not in df.columns or "y_true" not in df.columns:
        raise KeyError(f"{path} must contain sample_index and y_true.")

    prob_cols = [f"prob_{i}" for i in range(4)]
    if not all(c in df.columns for c in prob_cols):
        if "y_pred" not in df.columns:
            raise KeyError(f"{path} missing probability columns and y_pred.")
        for i in range(4):
            df[f"prob_{i}"] = (df["y_pred"].astype(int) == i).astype(float)

    df["sample_index"] = df["sample_index"].astype(int)
    df["y_true"] = df["y_true"].astype(int)

    p = df[prob_cols].to_numpy(dtype=float)
    s = p.sum(axis=1)
    s[s <= 0] = 1.0
    df[prob_cols] = p / s.reshape(-1, 1)
    return df.sort_values("sample_index").reset_index(drop=True)


def detect_targets(df: pd.DataFrame) -> List[str]:
    targets = []
    for c in df.columns:
        if c.startswith("y_reg_true_"):
            t = c[len("y_reg_true_"):]
            if f"y_reg_pred_{t}" in df.columns:
                targets.append(t)
    return [t for t in TARGET_ORDER if t in targets] + [t for t in targets if t not in TARGET_ORDER]


def align_dfs(dfs: List[pd.DataFrame]) -> Tuple[np.ndarray, List[pd.DataFrame]]:
    common = set(dfs[0]["sample_index"].tolist())
    for df in dfs[1:]:
        common &= set(df["sample_index"].tolist())
    idx = np.array(sorted(common), dtype=int)
    aligned = [df.set_index("sample_index").loc[idx].reset_index() for df in dfs]
    return idx, aligned


def softmax_weights(scores: np.ndarray, temp: float) -> np.ndarray:
    s = np.asarray(scores, dtype=float)
    s = s - np.max(s)
    z = np.exp(s / max(float(temp), 1e-12))
    return z / max(z.sum(), 1e-12)


def build_ensemble(
    dirs: List[str],
    split: str,
    mode: str,
    weighting: str,
    top_k: Optional[int],
    temp: float,
    out_dir: Path,
    prefix: str,
) -> Dict:
    root = Path.cwd()
    records = []
    dfs = []

    for d0 in dirs:
        d = Path(d0)
        if not d.is_absolute():
            d = root / d
        pred_path, metric_path = find_pair(d, split, mode)
        m = load_metrics(metric_path)

        records.append({
            "dir": str(d),
            "seed": m.get("seed", m.get("args", {}).get("seed")),
            "best_epoch": m.get("best_epoch"),
            "best_val_score": float(m.get("best_val_score", np.nan)),
            "pred_file": str(pred_path),
            "metrics_file": str(metric_path),
        })
        dfs.append(load_pred(pred_path))

    rank = pd.DataFrame(records).sort_values("best_val_score", ascending=False).reset_index(drop=True)
    rank["rank"] = np.arange(1, len(rank) + 1)
    use = rank.head(top_k).copy() if top_k is not None and top_k > 0 else rank.copy()

    pred_to_i = {r["pred_file"]: i for i, r in enumerate(records)}
    chosen = [dfs[pred_to_i[p]] for p in use["pred_file"].tolist()]
    idx, chosen = align_dfs(chosen)

    y_true = chosen[0]["y_true"].to_numpy(dtype=int)
    for df in chosen[1:]:
        if not np.array_equal(y_true, df["y_true"].to_numpy(dtype=int)):
            raise ValueError(f"y_true mismatch in {prefix}")

    scores = use["best_val_score"].to_numpy(dtype=float)
    if weighting == "equal":
        weights = np.ones(len(chosen), dtype=float) / len(chosen)
    elif weighting == "softmax":
        weights = softmax_weights(scores, temp)
    else:
        raise ValueError(f"Unsupported weighting: {weighting}")

    use["ensemble_weight"] = weights
    rank.to_csv(out_dir / f"{prefix}_all_models_ranked.csv", index=False, encoding="utf-8-sig")
    use.to_csv(out_dir / f"{prefix}_ensemble_composition.csv", index=False, encoding="utf-8-sig")

    prob_cols = [f"prob_{i}" for i in range(4)]
    prob_stack = np.stack([df[prob_cols].to_numpy(dtype=float) for df in chosen], axis=0)
    prob = np.sum(prob_stack * weights.reshape(-1, 1, 1), axis=0)
    y_pred = prob.argmax(axis=1)
    prob_high = prob[:, 2] + prob[:, 3]

    targets = detect_targets(chosen[0])
    true_cols = [f"y_reg_true_{t}" for t in targets]
    pred_cols = [f"y_reg_pred_{t}" for t in targets]
    y_reg_true = chosen[0][true_cols].to_numpy(dtype=float)
    pred_stack = np.stack([df[pred_cols].to_numpy(dtype=float) for df in chosen], axis=0)
    y_reg_pred = np.sum(pred_stack * weights.reshape(-1, 1, 1), axis=0)

    pred_out = pd.DataFrame({
        "sample_index": idx,
        "y_true": y_true,
        "y_pred": y_pred,
        "prob_high": prob_high,
    })
    for k in range(4):
        pred_out[f"prob_{k}"] = prob[:, k]
    for j, t in enumerate(targets):
        pred_out[f"y_reg_true_{t}"] = y_reg_true[:, j]
        pred_out[f"y_reg_pred_{t}"] = y_reg_pred[:, j]
        pred_out[f"y_reg_abs_error_{t}"] = np.abs(y_reg_pred[:, j] - y_reg_true[:, j])
    pred_out.to_csv(out_dir / f"{prefix}_ensemble_predictions.csv", index=False, encoding="utf-8-sig")

    return {
        "prefix": prefix,
        "composition": use,
        "sample_index": idx,
        "y_true": y_true,
        "y_pred": y_pred,
        "prob": prob,
        "prob_high": prob_high,
        "targets": targets,
        "y_reg_true": y_reg_true,
        "y_reg_pred": y_reg_pred,
        "pred_df": pred_out,
    }


def compute_metrics(e: Dict) -> Dict[str, float]:
    y_true = e["y_true"]
    y_pred = e["y_pred"]
    prob_high = e["prob_high"]
    targets = e["targets"]
    y_reg_true = e["y_reg_true"]
    y_reg_pred = e["y_reg_pred"]

    recalls = recall_score(y_true, y_pred, labels=[0, 1, 2, 3], average=None, zero_division=0)
    abs_err = np.abs(y_pred.astype(float) - y_true.astype(float))
    y_hr_true = (y_true >= 2).astype(int)
    y_hr_pred = (y_pred >= 2).astype(int)

    out = {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Macro-F1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "Weighted-F1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "QWK": float(cohen_kappa_score(y_true, y_pred, weights="quadratic")),
        "Recall0": float(recalls[0]),
        "Recall1": float(recalls[1]),
        "Recall2": float(recalls[2]),
        "Recall3": float(recalls[3]),
        "High-risk Accuracy": float(accuracy_score(y_hr_true, y_hr_pred)),
        "High-risk Recall": float(recall_score(y_hr_true, y_hr_pred, zero_division=0)),
        "High-risk Precision": float(precision_score(y_hr_true, y_hr_pred, zero_division=0)),
        "High-risk F1": float(f1_score(y_hr_true, y_hr_pred, zero_division=0)),
        "High-risk AUC": float(roc_auc_score(y_hr_true, prob_high)) if len(np.unique(y_hr_true)) == 2 else np.nan,
        "Ordinal MAE": float(np.mean(abs_err)),
        "Within-one-level Accuracy": float(np.mean(abs_err <= 1)),
        "Severe-error Rate": float(np.mean(abs_err >= 2)),
    }

    maes = []
    r2s = []
    for j, t in enumerate(targets):
        yt = y_reg_true[:, j]
        yp = y_reg_pred[:, j]
        out[f"MAE_{t}"] = float(mean_absolute_error(yt, yp))
        out[f"RMSE_{t}"] = float(mean_squared_error(yt, yp, squared=False))
        out[f"R2_{t}"] = float(r2_score(yt, yp))
        out[f"Pearson_{t}"] = float(pd.Series(yt).corr(pd.Series(yp), method="pearson"))
        out[f"Spearman_{t}"] = float(pd.Series(yt).corr(pd.Series(yp), method="spearman"))
        maes.append(out[f"MAE_{t}"])
        r2s.append(out[f"R2_{t}"])

    out["Regression MAE"] = float(np.mean(maes))
    out["Regression R2"] = float(np.mean(r2s))
    return out


def plot_confusion_matrix(e: Dict, out_base: Path, title: str, formats: List[str]) -> None:
    cm = confusion_matrix(e["y_true"], e["y_pred"], labels=[0, 1, 2, 3])
    row_sum = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, np.maximum(row_sum, 1))

    fig, ax = plt.subplots(figsize=(cm_to_inch(8.6), cm_to_inch(7.4)))
    im = ax.imshow(cm_norm, aspect="equal")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Row-normalized value")

    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(title)
    ax.set_xticks(np.arange(4))
    ax.set_yticks(np.arange(4))
    ax.set_xticklabels(["0", "1", "2", "3"])
    ax.set_yticklabels(["0", "1", "2", "3"])

    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{cm[i, j]}\n({cm_norm[i, j]:.2f})",
                    ha="center", va="center", fontsize=7.2)

    fig.tight_layout()
    save_figure(fig, out_base, formats)
    plt.close(fig)


def plot_roc(strict: Dict, update12: Dict, out_base: Path, formats: List[str]) -> None:
    fig, ax = plt.subplots(figsize=(cm_to_inch(8.6), cm_to_inch(7.2)))

    for label, e in [("Strict", strict), ("Update-12h", update12)]:
        y_hr = (e["y_true"] >= 2).astype(int)
        fpr, tpr, _ = roc_curve(y_hr, e["prob_high"])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{label} (AUC={roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=0.9)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("High-risk ROC curves")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.legend(frameon=False, loc="lower right")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)
    fig.tight_layout()
    save_figure(fig, out_base, formats)
    plt.close(fig)


def plot_regression_scatter(e: Dict, target: str, out_base: Path, title: str, formats: List[str]) -> None:
    if target not in e["targets"]:
        return
    j = e["targets"].index(target)
    yt = e["y_reg_true"][:, j]
    yp = e["y_reg_pred"][:, j]

    fig, ax = plt.subplots(figsize=(cm_to_inch(8.6), cm_to_inch(7.2)))
    ax.scatter(yt, yp, s=14, alpha=0.72, edgecolors="none")
    lo = float(min(np.min(yt), np.min(yp)))
    hi = float(max(np.max(yt), np.max(yp)))
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=0.9)
    ax.set_xlabel(f"True {TARGET_LABELS.get(target, target)}")
    ax.set_ylabel(f"Predicted {TARGET_LABELS.get(target, target)}")
    ax.set_title(title)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)
    fig.tight_layout()
    save_figure(fig, out_base, formats)
    plt.close(fig)


def plot_regression_residual(e: Dict, target: str, out_base: Path, title: str, formats: List[str]) -> None:
    if target not in e["targets"]:
        return
    j = e["targets"].index(target)
    residual = e["y_reg_pred"][:, j] - e["y_reg_true"][:, j]

    fig, ax = plt.subplots(figsize=(cm_to_inch(8.6), cm_to_inch(6.6)))
    ax.hist(residual, bins=28, alpha=0.88)
    ax.axvline(0, linestyle="--", linewidth=0.9)
    ax.set_xlabel(f"Residual of {TARGET_LABELS.get(target, target)}")
    ax.set_ylabel("Frequency")
    ax.set_title(title)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.35)
    fig.tight_layout()
    save_figure(fig, out_base, formats)
    plt.close(fig)


def plot_strict_update_error_bar(strict: Dict, update12: Dict, out_base: Path, formats: List[str]) -> None:
    targets = [t for t in TARGET_ORDER if t in strict["targets"] and t in update12["targets"]]
    labels = [TARGET_LABELS.get(t, t).replace(" (log)", "") for t in targets]

    strict_mae = []
    update_mae = []
    for t in targets:
        js = strict["targets"].index(t)
        ju = update12["targets"].index(t)
        strict_mae.append(mean_absolute_error(strict["y_reg_true"][:, js], strict["y_reg_pred"][:, js]))
        update_mae.append(mean_absolute_error(update12["y_reg_true"][:, ju], update12["y_reg_pred"][:, ju]))

    x = np.arange(len(targets))
    width = 0.36

    fig, ax = plt.subplots(figsize=(cm_to_inch(10.4), cm_to_inch(6.4)))
    ax.bar(x - width / 2, strict_mae, width, label="Strict")
    ax.bar(x + width / 2, update_mae, width, label="Update-12h")
    ax.set_ylabel("MAE")
    ax.set_title("Regression error comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.legend(frameon=False)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.35)
    fig.tight_layout()
    save_figure(fig, out_base, formats)
    plt.close(fig)


def plot_metric_bar(metrics_df: pd.DataFrame, out_base: Path, formats: List[str]) -> None:
    metrics = ["Accuracy", "Macro-F1", "QWK", "High-risk Recall", "High-risk AUC"]
    labels = ["Accuracy", "Macro-F1", "QWK", "HR Recall", "HR AUC"]
    x = np.arange(len(metrics))
    width = 0.36

    strict_vals = metrics_df.loc[metrics_df["Experiment"] == "Strict", metrics].iloc[0].to_numpy(dtype=float)
    update_vals = metrics_df.loc[metrics_df["Experiment"] == "Update-12h", metrics].iloc[0].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(cm_to_inch(10.4), cm_to_inch(6.4)))
    ax.bar(x - width / 2, strict_vals, width, label="Strict")
    ax.bar(x + width / 2, update_vals, width, label="Update-12h")
    ax.set_ylim(0.65, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Classification performance comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.35)
    fig.tight_layout()
    save_figure(fig, out_base, formats)
    plt.close(fig)


def markdown_table(df: pd.DataFrame) -> str:
    keep = [
        "Experiment", "Accuracy", "Macro-F1", "QWK", "High-risk Recall", "High-risk Precision",
        "High-risk F1", "High-risk AUC", "Ordinal MAE", "Within-one-level Accuracy",
        "Severe-error Rate", "Regression MAE", "MAE_peak_log", "MAE_sum_log",
        "MAE_mean_log", "MAE_count_high_log", "R2_peak_log", "R2_sum_log",
        "R2_mean_log", "R2_count_high_log", "Regression R2",
    ]
    keep = [c for c in keep if c in df.columns]
    view = df[keep].copy()
    for c in view.columns:
        if c != "Experiment":
            view[c] = view[c].map(lambda x: f"{float(x):.4f}")
    return view.to_markdown(index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict-dirs", nargs="+", required=True)
    parser.add_argument("--strict-top-k", type=int, default=7)
    parser.add_argument("--strict-weighting", choices=["equal", "softmax"], default="softmax")
    parser.add_argument("--strict-temp", type=float, default=0.05)

    parser.add_argument("--update12-dirs", nargs="+", required=True)
    parser.add_argument("--update12-top-k", type=int, default=0)
    parser.add_argument("--update12-weighting", choices=["equal", "softmax"], default="equal")
    parser.add_argument("--update12-temp", type=float, default=0.05)

    parser.add_argument("--split", default="stratified_random_9_1")
    parser.add_argument("--mode", default="ral_val")
    parser.add_argument("--out-dir", default="outputs_publication_figures")
    parser.add_argument("--formats", nargs="+", default=["svg", "pdf", "png"], choices=["svg", "pdf", "png", "tif", "tiff"])
    parser.add_argument("--font-size", type=float, default=8.5)
    args = parser.parse_args()

    configure_matplotlib(font_size=args.font_size)

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = Path.cwd() / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    strict = build_ensemble(
        args.strict_dirs,
        split=args.split,
        mode=args.mode,
        weighting=args.strict_weighting,
        top_k=args.strict_top_k,
        temp=args.strict_temp,
        out_dir=out_dir,
        prefix="strict_top7",
    )

    update12_top_k = args.update12_top_k if args.update12_top_k > 0 else None
    update12 = build_ensemble(
        args.update12_dirs,
        split=args.split,
        mode=args.mode,
        weighting=args.update12_weighting,
        top_k=update12_top_k,
        temp=args.update12_temp,
        out_dir=out_dir,
        prefix="update12_5seed",
    )

    strict_m = compute_metrics(strict)
    update_m = compute_metrics(update12)
    strict_m["Experiment"] = "Strict"
    update_m["Experiment"] = "Update-12h"
    metrics_df = pd.DataFrame([strict_m, update_m])
    metrics_df = metrics_df[["Experiment"] + [c for c in metrics_df.columns if c != "Experiment"]]
    metrics_df.to_csv(out_dir / "table_strict_vs_update12_publication.csv", index=False, encoding="utf-8-sig")

    md = markdown_table(metrics_df)
    (out_dir / "table_strict_vs_update12_publication_markdown.txt").write_text(md, encoding="utf-8")

    # Figures.
    plot_confusion_matrix(strict, out_dir / "fig1_strict_confusion_matrix", "Strict prediction", args.formats)
    plot_confusion_matrix(update12, out_dir / "fig2_update12_confusion_matrix", "Update-12h dynamic warning", args.formats)
    plot_roc(strict, update12, out_dir / "fig3_highrisk_roc", args.formats)
    plot_metric_bar(metrics_df, out_dir / "fig4_classification_metric_comparison", args.formats)
    plot_strict_update_error_bar(strict, update12, out_dir / "fig5_regression_mae_comparison", args.formats)

    for t in TARGET_ORDER:
        plot_regression_scatter(
            update12,
            t,
            out_dir / f"fig6_update12_scatter_{t}",
            f"Update-12h: {TARGET_LABELS.get(t, t)}",
            args.formats,
        )
        plot_regression_residual(
            update12,
            t,
            out_dir / f"fig7_update12_residual_{t}",
            f"Update-12h residuals: {TARGET_LABELS.get(t, t)}",
            args.formats,
        )

    print("\n=== Publication comparison metrics ===")
    print(md)

    print("\n=== Saved publication figures ===")
    for p in sorted(out_dir.iterdir()):
        if p.suffix.lower() in {".svg", ".pdf", ".png", ".tif", ".tiff", ".csv", ".txt"}:
            print(" ", p)


if __name__ == "__main__":
    main()
