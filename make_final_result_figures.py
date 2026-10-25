# -*- coding: utf-8 -*-
"""
Generate final figures and comparison tables for:
1) strict prediction: top-7 validation-weighted ensemble from 15 strict models
2) update-12h dynamic warning: 5-seed equal soft ensemble

Recommended run:
    python -m py_compile make_final_result_figures.py

    python make_final_result_figures.py ^
      --strict-dirs outputs_v3_probe_sumfocus_s42 outputs_v3_probe_sumfocus_s7 outputs_v3_probe_sumfocus_s123 outputs_v3_probe_sumfocus_s2026 outputs_v3_probe_sumfocus_s3407 outputs_v3_probe_clsboost_s42 outputs_v3_probe_clsboost_s7 outputs_v3_probe_clsboost_s123 outputs_v3_probe_clsboost_s2026 outputs_v3_probe_clsboost_s3407 outputs_v3_dualhead_sumfocus_s42 outputs_v3_dualhead_sumfocus_s7 outputs_v3_dualhead_sumfocus_s123 outputs_v3_dualhead_sumfocus_s2026 outputs_v3_dualhead_sumfocus_s3407 ^
      --strict-top-k 7 ^
      --strict-weighting softmax ^
      --strict-temp 0.05 ^
      --update12-dirs outputs_v3_dualhead_update12h_s42 outputs_v3_dualhead_update12h_s7 outputs_v3_dualhead_update12h_s123 outputs_v3_dualhead_update12h_s2026 outputs_v3_dualhead_update12h_s3407 ^
      --update12-weighting equal ^
      --split stratified_random_9_1 ^
      --mode ral_val ^
      --out-dir outputs_final_figures

Send me:
    === Final comparison metrics ===
    === Saved final figures ===
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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


def find_pair(d: Path, split: str, mode: str) -> Tuple[Path, Path]:
    pred_candidates = sorted(d.rglob(f"*{split}*{mode}*predictions*.csv"))
    if not pred_candidates:
        pred_candidates = sorted(d.rglob("*predictions*.csv"))
    if not pred_candidates:
        raise FileNotFoundError(f"No prediction csv found in {d}")
    pred = pred_candidates[0]

    metric_candidates = sorted(d.rglob(f"*{split}*{mode}*metrics*.json"))
    if not metric_candidates:
        metric_candidates = sorted(d.rglob("*metrics*.json"))
    if not metric_candidates:
        raise FileNotFoundError(f"No metrics json found in {d}")
    return pred, metric_candidates[0]


def load_metrics(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_pred(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "sample_index" not in df.columns or "y_true" not in df.columns:
        raise KeyError(f"{path} missing sample_index/y_true")
    prob_cols = [f"prob_{i}" for i in range(4)]
    if not all(c in df.columns for c in prob_cols):
        if "y_pred" not in df.columns:
            raise KeyError(f"{path} missing prob columns and y_pred")
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
    order = ["peak_log", "sum_log", "mean_log", "count_high_log"]
    return [t for t in order if t in targets] + [t for t in targets if t not in order]


def align_dfs(dfs: List[pd.DataFrame]) -> Tuple[np.ndarray, List[pd.DataFrame]]:
    common = set(dfs[0]["sample_index"].tolist())
    for df in dfs[1:]:
        common &= set(df["sample_index"].tolist())
    idx = np.array(sorted(common), dtype=int)
    out = [df.set_index("sample_index").loc[idx].reset_index() for df in dfs]
    return idx, out


def softmax_weights(scores: np.ndarray, temp: float) -> np.ndarray:
    s = np.asarray(scores, dtype=float)
    s = s - np.max(s)
    z = np.exp(s / max(temp, 1e-12))
    return z / max(z.sum(), 1e-12)


def build_ensemble(
    dirs: List[str],
    split: str,
    mode: str,
    weighting: str,
    top_k: int | None,
    temp: float,
    out_dir: Path,
    prefix: str,
) -> Dict:
    root = Path.cwd()
    records, dfs = [], []
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
            "test_accuracy_json": m.get("test", {}).get("accuracy", np.nan),
            "test_macro_f1_json": m.get("test", {}).get("macro_f1", np.nan),
            "test_qwk_json": m.get("test", {}).get("qwk", np.nan),
            "test_regression_mae_json": m.get("test", {}).get("regression_mae", np.nan),
            "pred_file": str(pred_path),
            "metrics_file": str(metric_path),
        })
        dfs.append(load_pred(pred_path))

    rank = pd.DataFrame(records).sort_values("best_val_score", ascending=False).reset_index(drop=True)
    rank["rank"] = np.arange(1, len(rank) + 1)
    rank_use = rank.head(top_k).copy() if top_k and top_k > 0 else rank.copy()

    pred_to_i = {r["pred_file"]: i for i, r in enumerate(records)}
    chosen_ids = [pred_to_i[p] for p in rank_use["pred_file"].tolist()]
    common_idx, chosen_dfs = align_dfs([dfs[i] for i in chosen_ids])

    y_true = chosen_dfs[0]["y_true"].to_numpy(dtype=int)
    for df in chosen_dfs[1:]:
        if not np.array_equal(y_true, df["y_true"].to_numpy(dtype=int)):
            raise ValueError(f"y_true mismatch in {prefix}")

    scores = rank_use["best_val_score"].to_numpy(dtype=float)
    if weighting == "equal":
        weights = np.ones(len(chosen_dfs), dtype=float) / float(len(chosen_dfs))
    elif weighting == "softmax":
        weights = softmax_weights(scores, temp)
    else:
        raise ValueError(f"Unknown weighting: {weighting}")

    rank_use["ensemble_weight"] = weights
    rank.to_csv(out_dir / f"{prefix}_all_models_ranked.csv", index=False, encoding="utf-8-sig")
    rank_use.to_csv(out_dir / f"{prefix}_ensemble_composition.csv", index=False, encoding="utf-8-sig")

    prob_cols = [f"prob_{i}" for i in range(4)]
    prob_stack = np.stack([df[prob_cols].to_numpy(dtype=float) for df in chosen_dfs], axis=0)
    prob = np.sum(prob_stack * weights.reshape(-1, 1, 1), axis=0)
    y_pred = prob.argmax(axis=1)
    prob_high = prob[:, 2] + prob[:, 3]

    targets = detect_targets(chosen_dfs[0])
    true_cols = [f"y_reg_true_{t}" for t in targets]
    pred_cols = [f"y_reg_pred_{t}" for t in targets]
    y_reg_true = chosen_dfs[0][true_cols].to_numpy(dtype=float)
    reg_stack = np.stack([df[pred_cols].to_numpy(dtype=float) for df in chosen_dfs], axis=0)
    y_reg_pred = np.sum(reg_stack * weights.reshape(-1, 1, 1), axis=0)

    pred_out = pd.DataFrame({
        "sample_index": common_idx,
        "y_true": y_true,
        "y_pred": y_pred,
        "prob_high": prob_high,
    })
    for c in range(4):
        pred_out[f"prob_{c}"] = prob[:, c]
    for j, t in enumerate(targets):
        pred_out[f"y_reg_true_{t}"] = y_reg_true[:, j]
        pred_out[f"y_reg_pred_{t}"] = y_reg_pred[:, j]
        pred_out[f"y_reg_abs_error_{t}"] = np.abs(y_reg_pred[:, j] - y_reg_true[:, j])
    pred_out.to_csv(out_dir / f"{prefix}_ensemble_predictions.csv", index=False, encoding="utf-8-sig")

    return {
        "prefix": prefix,
        "sample_index": common_idx,
        "y_true": y_true,
        "y_pred": y_pred,
        "prob": prob,
        "prob_high": prob_high,
        "targets": targets,
        "y_reg_true": y_reg_true,
        "y_reg_pred": y_reg_pred,
    }


def compute_metrics(e: Dict) -> Dict[str, float]:
    y_true, y_pred, prob_high = e["y_true"], e["y_pred"], e["prob_high"]
    targets, y_reg_true, y_reg_pred = e["targets"], e["y_reg_true"], e["y_reg_pred"]
    recalls = recall_score(y_true, y_pred, labels=[0, 1, 2, 3], average=None, zero_division=0)
    abs_err = np.abs(y_pred.astype(float) - y_true.astype(float))
    y_hr_true = (y_true >= 2).astype(int)
    y_hr_pred = (y_pred >= 2).astype(int)

    out = {
        "n": float(len(y_true)),
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
        "Severe-error Rate(|err|>=2)": float(np.mean(abs_err >= 2)),
    }

    maes, r2s = [], []
    for j, t in enumerate(targets):
        yt, yp = y_reg_true[:, j], y_reg_pred[:, j]
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


def plot_confusion_matrix(e: Dict, out_path: Path, title: str) -> None:
    cm = confusion_matrix(e["y_true"], e["y_pred"], labels=[0, 1, 2, 3])
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    im = ax.imshow(cm)
    ax.set_title(title)
    ax.set_xlabel("Predicted risk class")
    ax.set_ylabel("True risk class")
    ax.set_xticks(np.arange(4))
    ax.set_yticks(np.arange(4))
    for i in range(4):
        for j in range(4):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def plot_roc(strict_e: Dict, update_e: Dict, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    for label, e in [("Strict prediction", strict_e), ("Update-12h", update_e)]:
        y_hr = (e["y_true"] >= 2).astype(int)
        fpr, tpr, _ = roc_curve(y_hr, e["prob_high"])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, linewidth=1.8, label=f"{label} (AUC={roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.0)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("High-risk ROC curve")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def plot_regression_scatter(e: Dict, target: str, out_path: Path, title_prefix: str) -> None:
    if target not in e["targets"]:
        return
    j = e["targets"].index(target)
    yt, yp = e["y_reg_true"][:, j], e["y_reg_pred"][:, j]
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    ax.scatter(yt, yp, s=20, alpha=0.75)
    lo = float(min(np.min(yt), np.min(yp)))
    hi = float(max(np.max(yt), np.max(yp)))
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.0)
    ax.set_xlabel(f"True {target}")
    ax.set_ylabel(f"Predicted {target}")
    ax.set_title(f"{title_prefix}: true vs predicted {target}")
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def plot_regression_residual(e: Dict, target: str, out_path: Path, title_prefix: str) -> None:
    if target not in e["targets"]:
        return
    j = e["targets"].index(target)
    residual = e["y_reg_pred"][:, j] - e["y_reg_true"][:, j]
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    ax.hist(residual, bins=30, alpha=0.85)
    ax.axvline(0, linestyle="--", linewidth=1.0)
    ax.set_xlabel(f"Prediction residual for {target}")
    ax.set_ylabel("Frequency")
    ax.set_title(f"{title_prefix}: residual distribution of {target}")
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def plot_comparison_scatter(strict_e: Dict, update_e: Dict, target: str, out_path: Path) -> None:
    if target not in strict_e["targets"] or target not in update_e["targets"]:
        return
    js = strict_e["targets"].index(target)
    ju = update_e["targets"].index(target)
    ds = pd.DataFrame({
        "sample_index": strict_e["sample_index"],
        "true": strict_e["y_reg_true"][:, js],
        "strict_pred": strict_e["y_reg_pred"][:, js],
    }).set_index("sample_index")
    du = pd.DataFrame({
        "sample_index": update_e["sample_index"],
        "update_pred": update_e["y_reg_pred"][:, ju],
    }).set_index("sample_index")
    d = ds.join(du, how="inner").reset_index()
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    ax.scatter(d["true"], d["strict_pred"], s=18, alpha=0.65, label="Strict prediction")
    ax.scatter(d["true"], d["update_pred"], s=18, alpha=0.65, label="Update-12h")
    lo = float(min(d["true"].min(), d["strict_pred"].min(), d["update_pred"].min()))
    hi = float(max(d["true"].max(), d["strict_pred"].max(), d["update_pred"].max()))
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.0)
    ax.set_xlabel(f"True {target}")
    ax.set_ylabel(f"Predicted {target}")
    ax.set_title(f"Strict vs update-12h regression: {target}")
    ax.legend(loc="best")
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def markdown_table(metrics_df: pd.DataFrame) -> str:
    keep = [
        "Experiment", "Accuracy", "Macro-F1", "QWK", "High-risk Recall", "High-risk Precision",
        "High-risk F1", "High-risk AUC", "Ordinal MAE", "Within-one-level Accuracy",
        "Severe-error Rate(|err|>=2)", "Regression MAE", "MAE_peak_log", "MAE_sum_log",
        "MAE_mean_log", "MAE_count_high_log", "R2_peak_log", "R2_sum_log", "R2_mean_log",
        "R2_count_high_log", "Regression R2",
    ]
    keep = [c for c in keep if c in metrics_df.columns]
    view = metrics_df[keep].copy()
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
    parser.add_argument("--update12-weighting", choices=["equal", "softmax"], default="equal")
    parser.add_argument("--update12-top-k", type=int, default=0)
    parser.add_argument("--update12-temp", type=float, default=0.05)
    parser.add_argument("--split", default="stratified_random_9_1")
    parser.add_argument("--mode", default="ral_val")
    parser.add_argument("--out-dir", default="outputs_final_figures")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = Path.cwd() / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    strict = build_ensemble(
        args.strict_dirs, args.split, args.mode, args.strict_weighting,
        args.strict_top_k, args.strict_temp, out_dir, "strict_top7_weighted"
    )
    update_top_k = args.update12_top_k if args.update12_top_k > 0 else None
    update12 = build_ensemble(
        args.update12_dirs, args.split, args.mode, args.update12_weighting,
        update_top_k, args.update12_temp, out_dir, "update12_ensemble"
    )

    strict_m = compute_metrics(strict)
    update_m = compute_metrics(update12)
    strict_m["Experiment"] = "Strict top7 validation-weighted"
    update_m["Experiment"] = "Update-12h 5-seed ensemble"
    metrics_df = pd.DataFrame([strict_m, update_m])
    metrics_df = metrics_df[["Experiment"] + [c for c in metrics_df.columns if c != "Experiment"]]
    metrics_df.to_csv(out_dir / "strict_vs_update12_metrics.csv", index=False, encoding="utf-8-sig")

    md = markdown_table(metrics_df)
    (out_dir / "strict_vs_update12_metrics_markdown.txt").write_text(md, encoding="utf-8")

    plot_confusion_matrix(strict, out_dir / "strict_confusion_matrix.png", "Strict prediction confusion matrix")
    plot_confusion_matrix(update12, out_dir / "update12_confusion_matrix.png", "Update-12h confusion matrix")
    plot_roc(strict, update12, out_dir / "highrisk_roc_strict_vs_update12.png")

    targets = ["peak_log", "sum_log", "mean_log", "count_high_log"]
    for t in targets:
        plot_regression_scatter(update12, t, out_dir / f"update12_regression_scatter_{t}.png", "Update-12h")
        plot_regression_residual(update12, t, out_dir / f"update12_regression_residual_{t}.png", "Update-12h")
        plot_comparison_scatter(strict, update12, t, out_dir / f"strict_vs_update12_regression_scatter_{t}.png")

    print("\n=== Final comparison metrics ===")
    print(md)
    print("\n=== Saved final figures ===")
    files = [
        "strict_vs_update12_metrics.csv",
        "strict_vs_update12_metrics_markdown.txt",
        "strict_confusion_matrix.png",
        "update12_confusion_matrix.png",
        "highrisk_roc_strict_vs_update12.png",
    ]
    for t in targets:
        files += [
            f"update12_regression_scatter_{t}.png",
            f"update12_regression_residual_{t}.png",
            f"strict_vs_update12_regression_scatter_{t}.png",
        ]
    for f in files:
        print(" ", out_dir / f)


if __name__ == "__main__":
    main()
