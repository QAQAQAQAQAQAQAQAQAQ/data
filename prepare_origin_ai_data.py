# -*- coding: utf-8 -*-
"""
Prepare Origin / Adobe Illustrator figure data for the coal-rock disaster prediction paper.

Purpose
-------
This script reconstructs:
  1) Strict prediction ensemble:
     validation-selected top-k ensemble from strict prediction models
  2) Update-12h dynamic warning ensemble:
     5-seed soft ensemble from update-12h models

Then it exports clean CSV files that can be imported into Origin, Excel,
Adobe Illustrator workflows, or any other plotting software.

Outputs
-------
origin_data/
  00_table_strict_vs_update12_metrics.csv
  01_classification_metrics_long.csv
  02_regression_mae_long.csv
  03_regression_r2_long.csv
  04_highrisk_roc_curve.csv
  05_confusion_strict_counts.csv
  06_confusion_strict_row_norm.csv
  07_confusion_update12_counts.csv
  08_confusion_update12_row_norm.csv
  09_scatter_peak_log.csv
  10_scatter_sum_log.csv
  11_scatter_mean_log.csv
  12_scatter_count_high_log.csv
  13_residual_peak_log.csv
  14_residual_sum_log.csv
  15_residual_mean_log.csv
  16_residual_count_high_log.csv
  figure_manifest.json

Recommended command
-------------------
python prepare_origin_ai_data.py ^
  --strict-dirs outputs_v3_probe_sumfocus_s42 outputs_v3_probe_sumfocus_s7 outputs_v3_probe_sumfocus_s123 outputs_v3_probe_sumfocus_s2026 outputs_v3_probe_sumfocus_s3407 outputs_v3_probe_clsboost_s42 outputs_v3_probe_clsboost_s7 outputs_v3_probe_clsboost_s123 outputs_v3_probe_clsboost_s2026 outputs_v3_probe_clsboost_s3407 outputs_v3_dualhead_sumfocus_s42 outputs_v3_dualhead_sumfocus_s7 outputs_v3_dualhead_sumfocus_s123 outputs_v3_dualhead_sumfocus_s2026 outputs_v3_dualhead_sumfocus_s3407 ^
  --strict-top-k 7 ^
  --strict-weighting softmax ^
  --strict-temp 0.05 ^
  --update12-dirs outputs_v3_dualhead_update12h_s42 outputs_v3_dualhead_update12h_s7 outputs_v3_dualhead_update12h_s123 outputs_v3_dualhead_update12h_s2026 outputs_v3_dualhead_update12h_s3407 ^
  --update12-weighting equal ^
  --split stratified_random_9_1 ^
  --mode ral_val ^
  --out-dir origin_data
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd

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
    "peak_log": "Peak energy",
    "sum_log": "Cumulative energy",
    "mean_log": "Mean energy",
    "count_high_log": "High-energy count",
}


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


def load_json(path: Path) -> Dict:
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
        m = load_json(metric_path)

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

    pred_df = pd.DataFrame({
        "sample_index": idx,
        "y_true": y_true,
        "y_pred": y_pred,
        "prob_0": prob[:, 0],
        "prob_1": prob[:, 1],
        "prob_2": prob[:, 2],
        "prob_3": prob[:, 3],
        "prob_high": prob_high,
    })
    for j, t in enumerate(targets):
        pred_df[f"true_{t}"] = y_reg_true[:, j]
        pred_df[f"pred_{t}"] = y_reg_pred[:, j]
        pred_df[f"residual_{t}"] = y_reg_pred[:, j] - y_reg_true[:, j]
        pred_df[f"abs_error_{t}"] = np.abs(y_reg_pred[:, j] - y_reg_true[:, j])

    pred_df.to_csv(out_dir / f"{prefix}_ensemble_predictions.csv", index=False, encoding="utf-8-sig")

    return {
        "name": prefix,
        "sample_index": idx,
        "y_true": y_true,
        "y_pred": y_pred,
        "prob": prob,
        "prob_high": prob_high,
        "targets": targets,
        "y_reg_true": y_reg_true,
        "y_reg_pred": y_reg_pred,
        "pred_df": pred_df,
    }


def compute_metrics(e: Dict) -> Dict[str, float]:
    y_true = e["y_true"]
    y_pred = e["y_pred"]
    prob_high = e["prob_high"]
    targets = e["targets"]
    y_reg_true = e["y_reg_true"]
    y_reg_pred = e["y_reg_pred"]

    recalls = recall_score(y_true, y_pred, labels=[0, 1, 2, 3], average=None, zero_division=0)
    y_hr_true = (y_true >= 2).astype(int)
    y_hr_pred = (y_pred >= 2).astype(int)
    abs_cls_err = np.abs(y_pred.astype(float) - y_true.astype(float))

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
        "Ordinal MAE": float(np.mean(abs_cls_err)),
        "Within-one-level Accuracy": float(np.mean(abs_cls_err <= 1)),
        "Severe-error Rate": float(np.mean(abs_cls_err >= 2)),
    }

    maes, r2s = [], []
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


def export_metrics(strict: Dict, update12: Dict, out_dir: Path) -> pd.DataFrame:
    strict_m = compute_metrics(strict)
    update_m = compute_metrics(update12)
    strict_m["Experiment"] = "Strict"
    update_m["Experiment"] = "Update-12h"

    df = pd.DataFrame([strict_m, update_m])
    df = df[["Experiment"] + [c for c in df.columns if c != "Experiment"]]
    df.to_csv(out_dir / "00_table_strict_vs_update12_metrics.csv", index=False, encoding="utf-8-sig")

    cls_metrics = ["Accuracy", "Macro-F1", "QWK", "High-risk Recall", "High-risk Precision", "High-risk F1", "High-risk AUC"]
    rows = []
    for _, r in df.iterrows():
        for m in cls_metrics:
            rows.append({
                "Experiment": r["Experiment"],
                "Metric": m,
                "Score": float(r[m]),
            })
    pd.DataFrame(rows).to_csv(out_dir / "01_classification_metrics_long.csv", index=False, encoding="utf-8-sig")

    mae_metrics = {
        "peak_log": "Peak energy",
        "sum_log": "Cumulative energy",
        "mean_log": "Mean energy",
        "count_high_log": "High-energy count",
    }
    rows = []
    for _, r in df.iterrows():
        for t, label in mae_metrics.items():
            rows.append({
                "Experiment": r["Experiment"],
                "Target": label,
                "Target_key": t,
                "MAE": float(r[f"MAE_{t}"]),
            })
    pd.DataFrame(rows).to_csv(out_dir / "02_regression_mae_long.csv", index=False, encoding="utf-8-sig")

    rows = []
    for _, r in df.iterrows():
        for t, label in mae_metrics.items():
            rows.append({
                "Experiment": r["Experiment"],
                "Target": label,
                "Target_key": t,
                "R2": float(r[f"R2_{t}"]),
            })
    pd.DataFrame(rows).to_csv(out_dir / "03_regression_r2_long.csv", index=False, encoding="utf-8-sig")

    return df


def export_roc(strict: Dict, update12: Dict, out_dir: Path) -> None:
    rows = []
    for label, e in [("Strict", strict), ("Update-12h", update12)]:
        y_hr = (e["y_true"] >= 2).astype(int)
        fpr, tpr, thr = roc_curve(y_hr, e["prob_high"])
        roc_auc = auc(fpr, tpr)
        for i in range(len(fpr)):
            rows.append({
                "Experiment": label,
                "FPR": float(fpr[i]),
                "TPR": float(tpr[i]),
                "Threshold": float(thr[i]) if np.isfinite(thr[i]) else "",
                "AUC": float(roc_auc),
            })
    pd.DataFrame(rows).to_csv(out_dir / "04_highrisk_roc_curve.csv", index=False, encoding="utf-8-sig")


def export_confusion(e: Dict, out_dir: Path, prefix: str, count_file: str, norm_file: str) -> None:
    cm = confusion_matrix(e["y_true"], e["y_pred"], labels=[0, 1, 2, 3])
    row_sum = np.maximum(cm.sum(axis=1, keepdims=True), 1)
    norm = cm / row_sum

    count_df = pd.DataFrame(cm, columns=["Pred_0", "Pred_1", "Pred_2", "Pred_3"])
    count_df.insert(0, "True_class", [0, 1, 2, 3])
    count_df.to_csv(out_dir / count_file, index=False, encoding="utf-8-sig")

    norm_df = pd.DataFrame(norm, columns=["Pred_0", "Pred_1", "Pred_2", "Pred_3"])
    norm_df.insert(0, "True_class", [0, 1, 2, 3])
    norm_df.to_csv(out_dir / norm_file, index=False, encoding="utf-8-sig")

    long_rows = []
    for i in range(4):
        for j in range(4):
            long_rows.append({
                "Experiment": prefix,
                "True_class": i,
                "Predicted_class": j,
                "Count": int(cm[i, j]),
                "Row_normalized": float(norm[i, j]),
                "Cell_label": f"{int(cm[i,j])}\n({norm[i,j]:.2f})",
            })
    pd.DataFrame(long_rows).to_csv(out_dir / f"confusion_{prefix.lower()}_long.csv", index=False, encoding="utf-8-sig")


def export_scatter_residual(strict: Dict, update12: Dict, out_dir: Path) -> None:
    targets = [t for t in TARGET_ORDER if t in strict["targets"] and t in update12["targets"]]
    ds = strict["pred_df"].set_index("sample_index")
    du = update12["pred_df"].set_index("sample_index")
    idx = sorted(set(ds.index.tolist()) & set(du.index.tolist()))

    for k, t in enumerate(targets, start=9):
        label = TARGET_LABELS.get(t, t)
        d = pd.DataFrame({
            "sample_index": idx,
            "true": ds.loc[idx, f"true_{t}"].to_numpy(dtype=float),
            "strict_pred": ds.loc[idx, f"pred_{t}"].to_numpy(dtype=float),
            "update12_pred": du.loc[idx, f"pred_{t}"].to_numpy(dtype=float),
            "strict_abs_error": ds.loc[idx, f"abs_error_{t}"].to_numpy(dtype=float),
            "update12_abs_error": du.loc[idx, f"abs_error_{t}"].to_numpy(dtype=float),
            "target": label,
            "target_key": t,
        })
        d.to_csv(out_dir / f"{k:02d}_scatter_{t}.csv", index=False, encoding="utf-8-sig")

        res = pd.DataFrame({
            "sample_index": idx,
            "strict_residual": ds.loc[idx, f"residual_{t}"].to_numpy(dtype=float),
            "update12_residual": du.loc[idx, f"residual_{t}"].to_numpy(dtype=float),
            "target": label,
            "target_key": t,
        })
        res.to_csv(out_dir / f"{k+4:02d}_residual_{t}.csv", index=False, encoding="utf-8-sig")


def write_manifest(out_dir: Path, metrics_df: pd.DataFrame) -> None:
    manifest = {
        "purpose": "Origin / Illustrator plotting data package for strict prediction vs update-12h dynamic warning.",
        "csv_files": sorted([p.name for p in out_dir.glob("*.csv")]),
        "recommended_origin_figures": [
            {
                "figure": "Classification metric comparison",
                "file": "01_classification_metrics_long.csv",
                "plot": "Grouped column/bar chart",
                "x": "Metric",
                "y": "Score",
                "group": "Experiment",
            },
            {
                "figure": "Regression MAE comparison",
                "file": "02_regression_mae_long.csv",
                "plot": "Grouped column/bar chart",
                "x": "Target",
                "y": "MAE",
                "group": "Experiment",
            },
            {
                "figure": "High-risk ROC curve",
                "file": "04_highrisk_roc_curve.csv",
                "plot": "Line plot",
                "x": "FPR",
                "y": "TPR",
                "group": "Experiment",
            },
            {
                "figure": "Regression scatter plots",
                "file": "09_scatter_*.csv",
                "plot": "Scatter plot",
                "x": "true",
                "y": "strict_pred and update12_pred",
            },
        ],
        "metrics_preview": metrics_df.round(4).to_dict(orient="records"),
    }
    (out_dir / "figure_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


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
    parser.add_argument("--out-dir", default="origin_data")
    args = parser.parse_args()

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

    metrics_df = export_metrics(strict, update12, out_dir)
    export_roc(strict, update12, out_dir)
    export_confusion(strict, out_dir, "Strict", "05_confusion_strict_counts.csv", "06_confusion_strict_row_norm.csv")
    export_confusion(update12, out_dir, "Update12", "07_confusion_update12_counts.csv", "08_confusion_update12_row_norm.csv")
    export_scatter_residual(strict, update12, out_dir)
    write_manifest(out_dir, metrics_df)

    print("\n=== Origin/AI data package created ===")
    print("Output directory:", out_dir)
    print("\nMetrics preview:")
    keep = ["Experiment", "Accuracy", "Macro-F1", "QWK", "High-risk Recall", "High-risk AUC", "Regression MAE", "Regression R2"]
    print(metrics_df[keep].round(4).to_string(index=False))
    print("\nCSV files:")
    for p in sorted(out_dir.glob("*.csv")):
        print(" ", p)


if __name__ == "__main__":
    main()
