# -*- coding: utf-8 -*-
"""
Detailed analysis for a validation-selected ensemble.

What it does:
    1. Build a top-k validation-ranked ensemble from model prediction CSVs.
    2. Save ensemble predictions.
    3. Report:
        - standard 4-class metrics
        - high-risk binary metrics (class >= 2)
        - within-one-level accuracy
        - severe-error rate
        - gray-zone / stable-subset analysis near class thresholds

Recommended for the current best balanced strategy:
    python -m py_compile analyze_val_selected_ensemble.py

    python analyze_val_selected_ensemble.py ^
      --data camvnet_event_multimodal_v3.pt ^
      --dirs outputs_v3_probe_sumfocus_s42 outputs_v3_probe_sumfocus_s7 outputs_v3_probe_sumfocus_s123 outputs_v3_probe_sumfocus_s2026 outputs_v3_probe_sumfocus_s3407 outputs_v3_probe_clsboost_s42 outputs_v3_probe_clsboost_s7 outputs_v3_probe_clsboost_s123 outputs_v3_probe_clsboost_s2026 outputs_v3_probe_clsboost_s3407 outputs_v3_dualhead_sumfocus_s42 outputs_v3_dualhead_sumfocus_s7 outputs_v3_dualhead_sumfocus_s123 outputs_v3_dualhead_sumfocus_s2026 outputs_v3_dualhead_sumfocus_s3407 ^
      --split stratified_random_9_1 ^
      --mode ral_val ^
      --top-k 7 ^
      --weighting softmax ^
      --temp 0.05 ^
      --out-dir outputs_v3_final_top7_weighted_analysis

Send me:
    === Ensemble composition ===
    === Full-test metrics ===
    === Gray-zone analysis ===
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


def safe_torch_load(path: Path) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


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
    metrics = metric_candidates[0]
    return pred, metrics


def load_metrics(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_pred(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "sample_index" not in df.columns or "y_true" not in df.columns:
        raise KeyError(f"{path} missing sample_index/y_true")
    prob_cols = [f"prob_{i}" for i in range(4)]
    if not all(c in df.columns for c in prob_cols):
        if "y_pred" not in df.columns:
            raise KeyError(f"{path} has no prob columns and no y_pred")
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
            name = c[len("y_reg_true_"):]
            if f"y_reg_pred_{name}" in df.columns:
                targets.append(name)
    order = ["peak_log", "sum_log", "mean_log", "count_high_log"]
    return [t for t in order if t in targets] + [t for t in targets if t not in order]


def softmax_weights(scores: np.ndarray, temp: float) -> np.ndarray:
    s = np.asarray(scores, dtype=float)
    s = s - np.max(s)
    z = np.exp(s / max(temp, 1e-9))
    return z / max(z.sum(), 1e-12)


def align_dfs(dfs: List[pd.DataFrame]) -> Tuple[np.ndarray, List[pd.DataFrame]]:
    common = set(dfs[0]["sample_index"].tolist())
    for df in dfs[1:]:
        common &= set(df["sample_index"].tolist())
    idx = np.array(sorted(common), dtype=int)
    out = [df.set_index("sample_index").loc[idx].reset_index() for df in dfs]
    return idx, out


def metrics_for_arrays(y_true: np.ndarray, y_pred: np.ndarray, prob_high: np.ndarray | None = None) -> Dict[str, float]:
    recalls = recall_score(y_true, y_pred, labels=[0, 1, 2, 3], average=None, zero_division=0)
    abs_err = np.abs(y_pred.astype(float) - y_true.astype(float))
    y_hr_true = (y_true >= 2).astype(int)
    y_hr_pred = (y_pred >= 2).astype(int)
    out = {
        "n": int(len(y_true)),
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
        "Ordinal MAE": float(np.mean(abs_err)),
        "Within-one-level Accuracy": float(np.mean(abs_err <= 1)),
        "Severe-error Rate(|err|>=2)": float(np.mean(abs_err >= 2)),
    }
    if prob_high is not None and len(np.unique(y_hr_true)) == 2:
        out["High-risk AUC"] = float(roc_auc_score(y_hr_true, prob_high))
    return out


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, targets: List[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    maes, r2s = [], []
    for j, t in enumerate(targets):
        yt, yp = y_true[:, j], y_pred[:, j]
        mae = float(mean_absolute_error(yt, yp))
        rmse = float(mean_squared_error(yt, yp, squared=False))
        r2 = float(r2_score(yt, yp))
        pearson = float(pd.Series(yt).corr(pd.Series(yp), method="pearson"))
        spearman = float(pd.Series(yt).corr(pd.Series(yp), method="spearman"))
        out[f"MAE_{t}"] = mae
        out[f"RMSE_{t}"] = rmse
        out[f"R2_{t}"] = r2
        out[f"Pearson_{t}"] = pearson
        out[f"Spearman_{t}"] = spearman
        maes.append(mae)
        r2s.append(r2)
    out["Regression MAE"] = float(np.mean(maes))
    out["Regression R2"] = float(np.mean(r2s))
    return out


def print_metrics_block(title: str, m: Dict[str, float]) -> None:
    print(f"\n=== {title} ===")
    preferred = [
        "n", "Accuracy", "Macro-F1", "Weighted-F1", "QWK",
        "Recall0", "Recall1", "Recall2", "Recall3",
        "High-risk Accuracy", "High-risk Recall", "High-risk Precision", "High-risk F1", "High-risk AUC",
        "Ordinal MAE", "Within-one-level Accuracy", "Severe-error Rate(|err|>=2)",
        "Regression MAE", "MAE_peak_log", "MAE_sum_log", "MAE_mean_log", "MAE_count_high_log",
        "R2_peak_log", "R2_sum_log", "R2_mean_log", "R2_count_high_log",
        "Spearman_peak_log", "Spearman_sum_log", "Spearman_mean_log", "Spearman_count_high_log",
    ]
    for k in preferred:
        if k in m:
            v = m[k]
            if isinstance(v, (int, np.integer)):
                print(f"{k}: {v}")
            else:
                print(f"{k}: {float(v):.6f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--dirs", nargs="+", required=True)
    parser.add_argument("--split", default="stratified_random_9_1")
    parser.add_argument("--mode", default="ral_val")
    parser.add_argument("--top-k", type=int, default=7)
    parser.add_argument("--weighting", choices=["equal", "softmax"], default="softmax")
    parser.add_argument("--temp", type=float, default=0.05)
    parser.add_argument("--gray-deltas", nargs="+", type=float, default=[0.10, 0.20, 0.30])
    parser.add_argument("--out-dir", default="outputs_v3_final_top7_weighted_analysis")
    args = parser.parse_args()

    root = Path.cwd()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    dfs = []
    for d0 in args.dirs:
        d = Path(d0)
        if not d.is_absolute():
            d = root / d
        pred_path, metric_path = find_pair(d, args.split, args.mode)
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
    rank.to_csv(out_dir / "model_validation_rank.csv", index=False, encoding="utf-8-sig")

    selected_rank = rank.head(args.top_k).copy()
    selected_original_indices = selected_rank.index.to_numpy(dtype=int)

    # rank index after sort is not original df index, so recover via pred_file match.
    pred_to_i = {r["pred_file"]: i for i, r in enumerate(records)}
    chosen_ids = [pred_to_i[p] for p in selected_rank["pred_file"].tolist()]

    chosen_dfs_raw = [dfs[i] for i in chosen_ids]
    common_idx, chosen_dfs = align_dfs(chosen_dfs_raw)

    y_true = chosen_dfs[0]["y_true"].to_numpy(dtype=int)
    for df in chosen_dfs[1:]:
        if not np.array_equal(y_true, df["y_true"].to_numpy(dtype=int)):
            raise ValueError("y_true mismatch after alignment.")

    scores = selected_rank["best_val_score"].to_numpy(dtype=float)
    if args.weighting == "equal":
        weights = np.ones(len(chosen_dfs), dtype=float) / float(len(chosen_dfs))
    else:
        weights = softmax_weights(scores, args.temp)

    selected_rank["ensemble_weight"] = weights
    selected_rank.to_csv(out_dir / "ensemble_composition.csv", index=False, encoding="utf-8-sig")

    prob_cols = [f"prob_{i}" for i in range(4)]
    prob_stack = np.stack([df[prob_cols].to_numpy(dtype=float) for df in chosen_dfs], axis=0)
    prob = np.sum(prob_stack * weights.reshape(-1, 1, 1), axis=0)
    y_pred = prob.argmax(axis=1)
    prob_high = prob[:, 2] + prob[:, 3]

    targets = detect_targets(chosen_dfs[0])
    true_cols = [f"y_reg_true_{t}" for t in targets]
    pred_cols = [f"y_reg_pred_{t}" for t in targets]
    y_reg_true = chosen_dfs[0][true_cols].to_numpy(dtype=float)
    pred_stack = np.stack([df[pred_cols].to_numpy(dtype=float) for df in chosen_dfs], axis=0)
    y_reg_pred = np.sum(pred_stack * weights.reshape(-1, 1, 1), axis=0)

    full_metrics = metrics_for_arrays(y_true, y_pred, prob_high=prob_high)
    full_metrics.update(regression_metrics(y_reg_true, y_reg_pred, targets))

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
    pred_out.to_csv(out_dir / "ensemble_predictions.csv", index=False, encoding="utf-8-sig")

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
    pd.DataFrame(cm, index=["true_0", "true_1", "true_2", "true_3"], columns=["pred_0", "pred_1", "pred_2", "pred_3"]).to_csv(
        out_dir / "confusion_matrix.csv", encoding="utf-8-sig"
    )

    # Gray-zone analysis based on true peak log distance to thresholds.
    data = safe_torch_load(Path(args.data) if Path(args.data).is_absolute() else root / args.data)
    thresholds = data.get("thresholds", {}) or {}
    threshold_logs = []
    for key in ["class_0_1", "class_1_2", "class_2_3"]:
        if key in thresholds:
            threshold_logs.append(float(np.log1p(float(thresholds[key]))))

    gray_rows = []
    if threshold_logs and "peak_log" in targets:
        peak_j = targets.index("peak_log")
        peak_true = y_reg_true[:, peak_j]
        dist = np.min(np.abs(peak_true.reshape(-1, 1) - np.asarray(threshold_logs).reshape(1, -1)), axis=1)
        pred_out["gray_distance_to_nearest_threshold"] = dist
        for delta in args.gray_deltas:
            gray_mask = dist <= float(delta)
            stable_mask = ~gray_mask
            for subset_name, mask in [(f"gray_delta_{delta}", gray_mask), (f"stable_delta_{delta}", stable_mask)]:
                if int(mask.sum()) < 2:
                    continue
                sub_m = metrics_for_arrays(y_true[mask], y_pred[mask], prob_high=prob_high[mask])
                sub_reg = regression_metrics(y_reg_true[mask], y_reg_pred[mask], targets)
                row = {"subset": subset_name, "delta": float(delta), **sub_m, **sub_reg}
                gray_rows.append(row)
        pred_out.to_csv(out_dir / "ensemble_predictions.csv", index=False, encoding="utf-8-sig")

    gray_df = pd.DataFrame(gray_rows)
    if len(gray_df):
        gray_df.to_csv(out_dir / "gray_zone_analysis.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame([full_metrics]).to_csv(out_dir / "full_test_metrics.csv", index=False, encoding="utf-8-sig")

    print("\n=== Ensemble composition ===")
    show = ["rank", "seed", "best_epoch", "best_val_score", "ensemble_weight", "test_accuracy_json", "test_macro_f1_json", "test_qwk_json", "test_regression_mae_json", "dir"]
    print(selected_rank[show].to_string(index=False))

    print_metrics_block("Full-test metrics", full_metrics)

    print("\n=== Gray-zone analysis ===")
    if len(gray_df):
        cols = [
            "subset", "n", "Accuracy", "Macro-F1", "QWK", "High-risk Recall",
            "Within-one-level Accuracy", "Severe-error Rate(|err|>=2)",
            "Regression MAE", "MAE_peak_log", "MAE_sum_log", "MAE_mean_log", "MAE_count_high_log",
        ]
        cols = [c for c in cols if c in gray_df.columns]
        print(gray_df[cols].to_string(index=False))
    else:
        print("No gray-zone analysis produced. Check thresholds and peak_log target.")

    print("\nSaved:")
    print(" ", out_dir / "ensemble_composition.csv")
    print(" ", out_dir / "ensemble_predictions.csv")
    print(" ", out_dir / "full_test_metrics.csv")
    print(" ", out_dir / "confusion_matrix.csv")
    if len(gray_df):
        print(" ", out_dir / "gray_zone_analysis.csv")


if __name__ == "__main__":
    main()
