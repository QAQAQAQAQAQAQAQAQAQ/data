# -*- coding: utf-8 -*-
"""
Create the main comparison table with RMSE and normalized error metrics.

Recommended run:
    python -m py_compile make_main_comparison_with_rmse_normalized.py

    python make_main_comparison_with_rmse_normalized.py ^
      --traditional-dir baseline_results_strict ^
      --deep-dir baseline_results_strict_dl ^
      --proposed-pred outputs_final_figures/strict_top7_weighted_ensemble_predictions.csv ^
      --out-dir main_comparison_results_rmse
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, cohen_kappa_score, f1_score, mean_absolute_error,
    mean_squared_error, precision_score, r2_score, recall_score, roc_auc_score
)

REG_TARGETS = ["peak_log", "sum_log", "mean_log", "count_high_log"]

MODEL_MAP = {
    "RBF-SVM": ("traditional", "rbf_svm"),
    "Random Forest": ("traditional", "rf"),
    "ExtraTrees": ("traditional", "extratrees"),
    "HistGradientBoosting": ("traditional", "histgb"),
    "MLP": ("traditional", "mlp"),
    "LSTM": ("deep", "lstm"),
    "GRU": ("deep", "gru"),
    "TCN": ("deep", "tcn"),
    "Transformer": ("deep", "transformer"),
    "AMNet": ("proposed", "proposed"),
}


def safe_auc(y_true: np.ndarray, prob_high: np.ndarray) -> float:
    y_hr = (y_true >= 2).astype(int)
    if len(np.unique(y_hr)) < 2:
        return float("nan")
    return float(roc_auc_score(y_hr, prob_high))


def prob_high_from_df(df: pd.DataFrame, y_pred: np.ndarray) -> np.ndarray:
    prob_cols = [f"prob_{i}" for i in range(4)]
    if all(c in df.columns for c in prob_cols):
        p = df[prob_cols].to_numpy(float)
        s = p.sum(axis=1, keepdims=True)
        s[s <= 0] = 1.0
        p = p / s
        return p[:, 2] + p[:, 3]
    if "prob_high" in df.columns:
        return df["prob_high"].to_numpy(float)
    return (y_pred >= 2).astype(float)


def get_reg_columns(df: pd.DataFrame, target: str):
    candidates = [
        (f"y_reg_true_{target}", f"y_reg_pred_{target}"),
        (f"true_{target}", f"pred_{target}"),
    ]
    for tc, pc in candidates:
        if tc in df.columns and pc in df.columns:
            return tc, pc
    return None, None


def compute_from_prediction_csv(pred_path: Path, model_name: str) -> Dict[str, float]:
    if not pred_path.exists():
        raise FileNotFoundError(f"Prediction file not found for {model_name}: {pred_path}")

    df = pd.read_csv(pred_path)
    y_true = df["y_true"].to_numpy(int)
    y_pred = df["y_pred"].to_numpy(int)
    prob_high = prob_high_from_df(df, y_pred)

    y_hr_true = (y_true >= 2).astype(int)
    y_hr_pred = (y_pred >= 2).astype(int)
    recalls = recall_score(y_true, y_pred, labels=[0, 1, 2, 3], average=None, zero_division=0)
    ordinal_err = np.abs(y_pred - y_true).astype(float)

    out: Dict[str, float] = {
        "Model": model_name,
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Macro-F1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "Weighted-F1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "QWK": float(cohen_kappa_score(y_true, y_pred, weights="quadratic")),
        "Recall0": float(recalls[0]),
        "Recall1": float(recalls[1]),
        "Recall2": float(recalls[2]),
        "Recall3": float(recalls[3]),
        "High-risk Recall": float(recall_score(y_hr_true, y_hr_pred, zero_division=0)),
        "High-risk Precision": float(precision_score(y_hr_true, y_hr_pred, zero_division=0)),
        "High-risk F1": float(f1_score(y_hr_true, y_hr_pred, zero_division=0)),
        "High-risk AUC": safe_auc(y_true, prob_high),
        "Ordinal MAE": float(np.mean(ordinal_err)),
        "Within-one-level Accuracy": float(np.mean(ordinal_err <= 1)),
        "Severe-error Rate": float(np.mean(ordinal_err >= 2)),
    }

    maes, rmses, r2s, nmaes, nrmses = [], [], [], [], []
    for t in REG_TARGETS:
        tc, pc = get_reg_columns(df, t)
        if tc is None:
            continue

        yt = df[tc].to_numpy(float)
        yp = df[pc].to_numpy(float)

        mae = float(mean_absolute_error(yt, yp))
        rmse = float(mean_squared_error(yt, yp, squared=False))
        r2 = float(r2_score(yt, yp))
        y_range = float(np.max(yt) - np.min(yt))
        nmae = mae / y_range if y_range > 1e-12 else float("nan")
        nrmse = rmse / y_range if y_range > 1e-12 else float("nan")

        out[f"MAE_{t}"] = mae
        out[f"RMSE_{t}"] = rmse
        out[f"R2_{t}"] = r2
        out[f"NMAE_range_{t}"] = nmae
        out[f"NRMSE_range_{t}"] = nrmse

        maes.append(mae)
        rmses.append(rmse)
        r2s.append(r2)
        nmaes.append(nmae)
        nrmses.append(nrmse)

    out["Regression MAE"] = float(np.nanmean(maes))
    out["Regression RMSE"] = float(np.nanmean(rmses))
    out["Regression R2"] = float(np.nanmean(r2s))
    out["Regression NMAE_range"] = float(np.nanmean(nmaes))
    out["Regression NRMSE_range"] = float(np.nanmean(nrmses))
    return out


def markdown_table(df: pd.DataFrame, detail: bool = False) -> str:
    if not detail:
        keep = [
            "Model", "Accuracy", "Macro-F1", "QWK", "High-risk AUC",
            "Ordinal MAE", "Regression MAE", "Regression RMSE",
            "Regression NMAE_range", "Regression NRMSE_range", "Regression R2",
        ]
    else:
        keep = ["Model"]
        for t in REG_TARGETS:
            keep += [f"MAE_{t}", f"RMSE_{t}", f"NMAE_range_{t}", f"NRMSE_range_{t}"]
    keep = [c for c in keep if c in df.columns]

    view = df[keep].copy()
    for c in keep:
        if c != "Model":
            view[c] = view[c].map(lambda v: "" if pd.isna(v) else f"{float(v):.4f}")
    return view.to_markdown(index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traditional-dir", default="baseline_results_strict")
    parser.add_argument("--deep-dir", default="baseline_results_strict_dl")
    parser.add_argument("--proposed-pred", default="outputs_final_figures/strict_top7_weighted_ensemble_predictions.csv")
    parser.add_argument("--out-dir", default="main_comparison_results_rmse")
    args = parser.parse_args()

    trad_dir = Path(args.traditional_dir)
    deep_dir = Path(args.deep_dir)
    proposed_pred = Path(args.proposed_pred)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for display_name, (kind, raw_name) in MODEL_MAP.items():
        if kind == "traditional":
            pred_path = trad_dir / f"baseline_predictions_{raw_name}.csv"
        elif kind == "deep":
            pred_path = deep_dir / f"baseline_predictions_{raw_name}.csv"
        else:
            pred_path = proposed_pred

        try:
            rows.append(compute_from_prediction_csv(pred_path, display_name))
        except Exception as e:
            print(f"Warning: skip {display_name}: {e}")

    if not rows:
        raise RuntimeError("No prediction files were loaded.")

    df = pd.DataFrame(rows)
    df["Model_order"] = df["Model"].map({m: i for i, m in enumerate(MODEL_MAP.keys())})
    df = df.sort_values("Model_order").drop(columns=["Model_order"]).reset_index(drop=True)

    df.to_csv(out_dir / "main_model_comparison_with_rmse_normalized.csv", index=False, encoding="utf-8-sig")

    compact = markdown_table(df, detail=False)
    detail = markdown_table(df, detail=True)

    (out_dir / "main_model_comparison_with_rmse_compact.md").write_text(compact, encoding="utf-8")
    (out_dir / "main_model_comparison_with_rmse_detail.md").write_text(detail, encoding="utf-8")

    reg_rows = []
    for _, r in df.iterrows():
        for m in ["Regression MAE", "Regression RMSE", "Regression NMAE_range", "Regression NRMSE_range"]:
            reg_rows.append({"Model": r["Model"], "Metric": m, "Value": float(r[m])})
    pd.DataFrame(reg_rows).to_csv(out_dir / "regression_error_metrics_long.csv", index=False, encoding="utf-8-sig")

    print("\n=== Main comparison with RMSE and normalized errors ===")
    print(compact)
    print("\n=== Per-target MAE/RMSE/normalized errors ===")
    print(detail)
    print("\nSaved:")
    print(" ", out_dir / "main_model_comparison_with_rmse_normalized.csv")
    print(" ", out_dir / "main_model_comparison_with_rmse_compact.md")
    print(" ", out_dir / "main_model_comparison_with_rmse_detail.md")
    print(" ", out_dir / "regression_error_metrics_long.csv")


if __name__ == "__main__":
    main()
