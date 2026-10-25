# -*- coding: utf-8 -*-
"""
Merge baseline results with the proposed strict model for the main comparison table.

Inputs:
    baseline_results_strict/baseline_metrics.csv
    baseline_results_strict_dl/baseline_metrics.csv

Output:
    main_comparison_results/main_model_comparison_all.csv
    main_comparison_results/main_model_comparison_selected.csv
    main_comparison_results/main_model_comparison_markdown.txt
    main_comparison_results/main_classification_metrics_long.csv
    main_comparison_results/main_regression_mae_long.csv

Run:
    python -m py_compile merge_baselines_with_proposed.py
    python merge_baselines_with_proposed.py ^
      --traditional baseline_results_strict/baseline_metrics.csv ^
      --deep baseline_results_strict_dl/baseline_metrics.csv ^
      --out-dir main_comparison_results
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


PROPOSED_STRICT_TOP7 = {
    "Model": "Proposed",
    "Feature set": "RAL+VAL",
    "Accuracy": 0.795455,
    "Macro-F1": 0.783389,
    "Weighted-F1": 0.796746,
    "QWK": 0.746871,
    "Recall0": 0.674419,
    "Recall1": 0.729167,
    "Recall2": 0.852941,
    "Recall3": 0.868852,
    "High-risk Accuracy": 0.877273,
    "High-risk Recall": 0.899225,
    "High-risk Precision": 0.892308,
    "High-risk F1": 0.895753,
    "High-risk AUC": 0.937133,
    "Ordinal MAE": 0.318182,
    "Within-one-level Accuracy": 0.909091,
    "Severe-error Rate": 0.090909,
    "MAE_peak_log": 1.243361,
    "MAE_sum_log": 1.148687,
    "MAE_mean_log": 1.101776,
    "MAE_count_high_log": 0.321874,
    "R2_peak_log": 0.405151,
    "R2_sum_log": 0.401565,
    "R2_mean_log": 0.376619,
    "R2_count_high_log": 0.385402,
}
PROPOSED_STRICT_TOP7["Regression MAE"] = float(np.mean([
    PROPOSED_STRICT_TOP7["MAE_peak_log"],
    PROPOSED_STRICT_TOP7["MAE_sum_log"],
    PROPOSED_STRICT_TOP7["MAE_mean_log"],
    PROPOSED_STRICT_TOP7["MAE_count_high_log"],
]))
PROPOSED_STRICT_TOP7["Regression R2"] = float(np.mean([
    PROPOSED_STRICT_TOP7["R2_peak_log"],
    PROPOSED_STRICT_TOP7["R2_sum_log"],
    PROPOSED_STRICT_TOP7["R2_mean_log"],
    PROPOSED_STRICT_TOP7["R2_count_high_log"],
]))


def clean_model_name(x: str) -> str:
    mapping = {
        "logistic": "Logistic Regression",
        "linear_svm": "Linear SVM",
        "rbf_svm": "RBF-SVM",
        "knn": "KNN",
        "rf": "Random Forest",
        "extratrees": "ExtraTrees",
        "gradboost": "Gradient Boosting",
        "histgb": "HistGradientBoosting",
        "mlp": "MLP",
        "torch_mlp": "Torch-MLP",
        "lstm": "LSTM",
        "gru": "GRU",
        "tcn": "TCN",
        "transformer": "Transformer",
        "Proposed": "Proposed",
    }
    return mapping.get(str(x), str(x))


def markdown_table(df: pd.DataFrame) -> str:
    keep = [
        "Model",
        "Accuracy",
        "Macro-F1",
        "QWK",
        "High-risk Recall",
        "High-risk AUC",
        "Ordinal MAE",
        "MAE_peak_log",
        "MAE_sum_log",
        "MAE_mean_log",
        "MAE_count_high_log",
        "Regression MAE",
        "Regression R2",
    ]
    keep = [c for c in keep if c in df.columns]
    view = df[keep].copy()
    for c in keep:
        if c != "Model":
            view[c] = view[c].map(lambda v: "" if pd.isna(v) else f"{float(v):.4f}")
    return view.to_markdown(index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traditional", default="baseline_results_strict/baseline_metrics.csv")
    parser.add_argument("--deep", default="baseline_results_strict_dl/baseline_metrics.csv")
    parser.add_argument("--out-dir", default="main_comparison_results")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dfs = []
    for p0 in [args.traditional, args.deep]:
        p = Path(p0)
        if p.exists():
            dfs.append(pd.read_csv(p))
        else:
            print("Warning: missing file:", p)

    if not dfs:
        raise FileNotFoundError("No baseline metrics CSV files found.")

    df = pd.concat(dfs, ignore_index=True, sort=False)
    df = pd.concat([df, pd.DataFrame([PROPOSED_STRICT_TOP7])], ignore_index=True, sort=False)
    df["Model"] = df["Model"].map(clean_model_name)

    selected_order = [
        "RBF-SVM",
        "Random Forest",
        "ExtraTrees",
        "HistGradientBoosting",
        "MLP",
        "LSTM",
        "GRU",
        "TCN",
        "Transformer",
        "Proposed",
    ]

    df["Model_order"] = df["Model"].apply(lambda m: selected_order.index(m) if m in selected_order else 999)
    all_df = df.sort_values(["Model_order", "Model"]).reset_index(drop=True)
    all_df.to_csv(out_dir / "main_model_comparison_all.csv", index=False, encoding="utf-8-sig")

    selected = all_df[all_df["Model"].isin(selected_order)].copy()
    selected = selected.sort_values("Model_order").drop(columns=["Model_order"], errors="ignore")
    selected.to_csv(out_dir / "main_model_comparison_selected.csv", index=False, encoding="utf-8-sig")

    md = markdown_table(selected)
    (out_dir / "main_model_comparison_markdown.txt").write_text(md, encoding="utf-8")

    cls_metrics = ["Accuracy", "Macro-F1", "QWK", "High-risk Recall", "High-risk AUC"]
    rows = []
    for _, r in selected.iterrows():
        for m in cls_metrics:
            if m in selected.columns and not pd.isna(r[m]):
                rows.append({"Model": r["Model"], "Metric": m, "Score": float(r[m])})
    pd.DataFrame(rows).to_csv(out_dir / "main_classification_metrics_long.csv", index=False, encoding="utf-8-sig")

    target_labels = {
        "peak_log": "Peak",
        "sum_log": "Sum",
        "mean_log": "Mean",
        "count_high_log": "High-count",
    }
    rows = []
    for _, r in selected.iterrows():
        for k, label in target_labels.items():
            col = f"MAE_{k}"
            if col in selected.columns and not pd.isna(r[col]):
                rows.append({"Model": r["Model"], "Target": label, "Target_key": k, "MAE": float(r[col])})
    pd.DataFrame(rows).to_csv(out_dir / "main_regression_mae_long.csv", index=False, encoding="utf-8-sig")

    print("\n=== Main model comparison table ===")
    print(md)
    print("\nSaved:")
    print(" ", out_dir / "main_model_comparison_all.csv")
    print(" ", out_dir / "main_model_comparison_selected.csv")
    print(" ", out_dir / "main_model_comparison_markdown.txt")
    print(" ", out_dir / "main_classification_metrics_long.csv")
    print(" ", out_dir / "main_regression_mae_long.csv")


if __name__ == "__main__":
    main()
