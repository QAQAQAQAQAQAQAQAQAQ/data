# -*- coding: utf-8 -*-
"""
Build leakage/oracle-hint variants of camvnet_event_multimodal_v3.pt.

Purpose:
    This script is ONLY for upper-bound / oracle / leakage analysis.
    It intentionally injects information derived from the future prediction target
    into the input feature vector. Do NOT use the resulting dataset as the official
    prediction dataset.

Leakage modes:
    1) future_reg_hint:
        Append future regression targets to X_ral_stat_v2:
            oracle_hint_peak_log
            oracle_hint_sum_log
            oracle_hint_mean_log
            oracle_hint_count_high_log

        Since y_cls is derived from future energy thresholds, this also leaks strong
        information for classification.

    2) optional class one-hot:
        --include-class-onehot appends y_cls one-hot to X_ral_stat_v2.
        This is a strong oracle-label leakage and should only be used to estimate
        an unrealistic upper bound.

Noise:
    --noise-std adds Gaussian noise in log-target space to future_reg_hint.
    Examples:
        noise_std=0.50 : mild/weak future hint
        noise_std=0.20 : medium future hint
        noise_std=0.00 : exact oracle target hint

Recommended runs:
    python -m py_compile build_v3_oracle_leak_features.py

    # Medium leakage: future target summaries with 0.20 log-unit noise
    python build_v3_oracle_leak_features.py --data camvnet_event_multimodal_v3.pt --out camvnet_event_multimodal_v3_leak_future_reg_n02.pt --noise-std 0.20 --seed 42 --out-dir outputs_oracle_leak_n02

    # Strong leakage: exact future target summaries
    python build_v3_oracle_leak_features.py --data camvnet_event_multimodal_v3.pt --out camvnet_event_multimodal_v3_leak_future_reg_exact.pt --noise-std 0.00 --seed 42 --out-dir outputs_oracle_leak_exact

    # Extreme leakage: exact future targets + true class one-hot
    python build_v3_oracle_leak_features.py --data camvnet_event_multimodal_v3.pt --out camvnet_event_multimodal_v3_leak_future_reg_cls_exact.pt --noise-std 0.00 --include-class-onehot --seed 42 --out-dir outputs_oracle_leak_cls_exact

Send me:
    === Oracle leakage feature audit ===
    === Saved leakage dataset ===
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch


DEFAULT_TARGET_NAMES = ["peak_log", "sum_log", "mean_log", "count_high_log"]


def safe_torch_load(path: Path) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def get_y_reg_multi(data: Dict[str, Any]) -> tuple[np.ndarray, List[str]]:
    metadata = data.get("metadata", {}) or {}

    if "y_reg_multi" in data:
        y = to_numpy(data["y_reg_multi"]).astype(np.float32)
        names = (
            metadata.get("reg_target_names")
            or metadata.get("y_reg_multi_names")
            or metadata.get("v3_reg_target_names")
            or DEFAULT_TARGET_NAMES[: y.shape[1]]
        )
        names = list(names)
        if len(names) != y.shape[1]:
            names = DEFAULT_TARGET_NAMES[: y.shape[1]]
        return y, names

    # Fallback: single-target only.
    if "y_reg" in data:
        y = to_numpy(data["y_reg"]).astype(np.float32).reshape(-1, 1)
        return y, ["peak_log"]

    raise KeyError("Dataset must contain y_reg_multi or y_reg.")


def corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return float("nan")
    if np.std(a[mask]) < 1e-12 or np.std(b[mask]) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a[mask], b[mask])[0, 1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="camvnet_event_multimodal_v3.pt")
    parser.add_argument("--out", type=str, default="camvnet_event_multimodal_v3_leak_future_reg_n02.pt")
    parser.add_argument("--noise-std", type=float, default=0.20, help="Gaussian noise std added to leaked future regression hints in log space.")
    parser.add_argument("--hint-dropout", type=float, default=0.0, help="Probability of setting an oracle hint value to 0. Same fixed mask for stored dataset.")
    parser.add_argument("--include-class-onehot", action="store_true", help="Append true y_cls one-hot as an extreme oracle-label leakage.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=str, default="outputs_oracle_leak")
    args = parser.parse_args()

    root = Path.cwd()
    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = root / data_path
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = root / out_path
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    data = safe_torch_load(data_path)
    metadata = data.get("metadata", {}) or {}

    if "X_ral_stat_v2" not in data:
        raise KeyError("Dataset missing X_ral_stat_v2.")

    x_stat = to_numpy(data["X_ral_stat_v2"]).astype(np.float32)
    y_cls = to_numpy(data["y_cls"]).astype(int).reshape(-1)
    y_reg_multi, target_names = get_y_reg_multi(data)
    n = x_stat.shape[0]

    if y_reg_multi.shape[0] != n:
        raise ValueError(f"y_reg_multi n={y_reg_multi.shape[0]} but X_ral_stat_v2 n={n}")

    hints = y_reg_multi.astype(np.float32).copy()
    if args.noise_std > 0:
        noise = rng.normal(loc=0.0, scale=float(args.noise_std), size=hints.shape).astype(np.float32)
        hints = hints + noise
        # log targets should not go below zero in these datasets.
        hints = np.maximum(hints, 0.0)

    if args.hint_dropout > 0:
        mask = rng.random(hints.shape) < float(args.hint_dropout)
        hints[mask] = 0.0

    add_features = [f"oracle_hint_future_{name}" for name in target_names]
    add_arrays = [hints]

    if args.include_class_onehot:
        cls_oh = np.zeros((n, 4), dtype=np.float32)
        cls_oh[np.arange(n), y_cls] = 1.0
        add_arrays.append(cls_oh)
        add_features += [f"oracle_hint_y_cls_onehot_{i}" for i in range(4)]

    add_mat = np.concatenate(add_arrays, axis=1).astype(np.float32)
    x_stat_new = np.concatenate([x_stat, add_mat], axis=1).astype(np.float32)

    stat_features = list(metadata.get("ral_stat_features", []))
    new_stat_features = stat_features + add_features

    audit_rows = []
    for j, name in enumerate(target_names):
        audit_rows.append({
            "hint": f"oracle_hint_future_{name}",
            "true_target": name,
            "noise_std": float(args.noise_std),
            "hint_dropout": float(args.hint_dropout),
            "include_class_onehot": bool(args.include_class_onehot),
            "true_mean": float(np.mean(y_reg_multi[:, j])),
            "hint_mean": float(np.mean(hints[:, j])),
            "mae_hint_vs_true": float(np.mean(np.abs(hints[:, j] - y_reg_multi[:, j]))),
            "corr_hint_vs_true": corr(hints[:, j], y_reg_multi[:, j]),
        })
    audit = pd.DataFrame(audit_rows)
    audit.to_csv(out_dir / "oracle_leakage_feature_audit.csv", index=False, encoding="utf-8-sig")

    # Extra classification leakage audit: can the leaked peak hint alone reconstruct the class using official thresholds?
    thresholds = data.get("thresholds", {}) or {}
    cls_from_peak_available = False
    cls_from_peak_acc = np.nan
    if y_reg_multi.shape[1] >= 1 and all(k in thresholds for k in ["class_0_1", "class_1_2", "class_2_3"]):
        t = np.array([
            np.log1p(float(thresholds["class_0_1"])),
            np.log1p(float(thresholds["class_1_2"])),
            np.log1p(float(thresholds["class_2_3"])),
        ], dtype=np.float32)
        peak_hint = hints[:, 0]
        cls_pred = np.zeros(n, dtype=int)
        cls_pred[peak_hint > t[0]] = 1
        cls_pred[peak_hint > t[1]] = 2
        cls_pred[peak_hint > t[2]] = 3
        cls_from_peak_acc = float(np.mean(cls_pred == y_cls))
        cls_from_peak_available = True

    data_new = dict(data)
    data_new["X_ral_stat_v2_original_before_oracle_leak"] = data["X_ral_stat_v2"]
    data_new["X_ral_stat_v2"] = torch.tensor(x_stat_new, dtype=torch.float32)

    meta_new = dict(metadata)
    meta_new["oracle_leakage_dataset"] = True
    meta_new["oracle_leakage_warning"] = (
        "This dataset intentionally injects future target-derived hints into input features. "
        "Use only for upper-bound/leakage analysis, not as the official prediction result."
    )
    meta_new["oracle_leakage_source_dataset"] = str(data_path)
    meta_new["oracle_leakage_noise_std"] = float(args.noise_std)
    meta_new["oracle_leakage_hint_dropout"] = float(args.hint_dropout)
    meta_new["oracle_leakage_include_class_onehot"] = bool(args.include_class_onehot)
    meta_new["oracle_leakage_added_features"] = add_features
    meta_new["oracle_leakage_cls_from_peak_hint_available"] = bool(cls_from_peak_available)
    meta_new["oracle_leakage_cls_from_peak_hint_accuracy"] = float(cls_from_peak_acc) if cls_from_peak_available else None
    meta_new["ral_stat_features"] = new_stat_features
    data_new["metadata"] = meta_new

    torch.save(data_new, out_path)

    print("\n=== Oracle leakage feature audit ===")
    print(audit.to_string(index=False))
    if cls_from_peak_available:
        print(f"\nClass reconstruction from leaked peak hint using official thresholds: accuracy={cls_from_peak_acc:.6f}")

    print("\nShapes:")
    print(f"  X_ral_stat_v2 old: {tuple(x_stat.shape)}")
    print(f"  X_ral_stat_v2 new: {tuple(x_stat_new.shape)}")
    print(f"  added feature count: {len(add_features)}")

    print("\n=== Saved leakage dataset ===")
    print(" ", out_path)
    print(" ", out_dir / "oracle_leakage_feature_audit.csv")


if __name__ == "__main__":
    main()
