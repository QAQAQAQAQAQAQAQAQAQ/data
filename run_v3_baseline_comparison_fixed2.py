# -*- coding: utf-8 -*-
"""
Run baseline comparison experiments for CAMVNet / coal-rock dynamic disaster prediction.

Main purpose
------------
Compare the proposed model against traditional ML and lightweight DL baselines
under the SAME dataset split and SAME targets.

Default setting:
    data  = camvnet_event_multimodal_v3.pt
    split = stratified_random_9_1

Input features:
    --feature-set stat       : X_ral_stat_v2 only
    --feature-set ral        : flattened X_ral_v2 only
    --feature-set ral_stat   : flattened X_ral_v2 + X_ral_stat_v2

Targets:
    classification: y_cls, four classes 0/1/2/3
    regression: y_reg_multi if available; otherwise y_reg

Baselines:
    Traditional ML:
        LogisticRegression
        LinearSVM
        RBF-SVM
        KNN
        RandomForest
        ExtraTrees
        GradientBoosting
        HistGradientBoosting
        MLP

    Optional lightweight DL:
        MLP-Torch
        LSTM
        GRU
        TCN
        Transformer

Outputs:
    baseline_results/
        baseline_metrics.csv
        baseline_metrics_markdown.txt
        baseline_predictions_<model>.csv
        baseline_model_comparison_bar.csv

Recommended first run
---------------------
python -m py_compile run_v3_baseline_comparison.py

python run_v3_baseline_comparison.py ^
  --data camvnet_event_multimodal_v3.pt ^
  --split stratified_random_9_1 ^
  --feature-set ral_stat ^
  --out-dir baseline_results_strict ^
  --models logistic linear_svm rbf_svm knn rf extratrees gradboost histgb mlp ^
  --seed 42

Optional DL run:
python run_v3_baseline_comparison.py ^
  --data camvnet_event_multimodal_v3.pt ^
  --split stratified_random_9_1 ^
  --feature-set ral_stat ^
  --out-dir baseline_results_strict_dl ^
  --models torch_mlp lstm gru tcn transformer ^
  --epochs 120 ^
  --patience 20 ^
  --batch-size 64 ^
  --seed 42

For update-12h comparison:
python run_v3_baseline_comparison.py ^
  --data camvnet_event_multimodal_v3_update_12h.pt ^
  --split stratified_random_9_1 ^
  --feature-set ral_stat ^
  --out-dir baseline_results_update12h ^
  --models logistic linear_svm rbf_svm knn rf extratrees gradboost histgb mlp ^
  --seed 42
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.base import clone
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge, RidgeClassifier
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import ParameterGrid
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import LinearSVC, SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor

warnings.filterwarnings("ignore")


def safe_torch_load(path: Path) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _normalize_split_dict(sp: Dict[str, Any], n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize different split-key conventions to train/val/test index arrays."""
    def pick(keys: List[str]) -> np.ndarray:
        for k in keys:
            if k in sp:
                arr = to_numpy(sp[k]).astype(int).reshape(-1)
                # Some older time-ordered splits store [start, end] ranges.
                if arr.size == 2 and 0 <= arr[0] < arr[1] <= n:
                    return np.arange(arr[0], arr[1], dtype=int)
                return arr
        raise KeyError(f"Cannot find any split key among {keys}. Existing keys: {list(sp.keys())}")

    train_idx = pick(["train_idx", "train", "idx_train"])
    val_idx = pick(["val_idx", "valid_idx", "val", "valid", "idx_val"])
    test_idx = pick(["test_idx", "test", "idx_test"])

    for name, arr in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        if arr.size == 0:
            raise ValueError(f"{name} split is empty.")
        if arr.min() < 0 or arr.max() >= n:
            raise ValueError(f"{name} split index out of range: min={arr.min()}, max={arr.max()}, n={n}")
    return train_idx, val_idx, test_idx


def _stratified_split_indices(y: np.ndarray, seed: int, mode: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fallback that matches the original v2/v3 dataset builder split rule."""
    from sklearn.model_selection import train_test_split

    all_idx = np.arange(len(y), dtype=int)
    if mode == "stratified_random_9_1":
        train_val_idx, test_idx = train_test_split(
            all_idx,
            test_size=0.10,
            random_state=seed,
            shuffle=True,
            stratify=y,
        )
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=1.0 / 9.0,
            random_state=seed + 17,
            shuffle=True,
            stratify=y[train_val_idx],
        )
        return train_idx.astype(int), val_idx.astype(int), test_idx.astype(int)

    if mode == "stratified_random_7_1_2":
        train_val_idx, test_idx = train_test_split(
            all_idx,
            test_size=0.20,
            random_state=seed,
            shuffle=True,
            stratify=y,
        )
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=1.0 / 8.0,
            random_state=seed + 17,
            shuffle=True,
            stratify=y[train_val_idx],
        )
        return train_idx.astype(int), val_idx.astype(int), test_idx.astype(int)

    raise KeyError(f"Cannot reconstruct split mode: {mode}")


def get_split_indices(data: Dict[str, Any], split_name: str, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compatible split loader.

    The v3 training scripts use `split_idx_v2`, while this baseline script's
    first version only looked for `splits`. This function supports both and
    can also reconstruct the original stratified random split if split indices
    are not stored explicitly.
    """
    n = int(len(to_numpy(data["y_cls"]).reshape(-1)))

    # Current v2/v3 dataset convention.
    if "split_idx_v2" in data and split_name in data["split_idx_v2"]:
        return _normalize_split_dict(data["split_idx_v2"][split_name], n)

    # Other possible conventions.
    for key in ["splits", "split_idx", "split_indices"]:
        if key in data:
            raw = data[key]
            if isinstance(raw, dict) and split_name in raw and isinstance(raw[split_name], dict):
                return _normalize_split_dict(raw[split_name], n)
            if isinstance(raw, dict) and all(k in raw for k in ["train", "val", "test"]):
                return _normalize_split_dict(raw, n)

    # Last resort: reconstruct the split exactly as the original builder did.
    y = to_numpy(data["y_cls"]).astype(int).reshape(-1)
    train_idx, val_idx, test_idx = _stratified_split_indices(y, seed=seed, mode=split_name)
    print(
        f"[Split fallback] Dataset has no explicit split indices; reconstructed {split_name} "
        f"with seed={seed}: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}"
    )
    return train_idx, val_idx, test_idx


def get_reg_targets(data: Dict[str, Any], targets: List[str] | None = None) -> Tuple[np.ndarray, List[str]]:
    if "y_reg_multi" in data:
        y = to_numpy(data["y_reg_multi"]).astype(np.float32)
        names = ["peak_log", "sum_log", "mean_log", "count_high_log"][: y.shape[1]]
    elif "y_reg" in data:
        y = to_numpy(data["y_reg"]).astype(np.float32)
        if y.ndim == 1:
            y = y[:, None]
        names = ["peak_log"][: y.shape[1]]
    else:
        raise KeyError("Dataset missing regression target: y_reg_multi or y_reg")

    if targets:
        all_names = list(names)
        idx = []
        for t in targets:
            if t not in all_names:
                raise KeyError(f"Regression target {t!r} not available. Available: {all_names}")
            idx.append(all_names.index(t))
        y = y[:, idx]
        names = [all_names[i] for i in idx]

    return y, names


def make_features(data: Dict[str, Any], feature_set: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    parts = []
    meta: Dict[str, Any] = {}

    if feature_set in {"ral", "ral_stat"}:
        if "X_ral_v2" in data:
            x_ral = to_numpy(data["X_ral_v2"]).astype(np.float32)
        elif "X_ral" in data:
            x_ral = to_numpy(data["X_ral"]).astype(np.float32)
        else:
            raise KeyError("Dataset missing X_ral_v2 / X_ral")
        meta["X_ral_shape"] = tuple(x_ral.shape)
        parts.append(x_ral.reshape(x_ral.shape[0], -1))

    if feature_set in {"stat", "ral_stat"}:
        if "X_ral_stat_v2" in data:
            x_stat = to_numpy(data["X_ral_stat_v2"]).astype(np.float32)
        elif "X_stat" in data:
            x_stat = to_numpy(data["X_stat"]).astype(np.float32)
        else:
            raise KeyError("Dataset missing X_ral_stat_v2 / X_stat")
        meta["X_stat_shape"] = tuple(x_stat.shape)
        parts.append(x_stat.reshape(x_stat.shape[0], -1))

    if not parts:
        raise ValueError(f"Unsupported feature-set: {feature_set}")

    X = np.concatenate(parts, axis=1).astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    meta["X_shape"] = tuple(X.shape)
    return X, meta


def make_sequence_features(data: Dict[str, Any], use_stat_context: bool = True) -> Tuple[np.ndarray, np.ndarray | None]:
    if "X_ral_v2" in data:
        x_seq = to_numpy(data["X_ral_v2"]).astype(np.float32)
    elif "X_ral" in data:
        x_seq = to_numpy(data["X_ral"]).astype(np.float32)
    else:
        raise KeyError("Dataset missing X_ral_v2 / X_ral for DL sequence baselines.")

    x_seq = np.nan_to_num(x_seq, nan=0.0, posinf=1e6, neginf=-1e6)
    x_stat = None
    if use_stat_context:
        if "X_ral_stat_v2" in data:
            x_stat = to_numpy(data["X_ral_stat_v2"]).astype(np.float32)
        elif "X_stat" in data:
            x_stat = to_numpy(data["X_stat"]).astype(np.float32)
        if x_stat is not None:
            x_stat = np.nan_to_num(x_stat, nan=0.0, posinf=1e6, neginf=-1e6)
    return x_seq, x_stat


def safe_auc_high(y_true: np.ndarray, prob_high: np.ndarray) -> float:
    y_hr = (y_true >= 2).astype(int)
    if len(np.unique(y_hr)) < 2:
        return float("nan")
    return float(roc_auc_score(y_hr, prob_high))


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prob: np.ndarray | None,
    y_reg_true: np.ndarray,
    y_reg_pred: np.ndarray,
    reg_names: List[str],
) -> Dict[str, float]:
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    recalls = recall_score(y_true, y_pred, labels=[0, 1, 2, 3], average=None, zero_division=0)
    y_hr_true = (y_true >= 2).astype(int)
    y_hr_pred = (y_pred >= 2).astype(int)

    if prob is None:
        prob = np.zeros((len(y_true), 4), dtype=float)
        prob[np.arange(len(y_true)), np.clip(y_pred, 0, 3)] = 1.0
    prob_high = prob[:, 2] + prob[:, 3]

    cls_abs_err = np.abs(y_pred - y_true).astype(float)

    out: Dict[str, float] = {
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
        "High-risk AUC": safe_auc_high(y_true, prob_high),
        "Ordinal MAE": float(np.mean(cls_abs_err)),
        "Within-one-level Accuracy": float(np.mean(cls_abs_err <= 1)),
        "Severe-error Rate": float(np.mean(cls_abs_err >= 2)),
    }

    maes, rmses, r2s = [], [], []
    for j, name in enumerate(reg_names):
        yt = y_reg_true[:, j]
        yp = y_reg_pred[:, j]
        out[f"MAE_{name}"] = float(mean_absolute_error(yt, yp))
        out[f"RMSE_{name}"] = float(mean_squared_error(yt, yp, squared=False))
        out[f"R2_{name}"] = float(r2_score(yt, yp))
        out[f"Pearson_{name}"] = float(pd.Series(yt).corr(pd.Series(yp), method="pearson"))
        out[f"Spearman_{name}"] = float(pd.Series(yt).corr(pd.Series(yp), method="spearman"))
        maes.append(out[f"MAE_{name}"])
        rmses.append(out[f"RMSE_{name}"])
        r2s.append(out[f"R2_{name}"])

    out["Regression MAE"] = float(np.mean(maes))
    out["Regression RMSE"] = float(np.mean(rmses))
    out["Regression R2"] = float(np.mean(r2s))
    return out


def fit_predict_sklearn(
    name: str,
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    yr_train: np.ndarray,
    yr_val: np.ndarray,
    yr_test: np.ndarray,
    seed: int,
    tune_on_val: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Train sklearn classifier + regressor for a baseline name."""
    n_jobs = -1

    def build_pair(model_name: str, params: Dict[str, Any] | None = None):
        params = params or {}
        if model_name == "logistic":
            clf = make_pipeline(
                SimpleImputer(strategy="median"),
                RobustScaler(),
                LogisticRegression(max_iter=5000, class_weight="balanced", C=params.get("C", 1.0), solver="lbfgs", multi_class="auto"),
            )
            reg = make_pipeline(SimpleImputer(strategy="median"), RobustScaler(), Ridge(alpha=params.get("alpha", 1.0)))
        elif model_name == "linear_svm":
            clf = make_pipeline(
                SimpleImputer(strategy="median"),
                RobustScaler(),
                LinearSVC(max_iter=8000, class_weight="balanced", C=params.get("C", 1.0), dual=False),
            )
            reg = make_pipeline(SimpleImputer(strategy="median"), RobustScaler(), Ridge(alpha=params.get("alpha", 1.0)))
        elif model_name == "rbf_svm":
            clf = make_pipeline(
                SimpleImputer(strategy="median"),
                StandardScaler(),
                SVC(C=params.get("C", 3.0), gamma=params.get("gamma", "scale"), class_weight="balanced", probability=True),
            )
            reg = make_pipeline(
                SimpleImputer(strategy="median"),
                StandardScaler(),
                MultiOutputRegressor(SVR(C=params.get("reg_C", 3.0), gamma="scale", epsilon=0.05)),
            )
        elif model_name == "knn":
            k = params.get("n_neighbors", 9)
            clf = make_pipeline(SimpleImputer(strategy="median"), RobustScaler(), KNeighborsClassifier(n_neighbors=k, weights="distance"))
            reg = make_pipeline(SimpleImputer(strategy="median"), RobustScaler(), KNeighborsRegressor(n_neighbors=k, weights="distance"))
        elif model_name == "rf":
            clf = RandomForestClassifier(
                n_estimators=params.get("n_estimators", 500),
                max_depth=params.get("max_depth", None),
                min_samples_leaf=params.get("min_samples_leaf", 2),
                class_weight="balanced_subsample",
                random_state=seed,
                n_jobs=n_jobs,
            )
            reg = RandomForestRegressor(
                n_estimators=params.get("n_estimators", 500),
                max_depth=params.get("max_depth", None),
                min_samples_leaf=params.get("min_samples_leaf", 2),
                random_state=seed,
                n_jobs=n_jobs,
            )
        elif model_name == "extratrees":
            clf = ExtraTreesClassifier(
                n_estimators=params.get("n_estimators", 600),
                max_depth=params.get("max_depth", None),
                min_samples_leaf=params.get("min_samples_leaf", 2),
                class_weight="balanced",
                random_state=seed,
                n_jobs=n_jobs,
            )
            reg = ExtraTreesRegressor(
                n_estimators=params.get("n_estimators", 600),
                max_depth=params.get("max_depth", None),
                min_samples_leaf=params.get("min_samples_leaf", 2),
                random_state=seed,
                n_jobs=n_jobs,
            )
        elif model_name == "gradboost":
            clf = GradientBoostingClassifier(
                n_estimators=params.get("n_estimators", 250),
                learning_rate=params.get("learning_rate", 0.03),
                max_depth=params.get("max_depth", 2),
                random_state=seed,
            )
            reg = MultiOutputRegressor(
                GradientBoostingRegressor(
                    n_estimators=params.get("n_estimators", 250),
                    learning_rate=params.get("learning_rate", 0.03),
                    max_depth=params.get("max_depth", 2),
                    random_state=seed,
                )
            )
        elif model_name == "histgb":
            clf = HistGradientBoostingClassifier(
                learning_rate=params.get("learning_rate", 0.04),
                max_iter=params.get("max_iter", 250),
                l2_regularization=params.get("l2_regularization", 0.01),
                max_leaf_nodes=params.get("max_leaf_nodes", 31),
                random_state=seed,
            )
            reg = MultiOutputRegressor(
                HistGradientBoostingRegressor(
                    learning_rate=params.get("learning_rate", 0.04),
                    max_iter=params.get("max_iter", 250),
                    l2_regularization=params.get("l2_regularization", 0.01),
                    max_leaf_nodes=params.get("max_leaf_nodes", 31),
                    random_state=seed,
                )
            )
        elif model_name == "mlp":
            clf = make_pipeline(
                SimpleImputer(strategy="median"),
                StandardScaler(),
                MLPClassifier(
                    hidden_layer_sizes=params.get("hidden_layer_sizes", (128, 64)),
                    alpha=params.get("alpha", 1e-4),
                    learning_rate_init=params.get("lr", 1e-3),
                    max_iter=600,
                    early_stopping=True,
                    random_state=seed,
                ),
            )
            reg = make_pipeline(
                SimpleImputer(strategy="median"),
                StandardScaler(),
                MLPRegressor(
                    hidden_layer_sizes=params.get("hidden_layer_sizes", (128, 64)),
                    alpha=params.get("alpha", 1e-4),
                    learning_rate_init=params.get("lr", 1e-3),
                    max_iter=600,
                    early_stopping=True,
                    random_state=seed,
                ),
            )
        elif model_name == "ridge":
            clf = make_pipeline(SimpleImputer(strategy="median"), RobustScaler(), RidgeClassifier(alpha=params.get("alpha", 1.0)))
            reg = make_pipeline(SimpleImputer(strategy="median"), RobustScaler(), Ridge(alpha=params.get("alpha", 1.0)))
        else:
            raise ValueError(f"Unknown sklearn model: {model_name}")
        return clf, reg

    grids = {
        "logistic": [{"C": c, "alpha": a} for c in [0.3, 1.0, 3.0] for a in [0.3, 1.0, 3.0]],
        "linear_svm": [{"C": c, "alpha": a} for c in [0.3, 1.0, 3.0] for a in [0.3, 1.0, 3.0]],
        "rbf_svm": [{"C": c, "reg_C": c} for c in [1.0, 3.0, 10.0]],
        "knn": [{"n_neighbors": k} for k in [5, 9, 15]],
        "rf": [{"n_estimators": 500, "max_depth": d, "min_samples_leaf": leaf} for d in [None, 8, 14] for leaf in [1, 2, 4]],
        "extratrees": [{"n_estimators": 600, "max_depth": d, "min_samples_leaf": leaf} for d in [None, 8, 14] for leaf in [1, 2, 4]],
        "gradboost": [{"n_estimators": n, "learning_rate": lr, "max_depth": d} for n in [150, 250] for lr in [0.03, 0.05] for d in [2, 3]],
        "histgb": [{"max_iter": n, "learning_rate": lr, "l2_regularization": l2} for n in [150, 250] for lr in [0.03, 0.05] for l2 in [0.0, 0.01, 0.1]],
        "mlp": [
            {"hidden_layer_sizes": hs, "alpha": a, "lr": lr}
            for hs in [(128,), (128, 64), (256, 128)]
            for a in [1e-4, 1e-3]
            for lr in [5e-4, 1e-3]
        ],
        "ridge": [{"alpha": a} for a in [0.1, 0.3, 1.0, 3.0, 10.0]],
    }

    param_list = grids.get(name, [{}]) if tune_on_val else [{}]

    best = None
    best_payload = None

    for params in param_list:
        clf, reg = build_pair(name, params)
        t0 = time.time()
        clf.fit(X_train, y_train)
        reg.fit(X_train, yr_train)

        yv_pred = clf.predict(X_val).astype(int)
        if hasattr(clf, "predict_proba"):
            pv = clf.predict_proba(X_val)
            # Ensure 4 columns.
            if pv.shape[1] != 4:
                p4 = np.zeros((len(yv_pred), 4), dtype=float)
                classes = getattr(clf, "classes_", np.arange(pv.shape[1]))
                for j, c in enumerate(classes):
                    if int(c) < 4:
                        p4[:, int(c)] = pv[:, j]
                pv = p4
        else:
            pv = None
        yrv_pred = np.asarray(reg.predict(X_val), dtype=float)
        if yrv_pred.ndim == 1:
            yrv_pred = yrv_pred[:, None]

        m = compute_metrics(y_val, yv_pred, pv, yr_val, yrv_pred, [f"t{j}" for j in range(yr_val.shape[1])])
        score = m["Macro-F1"] + 0.35 * m["QWK"] + 0.25 * m["High-risk Recall"] - 0.03 * m["Regression MAE"]

        if best is None or score > best:
            best = score
            best_payload = {
                "params": params,
                "clf": clf,
                "reg": reg,
                "val_metrics": m,
                "fit_time": time.time() - t0,
            }

    assert best_payload is not None
    clf = best_payload["clf"]
    reg = best_payload["reg"]

    yt_pred = clf.predict(X_test).astype(int)
    if hasattr(clf, "predict_proba"):
        pt = clf.predict_proba(X_test)
        if pt.shape[1] != 4:
            p4 = np.zeros((len(yt_pred), 4), dtype=float)
            classes = getattr(clf, "classes_", np.arange(pt.shape[1]))
            for j, c in enumerate(classes):
                if int(c) < 4:
                    p4[:, int(c)] = pt[:, j]
            pt = p4
    else:
        pt = None

    yrt_pred = np.asarray(reg.predict(X_test), dtype=float)
    if yrt_pred.ndim == 1:
        yrt_pred = yrt_pred[:, None]

    info = {
        "best_params": best_payload["params"],
        "best_val_score": float(best),
        "val_metrics": best_payload["val_metrics"],
        "fit_time_sec": float(best_payload["fit_time"]),
    }
    return yt_pred, pt, yrt_pred, info


class MLPNet(nn.Module):
    def __init__(self, input_dim: int, hidden: int, n_reg: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.cls = nn.Linear(hidden, 4)
        self.reg = nn.Linear(hidden, n_reg)

    def forward(self, x, stat=None):
        h = self.net(x)
        return self.cls(h), self.reg(h)


class RNNNet(nn.Module):
    def __init__(self, input_dim: int, hidden: int, n_reg: int, kind: str, stat_dim: int = 0, dropout: float = 0.1):
        super().__init__()
        rnn_cls = nn.LSTM if kind == "lstm" else nn.GRU
        self.rnn = rnn_cls(input_dim, hidden, batch_first=True, num_layers=1, dropout=0.0)
        self.stat_proj = nn.Sequential(nn.Linear(stat_dim, hidden), nn.GELU()) if stat_dim > 0 else None
        fuse_dim = hidden * (2 if stat_dim > 0 else 1)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(fuse_dim, hidden), nn.GELU(), nn.Dropout(dropout))
        self.cls = nn.Linear(hidden, 4)
        self.reg = nn.Linear(hidden, n_reg)

    def forward(self, x, stat=None):
        out, _ = self.rnn(x)
        h = out[:, -1]
        if self.stat_proj is not None and stat is not None:
            h = torch.cat([h, self.stat_proj(stat)], dim=1)
        h = self.head(h)
        return self.cls(h), self.reg(h)


class TCNBlock(nn.Module):
    def __init__(self, ch: int, dilation: int, dropout: float):
        super().__init__()
        pad = dilation
        self.net = nn.Sequential(
            nn.Conv1d(ch, ch, kernel_size=3, padding=pad, dilation=dilation),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(ch, ch, kernel_size=3, padding=pad, dilation=dilation),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        y = self.net(x)
        # Padding may keep length same for k=3, pad=dilation.
        if y.shape[-1] != x.shape[-1]:
            y = y[..., : x.shape[-1]]
        return x + y


class TCNNet(nn.Module):
    def __init__(self, input_dim: int, hidden: int, n_reg: int, stat_dim: int = 0, dropout: float = 0.1):
        super().__init__()
        self.inp = nn.Conv1d(input_dim, hidden, kernel_size=1)
        self.blocks = nn.Sequential(TCNBlock(hidden, 1, dropout), TCNBlock(hidden, 2, dropout), TCNBlock(hidden, 4, dropout))
        self.stat_proj = nn.Sequential(nn.Linear(stat_dim, hidden), nn.GELU()) if stat_dim > 0 else None
        fuse_dim = hidden * (2 if stat_dim > 0 else 1)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(fuse_dim, hidden), nn.GELU())
        self.cls = nn.Linear(hidden, 4)
        self.reg = nn.Linear(hidden, n_reg)

    def forward(self, x, stat=None):
        z = self.inp(x.transpose(1, 2))
        z = self.blocks(z)
        h = z.mean(dim=-1)
        if self.stat_proj is not None and stat is not None:
            h = torch.cat([h, self.stat_proj(stat)], dim=1)
        h = self.head(h)
        return self.cls(h), self.reg(h)


class TransformerNet(nn.Module):
    def __init__(self, input_dim: int, hidden: int, n_reg: int, stat_dim: int = 0, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden)
        layer = nn.TransformerEncoderLayer(d_model=hidden, nhead=4, dim_feedforward=hidden * 2, dropout=dropout, batch_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, num_layers=2)
        self.stat_proj = nn.Sequential(nn.Linear(stat_dim, hidden), nn.GELU()) if stat_dim > 0 else None
        fuse_dim = hidden * (2 if stat_dim > 0 else 1)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(fuse_dim, hidden), nn.GELU())
        self.cls = nn.Linear(hidden, 4)
        self.reg = nn.Linear(hidden, n_reg)

    def forward(self, x, stat=None):
        z = self.proj(x)
        z = self.enc(z)
        h = z.mean(dim=1)
        if self.stat_proj is not None and stat is not None:
            h = torch.cat([h, self.stat_proj(stat)], dim=1)
        h = self.head(h)
        return self.cls(h), self.reg(h)


def standardize_train_val_test(X: np.ndarray, train_idx: np.ndarray, val_idx: np.ndarray, test_idx: np.ndarray):
    scaler = RobustScaler()
    Xtr = scaler.fit_transform(X[train_idx])
    Xva = scaler.transform(X[val_idx])
    Xte = scaler.transform(X[test_idx])
    return Xtr.astype(np.float32), Xva.astype(np.float32), Xte.astype(np.float32), scaler


def train_torch_model(
    model_name: str,
    X_flat: np.ndarray,
    X_seq: np.ndarray,
    X_stat: np.ndarray | None,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    y_cls: np.ndarray,
    y_reg: np.ndarray,
    reg_names: List[str],
    seed: int,
    epochs: int,
    patience: int,
    batch_size: int,
    hidden: int,
    dropout: float,
    lr: float,
    weight_decay: float,
    reg_weight: float,
    out_device: str = "auto",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if (out_device == "auto" and torch.cuda.is_available()) else "cpu")
    if out_device in {"cpu", "cuda"}:
        device = torch.device(out_device)

    if model_name == "torch_mlp":
        Xtr, Xva, Xte, scaler = standardize_train_val_test(X_flat, train_idx, val_idx, test_idx)
        S_tr = S_va = S_te = None
        model = MLPNet(Xtr.shape[1], hidden, y_reg.shape[1], dropout)
    else:
        # Standardize sequence feature channel-wise over train samples/timesteps.
        seq = X_seq.copy().astype(np.float32)
        tr_seq_flat = seq[train_idx].reshape(-1, seq.shape[-1])
        med = np.nanmedian(tr_seq_flat, axis=0, keepdims=True)
        q1 = np.nanpercentile(tr_seq_flat, 25, axis=0, keepdims=True)
        q3 = np.nanpercentile(tr_seq_flat, 75, axis=0, keepdims=True)
        scale = q3 - q1
        scale[scale < 1e-6] = 1.0
        seq = (seq - med.reshape(1, 1, -1)) / scale.reshape(1, 1, -1)
        seq = np.nan_to_num(seq, nan=0.0, posinf=10.0, neginf=-10.0).astype(np.float32)
        Xtr, Xva, Xte = seq[train_idx], seq[val_idx], seq[test_idx]

        if X_stat is not None:
            Str, Sva, Ste, stat_scaler = standardize_train_val_test(X_stat, train_idx, val_idx, test_idx)
        else:
            Str = Sva = Ste = None

        stat_dim = 0 if Str is None else Str.shape[1]
        if model_name == "lstm":
            model = RNNNet(Xtr.shape[-1], hidden, y_reg.shape[1], kind="lstm", stat_dim=stat_dim, dropout=dropout)
        elif model_name == "gru":
            model = RNNNet(Xtr.shape[-1], hidden, y_reg.shape[1], kind="gru", stat_dim=stat_dim, dropout=dropout)
        elif model_name == "tcn":
            model = TCNNet(Xtr.shape[-1], hidden, y_reg.shape[1], stat_dim=stat_dim, dropout=dropout)
        elif model_name == "transformer":
            model = TransformerNet(Xtr.shape[-1], hidden, y_reg.shape[1], stat_dim=stat_dim, dropout=dropout)
        else:
            raise ValueError(f"Unknown torch model: {model_name}")
        S_tr, S_va, S_te = Str, Sva, Ste

    model.to(device)

    def make_loader(Xd, Sd, ids, shuffle):
        xs = torch.tensor(Xd, dtype=torch.float32)
        yc = torch.tensor(y_cls[ids], dtype=torch.long)
        yr = torch.tensor(y_reg[ids], dtype=torch.float32)
        if Sd is None:
            sd = torch.zeros((len(ids), 1), dtype=torch.float32)
        else:
            sd = torch.tensor(Sd, dtype=torch.float32)
        return DataLoader(TensorDataset(xs, sd, yc, yr), batch_size=batch_size, shuffle=shuffle)

    train_loader = make_loader(Xtr, S_tr, train_idx, True)
    val_loader = make_loader(Xva, S_va, val_idx, False)
    test_loader = make_loader(Xte, S_te, test_idx, False)

    # Class weights.
    counts = np.bincount(y_cls[train_idx].astype(int), minlength=4).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    ce = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    mse = nn.SmoothL1Loss()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_state = None
    best_score = -1e18
    bad = 0
    best_epoch = -1
    best_val_metrics = None

    def eval_loader(loader, ids):
        model.eval()
        probs, preds, regs = [], [], []
        with torch.no_grad():
            for xb, sb, yc, yr in loader:
                xb, sb = xb.to(device), sb.to(device)
                logits, reg = model(xb, sb if (sb.shape[1] > 1 or model_name == "torch_mlp") else None)
                p = torch.softmax(logits, dim=1).detach().cpu().numpy()
                probs.append(p)
                preds.append(p.argmax(axis=1))
                regs.append(reg.detach().cpu().numpy())
        prob = np.concatenate(probs, axis=0)
        pred = np.concatenate(preds, axis=0).astype(int)
        regp = np.concatenate(regs, axis=0)
        return pred, prob, regp

    for ep in range(1, epochs + 1):
        model.train()
        for xb, sb, yc, yr in train_loader:
            xb, sb, yc, yr = xb.to(device), sb.to(device), yc.to(device), yr.to(device)
            opt.zero_grad(set_to_none=True)
            logits, reg = model(xb, sb if (sb.shape[1] > 1 or model_name == "torch_mlp") else None)
            loss = ce(logits, yc) + reg_weight * mse(reg, yr)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()

        if ep % 2 == 0 or ep == epochs:
            yp, pp, rp = eval_loader(val_loader, val_idx)
            vm = compute_metrics(y_cls[val_idx], yp, pp, y_reg[val_idx], rp, reg_names)
            score = vm["Macro-F1"] + 0.35 * vm["QWK"] + 0.25 * vm["High-risk Recall"] - 0.03 * vm["Regression MAE"]
            if score > best_score:
                best_score = score
                best_epoch = ep
                best_val_metrics = vm
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= patience:
                    break

    if best_state is not None:
        model.load_state_dict(best_state)

    yt_pred, pt, yrt_pred = eval_loader(test_loader, test_idx)
    info = {
        "best_epoch": int(best_epoch),
        "best_val_score": float(best_score),
        "val_metrics": best_val_metrics,
        "device": str(device),
    }
    return yt_pred, pt, yrt_pred, info


def write_predictions(
    out_path: Path,
    sample_idx: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prob: np.ndarray | None,
    y_reg_true: np.ndarray,
    y_reg_pred: np.ndarray,
    reg_names: List[str],
) -> None:
    df = pd.DataFrame({
        "sample_index": sample_idx,
        "y_true": y_true.astype(int),
        "y_pred": y_pred.astype(int),
    })
    if prob is not None:
        for c in range(4):
            df[f"prob_{c}"] = prob[:, c]
        df["prob_high"] = prob[:, 2] + prob[:, 3]
    for j, name in enumerate(reg_names):
        df[f"y_reg_true_{name}"] = y_reg_true[:, j]
        df[f"y_reg_pred_{name}"] = y_reg_pred[:, j]
        df[f"abs_error_{name}"] = np.abs(y_reg_true[:, j] - y_reg_pred[:, j])
    df.to_csv(out_path, index=False, encoding="utf-8-sig")


def markdown_table(df: pd.DataFrame) -> str:
    keep = [
        "Model", "Feature set", "Accuracy", "Macro-F1", "QWK", "High-risk Recall",
        "High-risk Precision", "High-risk F1", "High-risk AUC", "Ordinal MAE",
        "Regression MAE", "MAE_peak_log", "MAE_sum_log", "MAE_mean_log", "MAE_count_high_log",
        "Regression R2",
    ]
    keep = [c for c in keep if c in df.columns]
    view = df[keep].copy()
    for c in view.columns:
        if c not in {"Model", "Feature set"}:
            view[c] = view[c].map(lambda x: "" if pd.isna(x) else f"{float(x):.4f}")
    return view.to_markdown(index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="camvnet_event_multimodal_v3.pt")
    parser.add_argument("--split", default="stratified_random_9_1")
    parser.add_argument("--feature-set", choices=["stat", "ral", "ral_stat"], default="ral_stat")
    parser.add_argument("--out-dir", default="baseline_results")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["logistic", "linear_svm", "rbf_svm", "knn", "rf", "extratrees", "gradboost", "histgb", "mlp"],
        help="Models: logistic linear_svm rbf_svm knn rf extratrees gradboost histgb mlp ridge torch_mlp lstm gru tcn transformer",
    )
    parser.add_argument("--no-tune", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    # DL options
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--reg-weight", type=float, default=0.03)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])

    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = Path.cwd() / data_path
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = Path.cwd() / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading dataset:", data_path)
    data = safe_torch_load(data_path)
    train_idx, val_idx, test_idx = get_split_indices(data, args.split, seed=args.seed)

    y_cls = to_numpy(data["y_cls"]).astype(int).reshape(-1)
    y_reg, reg_names = get_reg_targets(data)

    X, feature_meta = make_features(data, args.feature_set)
    X_seq, X_stat = make_sequence_features(data, use_stat_context=True)

    print("Split:", args.split, "train/val/test =", len(train_idx), len(val_idx), len(test_idx))
    print("Feature set:", args.feature_set, feature_meta)
    print("Regression targets:", reg_names)
    print("Models:", args.models)

    results = []
    run_info = {
        "data": str(data_path),
        "split": args.split,
        "feature_set": args.feature_set,
        "feature_meta": feature_meta,
        "models": args.models,
        "regression_targets": reg_names,
        "seed": args.seed,
    }

    X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
    y_train, y_val, y_test = y_cls[train_idx], y_cls[val_idx], y_cls[test_idx]
    yr_train, yr_val, yr_test = y_reg[train_idx], y_reg[val_idx], y_reg[test_idx]

    sklearn_models = {"logistic", "linear_svm", "rbf_svm", "knn", "rf", "extratrees", "gradboost", "histgb", "mlp", "ridge"}
    torch_models = {"torch_mlp", "lstm", "gru", "tcn", "transformer"}

    for model_name in args.models:
        print("\n=== Running baseline:", model_name, "===")
        t0 = time.time()
        if model_name in sklearn_models:
            y_pred, prob, yreg_pred, info = fit_predict_sklearn(
                model_name,
                X_train,
                X_val,
                X_test,
                y_train,
                y_val,
                y_test,
                yr_train,
                yr_val,
                yr_test,
                seed=args.seed,
                tune_on_val=not args.no_tune,
            )
        elif model_name in torch_models:
            y_pred, prob, yreg_pred, info = train_torch_model(
                model_name,
                X,
                X_seq,
                X_stat,
                train_idx,
                val_idx,
                test_idx,
                y_cls,
                y_reg,
                reg_names,
                seed=args.seed,
                epochs=args.epochs,
                patience=args.patience,
                batch_size=args.batch_size,
                hidden=args.hidden,
                dropout=args.dropout,
                lr=args.lr,
                weight_decay=args.weight_decay,
                reg_weight=args.reg_weight,
                out_device=args.device,
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")

        metrics = compute_metrics(y_test, y_pred, prob, yr_test, yreg_pred, reg_names)
        metrics["Model"] = model_name
        metrics["Feature set"] = args.feature_set
        metrics["Seed"] = args.seed
        metrics["Runtime sec"] = float(time.time() - t0)
        results.append(metrics)

        write_predictions(
            out_dir / f"baseline_predictions_{model_name}.csv",
            test_idx,
            y_test,
            y_pred,
            prob,
            yr_test,
            yreg_pred,
            reg_names,
        )

        with open(out_dir / f"baseline_info_{model_name}.json", "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)

        print(
            f"{model_name}: Acc={metrics['Accuracy']:.4f}, Macro-F1={metrics['Macro-F1']:.4f}, "
            f"QWK={metrics['QWK']:.4f}, HR Recall={metrics['High-risk Recall']:.4f}, "
            f"Reg MAE={metrics['Regression MAE']:.4f}"
        )

    res_df = pd.DataFrame(results)
    # Sort by Macro-F1 + QWK + HR Recall, as a balanced classification rank.
    res_df["rank_score"] = res_df["Macro-F1"] + 0.35 * res_df["QWK"] + 0.25 * res_df["High-risk Recall"] - 0.03 * res_df["Regression MAE"]
    res_df = res_df.sort_values("rank_score", ascending=False).reset_index(drop=True)

    res_df.to_csv(out_dir / "baseline_metrics.csv", index=False, encoding="utf-8-sig")
    md = markdown_table(res_df)
    (out_dir / "baseline_metrics_markdown.txt").write_text(md, encoding="utf-8")
    with open(out_dir / "baseline_run_info.json", "w", encoding="utf-8") as f:
        json.dump(run_info, f, ensure_ascii=False, indent=2)

    # Long format for model comparison figures.
    cls_rows = []
    for _, r in res_df.iterrows():
        for m in ["Accuracy", "Macro-F1", "QWK", "High-risk Recall", "High-risk AUC"]:
            cls_rows.append({"Model": r["Model"], "Metric": m, "Score": r[m]})
    pd.DataFrame(cls_rows).to_csv(out_dir / "baseline_classification_metrics_long.csv", index=False, encoding="utf-8-sig")

    reg_rows = []
    for _, r in res_df.iterrows():
        for t in reg_names:
            reg_rows.append({"Model": r["Model"], "Target": t, "MAE": r[f"MAE_{t}"]})
    pd.DataFrame(reg_rows).to_csv(out_dir / "baseline_regression_mae_long.csv", index=False, encoding="utf-8-sig")

    print("\n=== Baseline comparison summary ===")
    print(md)
    print("\nSaved to:", out_dir)
    print("Main table:", out_dir / "baseline_metrics.csv")
    print("Markdown:", out_dir / "baseline_metrics_markdown.txt")


if __name__ == "__main__":
    main()
