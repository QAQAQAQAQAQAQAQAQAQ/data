# -*- coding: utf-8 -*-
"""
Oracle/global-normalization upper-bound variant for multimodal v3 probe.

Put this file in the project root, for example:
    C:/Users/19735/Documents/Codex/2026-07-04/n-s/

Recommended first run:
    python -m py_compile train_multimodal_v2_probe.py
    python train_multimodal_v2_probe.py --split stratified_random_9_1 --seed 42

Modes:
    ral
    val
    tal_rule
    ral_val
    ral_tal
    ral_val_tal

Outputs:
    outputs/multimodal_v2_probe_summary.csv
    outputs/multimodal_v2_probe_<split>_<mode>_metrics.json
    outputs/multimodal_v2_probe_<split>_<mode>_confusion_matrix.csv
    outputs/multimodal_v2_probe_<split>_<mode>_predictions.csv
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    r2_score,
)


DEFAULT_MODES = ["ral", "val", "tal_rule", "ral_val", "ral_tal", "ral_val_tal"]

REG_TARGET_MAP = {"peak": 0, "sum": 1, "mean": 2, "count": 3}
REG_TARGET_FULL_NAMES = {
    "peak": "peak_log",
    "sum": "sum_log",
    "mean": "mean_log",
    "count": "count_high_log",
}


# -----------------------------
# Basic utilities
# -----------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)


def get_device() -> torch.device:
    """Prefer DirectML when available; otherwise use CPU. CUDA is intentionally avoided."""
    try:
        import torch_directml  # type: ignore

        return torch_directml.device()
    except Exception:
        return torch.device("cpu")


def safe_torch_load(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def to_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def ensure_2d_y_reg(y_reg: np.ndarray, n: int) -> np.ndarray:
    y_reg = np.asarray(y_reg, dtype=np.float32)
    if y_reg.ndim == 0:
        y_reg = np.zeros((n, 1), dtype=np.float32)
    elif y_reg.ndim == 1:
        y_reg = y_reg.reshape(-1, 1)
    elif y_reg.ndim > 2:
        y_reg = y_reg.reshape(y_reg.shape[0], -1)
    if y_reg.shape[0] != n:
        raise ValueError(f"y_reg first dimension mismatch: {y_reg.shape[0]} vs n={n}")
    return np.nan_to_num(y_reg, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)




def select_y_reg_targets(data: dict, n: int, reg_targets: List[str]) -> Tuple[np.ndarray, List[str]]:
    """Select official v3 multi-regression targets while keeping v2 compatibility."""
    if "y_reg_multi" in data:
        y_all = ensure_2d_y_reg(to_numpy(data["y_reg_multi"]), n)
        cols = []
        names = []
        for t in reg_targets:
            if t not in REG_TARGET_MAP:
                raise ValueError(f"Unknown regression target: {t}. Choose from {sorted(REG_TARGET_MAP)}")
            j = REG_TARGET_MAP[t]
            if j >= y_all.shape[1]:
                raise ValueError(f"Target {t} requires column {j}, but y_reg_multi shape is {y_all.shape}")
            cols.append(j)
            names.append(REG_TARGET_FULL_NAMES[t])
        return y_all[:, cols].astype(np.float32), names

    # v2 fallback: only peak target exists.
    y = ensure_2d_y_reg(to_numpy(data.get("y_reg", np.zeros((n, 1), dtype=np.float32))), n)
    return y[:, :1].astype(np.float32), ["peak_log"]


def get_split(data: dict, split_name: str, n: int) -> Dict[str, np.ndarray]:
    if "split_idx_v2" in data and split_name in data["split_idx_v2"]:
        split = data["split_idx_v2"][split_name]
        return {k: to_numpy(v).astype(int).reshape(-1) for k, v in split.items()}

    if split_name == "time_ordered_70_10_20":
        raw = data.get("split_idx") or data.get("splits")
        if raw is not None:
            out = {}
            for key in ["train", "val", "test"]:
                arr = to_numpy(raw[key]).astype(int).reshape(-1)
                if arr.size == 2 and arr[0] >= 0 and arr[1] <= n and arr[0] < arr[1]:
                    out[key] = np.arange(arr[0], arr[1], dtype=int)
                else:
                    out[key] = arr
            return out

    # fallback time split
    train_end = int(n * 0.70)
    val_end = int(n * 0.80)
    return {
        "train": np.arange(0, train_end, dtype=int),
        "val": np.arange(train_end, val_end, dtype=int),
        "test": np.arange(val_end, n, dtype=int),
    }


# -----------------------------
# Train-only standardization
# -----------------------------


def standardize_3d_train_only(x: np.ndarray, train_idx: np.ndarray, eps: float = 1e-6):
    x = x.astype(np.float32, copy=True)
    train_values = x[train_idx].reshape(-1, x.shape[-1])
    mean = train_values.mean(axis=0)
    std = train_values.std(axis=0)
    std = np.where(std < eps, 1.0, std)
    x = (x - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return x, mean.astype(np.float32), std.astype(np.float32)


def standardize_2d_train_only(x: np.ndarray, train_idx: np.ndarray, eps: float = 1e-6):
    x = x.astype(np.float32, copy=True)
    if x.shape[1] == 0:
        return x, np.zeros((0,), dtype=np.float32), np.ones((0,), dtype=np.float32)
    train_values = x[train_idx]
    mean = train_values.mean(axis=0)
    std = train_values.std(axis=0)
    std = np.where(std < eps, 1.0, std)
    x = (x - mean.reshape(1, -1)) / std.reshape(1, -1)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return x, mean.astype(np.float32), std.astype(np.float32)


def standardize_val_train_only(x: np.ndarray, train_idx: np.ndarray, eps: float = 1e-6):
    """Channel-wise image standardization using train samples only."""
    x = x.astype(np.float32, copy=True)
    # x: N,C,H,W
    train_values = x[train_idx]
    mean = train_values.mean(axis=(0, 2, 3))
    std = train_values.std(axis=(0, 2, 3))
    std = np.where(std < eps, 1.0, std)
    x = (x - mean.reshape(1, -1, 1, 1)) / std.reshape(1, -1, 1, 1)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return x, mean.astype(np.float32), std.astype(np.float32)


def standardize_y_train_only(y: np.ndarray, train_idx: np.ndarray, eps: float = 1e-6):
    y = y.astype(np.float32, copy=True)
    mean = y[train_idx].mean(axis=0)
    std = y[train_idx].std(axis=0)
    std = np.where(std < eps, 1.0, std)
    y_std = (y - mean.reshape(1, -1)) / std.reshape(1, -1)
    y_std = np.nan_to_num(y_std, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return y_std, mean.astype(np.float32), std.astype(np.float32)


# -----------------------------
# Dataset
# -----------------------------


class MultiModalDataset(Dataset):
    def __init__(
        self,
        x_ral: np.ndarray,
        x_stat: np.ndarray,
        x_val: np.ndarray,
        x_tal: np.ndarray,
        y_cls: np.ndarray,
        y_reg_std: np.ndarray,
        indices: np.ndarray,
    ):
        self.x_ral = torch.from_numpy(x_ral[indices]).float()
        self.x_stat = torch.from_numpy(x_stat[indices]).float()
        self.x_val = torch.from_numpy(x_val[indices]).float()
        self.x_tal = torch.from_numpy(x_tal[indices]).float()
        self.y_cls = torch.from_numpy(y_cls[indices]).long()
        self.y_reg_std = torch.from_numpy(y_reg_std[indices]).float()
        self.indices = torch.from_numpy(indices.astype(np.int64)).long()

    def __len__(self):
        return len(self.y_cls)

    def __getitem__(self, idx):
        return {
            "x_ral": self.x_ral[idx],
            "x_stat": self.x_stat[idx],
            "x_val": self.x_val[idx],
            "x_tal": self.x_tal[idx],
            "y_cls": self.y_cls[idx],
            "y_reg": self.y_reg_std[idx],
            "index": self.indices[idx],
        }


# -----------------------------
# Model blocks
# -----------------------------


class TemporalBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation),
            nn.BatchNorm1d(channels),
        )
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.act(x + self.net(x)))


class RALBranch(nn.Module):
    def __init__(self, in_dim: int, stat_dim: int, hidden: int, dropout: float):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Conv1d(in_dim, hidden, kernel_size=1),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
        )
        self.tcn = nn.Sequential(
            TemporalBlock(hidden, 1, dropout),
            TemporalBlock(hidden, 2, dropout),
            TemporalBlock(hidden, 4, dropout),
        )
        self.attn = nn.Sequential(
            nn.Linear(hidden, max(hidden // 2, 8)),
            nn.Tanh(),
            nn.Linear(max(hidden // 2, 8), 1),
        )
        self.stat_dim = stat_dim
        if stat_dim > 0:
            self.stat_encoder = nn.Sequential(
                nn.Linear(stat_dim, hidden // 2),
                nn.LayerNorm(hidden // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden // 2, hidden // 2),
                nn.GELU(),
            )
            fused_dim = hidden + hidden // 2
        else:
            self.stat_encoder = None
            fused_dim = hidden
        self.out = nn.Sequential(
            nn.Linear(fused_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x_ral, x_stat):
        h = x_ral.transpose(1, 2)
        h = self.input_proj(h)
        h = self.tcn(h).transpose(1, 2)
        a = self.attn(h).squeeze(-1)
        w = torch.softmax(a, dim=1).unsqueeze(-1)
        z = (h * w).sum(dim=1)
        if self.stat_encoder is not None and x_stat.shape[1] > 0:
            z = torch.cat([z, self.stat_encoder(x_stat)], dim=-1)
        return self.out(z)


class VALBranch(nn.Module):
    def __init__(self, in_channels: int, hidden: int, dropout: float):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 24, kernel_size=3, padding=1),
            nn.BatchNorm2d(24),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout * 0.5),
            nn.Conv2d(24, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout * 0.5),
            nn.Conv2d(48, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.out = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.out(self.cnn(x))


class TALRuleBranch(nn.Module):
    def __init__(self, in_dim: int, hidden: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class GatedFusionProbe(nn.Module):
    def __init__(
        self,
        mode: str,
        ral_dim: int,
        stat_dim: int,
        val_channels: int,
        tal_dim: int,
        reg_dim: int,
        hidden: int = 64,
        dropout: float = 0.20,
        n_classes: int = 4,
        gate_bias: float = -1.0,
    ):
        super().__init__()
        self.mode = mode
        self.use_ral = "ral" in mode.split("_")
        self.use_val = "val" in mode.split("_")
        self.use_tal = "tal" in mode.split("_") or "tal_rule" in mode

        # Robust mode parsing for names like ral_tal and tal_rule.
        self.use_ral = mode in ["ral", "ral_val", "ral_tal", "ral_val_tal"]
        self.use_val = mode in ["val", "ral_val", "ral_val_tal"]
        self.use_tal = mode in ["tal_rule", "ral_tal", "ral_val_tal"]

        if self.use_ral:
            self.ral_branch = RALBranch(ral_dim, stat_dim, hidden, dropout)
            self.ral_aux = nn.Linear(hidden, n_classes)
        else:
            self.ral_branch = None
            self.ral_aux = None

        if self.use_val:
            self.val_branch = VALBranch(val_channels, hidden, dropout)
            self.val_aux = nn.Linear(hidden, n_classes)
        else:
            self.val_branch = None
            self.val_aux = None

        if self.use_tal:
            self.tal_branch = TALRuleBranch(tal_dim, hidden, dropout)
            self.tal_aux = nn.Linear(hidden, n_classes)
        else:
            self.tal_branch = None
            self.tal_aux = None

        # If RAL exists, use RAL-dominant residual gates.
        if self.use_ral and self.use_val:
            self.gate_val = nn.Sequential(nn.Linear(hidden * 2, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1))
            nn.init.constant_(self.gate_val[-1].bias, gate_bias)
        else:
            self.gate_val = None

        if self.use_ral and self.use_tal:
            self.gate_tal = nn.Sequential(nn.Linear(hidden * 2, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1))
            nn.init.constant_(self.gate_tal[-1].bias, gate_bias)
        else:
            self.gate_tal = None

        # If no RAL and more than one branch, concatenate and project.
        branch_count = int(self.use_ral) + int(self.use_val) + int(self.use_tal)
        if self.use_ral:
            fusion_in = hidden
        else:
            fusion_in = hidden * branch_count

        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.cls_head = nn.Linear(hidden, n_classes)
        self.highrisk_head = nn.Linear(hidden, 1)
        self.reg_head = nn.Linear(hidden, reg_dim)

    def forward(self, batch: dict) -> dict:
        z_ral = z_val = z_tal = None
        aux_logits = {}
        gate_values = {}

        if self.use_ral:
            z_ral = self.ral_branch(batch["x_ral"], batch["x_stat"])
            aux_logits["ral"] = self.ral_aux(z_ral)
        if self.use_val:
            z_val = self.val_branch(batch["x_val"])
            aux_logits["val"] = self.val_aux(z_val)
        if self.use_tal:
            z_tal = self.tal_branch(batch["x_tal"])
            aux_logits["tal_rule"] = self.tal_aux(z_tal)

        if self.use_ral:
            fused = z_ral
            if self.use_val:
                g = torch.sigmoid(self.gate_val(torch.cat([z_ral, z_val], dim=-1)))
                fused = fused + g * z_val
                gate_values["gate_val"] = g
            if self.use_tal:
                g = torch.sigmoid(self.gate_tal(torch.cat([z_ral, z_tal], dim=-1)))
                fused = fused + g * z_tal
                gate_values["gate_tal"] = g
        else:
            parts = []
            if self.use_val:
                parts.append(z_val)
            if self.use_tal:
                parts.append(z_tal)
            if len(parts) == 1:
                fused = parts[0]
            else:
                fused = torch.cat(parts, dim=-1)

        h = self.fusion(fused)
        return {
            "cls_logits": self.cls_head(h),
            "highrisk_logits": self.highrisk_head(h),
            "reg_pred": self.reg_head(h),
            "aux_logits": aux_logits,
            "gate_values": gate_values,
        }


# -----------------------------
# Loss and metrics
# -----------------------------


class ProbeLoss:
    def __init__(
        self,
        y_train: np.ndarray,
        device: torch.device,
        aux_weight: float = 0.15,
        reg_weight: float = 0.03,
        gate_l1: float = 0.005,
        reg_target_weights: np.ndarray | None = None,
    ):
        counts = np.bincount(y_train, minlength=4).astype(np.float32)
        # mild class balancing for stratified setting and time-ordered setting
        inv_sqrt = np.sqrt(counts.sum() / np.maximum(counts, 1.0))
        inv_sqrt = inv_sqrt / inv_sqrt.mean()
        self.class_weight = torch.tensor(inv_sqrt, dtype=torch.float32, device=device)
        self.ce = nn.CrossEntropyLoss(weight=self.class_weight)
        # Huber/SmoothL1 with per-target weighting, better for volatile log-energy targets.
        self.reg = nn.SmoothL1Loss(reduction="none")
        if reg_target_weights is None:
            reg_target_weights = np.ones((1,), dtype=np.float32)
        reg_target_weights = np.asarray(reg_target_weights, dtype=np.float32).reshape(-1)
        reg_target_weights = reg_target_weights / max(float(reg_target_weights.sum()), 1e-6)
        self.reg_target_weights = torch.tensor(reg_target_weights, dtype=torch.float32, device=device).view(1, -1)
        pos = float((y_train >= 2).sum())
        neg = float(len(y_train) - pos)
        pos_weight = min(max(neg / max(pos, 1.0), 0.5), 20.0)
        self.bce_high = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32, device=device))
        self.aux_weight = float(aux_weight)
        self.reg_weight = float(reg_weight)
        self.gate_l1 = float(gate_l1)
        self.class_values = torch.arange(4, dtype=torch.float32, device=device)

    def __call__(self, out: dict, y_cls: torch.Tensor, y_reg: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        logits = out["cls_logits"]
        loss_cls = self.ce(logits, y_cls)
        reg_each = self.reg(out["reg_pred"], y_reg)
        if self.reg_target_weights.shape[1] != reg_each.shape[1]:
            weights = torch.ones((1, reg_each.shape[1]), dtype=reg_each.dtype, device=reg_each.device) / reg_each.shape[1]
        else:
            weights = self.reg_target_weights.to(device=reg_each.device, dtype=reg_each.dtype)
        loss_reg = torch.mean(torch.sum(reg_each * weights, dim=1))
        y_high = (y_cls >= 2).float().view(-1, 1)
        loss_high = self.bce_high(out["highrisk_logits"], y_high)

        prob = torch.softmax(logits, dim=1)
        pred_expectation = (prob * self.class_values.view(1, -1)).sum(dim=1)
        loss_ord = nn.functional.smooth_l1_loss(pred_expectation, y_cls.float())

        aux_loss = torch.zeros((), dtype=torch.float32, device=logits.device)
        aux_count = 0
        for aux in out.get("aux_logits", {}).values():
            aux_loss = aux_loss + self.ce(aux, y_cls)
            aux_count += 1
        if aux_count > 0:
            aux_loss = aux_loss / aux_count

        gate_loss = torch.zeros((), dtype=torch.float32, device=logits.device)
        gate_count = 0
        for g in out.get("gate_values", {}).values():
            gate_loss = gate_loss + g.mean()
            gate_count += 1
        if gate_count > 0:
            gate_loss = gate_loss / gate_count

        total = (
            loss_cls
            + self.reg_weight * loss_reg
            + 0.08 * loss_high
            + 0.10 * loss_ord
            + self.aux_weight * aux_loss
            + self.gate_l1 * gate_loss
        )
        return total, {
            "loss": float(total.detach().cpu()),
            "loss_cls": float(loss_cls.detach().cpu()),
            "loss_reg": float(loss_reg.detach().cpu()),
            "loss_high": float(loss_high.detach().cpu()),
            "loss_ord": float(loss_ord.detach().cpu()),
            "loss_aux": float(aux_loss.detach().cpu()),
            "loss_gate": float(gate_loss.detach().cpu()),
        }


def distribution_penalty(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 4) -> float:
    true_counts = np.bincount(y_true, minlength=n_classes).astype(np.float64)
    pred_counts = np.bincount(y_pred, minlength=n_classes).astype(np.float64)
    true_dist = true_counts / max(true_counts.sum(), 1.0)
    pred_dist = pred_counts / max(pred_counts.sum(), 1.0)
    return float(np.abs(pred_dist - true_dist).sum())


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_reg_true: np.ndarray | None = None, y_reg_pred: np.ndarray | None = None, target_names: List[str] | None = None) -> Dict:
    recalls = recall_score(y_true, y_pred, labels=[0, 1, 2, 3], average=None, zero_division=0)
    precisions = precision_score(y_true, y_pred, labels=[0, 1, 2, 3], average=None, zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    try:
        qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
        if qwk is None or math.isnan(qwk):
            qwk = 0.0
    except Exception:
        qwk = 0.0

    highrisk_mask = y_true >= 2
    highrisk_recall = float(((y_pred[highrisk_mask] >= 2).sum() / max(highrisk_mask.sum(), 1)))
    highrisk_precision = float((((y_true >= 2) & (y_pred >= 2)).sum() / max((y_pred >= 2).sum(), 1)))
    ordinal_mae = float(np.mean(np.abs(y_pred.astype(float) - y_true.astype(float))))
    dist_pen = distribution_penalty(y_true, y_pred)
    min_recall = float(np.min(recalls))

    val_score = (
        0.30 * float(accuracy_score(y_true, y_pred))
        + 0.30 * float(macro_f1)
        + 0.20 * float(qwk)
        + 0.10 * float(recalls[3])
        + 0.05 * float(highrisk_recall)
        + 0.05 * min_recall
        - 0.05 * dist_pen
    )

    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "qwk": float(qwk),
        "ordinal_mae": ordinal_mae,
        "class_recall": {str(i): float(recalls[i]) for i in range(4)},
        "class_precision": {str(i): float(precisions[i]) for i in range(4)},
        "class0_recall": float(recalls[0]),
        "class3_recall": float(recalls[3]),
        "highrisk_recall": highrisk_recall,
        "highrisk_precision": highrisk_precision,
        "min_class_recall": min_recall,
        "distribution_penalty": float(dist_pen),
        "predicted_distribution": {str(i): int(v) for i, v in enumerate(np.bincount(y_pred, minlength=4))},
        "true_distribution": {str(i): int(v) for i, v in enumerate(np.bincount(y_true, minlength=4))},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3]).tolist(),
        "classification_report": classification_report(y_true, y_pred, labels=[0, 1, 2, 3], zero_division=0, output_dict=True),
        "val_score": float(val_score),
    }

    if y_reg_true is not None and y_reg_pred is not None:
        err = y_reg_pred.reshape(y_reg_true.shape) - y_reg_true
        result["regression_mae"] = float(np.mean(np.abs(err)))
        result["regression_rmse"] = float(np.sqrt(np.mean(err ** 2)))
        if y_reg_true.ndim == 2 and y_reg_true.shape[1] > 1:
            per_dim_mae = np.mean(np.abs(err), axis=0)
            result["regression_mae_mean_over_targets"] = float(per_dim_mae.mean())
        if target_names is not None:
            for j, name in enumerate(target_names):
                if j >= y_reg_true.shape[1]:
                    continue
                yt = y_reg_true[:, j]
                yp = y_reg_pred[:, j]
                e = yp - yt
                result[f"regression_mae_{name}"] = float(np.mean(np.abs(e)))
                result[f"regression_rmse_{name}"] = float(np.sqrt(np.mean(e ** 2)))
                try:
                    result[f"regression_r2_{name}"] = float(r2_score(yt, yp))
                except Exception:
                    result[f"regression_r2_{name}"] = float("nan")
                try:
                    result[f"regression_spearman_{name}"] = float(pd.Series(yt).corr(pd.Series(yp), method="spearman"))
                    result[f"regression_pearson_{name}"] = float(pd.Series(yt).corr(pd.Series(yp), method="pearson"))
                except Exception:
                    result[f"regression_spearman_{name}"] = float("nan")
                    result[f"regression_pearson_{name}"] = float("nan")
    return result


# -----------------------------
# Train / evaluate
# -----------------------------


def move_batch(batch: dict, device: torch.device) -> dict:
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, y_reg_mean: np.ndarray, y_reg_std: np.ndarray, target_names: List[str] | None = None):
    model.eval()
    all_true, all_pred = [], []
    all_reg_true_std, all_reg_pred_std = [], []
    all_prob = []
    gate_sums = {}
    gate_counts = {}
    indices_all = []

    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            out = model(batch)
            prob = torch.softmax(out["cls_logits"], dim=1)
            pred = prob.argmax(dim=1)
            all_true.append(batch["y_cls"].detach().cpu().numpy())
            all_pred.append(pred.detach().cpu().numpy())
            all_prob.append(prob.detach().cpu().numpy())
            all_reg_true_std.append(batch["y_reg"].detach().cpu().numpy())
            all_reg_pred_std.append(out["reg_pred"].detach().cpu().numpy())
            indices_all.append(batch["index"].detach().cpu().numpy())
            for name, g in out.get("gate_values", {}).items():
                gate_sums[name] = gate_sums.get(name, 0.0) + float(g.detach().cpu().sum())
                gate_counts[name] = gate_counts.get(name, 0) + int(g.numel())

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    y_prob = np.concatenate(all_prob, axis=0)
    reg_true_std = np.concatenate(all_reg_true_std, axis=0)
    reg_pred_std = np.concatenate(all_reg_pred_std, axis=0)
    y_reg_true = reg_true_std * y_reg_std.reshape(1, -1) + y_reg_mean.reshape(1, -1)
    y_reg_pred = reg_pred_std * y_reg_std.reshape(1, -1) + y_reg_mean.reshape(1, -1)

    metrics = compute_metrics(y_true, y_pred, y_reg_true, y_reg_pred, target_names=target_names)
    metrics["gate_mean"] = {name: gate_sums[name] / max(gate_counts[name], 1) for name in gate_sums}
    return metrics, y_true, y_pred, y_reg_true, y_reg_pred, np.concatenate(indices_all), y_prob


def train_one_mode(
    mode: str,
    data_arrays: dict,
    splits: Dict[str, np.ndarray],
    args,
    device: torch.device,
    out_dir: Path,
) -> dict:
    print(f"\n=== Training mode: {mode} | split={args.split} | seed={args.seed} ===")

    # Only pass fields required by MultiModalDataset.
    # data_arrays also contains y_reg_mean / y_reg_target_std for inverse-transforming
    # regression outputs during evaluation, and those should not be passed into Dataset.
    dataset_kwargs = {
        "x_ral": data_arrays["x_ral"],
        "x_stat": data_arrays["x_stat"],
        "x_val": data_arrays["x_val"],
        "x_tal": data_arrays["x_tal"],
        "y_cls": data_arrays["y_cls"],
        "y_reg_std": data_arrays["y_reg_std"],
    }
    train_ds = MultiModalDataset(**dataset_kwargs, indices=splits["train"])
    val_ds = MultiModalDataset(**dataset_kwargs, indices=splits["val"])
    test_ds = MultiModalDataset(**dataset_kwargs, indices=splits["test"])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    model = GatedFusionProbe(
        mode=mode,
        ral_dim=data_arrays["x_ral"].shape[-1],
        stat_dim=data_arrays["x_stat"].shape[-1],
        val_channels=data_arrays["x_val"].shape[1],
        tal_dim=data_arrays["x_tal"].shape[-1],
        reg_dim=data_arrays["y_reg_std"].shape[-1],
        hidden=args.hidden,
        dropout=args.dropout,
        gate_bias=args.gate_bias,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=4)
    loss_fn = ProbeLoss(
        y_train=data_arrays["y_cls"][splits["train"]],
        device=device,
        aux_weight=args.aux_weight,
        reg_weight=args.reg_weight,
        gate_l1=args.gate_l1,
        reg_target_weights=data_arrays.get("reg_target_weights"),
    )

    best_val_score = -1e9
    best_epoch = -1
    best_state = None
    bad_epochs = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        parts_sum = {}

        for batch in train_loader:
            batch = move_batch(batch, device)
            opt.zero_grad(set_to_none=True)
            out = model(batch)
            loss, parts = loss_fn(out, batch["y_cls"], batch["y_reg"])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            opt.step()
            loss_sum += parts["loss"]
            for k, v in parts.items():
                parts_sum[k] = parts_sum.get(k, 0.0) + float(v)

        val_metrics, *_ = evaluate(
            model,
            val_loader,
            device,
            y_reg_mean=data_arrays["y_reg_mean"],
            y_reg_std=data_arrays["y_reg_target_std"],
            target_names=data_arrays.get("target_names"),
        )
        scheduler.step(val_metrics["val_score"])

        history.append(
            {
                "epoch": epoch,
                "lr": float(opt.param_groups[0]["lr"]),
                "train_loss": float(loss_sum / max(len(train_loader), 1)),
                "val": val_metrics,
            }
        )

        print(
            f"epoch={epoch:03d} "
            f"loss={loss_sum / max(len(train_loader), 1):.4f} "
            f"val_score={val_metrics['val_score']:.4f} "
            f"acc={val_metrics['accuracy']:.4f} "
            f"macroF1={val_metrics['macro_f1']:.4f} "
            f"QWK={val_metrics['qwk']:.4f} "
            f"R3={val_metrics['class3_recall']:.4f} "
            f"HR={val_metrics['highrisk_recall']:.4f} "
            f"gate={val_metrics.get('gate_mean', {})} "
            f"pred={val_metrics['predicted_distribution']}"
        )

        if val_metrics["val_score"] > best_val_score:
            best_val_score = val_metrics["val_score"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping at epoch {epoch}; best_epoch={best_epoch}, best_val_score={best_val_score:.4f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics, y_true, y_pred, y_reg_true, y_reg_pred, indices, y_prob = evaluate(
        model,
        test_loader,
        device,
        y_reg_mean=data_arrays["y_reg_mean"],
        y_reg_std=data_arrays["y_reg_target_std"],
        target_names=data_arrays.get("target_names"),
    )

    prefix = f"multimodal_v3_probe_{args.split}_{mode}"
    metrics_path = out_dir / f"{prefix}_metrics.json"
    cm_path = out_dir / f"{prefix}_confusion_matrix.csv"
    pred_path = out_dir / f"{prefix}_predictions.csv"
    best_path = out_dir / f"{prefix}_best.pt"

    result = {
        "mode": mode,
        "split": args.split,
        "seed": args.seed,
        "device": str(device),
        "best_epoch": int(best_epoch),
        "best_val_score": float(best_val_score),
        "split_sizes": {k: int(len(v)) for k, v in splits.items()},
        "history": history,
        "test": test_metrics,
        "args": vars(args),
        "target_names": data_arrays.get("target_names", []),
    }

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    pd.DataFrame(
        test_metrics["confusion_matrix"],
        index=["true_0", "true_1", "true_2", "true_3"],
        columns=["pred_0", "pred_1", "pred_2", "pred_3"],
    ).to_csv(cm_path, encoding="utf-8-sig")

    pred_df = pd.DataFrame({"sample_index": indices, "y_true": y_true, "y_pred": y_pred})
    for c in range(y_prob.shape[1]):
        pred_df[f"prob_{c}"] = y_prob[:, c]
    target_names = data_arrays.get("target_names") or [f"target{j}" for j in range(y_reg_true.shape[1])]
    # Keep old first-dimension columns for compatibility with previous ensemble scripts.
    pred_df["y_reg_true_first"] = y_reg_true[:, 0]
    pred_df["y_reg_pred_first"] = y_reg_pred[:, 0]
    for j, name in enumerate(target_names):
        pred_df[f"y_reg_true_{name}"] = y_reg_true[:, j]
        pred_df[f"y_reg_pred_{name}"] = y_reg_pred[:, j]
        pred_df[f"y_reg_abs_error_{name}"] = np.abs(y_reg_pred[:, j] - y_reg_true[:, j])
    pred_df["y_reg_abs_error_mean"] = np.mean(np.abs(y_reg_pred - y_reg_true), axis=1)
    pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "mode": mode,
            "split": args.split,
            "seed": args.seed,
            "best_epoch": int(best_epoch),
            "test_metrics": test_metrics,
        },
        best_path,
    )

    print(
        f"TEST {mode}: "
        f"acc={test_metrics['accuracy']:.4f}, "
        f"macroF1={test_metrics['macro_f1']:.4f}, "
        f"QWK={test_metrics['qwk']:.4f}, "
        f"R0={test_metrics['class0_recall']:.4f}, "
        f"R3={test_metrics['class3_recall']:.4f}, "
        f"HR={test_metrics['highrisk_recall']:.4f}, "
        f"MAE={test_metrics.get('regression_mae', float('nan')):.4f}, "
        f"RMSE={test_metrics.get('regression_rmse', float('nan')):.4f}, "
        f"gate={test_metrics.get('gate_mean', {})}, "
        f"pred={test_metrics['predicted_distribution']}"
    )
    return result


# -----------------------------
# Main
# -----------------------------


def main():
    parser = argparse.ArgumentParser(description="Oracle/global-normalization upper-bound variant for multimodal v3 probe.")
    parser.add_argument("--data", type=str, default="camvnet_event_multimodal_v3.pt")
    parser.add_argument("--out-dir", type=str, default="outputs")
    parser.add_argument("--split", type=str, default="stratified_random_9_1")
    parser.add_argument("--modes", type=str, nargs="*", default=DEFAULT_MODES)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--aux-weight", type=float, default=0.15)
    parser.add_argument("--reg-weight", type=float, default=0.03)
    parser.add_argument("--gate-l1", type=float, default=0.005)
    parser.add_argument("--gate-bias", type=float, default=-1.0)
    parser.add_argument("--reg-targets", type=str, nargs="+", default=["peak", "sum", "mean", "count"], choices=sorted(REG_TARGET_MAP))
    parser.add_argument("--reg-target-weights", type=float, nargs="+", default=[0.30, 0.35, 0.20, 0.15])
    parser.add_argument("--norm-scope", type=str, default="train", choices=["train", "train_val", "all"], help="Normalization fitting scope. train=strict; all=oracle/transductive upper-bound.")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print("Device:", device)

    root = Path.cwd()
    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = root / data_path
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    data = safe_torch_load(data_path)
    metadata = data.get("metadata", {}) or {}

    required_keys = ["X_ral_v2", "X_ral_stat_v2", "X_val_v2", "X_text_rule_v2", "y_cls"]
    missing = [k for k in required_keys if k not in data]
    if missing:
        raise KeyError(f"Missing keys in v2 dataset: {missing}")

    x_ral = to_numpy(data["X_ral_v2"]).astype(np.float32)
    x_stat = to_numpy(data["X_ral_stat_v2"]).astype(np.float32)
    x_val = to_numpy(data["X_val_v2"]).astype(np.float32)
    x_tal = to_numpy(data["X_text_rule_v2"]).astype(np.float32)
    y_cls = to_numpy(data["y_cls"]).astype(np.int64)

    n = len(y_cls)
    y_reg_raw, target_names = select_y_reg_targets(data, n, args.reg_targets)
    if len(args.reg_target_weights) != y_reg_raw.shape[1]:
        raise ValueError(
            f"--reg-target-weights length {len(args.reg_target_weights)} does not match selected targets {target_names}. "
            f"Use one weight per target."
        )
    reg_target_weights = np.asarray(args.reg_target_weights, dtype=np.float32)
    reg_target_weights = reg_target_weights / max(float(reg_target_weights.sum()), 1e-6)

    splits = get_split(data, args.split, n)

    print("Dataset:", data_path)
    print("Samples:", n)
    print("Split:", args.split, {k: int(len(v)) for k, v in splits.items()})
    print("Class distribution:", {str(i): int(v) for i, v in enumerate(np.bincount(y_cls, minlength=4))})
    print("X_ral_v2:", x_ral.shape, metadata.get("ral_v2_features", ""))
    print("X_ral_stat_v2:", x_stat.shape, metadata.get("ral_stat_features", ""))
    print("X_val_v2:", x_val.shape, metadata.get("val_v2_channels", ""))
    print("X_text_rule_v2:", x_tal.shape, metadata.get("text_v2_feature_names", ""))
    print("y_reg:", y_reg_raw.shape, "targets=", target_names, "weights=", reg_target_weights.tolist())

    # Normalization scope.
    # train     = strict setting, no validation/test distribution leakage.
    # train_val = semi-strict setting, validation distribution is included.
    # all       = oracle/transductive upper-bound setting, test distribution is included.
    if args.norm_scope == "train":
        norm_idx = splits["train"]
    elif args.norm_scope == "train_val":
        norm_idx = np.concatenate([splits["train"], splits["val"]]).astype(int)
    elif args.norm_scope == "all":
        norm_idx = np.arange(n, dtype=int)
    else:
        raise ValueError(f"Unsupported norm scope: {args.norm_scope}")

    print("Normalization scope:", args.norm_scope, "n_fit=", int(len(norm_idx)))
    if args.norm_scope == "all":
        print("WARNING: norm_scope=all uses test-set distribution information. Treat this as an oracle/transductive upper-bound experiment, not the official prediction setting.")

    x_ral_std, ral_mean, ral_std = standardize_3d_train_only(x_ral, norm_idx)
    x_stat_std, stat_mean, stat_std = standardize_2d_train_only(x_stat, norm_idx)
    x_val_std, val_mean, val_std = standardize_val_train_only(x_val, norm_idx)
    x_tal_std, tal_mean, tal_std = standardize_2d_train_only(x_tal, norm_idx)
    y_reg_std, y_reg_mean, y_reg_target_std = standardize_y_train_only(y_reg_raw, norm_idx)

    for name, arr in [
        ("x_ral_std", x_ral_std),
        ("x_stat_std", x_stat_std),
        ("x_val_std", x_val_std),
        ("x_tal_std", x_tal_std),
        ("y_reg_std", y_reg_std),
    ]:
        if not np.isfinite(arr).all():
            raise ValueError(f"Non-finite values found in {name}")

    data_arrays = {
        "x_ral": x_ral_std,
        "x_stat": x_stat_std,
        "x_val": x_val_std,
        "x_tal": x_tal_std,
        "y_cls": y_cls,
        "y_reg_std": y_reg_std,
        "y_reg_mean": y_reg_mean,
        "y_reg_target_std": y_reg_target_std,
        "target_names": target_names,
        "reg_target_weights": reg_target_weights,
    }

    allowed = set(DEFAULT_MODES + ["val_tal"])
    for mode in args.modes:
        if mode not in allowed:
            raise ValueError(f"Unsupported mode: {mode}. Allowed: {sorted(allowed)}")

    results = []
    for mode in args.modes:
        result = train_one_mode(mode, data_arrays, splits, args, device, out_dir)
        results.append(result)

    rows = []
    for r in results:
        t = r["test"]
        rows.append(
            {
                "split": r["split"],
                "mode": r["mode"],
                "seed": r["seed"],
                "norm_scope": args.norm_scope,
                "best_epoch": r["best_epoch"],
                "best_val_score": r["best_val_score"],
                "Accuracy": t["accuracy"],
                "Macro-F1": t["macro_f1"],
                "Weighted-F1": t["weighted_f1"],
                "QWK": t["qwk"],
                "Ordinal MAE": t["ordinal_mae"],
                "Recall0": t["class_recall"]["0"],
                "Recall1": t["class_recall"]["1"],
                "Recall2": t["class_recall"]["2"],
                "Recall3": t["class_recall"]["3"],
                "High-risk Recall": t["highrisk_recall"],
                "High-risk Precision": t["highrisk_precision"],
                "Regression MAE": t.get("regression_mae", np.nan),
                "Regression RMSE": t.get("regression_rmse", np.nan),
                **{f"MAE_{name}": t.get(f"regression_mae_{name}", np.nan) for name in data_arrays.get("target_names", [])},
                **{f"RMSE_{name}": t.get(f"regression_rmse_{name}", np.nan) for name in data_arrays.get("target_names", [])},
                **{f"R2_{name}": t.get(f"regression_r2_{name}", np.nan) for name in data_arrays.get("target_names", [])},
                **{f"Spearman_{name}": t.get(f"regression_spearman_{name}", np.nan) for name in data_arrays.get("target_names", [])},
                "gate_mean": json.dumps(t.get("gate_mean", {}), ensure_ascii=False),
                "predicted_distribution": json.dumps(t["predicted_distribution"], ensure_ascii=False),
                "true_distribution": json.dumps(t["true_distribution"], ensure_ascii=False),
            }
        )
    summary = pd.DataFrame(rows)
    summary_path = out_dir / f"multimodal_v3_probe_summary_{args.split}_seed{args.seed}.csv"
    latest_path = out_dir / "multimodal_v3_probe_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    summary.to_csv(latest_path, index=False, encoding="utf-8-sig")

    print("\n=== Multimodal v3 probe summary ===")
    show_cols = [
        "mode",
        "Accuracy",
        "Macro-F1",
        "QWK",
        "Recall0",
        "Recall3",
        "High-risk Recall",
        "Regression MAE",
        "Regression RMSE",
        *[f"MAE_{name}" for name in data_arrays.get("target_names", [])],
        *[f"R2_{name}" for name in data_arrays.get("target_names", [])],
        "gate_mean",
    ]
    print(summary[show_cols].to_string(index=False))
    print(f"\nSaved summary to: {summary_path}")
    print(f"Saved latest summary to: {latest_path}")


if __name__ == "__main__":
    main()
