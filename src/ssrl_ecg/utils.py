from __future__ import annotations

import random
from typing import Dict

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def choose_device(requested: str = "auto") -> torch.device:
    """Prefer CUDA or Apple's Metal backend; never silently demote an explicit device."""
    if requested == "auto":
        requested = (
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError(
            "Apple MPS is unavailable; use an unrestricted local process on Apple Silicon"
        )
    if requested not in ("cpu", "cuda", "mps"):
        raise ValueError(f"Unknown device: {requested}")
    if requested == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    device = torch.device(requested)
    print(f"Using {device} with PyTorch {torch.__version__}", flush=True)
    return device


def multilabel_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5
) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    metrics: Dict[str, float] = {}

    # Macro-F1 across classes with support.
    metrics["f1_macro"] = float(
        f1_score(y_true, y_pred, average="binary" if y_true.shape[1] == 1 else "macro", zero_division=0)
    )

    per_class_auc = []
    for c in range(y_true.shape[1]):
        yc = y_true[:, c]
        if len(np.unique(yc)) < 2:
            continue
        per_class_auc.append(roc_auc_score(yc, y_prob[:, c]))
    metrics["auroc_macro"] = (
        float(np.mean(per_class_auc)) if per_class_auc else None
    )

    # Micro sensitivity/specificity for practical clinical reporting.
    tp = float(((y_pred == 1) & (y_true == 1)).sum())
    tn = float(((y_pred == 0) & (y_true == 0)).sum())
    fp = float(((y_pred == 1) & (y_true == 0)).sum())
    fn = float(((y_pred == 0) & (y_true == 1)).sum())

    metrics["sensitivity_micro"] = tp / (tp + fn + 1e-8)
    metrics["specificity_micro"] = tn / (tn + fp + 1e-8)
    return metrics


def apply_random_mask(
    x: torch.Tensor, mask_ratio: float, block_size: int = 50
) -> torch.Tensor:
    """Mask contiguous temporal blocks in each sample for reconstruction pretraining."""
    if mask_ratio <= 0:
        return x

    x_masked = x.clone()
    b, _, t = x.shape
    n_mask = max(1, int((t * mask_ratio) // block_size))

    for i in range(b):
        for _ in range(n_mask):
            start = np.random.randint(0, max(1, t - block_size))
            x_masked[i, :, start : start + block_size] = 0.0
    return x_masked
