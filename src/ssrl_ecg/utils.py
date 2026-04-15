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


def choose_device() -> torch.device:
    """Choose optimal device (GPU if available, else CPU) and configure it."""
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        
        # Enable GPU memory optimization
        torch.cuda.set_per_process_memory_fraction(0.95, device=device)  # Use 95% of GPU memory
        torch.cuda.empty_cache()  # Clear cache before training
        
        # Print GPU info
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        cuda_version = torch.version.cuda
        cudnn_version = torch.backends.cudnn.version()
        
        print(f"\n[GPU CONFIGURATION]")
        print(f"  Device: {gpu_name}")
        print(f"  GPU Memory: {gpu_memory:.1f} GB")
        print(f"  CUDA Version: {cuda_version}")
        print(f"  cuDNN Version: {cudnn_version}")
        print(f"  PyTorch Version: {torch.__version__}")
        
        # Enable cuDNN auto-tuner for optimal performance
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False  # Trade determinism for speed
        
        print(f"  [STATUS] Using GPU mode - optimal for RTX 5070 Ti\n")
        return device
    else:
        print("\n[GPU CONFIGURATION]")
        print("  [WARNING] No CUDA-capable GPU detected!")
        print("  [STATUS] Falling back to CPU mode - training will be slow\n")
        return torch.device("cpu")


def multilabel_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    metrics: Dict[str, float] = {}

    # Macro-F1 across classes with support.
    metrics["f1_macro"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    per_class_auc = []
    for c in range(y_true.shape[1]):
        yc = y_true[:, c]
        if len(np.unique(yc)) < 2:
            continue
        per_class_auc.append(roc_auc_score(yc, y_prob[:, c]))
    metrics["auroc_macro"] = float(np.mean(per_class_auc)) if per_class_auc else float("nan")

    # Micro sensitivity/specificity for practical clinical reporting.
    tp = float(((y_pred == 1) & (y_true == 1)).sum())
    tn = float(((y_pred == 0) & (y_true == 0)).sum())
    fp = float(((y_pred == 1) & (y_true == 0)).sum())
    fn = float(((y_pred == 0) & (y_true == 1)).sum())

    metrics["sensitivity_micro"] = tp / (tp + fn + 1e-8)
    metrics["specificity_micro"] = tn / (tn + fp + 1e-8)
    return metrics


def apply_random_mask(x: torch.Tensor, mask_ratio: float, block_size: int = 50) -> torch.Tensor:
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
