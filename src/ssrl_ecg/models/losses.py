"""Advanced loss functions for handling class imbalance in multi-label ECG classification."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional


class FocalLoss(nn.Module):
    """Focal Loss for multi-label classification.
    
    Addresses class imbalance by down-weighting easy negatives and focusing on hard examples.
    Reference: Lin et al. "Focal Loss for Dense Object Detection" (RetinaNet)
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        """
        Args:
            alpha: Weighting factor in range (0,1) to balance positive/negative
            gamma: Exponent of the modulating factor (1 - p_t) to balance easy/hard examples
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Model predictions [batch_size, num_classes]
            targets: Ground truth binary labels [batch_size, num_classes]
        
        Returns:
            Focal loss value
        """
        # Sigmoid + BCE focal loss for multi-label
        p = torch.sigmoid(logits)
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        # Focal term: (1 - p_t)^gamma
        p_t = torch.where(targets == 1, p, 1 - p)
        focal_weight = (1 - p_t) ** self.gamma
        
        # Apply focal weighting
        focal_loss = self.alpha * focal_weight * ce_loss
        
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class WeightedBCELoss(nn.Module):
    """Weighted Binary Cross-Entropy for class imbalance.
    
    Applies per-class weights based on inverse frequency or custom weights.
    """
    
    def __init__(self, 
                 class_weights: Optional[torch.Tensor] = None,
                 reduction: str = "mean",
                 pos_weight: Optional[torch.Tensor] = None):
        """
        Args:
            class_weights: Per-class weights [num_classes]. If None, uses equal weights
            reduction: 'mean', 'sum', or 'none'
            pos_weight: Alternative to class_weights, passed to BCEWithLogitsLoss
        """
        super().__init__()
        self.class_weights = class_weights
        self.pos_weight = pos_weight
        self.reduction = reduction
        
        if pos_weight is not None:
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')
        else:
            self.criterion = nn.BCEWithLogitsLoss(reduction='none')
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Model predictions [batch_size, num_classes]
            targets: Ground truth binary labels [batch_size, num_classes]
        
        Returns:
            Weighted BCE loss
        """
        loss = self.criterion(logits, targets)  # [batch_size, num_classes]
        
        # Apply per-class weights if provided
        if self.class_weights is not None:
            weights = self.class_weights.view(1, -1).expand_as(loss)
            loss = loss * weights
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss.mean(dim=1)  # Return per-sample loss


class ClassBalancedLoss(nn.Module):
    """Class Balanced Loss using effective number of samples.
    
    Reference: Cui et al. "Class-Balanced Loss Based on Effective Number of Samples"
    """
    
    def __init__(self, class_counts: torch.Tensor, beta: float = 0.9999, reduction: str = "mean"):
        """
        Args:
            class_counts: Number of samples per class [num_classes]
            beta: Hyperparameter (0, 1) controlling re-weighting strength
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.reduction = reduction
        
        # Compute effective number of samples
        # e_n = (1 - beta^n) / (1 - beta)
        effective_num = 1.0 - torch.pow(beta, class_counts.float())
        weights = (1.0 - beta) / effective_num
        weights = weights / weights.sum() * len(weights)  # Normalize
        
        self.register_buffer('class_weights', weights)
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Model predictions [batch_size, num_classes]
            targets: Ground truth binary labels [batch_size, num_classes]
        
        Returns:
            Class-balanced loss
        """
        loss = self.criterion(logits, targets)  # [batch_size, num_classes]
        weights = self.class_weights.view(1, -1).expand_as(loss)
        loss = loss * weights
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss.mean(dim=1)


class DynamicWeightedLoss(nn.Module):
    """Dynamically weighted loss that adjusts weights during training.
    
    Useful for curriculum learning or hard example mining.
    """
    
    def __init__(self, num_classes: int, base_criteria: str = "bce", reduction: str = "mean"):
        """
        Args:
            num_classes: Number of classes
            base_criteria: 'bce', 'focal', or 'class_balanced'
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.num_classes = num_classes
        self.reduction = reduction
        
        # Initialize with uniform weights
        self.register_buffer('class_weights', torch.ones(num_classes))
        
        if base_criteria == "bce":
            self.criterion = nn.BCEWithLogitsLoss(reduction='none')
        elif base_criteria == "focal":
            self.focal_loss = FocalLoss(reduction='none')
            self.criterion = None
        else:
            raise ValueError(f"Unknown base_criteria: {base_criteria}")
        
        self.base_criteria = base_criteria
    
    def update_weights(self, per_class_metrics: torch.Tensor):
        """Update class weights based on per-class performance metrics.
        
        Args:
            per_class_metrics: Per-class metrics (e.g., F1 scores) [num_classes]
        """
        # Invert metrics: lower performance -> higher weight
        weights = 1.0 / (per_class_metrics + 1e-8)
        self.class_weights = weights / weights.sum() * len(weights)
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Model predictions [batch_size, num_classes]
            targets: Ground truth binary labels [batch_size, num_classes]
        
        Returns:
            Weighted loss
        """
        if self.base_criteria == "focal":
            loss = self.focal_loss(logits, targets)
        else:
            loss = self.criterion(logits, targets)
        
        weights = self.class_weights.view(1, -1).expand_as(loss)
        loss = loss * weights
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss.mean(dim=1)


def compute_class_weights(targets: torch.Tensor, method: str = "inverse_frequency") -> torch.Tensor:
    """Compute class weights from label distribution.
    
    Args:
        targets: Binary labels [num_samples, num_classes]
        method: 'inverse_frequency', 'log_inverse', or 'sqrt_inverse'
    
    Returns:
        Class weights [num_classes]
    """
    # Ensure targets is a torch tensor
    if not isinstance(targets, torch.Tensor):
        targets = torch.from_numpy(targets).float() if isinstance(targets, np.ndarray) else torch.tensor(targets).float()
    
    num_classes = targets.shape[1]
    class_counts = targets.sum(dim=0).float()
    
    if method == "inverse_frequency":
        # w_c = 1 / n_c
        weights = 1.0 / (class_counts + 1e-8)
    elif method == "log_inverse":
        # w_c = 1 / log(n_c + 1)
        weights = 1.0 / torch.log(class_counts + 2)
    elif method == "sqrt_inverse":
        # w_c = 1 / sqrt(n_c)
        weights = 1.0 / torch.sqrt(class_counts + 1e-8)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Normalize weights
    weights = weights / weights.sum() * num_classes
    return weights
