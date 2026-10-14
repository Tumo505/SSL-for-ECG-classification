"""SimCLR: A Simple Framework for Contrastive Learning of Visual Representations.

Adapted for ECG signals. Reference: Chen et al., ICML 2020
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class SimCLRProjectionHead(nn.Module):
    """Projection head for SimCLR contrastive learning."""

    def __init__(
        self, input_dim: int = 256, hidden_dim: int = 2048, output_dim: int = 128
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SimCLRModel(nn.Module):
    """SimCLR model for self-supervised ECG learning."""

    def __init__(self, encoder: nn.Module, projection_dim: int = 128):
        """
        Args:
            encoder: Feature extraction backbone (e.g., ECGEncoder1DCNN)
            projection_dim: Dimension of projection head output
        """
        super().__init__()
        self.encoder = encoder
        self.projection_head = SimCLRProjectionHead(
            input_dim=encoder.out_channels, hidden_dim=2048, output_dim=projection_dim
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input tensor [batch_size, channels, length]

        Returns:
            Tuple of (features, projections)
        """
        # Backbone
        h = self.encoder(x)

        # Global average pooling
        h = F.adaptive_avg_pool1d(h, 1).squeeze(-1)  # [batch_size, encoder_dim]

        # Projection
        z = self.projection_head(h)

        return h, z


class NTXentLoss(nn.Module):
    """Normalized Temperature-scaled Cross Entropy Loss (NT-Xent).

    Used in SimCLR for contrastive learning.
    """

    def __init__(self, temperature: float = 0.07, batch_size: int = 256):
        """
        Args:
            temperature: Temperature parameter for scaling
            batch_size: Batch size (used for constructing similarity matrix)
        """
        super().__init__()
        self.temperature = temperature
        self.batch_size = batch_size

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """Compute NT-Xent loss.

        Args:
            z_i: Projections from first augmentation [batch_size, proj_dim]
            z_j: Projections from second augmentation [batch_size, proj_dim]

        Returns:
            Scalar loss value
        """
        if z_i.shape != z_j.shape or z_i.ndim != 2 or z_i.shape[0] < 2:
            raise ValueError("NT-Xent requires matching [N,D] views with N >= 2")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        z = F.normalize(torch.cat([z_i, z_j], dim=0), dim=1)
        n = z_i.shape[0]
        logits = z @ z.T / self.temperature
        logits = logits.masked_fill(
            torch.eye(2 * n, dtype=torch.bool, device=z.device), -torch.inf
        )
        positives = (torch.arange(2 * n, device=z.device) + n) % (2 * n)
        return F.cross_entropy(logits, positives)


class SimCLRAugmentations:
    """Enhanced augmentation pipeline for ECG signals in SimCLR."""

    def __init__(self, signal_length: int = 5000, prob: float = 0.8):
        """
        Args:
            signal_length: Length of ECG signal
            prob: Probability of applying strong augmentations
        """
        self.signal_length = signal_length
        self.prob = prob
        # Import here to avoid circular imports
        from ssrl_ecg.augmentations import ECGAugmentations

        self.augmentations = ECGAugmentations(
            signal_length=signal_length, sampling_rate=signal_length // 10
        )

    def __call__(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply advanced augmentation pipeline to get two views.

        Args:
            x: Input tensor [batch?, channels, length]

        Returns:
            Tuple of two augmented views
        """
        # Use the enhanced augmentation pipeline
        return self.augmentations(x)
