"""SimCLR: A Simple Framework for Contrastive Learning of Visual Representations.

Adapted for ECG signals. Reference: Chen et al., ICML 2020
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class SimCLRProjectionHead(nn.Module):
    """Projection head for SimCLR contrastive learning."""
    
    def __init__(self, input_dim: int = 256, hidden_dim: int = 2048, output_dim: int = 128):
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
            input_dim=encoder.out_channels,
            hidden_dim=2048,
            output_dim=projection_dim
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
        # Normalize projections
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        
        # Concatenate: [2*batch_size, proj_dim]
        representations = torch.cat([z_i, z_j], dim=0)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(representations, representations.t())  # [2N, 2N]
        
        # Create mask for positive pairs (diagonal)
        batch_size = z_i.shape[0]
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z_i.device)
        
        # Extract positive pairs (similarity of z_i with z_j)
        pos_sim = torch.diag(similarity_matrix, batch_size)  # [batch_size]
        pos_sim = torch.cat([pos_sim, torch.diag(similarity_matrix, -batch_size)])  # [2*batch_size]
        
        # Scale by temperature
        logits = similarity_matrix / self.temperature
        
        # Remove diagonal (self-similarity)
        logits = logits[~mask].view(2 * batch_size, -1)  # [2N, 2N-1]
        
        # Labels: 0-indexed position of positive pair
        labels = torch.arange(batch_size, device=z_i.device)
        labels = torch.cat([labels, labels])
        
        # Cross-entropy loss
        loss = F.cross_entropy(logits, labels, reduction='mean')
        
        return loss


class SimCLRAugmentations:
    """Augmentation pipeline for ECG signals in SimCLR."""
    
    def __init__(self, signal_length: int = 1000, prob: float = 0.5):
        """
        Args:
            signal_length: Length of ECG signal
            prob: Probability of applying each augmentation
        """
        self.signal_length = signal_length
        self.prob = prob
    
    def time_shift(self, x: torch.Tensor, max_shift: int = 50) -> torch.Tensor:
        """Random time shift augmentation."""
        if torch.rand(1) < self.prob:
            shift = torch.randint(-max_shift, max_shift, (1,)).item()
            if shift > 0:
                x = F.pad(x, (shift, 0))[:, :self.signal_length]
            elif shift < 0:
                x = F.pad(x, (0, -shift))[:, -self.signal_length:]
        return x
    
    def scale(self, x: torch.Tensor, scale_range: Tuple[float, float] = (0.8, 1.2)) -> torch.Tensor:
        """Random amplitude scaling."""
        if torch.rand(1) < self.prob:
            scale = torch.empty(1).uniform_(*scale_range).item()
            x = x * scale
        return x
    
    def jitter(self, x: torch.Tensor, jitter_std: float = 0.001) -> torch.Tensor:
        """Add Gaussian noise (jitter)."""
        if torch.rand(1) < self.prob:
            noise = torch.randn_like(x) * jitter_std
            x = x + noise
        return x
    
    def __call__(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply augmentation pipeline to get two views.
        
        Args:
            x: Input tensor [channels, length]
        
        Returns:
            Tuple of two augmented views
        """
        # First view
        x1 = self.time_shift(x.clone())
        x1 = self.scale(x1)
        x1 = self.jitter(x1)
        
        # Second view (independent augmentations)
        x2 = self.time_shift(x.clone())
        x2 = self.scale(x2)
        x2 = self.jitter(x2)
        
        return x1, x2
