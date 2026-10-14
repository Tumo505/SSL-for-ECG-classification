#!/usr/bin/env python3
"""BYOL (Bootstrap Your Own Latent) baseline for SSL pre-training on ECG."""

import argparse
import copy
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ssrl_ecg.data.ptbxl import (
    PTBXLRecordDataset,
    load_ptbxl_metadata,
    make_default_splits,
)
from ssrl_ecg.models.cnn import ECGEncoder1DCNN
from ssrl_ecg.utils import choose_device, set_seed
from ssrl_ecg.augmentations import ECGAugmentations  # Use enhanced augmentations


class BYOLAugmentations:
    """Wrapper for enhanced ECG augmentations."""

    def __init__(self, signal_length=5000, prob=0.8):
        self.signal_length = signal_length
        self.augmentations = ECGAugmentations(
            signal_length=signal_length, sampling_rate=signal_length // 10
        )
        self.prob = prob

    def __call__(self, x):
        """Return two augmented views using advanced augmentations."""
        return self.augmentations(x)


class BYOLProjector(nn.Module):
    """MLP projector for BYOL."""

    def __init__(self, in_features, hidden_dim=2048, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class BYOLPredictor(nn.Module):
    """Predictor network for BYOL (momentum contrast)."""

    def __init__(self, in_features, hidden_dim=2048, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class BYOLModel(nn.Module):
    """BYOL model for ECG SSL pre-training."""

    def __init__(self, encoder, projection_dim=256, hidden_dim=2048):
        super().__init__()
        self.encoder = encoder

        encoder_dim = encoder.out_channels

        # Online and target networks
        self.online_projector = BYOLProjector(encoder_dim, hidden_dim, projection_dim)
        self.online_predictor = BYOLPredictor(
            projection_dim, hidden_dim, projection_dim
        )

        self.target_encoder = copy.deepcopy(encoder)
        self.target_projector = copy.deepcopy(self.online_projector)

        # Disable gradients for target network
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        for param in self.target_projector.parameters():
            param.requires_grad = False

    def forward(self, x1, x2):
        """Forward pass with online and target networks."""

        # Online network
        z1 = self.encoder(x1)
        # Global average pooling if needed
        if len(z1.shape) > 2:
            z1 = z1.mean(dim=-1)  # Average over time dimension
        p1 = self.online_projector(z1)
        pred1 = self.online_predictor(p1)

        z2 = self.encoder(x2)
        if len(z2.shape) > 2:
            z2 = z2.mean(dim=-1)
        p2 = self.online_projector(z2)
        pred2 = self.online_predictor(p2)

        # Target network (no grad)
        with torch.no_grad():
            z1_target = self.target_encoder(x1)
            if len(z1_target.shape) > 2:
                z1_target = z1_target.mean(dim=-1)
            p1_target = self.target_projector(z1_target)

            z2_target = self.target_encoder(x2)
            if len(z2_target.shape) > 2:
                z2_target = z2_target.mean(dim=-1)
            p2_target = self.target_projector(z2_target)

        return pred1, p1_target, pred2, p2_target

    @torch.no_grad()
    def update_target_network(self, tau=0.999):
        """Update target network with EMA."""
        for online_p, target_p in zip(
            self.encoder.parameters(), self.target_encoder.parameters()
        ):
            target_p.data = tau * target_p.data + (1 - tau) * online_p.data

        for online_p, target_p in zip(
            self.online_projector.parameters(), self.target_projector.parameters()
        ):
            target_p.data = tau * target_p.data + (1 - tau) * online_p.data


def byol_loss(pred1, target1, pred2, target2):
    """BYOL loss (L2 distance between predictions and targets)."""

    def regression_loss(pred, target):
        # Normalize
        pred = nn.functional.normalize(pred, dim=1)
        target = nn.functional.normalize(target, dim=1)
        return 2 - 2 * (pred * target).sum(dim=1).mean()

    return regression_loss(pred1, target2.detach()) + regression_loss(
        pred2, target1.detach()
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train BYOL baseline for ECG SSL pre-training."
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/PTB-XL"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--signal-length", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("checkpoints/ssl_byol.pt"))
    return parser.parse_args()


def main():
    from ssrl_ecg.cli import run_training_cli

    run_training_cli("byol")


if __name__ == "__main__":
    main()
