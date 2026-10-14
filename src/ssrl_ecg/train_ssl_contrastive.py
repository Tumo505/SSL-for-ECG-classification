from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from ssrl_ecg.data.ptbxl import PTBXLRecordDataset, load_ptbxl_metadata, make_default_splits
from ssrl_ecg.models.cnn import ECGEncoder1DCNN
from ssrl_ecg.utils import choose_device, set_seed


class ContrastiveProjectionHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 256, out_dim: int = 128):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj(x), dim=1)


class SSLContrastiveModel(nn.Module):
    def __init__(self, in_ch: int = 12, width: int = 64, proj_dim: int = 128):
        super().__init__()
        self.encoder = ECGEncoder1DCNN(in_ch=in_ch, width=width)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj_head = ContrastiveProjectionHead(self.encoder.out_channels, out_dim=proj_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        pooled = self.pool(z).squeeze(-1)
        proj = self.proj_head(pooled)
        return proj


def random_augment_signal(x: torch.Tensor, aug_type: str = "mix") -> torch.Tensor:
    """Apply simple augmentations: noise, time-shift, scaling."""
    if aug_type == "noise":
        return x + torch.randn_like(x) * 0.05
    elif aug_type == "scale":
        scale = torch.rand((x.shape[0], 1, 1), device=x.device) * 0.2 + 0.9
        return x * scale
    elif aug_type == "shift":
        shift = torch.randint(-10, 11, (1,)).item()
        if shift > 0:
            return torch.cat([torch.zeros(x.shape[0], x.shape[1], shift, device=x.device), x[:, :, :-shift]], dim=2)
        elif shift < 0:
            return torch.cat([x[:, :, -shift:], torch.zeros(x.shape[0], x.shape[1], -shift, device=x.device)], dim=2)
    return x


from ssrl_ecg.models.simclr import NTXentLoss


def main() -> None:
    from ssrl_ecg.cli import run_training_cli
    run_training_cli("simclr")


if __name__ == "__main__":
    main()
