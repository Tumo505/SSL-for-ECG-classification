#!/usr/bin/env python3
"""BYOL (Bootstrap Your Own Latent) baseline for SSL pre-training on ECG."""

import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ssrl_ecg.data.ptbxl import PTBXLRecordDataset, load_ptbxl_metadata, make_default_splits
from ssrl_ecg.models.cnn import ECGEncoder1DCNN
from ssrl_ecg.utils import choose_device, set_seed


class BYOLAugmentations:
    """Simple augmentations for BYOL."""
    
    def __init__(self, signal_length=1000, prob=0.7):
        self.signal_length = signal_length
        self.prob = prob
    
    def __call__(self, x):
        """Return two augmented views."""
        x1 = self._augment(x.clone())
        x2 = self._augment(x.clone())
        return x1, x2
    
    def _augment(self, x):
        """Apply random augmentations."""
        # Random time shift
        if torch.rand(1).item() < self.prob:
            shift = torch.randint(-50, 51, (1,)).item()
            if shift > 0:
                # Pad left with zeros, keep right part: [batch, channels, time]
                x = torch.cat([torch.zeros_like(x[:, :, :shift]), x[:, :, :self.signal_length-shift]], dim=2)
            elif shift < 0:
                # Keep left part, pad right with zeros
                x = torch.cat([x[:, :, -shift:self.signal_length], torch.zeros_like(x[:, :, :shift])], dim=2)
        
        # Random scaling
        if torch.rand(1).item() < self.prob:
            scale = (torch.randn(1) * 0.1 + 1.0).item()
            x = x * scale
        
        # Random jitter
        if torch.rand(1).item() < self.prob:
            jitter = torch.randn_like(x) * 0.05
            x = x + jitter
        
        return x


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
        
        # Get encoder output dimension by forward pass
        dummy_input = torch.randn(1, 12, 1000)
        with torch.no_grad():
            encoder_out = encoder(dummy_input)
            encoder_dim = encoder_out.shape[1]
        
        # Online and target networks
        self.online_projector = BYOLProjector(encoder_dim, hidden_dim, projection_dim)
        self.online_predictor = BYOLPredictor(projection_dim, hidden_dim, projection_dim)
        
        self.target_encoder = encoder.__class__(in_ch=12, width=64)
        self.target_encoder.load_state_dict(encoder.state_dict())
        self.target_projector = BYOLProjector(encoder_dim, hidden_dim, projection_dim)  # Fixed: use same encoder_dim
        
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
        for online_p, target_p in zip(self.encoder.parameters(), 
                                      self.target_encoder.parameters()):
            target_p.data = tau * target_p.data + (1 - tau) * online_p.data
        
        for online_p, target_p in zip(self.online_projector.parameters(),
                                      self.target_projector.parameters()):
            target_p.data = tau * target_p.data + (1 - tau) * online_p.data


def byol_loss(pred1, target1, pred2, target2):
    """BYOL loss (L2 distance between predictions and targets)."""
    
    def regression_loss(pred, target):
        # Normalize
        pred = nn.functional.normalize(pred, dim=1)
        target = nn.functional.normalize(target, dim=1)
        return 2 - 2 * (pred * target).sum(dim=1).mean()
    
    return regression_loss(pred1, target1) + regression_loss(pred2, target2)


def parse_args():
    parser = argparse.ArgumentParser(description="Train BYOL baseline for ECG SSL pre-training.")
    parser.add_argument("--data-root", type=Path, default=Path("data/PTB-XL"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--signal-length", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("checkpoints/ssl_byol.pt"))
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = choose_device()
    
    # Load data
    db_df, labels = load_ptbxl_metadata(args.data_root)
    splits = make_default_splits(db_df)
    
    train_ds = PTBXLRecordDataset(
        args.data_root, db_df, labels, splits.train_idx,
        signal_length=args.signal_length, return_labels=False
    )
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, 
                            shuffle=True, num_workers=0)
    
    # Model
    encoder = ECGEncoder1DCNN(in_ch=12, width=64)
    model = BYOLModel(encoder).to(device)
    
    # Optimizer (only for online network)
    optimizer = torch.optim.Adam(
        list(model.encoder.parameters()) +
        list(model.online_projector.parameters()) +
        list(model.online_predictor.parameters()),
        lr=args.lr
    )
    
    print("\n[BYOL TRAINING]")
    print(f"  Encoder: ECGEncoder1DCNN(width=64)")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Training samples: {len(train_ds)}")
    
    model.train()
    for epoch in range(1, args.epochs + 1):
        running_loss = 0.0
        n = 0
        
        pbar = tqdm(train_loader, desc=f"BYOL Epoch {epoch}/{args.epochs}")
        for x in pbar:
            x = x.to(device)
            
            # Augment
            aug = BYOLAugmentations(signal_length=args.signal_length, prob=0.7)
            x1, x2 = aug(x)
            x1, x2 = x1.to(device), x2.to(device)
            
            # Forward
            pred1, p1_target, pred2, p2_target = model(x1, x2)
            loss = byol_loss(pred1, p1_target, pred2, p2_target)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update target network with EMA
            model.update_target_network(tau=0.999)
            
            running_loss += loss.item() * x.size(0)
            n += x.size(0)
            
            pbar.set_postfix({'loss': running_loss / n})
        
        epoch_loss = running_loss / n
        print(f"Epoch {epoch} loss: {epoch_loss:.4f}")
    
    # Save checkpoint
    args.out.parent.mkdir(exist_ok=True)
    torch.save({'encoder': model.encoder.state_dict()}, args.out)
    print(f"\nSaved BYOL encoder checkpoint to: {args.out}")


if __name__ == "__main__":
    main()
