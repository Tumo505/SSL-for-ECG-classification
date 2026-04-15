#!/usr/bin/env python3
"""Train SimCLR baseline for ECG SSL pre-training."""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ssrl_ecg.data.ptbxl import PTBXLRecordDataset, load_ptbxl_metadata, make_default_splits
from ssrl_ecg.models.cnn import ECGEncoder1DCNN
from ssrl_ecg.models.simclr import SimCLRModel, NTXentLoss, SimCLRAugmentations
from ssrl_ecg.utils import choose_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SimCLR baseline for ECG SSL pre-training.")
    parser.add_argument("--data-root", type=Path, default=Path("data/PTB-XL"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--projection-dim", type=int, default=128)
    parser.add_argument("--signal-length", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("checkpoints/ssl_simclr.pt"))
    return parser.parse_args()


class SimCLRDataset(torch.utils.data.Dataset):
    """Wrapper for SimCLR with on-the-fly augmentation."""
    
    def __init__(self, base_dataset, augmentor):
        self.base_dataset = base_dataset
        self.augmentor = augmentor
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        x = self.base_dataset[idx]  # Returns only x when PTBXLRecordDataset(return_labels=False)
        x1, x2 = self.augmentor(x)
        return x1, x2


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = choose_device()

    db_df, labels = load_ptbxl_metadata(args.data_root)
    splits = make_default_splits(db_df)

    base_ds = PTBXLRecordDataset(
        data_root=args.data_root,
        db_df=db_df,
        labels=labels,
        indices=splits.train_idx,
        use_high_resolution=False,
        signal_length=args.signal_length,
        return_labels=False,
    )
    
    # Wrap with augmentation
    augmentor = SimCLRAugmentations(signal_length=args.signal_length, prob=0.7)
    train_ds = SimCLRDataset(base_ds, augmentor)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    # Model
    encoder = ECGEncoder1DCNN(in_ch=12, width=64)
    model = SimCLRModel(encoder=encoder, projection_dim=args.projection_dim).to(device)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Loss
    criterion = NTXentLoss(temperature=args.temperature, batch_size=args.batch_size)

    print("\n[SIMCLR TRAINING]")
    print(f"  Encoder: ECGEncoder1DCNN(width=64)")
    print(f"  Projection dim: {args.projection_dim}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Batch size: {args.batch_size}")
    print()

    model.train()
    for epoch in range(1, args.epochs + 1):
        running_loss = 0.0
        n = 0
        pbar = tqdm(train_loader, desc=f"SimCLR Epoch {epoch}/{args.epochs}")
        
        for x1, x2 in pbar:
            x1 = x1.to(device)
            x2 = x2.to(device)

            optimizer.zero_grad(set_to_none=True)
            
            # Forward pass for both views
            _, z1 = model(x1)
            _, z2 = model(x2)
            
            # Compute NT-Xent loss
            loss = criterion(z1, z2)
            
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x1.size(0)
            n += x1.size(0)
            pbar.set_postfix(loss=f"{running_loss / max(1, n):.4f}")

    # Save encoder only (like other SSL methods)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"encoder": model.encoder.state_dict()}, args.out)
    print(f"\nSaved SimCLR encoder checkpoint to: {args.out}")


if __name__ == "__main__":
    main()
