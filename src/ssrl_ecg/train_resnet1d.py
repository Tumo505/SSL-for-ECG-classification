#!/usr/bin/env python3
"""Train ResNet-1D baseline on PTB-XL."""

import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ssrl_ecg.data.ptbxl import PTBXLRecordDataset, load_ptbxl_metadata, make_default_splits, sample_labelled_indices
from ssrl_ecg.models.resnet1d import ResNet1D
from ssrl_ecg.models.losses import compute_class_weights
from ssrl_ecg.data.balancing import create_balanced_dataloader, report_class_imbalance
from ssrl_ecg.utils import choose_device, multilabel_metrics, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train ResNet-1D baseline on PTB-XL.")
    parser.add_argument("--data-root", type=Path, default=Path("data/PTB-XL"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--label-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("checkpoints/resnet1d.pt"))
    return parser.parse_args()


def evaluate(model, loader, device):
    """Evaluate model on validation set."""
    model.eval()
    y_true_list = []
    y_prob_list = []
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            y_prob = torch.sigmoid(logits).cpu().numpy()
            y_prob_list.append(y_prob)
            y_true_list.append(y.numpy())
    
    import numpy as np
    y_true = np.concatenate(y_true_list, axis=0)
    y_prob = np.concatenate(y_prob_list, axis=0)
    
    return multilabel_metrics(y_true, y_prob)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = choose_device()
    
    # Load data
    db_df, labels = load_ptbxl_metadata(args.data_root)
    splits = make_default_splits(db_df)
    sampled_train_idx = sample_labelled_indices(splits.train_idx, labels, args.label_fraction, args.seed)
    
    train_ds = PTBXLRecordDataset(args.data_root, db_df, labels, sampled_train_idx, 
                                  signal_length=1000)
    val_ds = PTBXLRecordDataset(args.data_root, db_df, labels, splits.val_idx, 
                                signal_length=1000)
    
    train_labels = labels[sampled_train_idx]
    
    # Report class imbalance
    print("\n[TRAIN SET CLASS DISTRIBUTION]")
    report_class_imbalance(train_labels, "Training Set")
    
    # Create data loaders with balancing
    train_loader = create_balanced_dataloader(
        train_ds, train_labels, batch_size=args.batch_size,
        strategy="oversample", shuffle=True, num_workers=0
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # Model
    model = ResNet1D(in_channels=12, num_classes=5, width=64).to(device)
    
    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    weights = compute_class_weights(train_labels)
    if isinstance(weights, torch.Tensor):
        class_weights = weights.float().to(device)
    else:
        class_weights = torch.from_numpy(weights).float().to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=class_weights)
    
    print(f"\n[RESNET-1D TRAINING]")
    print(f"  Model: ResNet-1D(width=64)")
    print(f"  Loss: BCEWithLogitsLoss (weighted)")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    
    best_auroc = 0.0
    best_checkpoint = None
    
    model.train()
    for epoch in range(1, args.epochs + 1):
        running_loss = 0.0
        n = 0
        
        pbar = tqdm(train_loader, desc=f"ResNet Epoch {epoch}/{args.epochs}")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            
            logits = model(x)
            loss = criterion(logits, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * x.size(0)
            n += x.size(0)
            pbar.set_postfix({'loss': running_loss / n})
        
        # Validate
        val_metrics = evaluate(model, val_loader, device)
        print(f"Epoch {epoch}: AUROC={val_metrics['auroc_macro']:.4f}, F1={val_metrics['f1_macro']:.4f}")
        
        # Save best checkpoint
        if val_metrics['auroc_macro'] > best_auroc:
            best_auroc = val_metrics['auroc_macro']
            best_checkpoint = {
                'model': model.state_dict(),
                'epoch': epoch,
                'metrics': val_metrics,
            }
    
    # Save best model
    if best_checkpoint:
        args.out.parent.mkdir(exist_ok=True)
        torch.save(best_checkpoint, args.out)
        print(f"\nSaved best ResNet-1D checkpoint to: {args.out}")
        print(f"Best validation AUROC: {best_auroc:.4f}")


if __name__ == "__main__":
    main()
