from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ssrl_ecg.data.ptbxl import PTBXLRecordDataset, load_ptbxl_metadata, make_default_splits, sample_labelled_indices
from ssrl_ecg.models.cnn import ECGClassifier, ECGEncoder1DCNN
from ssrl_ecg.models.losses import FocalLoss, WeightedBCELoss, ClassBalancedLoss, compute_class_weights
from ssrl_ecg.data.balancing import create_balanced_dataloader, report_class_imbalance
from ssrl_ecg.utils import choose_device, multilabel_metrics, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Supervised PTB-XL baseline from scratch.")
    parser.add_argument("--data-root", type=Path, default=Path("data/PTB-XL"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--label-fraction", type=float, default=0.1)
    parser.add_argument("--signal-length", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("checkpoints/supervised.pt"))
    parser.add_argument("--loss", type=str, default="focal", choices=["bce", "focal", "weighted", "class_balanced"],
                       help="Loss function to use")
    parser.add_argument("--balance-strategy", type=str, default="oversample", choices=["standard", "stratified", "oversample"],
                       help="Data balancing strategy")
    return parser.parse_args()


def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    y_true = []
    y_prob = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            prob = torch.sigmoid(logits).cpu().numpy()
            y_prob.append(prob)
            y_true.append(y.numpy())
    y_true_arr = np.concatenate(y_true, axis=0)
    y_prob_arr = np.concatenate(y_prob, axis=0)
    return multilabel_metrics(y_true_arr, y_prob_arr)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = choose_device()

    db_df, labels = load_ptbxl_metadata(args.data_root)
    splits = make_default_splits(db_df)
    sampled_train_idx = sample_labelled_indices(splits.train_idx, labels, args.label_fraction, args.seed)

    train_ds = PTBXLRecordDataset(args.data_root, db_df, labels, sampled_train_idx, signal_length=args.signal_length)
    val_ds = PTBXLRecordDataset(args.data_root, db_df, labels, splits.val_idx, signal_length=args.signal_length)

    train_labels = labels[sampled_train_idx]
    # Ensure train_labels is a torch tensor
    if not isinstance(train_labels, torch.Tensor):
        train_labels = torch.from_numpy(train_labels).float()
    
    # Report class imbalance
    print("\n[TRAIN SET CLASS DISTRIBUTION]")
    report_class_imbalance(train_labels, "Training Set")

    # Create balanced data loaders
    train_loader = create_balanced_dataloader(
        train_ds, train_labels, batch_size=args.batch_size,
        strategy=args.balance_strategy, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    encoder = ECGEncoder1DCNN(in_ch=12, width=64)
    model = ECGClassifier(encoder=encoder, n_classes=labels.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Select loss function
    print(f"\n[LOSS CONFIGURATION]")
    print(f"  Loss function: {args.loss}")
    print(f"  Balancing strategy: {args.balance_strategy}")
    
    if args.loss == "focal":
        criterion = FocalLoss(alpha=0.25, gamma=2.0, reduction="mean")
    elif args.loss == "weighted":
        class_weights = compute_class_weights(train_labels, method="inverse_frequency").to(device)
        criterion = WeightedBCELoss(class_weights=class_weights, reduction="mean")
    elif args.loss == "class_balanced":
        if not isinstance(train_labels, torch.Tensor):
            train_labels_tensor = torch.from_numpy(train_labels).float()
        else:
            train_labels_tensor = train_labels
        class_counts = (train_labels_tensor == 1).sum(dim=0)
        criterion = ClassBalancedLoss(class_counts, beta=0.9999, reduction="mean")
    else:
        criterion = torch.nn.BCEWithLogitsLoss()

    best_f1 = -1.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        n = 0
        pbar = tqdm(train_loader, desc=f"Sup Epoch {epoch}/{args.epochs}")
        for x, y in pbar:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running += loss.item() * x.size(0)
            n += x.size(0)
            pbar.set_postfix(loss=f"{running / max(1, n):.4f}")

        metrics = evaluate(model, val_loader, device)
        print({"epoch": epoch, **metrics})
        if metrics["f1_macro"] > best_f1:
            best_f1 = metrics["f1_macro"]
            best_state = model.state_dict()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": best_state}, args.out)
    print(f"Saved best supervised checkpoint to: {args.out}")


if __name__ == "__main__":
    main()
