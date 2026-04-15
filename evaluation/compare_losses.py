#!/usr/bin/env python3
"""Comprehensive comparison of loss functions and balancing strategies on PTB-XL test set."""

from pathlib import Path
import torch
from torch.utils.data import DataLoader

from ssrl_ecg.data.ptbxl import PTBXLRecordDataset, load_ptbxl_metadata, make_default_splits
from ssrl_ecg.models.cnn import ECGClassifier, ECGEncoder1DCNN
from ssrl_ecg.utils import choose_device, multilabel_metrics


def evaluate_checkpoint(checkpoint_path, checkpoint_name, data_root, device):
    """Evaluate a single checkpoint on test set."""
    
    if not Path(checkpoint_path).exists():
        print(f" Checkpoint not found: {checkpoint_path}")
        return None
    
    print(f"\n[{checkpoint_name}]")
    print(f"  Path: {checkpoint_path}")
    
    # Load metadata and create test dataset
    db_df, labels = load_ptbxl_metadata(data_root)
    splits = make_default_splits(db_df)
    
    test_ds = PTBXLRecordDataset(data_root, db_df, labels, splits.test_idx, signal_length=1000)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)
    
    # Load model
    encoder = ECGEncoder1DCNN(in_ch=12, width=64)
    classifier = ECGClassifier(encoder, n_classes=5)
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        classifier.load_state_dict(checkpoint["model"])
    elif isinstance(checkpoint, dict) and "encoder" in checkpoint:
        encoder.load_state_dict(checkpoint["encoder"])
    else:
        classifier.load_state_dict(checkpoint)
    
    classifier = classifier.to(device)
    classifier.eval()
    
    # Collect predictions
    y_true_list = []
    y_prob_list = []
    
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            logits = classifier(x)
            y_prob = torch.sigmoid(logits).cpu().numpy()
            y_prob_list.append(y_prob)
            y_true_list.append(y.numpy())
    
    y_true = np.concatenate(y_true_list, axis=0)
    y_prob = np.concatenate(y_prob_list, axis=0)
    
    # Compute metrics
    metrics = multilabel_metrics(y_true, y_prob)
    
    # Print results
    print(f"  Macro AUROC: {metrics['auroc_macro']:.4f}")
    print(f"  Macro F1: {metrics['f1_macro']:.4f}")
    print(f"  Sensitivity (micro): {metrics['sensitivity_micro']:.4f}")
    print(f"  Specificity (micro): {metrics['specificity_micro']:.4f}")
    
    # Per-class metrics
    print("\n  Per-class AUROC:")
    for c, auroc in enumerate(metrics.get('auroc_per_class', [])):
        if auroc is not None:
            print(f"    Class {c}: {auroc:.4f}")
    
    return metrics


def main():
    import numpy as np
    
    device = choose_device()
    data_root = Path("data/PTB-XL")
    checkpoint_dir = Path("checkpoints")
    
    print("\n" + "="*70)
    print("TEST SET EVALUATION: LOSS FUNCTION COMPARISON")
    print("="*70)
    
    results = {}
    
    # Evaluate BCE baseline
    bce_metrics = evaluate_checkpoint(
        checkpoint_dir / "supervised_bce_baseline.pt",
        "BCE Baseline (No Balancing)",
        data_root, device
    )
    if bce_metrics:
        results['BCE Baseline'] = bce_metrics
    
    # Evaluate focal loss + oversampling
    focal_metrics = evaluate_checkpoint(
        checkpoint_dir / "supervised.pt",
        "Focal Loss + Oversampling",
        data_root, device
    )
    if focal_metrics:
        results['Focal Loss + Oversampling'] = focal_metrics
    
    # Summary comparison
    print("\n" + "="*70)
    print("SUMMARY COMPARISON")
    print("="*70)
    
    if len(results) >= 2:
        names = list(results.keys())
        m1, m2 = results[names[0]], results[names[1]]
        
        auroc_diff = m2['auroc_macro'] - m1['auroc_macro']
        f1_diff = m2['f1_macro'] - m1['f1_macro']
        
        print(f"\n{names[0]}:")
        print(f"  AUROC: {m1['auroc_macro']:.4f}, F1: {m1['f1_macro']:.4f}")
        
        print(f"\n{names[1]}:")
        print(f"  AUROC: {m2['auroc_macro']:.4f}, F1: {m2['f1_macro']:.4f}")
        
        print(f"\nImprovement (Absolute Difference):")
        print(f"  AUROC: {auroc_diff:+.4f} ({auroc_diff*100:+.2f}%)")
        print(f"  F1: {f1_diff:+.4f} ({f1_diff*100:+.2f}%)")
        
        if auroc_diff > 0 and f1_diff > 0:
            print(f"\n[OK] Focal loss + oversampling shows improvements on both metrics!")
        elif auroc_diff > 0 or f1_diff > 0:
            print(f"\n[WARNING] Mixed results - improvements on some metrics, check details.")
        else:
            print(f"\n[FAILED] No improvement - further investigation needed.")


if __name__ == "__main__":
    main()
