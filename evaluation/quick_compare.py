#!/usr/bin/env python3
"""Quick evaluation - verify checkpoints load and make predictions."""

from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from ssrl_ecg.data.ptbxl import PTBXLRecordDataset, load_ptbxl_metadata, make_default_splits
from ssrl_ecg.models.cnn import ECGClassifier, ECGEncoder1DCNN
from ssrl_ecg.utils import choose_device, multilabel_metrics


def quick_eval(checkpoint_path, name, data_root, device):
    """Load checkpoint and test on subset of test data."""
    
    if not Path(checkpoint_path).exists():
        print(f"  [WARNING] Not found: {checkpoint_path}")
        return None
    
    # Load data
    db_df, labels = load_ptbxl_metadata(data_root)
    splits = make_default_splits(db_df)
    
    # Create dataset (full test set)
    test_ds = PTBXLRecordDataset(data_root, db_df, labels, splits.test_idx, signal_length=1000)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)
    
    # Load model
    encoder = ECGEncoder1DCNN(in_ch=12, width=64)
    classifier = ECGClassifier(encoder, n_classes=5)
    
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(ckpt, dict) and "model" in ckpt:
        classifier.load_state_dict(ckpt["model"])
    else:
        classifier.load_state_dict(ckpt)
    
    classifier = classifier.to(device)
    classifier.eval()
    
    print(f"\n[OK] {name}")
    print(f"   File: {Path(checkpoint_path).name}")
    
    # Get predictions on full test set
    y_true = []
    y_prob = []
    
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            logits = classifier(x)
            y_prob.append(torch.sigmoid(logits).cpu().numpy())
            y_true.append(y.numpy())
    
    y_true_arr = np.concatenate(y_true)
    y_prob_arr = np.concatenate(y_prob)
    
    # Calculate metrics
    metrics = multilabel_metrics(y_true_arr, y_prob_arr)
    
    print(f"   AUROC (macro): {metrics['auroc_macro']:.4f}")
    print(f"   F1 (macro): {metrics['f1_macro']:.4f}")
    
    # Per-class AUROC (most important for class imbalance analysis)
    if 'auroc_per_class' in metrics:
        print(f"   Per-class AUROC:")
        for i, auroc_i in enumerate(metrics['auroc_per_class']):
            if auroc_i is not None:
                print(f"     Class {i}: {auroc_i:.4f}")
    
    return metrics


def main():
    device = choose_device()
    data_root = Path("data/PTB-XL")
    ckpt_dir = Path("checkpoints")
    
    print("\n" + "="*60)
    print("QUICK EVALUATION: Loss Function Comparison")
    print("="*60)
    
    results = {}
    
    # BCE baseline
    bce_m = quick_eval(ckpt_dir / "supervised_bce_baseline.pt", 
                       "BCE Baseline (no balancing)", data_root, device)
    if bce_m:
        results['BCE'] = bce_m
    
    # Focal loss + oversampling  
    focal_m = quick_eval(ckpt_dir / "supervised.pt",
                         "Focal Loss + Oversampling", data_root, device)
    if focal_m:
        results['Focal+OS'] = focal_m
    
    # Summary
    if len(results) == 2:
        print("\n" + "="*60)
        print("COMPARISON SUMMARY")
        print("="*60)
        
        bce_m = results['BCE']
        focal_m = results['Focal+OS']
        
        auroc_imp = focal_m['auroc_macro'] - bce_m['auroc_macro']
        f1_imp = focal_m['f1_macro'] - bce_m['f1_macro']
        
        print(f"\nAUROC Improvement: {auroc_imp:+.4f} ({auroc_imp*100:+.2f}%)")
        print(f"F1 Improvement: {f1_imp:+.4f} ({f1_imp*100:+.2f}%)")
        
        if auroc_imp > 0.01:
            print("[OK] Significant AUROC improvement!")
        if f1_imp > 0.01:
            print("[OK] Significant F1 improvement!")


if __name__ == "__main__":
    main()
