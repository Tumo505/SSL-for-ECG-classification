#!/usr/bin/env python3
"""Per-class evaluation - check improvements for underrepresented classes."""

from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from ssrl_ecg.data.ptbxl import PTBXLRecordDataset, load_ptbxl_metadata, make_default_splits, DIAGNOSTIC_CLASSES
from ssrl_ecg.models.cnn import ECGClassifier, ECGEncoder1DCNN
from ssrl_ecg.utils import choose_device


def evaluate_per_class(checkpoint_path, name, data_root, device):
    """Evaluate checkpoint on per-class basis."""
    
    if not Path(checkpoint_path).exists():
        print(f"  [WARNING] Not found: {checkpoint_path}")
        return None
    
    # Load data
    db_df, labels = load_ptbxl_metadata(data_root)
    splits = make_default_splits(db_df)
    
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
    
    # Get predictions
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
    
    # Calculate per-class AUROC
    per_class_auroc = []
    for i in range(5):
        try:
            auroc = roc_auc_score(y_true_arr[:, i], y_prob_arr[:, i])
        except:
            auroc = np.nan
        per_class_auroc.append(auroc)
    
    print(f"\n{name}")
    print(f"  {'Class':<10} {'AUROC':<8}")
    print(f"  {'-'*18}")
    for i, auroc in enumerate(per_class_auroc):
        class_name = DIAGNOSTIC_CLASSES[i] if i < len(DIAGNOSTIC_CLASSES) else f"Class {i}"
        if np.isnan(auroc):
            print(f"  {class_name:<10} {'N/A':<8}")
        else:
            print(f"  {class_name:<10} {auroc:>7.4f}")
    
    macro_auroc = np.nanmean(per_class_auroc)
    print(f"  {'Macro':<10} {macro_auroc:>7.4f}")
    
    return per_class_auroc


def main():
    device = choose_device()
    data_root = Path("data/PTB-XL")
    ckpt_dir = Path("checkpoints")
    
    print("\n" + "="*50)
    print("PER-CLASS AUROC COMPARISON")
    print("="*50)
    
    print("\nTarget: Improve Class 3 (CD) - originally ~0.37")
    print("        Improve Class 4 (HYP) - originally ~0.40")
    
    # BCE baseline
    bce_auroc = evaluate_per_class(ckpt_dir / "supervised_bce_baseline.pt",
                                   "BCE Baseline (No Balancing)", data_root, device)
    
    # Focal + oversampling
    focal_auroc = evaluate_per_class(ckpt_dir / "supervised.pt",
                                     "Focal Loss + Oversampling", data_root, device)
    
    # Comparison
    if bce_auroc and focal_auroc:
        print("\n" + "="*50)
        print("IMPROVEMENT ANALYSIS")
        print("="*50)
        
        improvements = []
        for i, (bce, focal) in enumerate(zip(bce_auroc, focal_auroc)):
            if np.isnan(bce) or np.isnan(focal):
                continue
            
            diff = focal - bce
            pct_diff = (diff / bce) * 100 if bce > 0 else 0
            
            class_name = DIAGNOSTIC_CLASSES[i] if i < len(DIAGNOSTIC_CLASSES) else f"Class {i}"
            improvement = "[UP] " if diff > 0 else "[DOWN] "
            print(f"\n{class_name}: {improvement}{diff:+.4f} ({pct_diff:+.2f}%)")
            print(f"  {bce:.4f} → {focal:.4f}")
            
            improvements.append((class_name, diff, focal))
        
        # Check critical classes
        print("\n" + "="*50)
        print("CRITICAL CLASSES (Original Concern)")
        
        if focal_auroc[3] is not None and not np.isnan(focal_auroc[3]):
            print(f"Class 3 (CD): {focal_auroc[3]:.4f}")
            if focal_auroc[3] >= 0.60:
                print("  [OK] Target achieved (>0.60)")
            else:
                print(f"  [FAILED] Still below target (need {0.60 - focal_auroc[3]:.4f} more)")
        
        if focal_auroc[4] is not None and not np.isnan(focal_auroc[4]):
            print(f"Class 4 (HYP): {focal_auroc[4]:.4f}")
            if focal_auroc[4] >= 0.60:
                print("  [OK] Target achieved (>0.60)")
            else:
                print(f"  [FAILED] Still below target (need {0.60 - focal_auroc[4]:.4f} more)")


if __name__ == "__main__":
    main()
