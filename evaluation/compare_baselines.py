#!/usr/bin/env python3
"""
Phase 3: Baseline Comparisons
Evaluate SimCLR, BYOL, and ResNet-1D baselines against supervised methods.
"""

from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from ssrl_ecg.data.ptbxl import PTBXLRecordDataset, load_ptbxl_metadata, make_default_splits, sample_labelled_indices
from ssrl_ecg.models.cnn import ECGClassifier, ECGEncoder1DCNN
from ssrl_ecg.models.resnet1d import ResNet1D
from ssrl_ecg.utils import choose_device, multilabel_metrics


def fine_tune_ssl_encoder(ssl_encoder, train_loader, val_loader, device, epochs=30, lr=1e-3):
    """Fine-tune SSL pre-trained encoder on labeled data."""
    
    # Freeze encoder, train only classifier head
    classifier = ECGClassifier(ssl_encoder, n_classes=5).to(device)
    
    # Freeze encoder
    for param in ssl_encoder.parameters():
        param.requires_grad = False
    
    optimizer = torch.optim.Adam(classifier.classifier.parameters(), lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    best_auroc = 0.0
    best_state = None
    
    for epoch in range(epochs):
        classifier.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            logits = classifier(x)
            loss = criterion(logits, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Validate
        classifier.eval()
        y_true_list = []
        y_prob_list = []
        
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                logits = classifier(x)
                y_prob = torch.sigmoid(logits).cpu().numpy()
                y_prob_list.append(y_prob)
                y_true_list.append(y.numpy())
        
        y_true = np.concatenate(y_true_list, axis=0)
        y_prob = np.concatenate(y_prob_list, axis=0)
        metrics = multilabel_metrics(y_true, y_prob)
        
        if metrics['auroc_macro'] > best_auroc:
            best_auroc = metrics['auroc_macro']
            best_state = classifier.state_dict()
    
    classifier.load_state_dict(best_state)
    return classifier


def evaluate_checkpoint(checkpoint_path, model_type, data_root, device, is_ssl=False):
    """Evaluate a checkpoint on test set."""
    
    if not Path(checkpoint_path).exists():
        print(f"  [WARNING] Not found: {checkpoint_path}")
        return None
    
    # Load data
    db_df, labels = load_ptbxl_metadata(data_root)
    splits = make_default_splits(db_df)
    
    sampled_train_idx = sample_labelled_indices(splits.train_idx, labels, 0.1, seed=42)
    train_ds = PTBXLRecordDataset(data_root, db_df, labels, sampled_train_idx, signal_length=1000)
    val_ds = PTBXLRecordDataset(data_root, db_df, labels, splits.val_idx, signal_length=1000)
    test_ds = PTBXLRecordDataset(data_root, db_df, labels, splits.test_idx, signal_length=1000)
    
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)
    
    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    
    if is_ssl:
        # SSL encoder - fine-tune on labeled data
        encoder = ECGEncoder1DCNN(in_ch=12, width=64)
        if isinstance(ckpt, dict) and "encoder" in ckpt:
            encoder.load_state_dict(ckpt["encoder"])
        else:
            encoder.load_state_dict(ckpt)
        
        encoder = encoder.to(device)
        
        # Fine-tune
        model = fine_tune_ssl_encoder(encoder, train_loader, val_loader, device, epochs=30)
        model = model.to(device)
    else:
        # Supervised model - direct evaluation
        if model_type == "resnet1d":
            model = ResNet1D(in_channels=12, num_classes=5, width=64)
        else:
            encoder = ECGEncoder1DCNN(in_ch=12, width=64)
            model = ECGClassifier(encoder, n_classes=5)
        
        if isinstance(ckpt, dict) and "model" in ckpt:
            model.load_state_dict(ckpt["model"])
        else:
            model.load_state_dict(ckpt)
        
        model = model.to(device)
    
    # Evaluate on test set
    model.eval()
    y_true_list = []
    y_prob_list = []
    
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            logits = model(x)
            y_prob = torch.sigmoid(logits).cpu().numpy()
            y_prob_list.append(y_prob)
            y_true_list.append(y.numpy())
    
    y_true = np.concatenate(y_true_list, axis=0)
    y_prob = np.concatenate(y_prob_list, axis=0)
    
    metrics = multilabel_metrics(y_true, y_prob)
    
    return metrics


def main():
    """Evaluate Phase 3 baselines."""
    
    device = choose_device()
    data_root = Path("data/PTB-XL")
    ckpt_dir = Path("checkpoints")
    
    print("\n" + "="*70)
    print("PHASE 3: BASELINE COMPARISONS")
    print("="*70)
    
    baselines = {
        "Supervised (10% BCE)": (ckpt_dir / "supervised_bce_baseline.pt", "cnn", False),
        "Supervised (10% Focal+OS)": (ckpt_dir / "supervised.pt", "cnn", False),
        "SimCLR (pre-trained + fine-tune)": (ckpt_dir / "ssl_simclr.pt", "cnn", True),
        "ResNet-1D (10%)": (ckpt_dir / "resnet1d.pt", "resnet1d", False),
    }
    
    results = {}
    
    for name, (checkpoint_path, model_type, is_ssl) in baselines.items():
        print(f"\n{name}...")
        
        if not checkpoint_path.exists():
            print(f"  [WARNING] Checkpoint not found")
            continue
        
        try:
            metrics = evaluate_checkpoint(checkpoint_path, model_type, data_root, device, is_ssl)
            if metrics:
                results[name] = metrics
                print(f"  [OK] AUROC: {metrics['auroc_macro']:.4f}, F1: {metrics['f1_macro']:.4f}")
        except Exception as e:
            print(f"  [ERROR] Error: {str(e)[:100]}")
    
    # Summary table
    if results:
        print("\n" + "="*70)
        print("SUMMARY TABLE")
        print("="*70)
        print(f"{'Method':<40} {'AUROC':>10} {'F1':>10}")
        print("-"*60)
        
        for name, metrics in sorted(results.items(), 
                                   key=lambda x: x[1]['auroc_macro'], 
                                   reverse=True):
            print(f"{name:<40} {metrics['auroc_macro']:>10.4f} {metrics['f1_macro']:>10.4f}")
        
        # Find best
        best = max(results.items(), key=lambda x: x[1]['auroc_macro'])
        print(f"\n🏆 Best baseline: {best[0]} (AUROC: {best[1]['auroc_macro']:.4f})")


if __name__ == "__main__":
    main()
