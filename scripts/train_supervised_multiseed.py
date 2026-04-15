#!/usr/bin/env python3
"""
Phase 2 Multi-Seed Trainer (Direct Python version)
Runs training directly without subprocess for better error handling.
"""

import json
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
import argparse

# Import training setup functions
from ssrl_ecg.data.ptbxl import PTBXLRecordDataset, load_ptbxl_metadata, make_default_splits, sample_labelled_indices
from ssrl_ecg.models.cnn import ECGClassifier, ECGEncoder1DCNN
from ssrl_ecg.models.losses import FocalLoss, WeightedBCELoss, compute_class_weights
from ssrl_ecg.data.balancing import create_balanced_dataloader, report_class_imbalance
from ssrl_ecg.utils import choose_device, multilabel_metrics, set_seed, choose_device
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_single_seed(loss, strategy, seed, epochs=30, batch_size=64, data_root="data/PTB-XL"):
    """Train model with specific seed and configuration."""
    
    set_seed(seed)
    device = choose_device()
    
    # Load data
    db_df, labels = load_ptbxl_metadata(Path(data_root))
    splits = make_default_splits(db_df)
    sampled_train_idx = sample_labelled_indices(splits.train_idx, labels, 0.1, seed)
    
    train_ds = PTBXLRecordDataset(Path(data_root), db_df, labels, sampled_train_idx, 
                                  signal_length=1000)
    val_ds = PTBXLRecordDataset(Path(data_root), db_df, labels, splits.val_idx, 
                                signal_length=1000)
    
    train_labels = labels[sampled_train_idx]
    
    # Create balanced dataloader
    train_loader = create_balanced_dataloader(
        train_ds, train_labels, 
        batch_size=batch_size,
        strategy=strategy,
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Model
    encoder = ECGEncoder1DCNN(in_ch=12, width=64)
    classifier = ECGClassifier(encoder, n_classes=5).to(device)
    
    # Loss function
    if loss == "bce":
        criterion = torch.nn.BCEWithLogitsLoss()
    elif loss == "focal":
        class_weights = compute_class_weights(train_labels)
        criterion = FocalLoss(alpha=class_weights, gamma=2.0)
    else:
        criterion = torch.nn.BCEWithLogitsLoss()
    
    # Optimizer
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)
    
    # Training loop
    best_auroc = 0.0
    best_metrics = None
    
    classifier.train()
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        n = 0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            logits = classifier(x)
            loss_val = criterion(logits, y)
            
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            
            running_loss += loss_val.item() * x.size(0)
            n += x.size(0)
        
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
            best_metrics = metrics
        
        classifier.train()
    
    # Save checkpoint
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    
    out_path = checkpoint_dir / f"multiseed_{loss}_{strategy}_seed{seed:03d}.pt"
    torch.save({'model': classifier.state_dict()}, out_path)
    
    return {
        'seed': seed,
        'auroc_macro': best_metrics['auroc_macro'],
        'f1_macro': best_metrics['f1_macro'],
        'sensitivity_micro': best_metrics['sensitivity_micro'],
        'specificity_micro': best_metrics['specificity_micro'],
        'checkpoint': str(out_path),
    }


def compute_statistics(results):
    """Compute statistics from results."""
    
    if not results:
        return None
    
    aurocs = np.array([r['auroc_macro'] for r in results])
    f1s = np.array([r['f1_macro'] for r in results])
    sens = np.array([r['sensitivity_micro'] for r in results])
    specs = np.array([r['specificity_micro'] for r in results])
    
    return {
        'auroc': {
            'mean': float(np.mean(aurocs)),
            'std': float(np.std(aurocs)),
            'min': float(np.min(aurocs)),
            'max': float(np.max(aurocs)),
            'ci_95': (float(np.percentile(aurocs, 2.5)), float(np.percentile(aurocs, 97.5))),
        },
        'f1': {
            'mean': float(np.mean(f1s)),
            'std': float(np.std(f1s)),
            'min': float(np.min(f1s)),
            'max': float(np.max(f1s)),
            'ci_95': (float(np.percentile(f1s, 2.5)), float(np.percentile(f1s, 97.5))),
        },
        'sensitivity': {
            'mean': float(np.mean(sens)),
            'std': float(np.std(sens)),
        },
        'specificity': {
            'mean': float(np.mean(specs)),
            'std': float(np.std(specs)),
        },
        'n_seeds': len(results),
    }


def perform_ttest(results1, results2):
    """Perform t-test between configurations."""
    from scipy import stats
    
    aurocs1 = np.array([r['auroc_macro'] for r in results1])
    aurocs2 = np.array([r['auroc_macro'] for r in results2])
    
    t_stat, p_value = stats.ttest_ind(aurocs1, aurocs2)
    
    # Cohen's d
    n1, n2 = len(aurocs1), len(aurocs2)
    var1, var2 = np.var(aurocs1, ddof=1), np.var(aurocs2, ddof=1)
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    cohens_d = (np.mean(aurocs2) - np.mean(aurocs1)) / pooled_std if pooled_std > 0 else 0
    
    return {
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'cohens_d': float(cohens_d),
        'significant': p_value < 0.05,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Quick run with 3 seeds")
    args = parser.parse_args()
    
    if args.quick:
        seeds = [42, 52, 62]
        print("[QUICK MODE] 3 seeds per configuration")
    else:
        seeds = [42, 52, 62, 72, 82, 92, 102, 112, 122, 132]
    
    configurations = [
        ("bce", "standard"),
        ("focal", "oversample"),
    ]
    
    print("\n" + "="*70)
    print("PHASE 2: MULTI-SEED EXPERIMENTS")
    print("="*70)
    print(f"Seeds: {seeds}")
    print(f"Configurations: {configurations}")
    print(f"Total runs: {len(seeds) * len(configurations)}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    all_results = {}
    
    for loss, strategy in configurations:
        config_name = f"{loss}_{strategy}"
        all_results[config_name] = []
        
        print(f"\n{'='*70}")
        print(f"Configuration: {loss} + {strategy}")
        print(f"{'='*70}")
        
        for i, seed in enumerate(seeds, 1):
            print(f"  Seed {i:2d}/{len(seeds)} (seed={seed:3d})...", end=" ", flush=True)
            try:
                result = train_single_seed(loss, strategy, seed, epochs=30 if not args.quick else 5)
                if result:
                    all_results[config_name].append(result)
                    print(f"[OK] AUROC: {result['auroc_macro']:.4f}")
            except Exception as e:
                print(f"[ERROR] ERROR: {str(e)[:50]}")
    
    # Statistics
    print(f"\n{'='*70}")
    print("STATISTICAL SUMMARY")
    print("="*70)
    
    stats_results = {}
    for config_name, results in all_results.items():
        if results:
            stats = compute_statistics(results)
            stats_results[config_name] = stats
            
            print(f"\n{config_name.upper()}")
            print(f"  Completed: {len(results)}/{len(seeds)}")
            print(f"  AUROC: {stats['auroc']['mean']:.4f} ± {stats['auroc']['std']:.4f}")
            print(f"  95% CI: [{stats['auroc']['ci_95'][0]:.4f}, {stats['auroc']['ci_95'][1]:.4f}]")
            print(f"  F1:    {stats['f1']['mean']:.4f} ± {stats['f1']['std']:.4f}")
    
    # Significance test
    if len(stats_results) == 2:
        config_names = list(all_results.keys())
        print(f"\n{'='*70}")
        print("SIGNIFICANCE TEST")
        print("="*70)
        
        test_res = perform_ttest(all_results[config_names[0]], all_results[config_names[1]])
        print(f"\nComparing: {config_names[0]} vs {config_names[1]}")
        print(f"  t-stat: {test_res['t_statistic']:>8.4f}")
        print(f"  p-value: {test_res['p_value']:>10.6f}")
        print(f"  Cohen's d: {test_res['cohens_d']:>8.4f}")
        
        if test_res['significant']:
            print(f"  [OK] SIGNIFICANT (p < 0.05)")
        else:
            print(f"  [WARNING] Not significant")
        
        stats_results['significance_test'] = test_res
    
    # Save results
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    results_path = results_dir / "phase2_multiseed_results.json"
    
    with open(results_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'quick_mode': args.quick,
            'configurations': configurations,
            'seeds': seeds,
            'individual_results': all_results,
            'statistics': stats_results,
        }, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"[OK] Results saved to: {results_path}")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
