"""Script to retrain with enhanced augmentations and optimized epochs.

Compares:
1. Current BYOL (basic augmentations, 20 epochs)
2. Enhanced BYOL (advanced augmentations, 30 epochs)
3. Supervised with optimized epochs (20 vs 30)
"""

import torch
import json
from pathlib import Path
from argparse import ArgumentParser


def main():
    parser = ArgumentParser()
    parser.add_argument('--experiment', choices=['all', 'byol', 'supervised'], default='all')
    parser.add_argument('--force', action='store_true', help='Force retrain even if checkpoint exists')
    args = parser.parse_args()

    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)

    experiments = []

    print("=" * 80)
    print("RETRAINING ANALYSIS: EPOCHS & AUGMENTATIONS")
    print("=" * 80)

    # ============ EXPERIMENT 1: BYOL Enhanced ============
    if args.experiment in ['all', 'byol']:
        print("\n[1] BYOL with ENHANCED AUGMENTATIONS (30 epochs)")
        print("-" * 80)
        cmd_byol_enhanced = (
            "python -m ssrl_ecg.train_ssl_byol "
            "--epochs 30 --batch-size 256 --seed 42 "
            "--use-advanced-augmentations True "
            "--out checkpoints/ssl_byol_enhanced.pt "
            "--log-dir results/byol_enhanced"
        )
        print(f"Command: {cmd_byol_enhanced}\n")
        experiments.append({
            'name': 'BYOL Enhanced (30 epochs, advanced augmentations)',
            'command': cmd_byol_enhanced,
            'checkpoint': 'checkpoints/ssl_byol_enhanced.pt',
            'expected_improvement': '+2-5% AUROC (vs current 0.8594)'
        })
    
    # ============ EXPERIMENT 2: Supervised (Optimized Epochs) ============
    if args.experiment in ['all', 'supervised']:
        print("\n[2] SUPERVISED: Epoch Optimization (20 vs 30)")
        print("-" * 80)
        
        experiments.append({
            'name': 'Supervised Optimized (20 epochs, Focal Loss)',
            'command': (
                "python -m ssrl_ecg.train_supervised "
                "--epochs 20 --batch-size 64 --seed 42 "
                "--label-fraction 0.1 --loss focal --balance-strategy oversample "
                "--out checkpoints/supervised_focal_20epochs.pt"
            ),
            'checkpoint': 'checkpoints/supervised_focal_20epochs.pt',
            'expected_improvement': 'Reduced overfitting, higher generalization'
        })

    print(summary)

    # Print experiment details
    print("\nDETAILED EXPERIMENT SETUP:")
    print("-" * 80)
    for i, exp in enumerate(experiments, 1):
        print(f"\n[Experiment {i}] {exp['name']}")
        print(f"  Command: {exp['command']}")
        print(f"  Output: {exp['checkpoint']}")
        print(f"  Expected: {exp['expected_improvement']}")

    # Save recommendations to file
    with open(results_dir / 'retraining_recommendations.json', 'w') as f:
        json.dump({
            'epochs_recommendation': {
                'byol': 'Increase from 20 to 30 epochs (5 min training)',
                'simclr': 'Keep 15 epochs (converged)',
                'supervised': 'Reduce from 30 to 20 epochs (reduce overfitting)'
            },
            'augmentation_recommendation': {
                'upgrade_from': ['time_shift', 'scaling', 'jitter'],
                'add_these': [
                    'time_warping (frequency domain)',
                    'temporal_masking/dropout',
                    'per_channel_variation',
                    'frequency_filtering_variation'
                ],
                'expected_improvement': '2-5% AUROC'
            },
            'experiments_to_run': experiments
        }, f, indent=2)
    
    print(f"\n✓ Recommendations saved to {results_dir / 'retraining_recommendations.json'}")


if __name__ == '__main__':
    main()
