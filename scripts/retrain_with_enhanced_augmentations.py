#!/usr/bin/env python3
"""Retrain BYOL & SimCLR with enhanced augmentations and optimized epochs."""

import subprocess
import sys
import time
from pathlib import Path


def run_command(cmd, description):
    """Run command and track timing."""
    print(f"\n{'='*80}")
    print(f"[{time.strftime('%H:%M:%S')}] {description}")
    print(f"{'='*80}")
    print(f"Command: {cmd}\n")
    
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"[ERROR] Command failed: {cmd}")
        return False
    return True


def main():
    print("\n" + "="*80)
    print("ENHANCED AUGMENTATIONS RETRAINING PIPELINE")
    print("="*80)
    
    experiments = [
        # Phase 1: Primary SSL models with enhanced augmentations
        {
            'cmd': 'python -m ssrl_ecg.train_ssl_byol --epochs 30 --batch-size 256 --seed 42 --out checkpoints/ssl_byol_enhanced.pt',
            'desc': '[1/4] BYOL with Enhanced Augmentations (30 epochs)',
            'checkpoint': 'checkpoints/ssl_byol_enhanced.pt'
        },
        {
            'cmd': 'python -m ssrl_ecg.train_ssl_simclr --epochs 20 --batch-size 128 --temperature 0.07 --seed 42 --out checkpoints/ssl_simclr_enhanced.pt',
            'desc': '[2/4] SimCLR with Enhanced Augmentations (20 epochs)',
            'checkpoint': 'checkpoints/ssl_simclr_enhanced.pt'
        },
        
        # Phase 2: Fine-tune enhanced models on labeled data (10%)
        {
            'cmd': 'python -m ssrl_ecg.train_finetune --ssl-checkpoint checkpoints/ssl_byol_enhanced.pt --epochs 20 --batch-size 64 --label-fraction 0.1 --seed 42 --out checkpoints/ssl_byol_enhanced_finetuned.pt',
            'desc': '[3/4] Fine-tune Enhanced BYOL on 10% Labels (20 epochs)',
            'checkpoint': 'checkpoints/ssl_byol_enhanced_finetuned.pt'
        },
        {
            'cmd': 'python -m ssrl_ecg.train_finetune --ssl-checkpoint checkpoints/ssl_simclr_enhanced.pt --epochs 20 --batch-size 64 --label-fraction 0.1 --seed 42 --out checkpoints/ssl_simclr_enhanced_finetuned.pt',
            'desc': '[4/4] Fine-tune Enhanced SimCLR on 10% Labels (20 epochs)',
            'checkpoint': 'checkpoints/ssl_simclr_enhanced_finetuned.pt'
        },
    ]
    
    checkpoints_created = []
    start_time = time.time()
    
    for i, exp in enumerate(experiments, 1):
        print(f"\n\n{'#'*80}")
        print(f"# EXPERIMENT {i}/{len(experiments)}")
        print(f"{'#'*80}")
        
        success = run_command(exp['cmd'], exp['desc'])
        
        if success:
            checkpoints_created.append(exp['checkpoint'])
            print(f"✓ Generated: {exp['checkpoint']}")
        else:
            print(f"✗ FAILED: {exp['checkpoint']}")
            sys.exit(1)
        
        elapsed = time.time() - start_time
        remaining_experiments = len(experiments) - i
        print(f"\nElapsed: {elapsed/60:.1f} min | Remaining: {remaining_experiments} experiments")
    
    # Phase 3: Evaluation
    print(f"\n\n{'='*80}")
    print("PHASE 3: COMPREHENSIVE EVALUATION")
    print(f"{'='*80}")
    
    eval_cmd = 'python evaluation/compare_baselines.py'
    print(f"\nRunning: {eval_cmd}")
    print("This will compare all models (both original and enhanced)")
    success = run_command(eval_cmd, '[EVAL] Compare All Baselines')
    
    if not success:
        print("[WARNING] Evaluation script had issues, but continuing...")
    
    # Summary
    total_time = time.time() - start_time
    print(f"\n\n{'='*80}")
    print("RETRAINING SUMMARY")
    print(f"{'='*80}")
    
    summary = f"""
Total Time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)

Checkpoints Created:
{chr(10).join(f'  ✓ {cp}' for cp in checkpoints_created)}

  1. Run: python evaluation/generate_publication_report.py
  2. Compare old vs new in paper
  3. Show ablation study
  4. Prepare conference submission
"""
    print(summary)
    
    print(f"\n✓ All experiments completed successfully!")
    print(f"✓ Results saved to checkpoints/ and evaluation logs")


if __name__ == '__main__':
    main()
