"""
Label scarcity benchmark: Comprehensive evaluation of SSL vs supervised learning.

Demonstrates that self-supervised pre-training provides superior performance
under limited labeled data for cardiovascular risk prediction.
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader
from torch.optim import Adam
import json

from ssrl_ecg.data.ptbxl import (
    load_ptbxl_metadata, make_default_splits, sample_labelled_indices,
    PTBXLRecordDataset, DIAGNOSTIC_CLASSES
)
from ssrl_ecg.data.preprocessing import ECGPreprocessor
from ssrl_ecg.models.improved_cnn import ImprovedECGEncoder, ImprovedECGClassifier
from ssrl_ecg.utils import choose_device, multilabel_metrics


class PreprocessedDataset:
    """Wraps dataset with preprocessing."""
    def __init__(self, ptbxl_dataset, preprocessor=None):
        self.dataset = ptbxl_dataset
        self.preprocessor = preprocessor or ECGPreprocessor()
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        x_np = x.numpy() if isinstance(x, torch.Tensor) else x
        x_np = self.preprocessor.preprocess(x_np)
        y_np = y.numpy() if isinstance(y, torch.Tensor) else y
        return torch.from_numpy(x_np).float(), torch.from_numpy(y_np).float()


class LabelScarcityBenchmark:
    """Benchmark SSL vs supervised learning at different label fractions."""
    
    def __init__(self, data_root: Path, checkpoint_dir: Path, results_dir: Path):
        self.data_root = Path(data_root)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        self.device = choose_device()
        self.preprocessor = ECGPreprocessor()
        
        # Load data
        self.db_df, self.labels = load_ptbxl_metadata(self.data_root)
        self.splits = make_default_splits(self.db_df)
        
        print("[Label Scarcity Benchmark: SSL vs Supervised]")
        print(f"Test set size: {len(self.splits.test_idx)}")
        print(f"Val set size: {len(self.splits.val_idx)}")
    
    def train_supervised_baseline(self,
                                 label_fraction: float,
                                 seed: int = 42,
                                 epochs: int = 50,
                                 batch_size: int = 32):
        """Train supervised model from scratch (no pre-training)."""
        
        train_indices = sample_labelled_indices(
            self.splits.train_idx, self.labels, label_fraction, seed
        )
        
        # Create datasets
        train_ds = PreprocessedDataset(
            PTBXLRecordDataset(self.data_root, self.db_df, self.labels, train_indices),
            self.preprocessor
        )
        val_ds = PreprocessedDataset(
            PTBXLRecordDataset(self.data_root, self.db_df, self.labels, self.splits.val_idx),
            self.preprocessor
        )
        
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
        
        # Model from scratch
        encoder = ImprovedECGEncoder(in_ch=12, width=64, dropout=0.2)
        model = ImprovedECGClassifier(encoder, n_classes=5, dropout=0.3).to(self.device)
        
        criterion = nn.BCEWithLogitsLoss()
        optimizer = Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Train
            model.train()
            train_loss = 0
            for x, y_batch in train_loader:
                x, y_batch = x.to(self.device), y_batch.to(self.device)
                logits = model(x)
                loss = criterion(logits, y_batch)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Val
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for x, y_batch in val_loader:
                    x, y_batch = x.to(self.device), y_batch.to(self.device)
                    logits = model(x)
                    loss = criterion(logits, y_batch)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 10:
                    break
        
        return model
    
    def load_ssl_pretrained(self, checkpoint_path: Path):
        """Load SSL pre-trained encoder."""
        if not checkpoint_path.exists():
            print(f"WARNING: SSL checkpoint not found: {checkpoint_path}")
            return None
        
        encoder = ImprovedECGEncoder(in_ch=12, width=64, dropout=0.2)
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        
        if "encoder" in ckpt:
            encoder.load_state_dict(ckpt["encoder"])
        elif "model" in ckpt:
            state = {k.replace("encoder.", ""): v for k, v in ckpt["model"].items() 
                    if k.startswith("encoder.")}
            encoder.load_state_dict(state)
        
        return encoder
    
    def finetune_ssl(self,
                    ssl_encoder: ImprovedECGEncoder,
                    label_fraction: float,
                    seed: int = 42,
                    epochs: int = 50,
                    batch_size: int = 32,
                    freeze_encoder: bool = False):
        """Fine-tune SSL pre-trained model on labeled data."""
        
        train_indices = sample_labelled_indices(
            self.splits.train_idx, self.labels, label_fraction, seed
        )
        
        train_ds = PreprocessedDataset(
            PTBXLRecordDataset(self.data_root, self.db_df, self.labels, train_indices),
            self.preprocessor
        )
        val_ds = PreprocessedDataset(
            PTBXLRecordDataset(self.data_root, self.db_df, self.labels, self.splits.val_idx),
            self.preprocessor
        )
        
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
        
        # Use pre-trained encoder
        model = ImprovedECGClassifier(ssl_encoder, n_classes=5, dropout=0.3).to(self.device)
        
        # Optionally freeze encoder
        if freeze_encoder:
            for param in model.encoder.parameters():
                param.requires_grad = False
        
        criterion = nn.BCEWithLogitsLoss()
        optimizer = Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            model.train()
            train_loss = 0
            for x, y_batch in train_loader:
                x, y_batch = x.to(self.device), y_batch.to(self.device)
                logits = model(x)
                loss = criterion(logits, y_batch)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for x, y_batch in val_loader:
                    x, y_batch = x.to(self.device), y_batch.to(self.device)
                    logits = model(x)
                    loss = criterion(logits, y_batch)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 10:
                    break
        
        return model
    
    def evaluate_model(self, model):
        """Evaluate model on test set."""
        test_ds = PreprocessedDataset(
            PTBXLRecordDataset(self.data_root, self.db_df, self.labels, 
                             self.splits.test_idx),
            self.preprocessor
        )
        test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)
        
        model.eval()
        y_true = []
        y_prob = []
        
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(self.device)
                logits = model(x)
                prob = torch.sigmoid(logits).cpu().numpy()
                y_prob.append(prob)
                y_true.append(y.numpy())
        
        y_true_arr = np.concatenate(y_true, axis=0)
        y_prob_arr = np.concatenate(y_prob, axis=0)
        metrics = multilabel_metrics(y_true_arr, y_prob_arr)
        
        return metrics
    
    def run_label_scarcity_benchmark(self, label_fractions=[0.01, 0.05, 0.1, 0.25, 1.0], 
                                     seeds=[42, 52, 62], epochs=40):
        """Run full benchmark comparing SSL vs supervised at different label fractions."""
        
        results = {
            "label_scarcity": {},
            "seeds": seeds,
            "label_fractions": label_fractions
        }
        
        # Load SSL pre-trained model (if available)
        ssl_encoder = self.load_ssl_pretrained(self.checkpoint_dir / "ssl_masked.pt")
        
        for frac in label_fractions:
            print(f"\n[Label Fraction: {frac:.1%}]")
            results["label_scarcity"][frac] = {
                "supervised": {"auroc": [], "f1": []},
                "ssl_finetuned": {"auroc": [], "f1": []},
                "ssl_frozen": {"auroc": [], "f1": []},
            }
            
            for seed in seeds:
                print(f"  Seed {seed}...")
                
                # Supervised baseline
                sup_model = self.train_supervised_baseline(frac, seed, epochs)
                sup_metrics = self.evaluate_model(sup_model)
                results["label_scarcity"][frac]["supervised"]["auroc"].append(
                    sup_metrics.get("auroc_macro", 0)
                )
                results["label_scarcity"][frac]["supervised"]["f1"].append(
                    sup_metrics.get("f1_macro", 0)
                )
                
                # SSL fine-tuned (if encoder available)
                if ssl_encoder is not None:
                    ssl_ft_model = self.finetune_ssl(ssl_encoder, frac, seed, epochs, 
                                                     freeze_encoder=False)
                    ssl_ft_metrics = self.evaluate_model(ssl_ft_model)
                    results["label_scarcity"][frac]["ssl_finetuned"]["auroc"].append(
                        ssl_ft_metrics.get("auroc_macro", 0)
                    )
                    results["label_scarcity"][frac]["ssl_finetuned"]["f1"].append(
                        ssl_ft_metrics.get("f1_macro", 0)
                    )
                    
                    ssl_frozen_model = self.finetune_ssl(ssl_encoder, frac, seed, epochs,
                                                         freeze_encoder=True)
                    ssl_frozen_metrics = self.evaluate_model(ssl_frozen_model)
                    results["label_scarcity"][frac]["ssl_frozen"]["auroc"].append(
                        ssl_frozen_metrics.get("auroc_macro", 0)
                    )
                    results["label_scarcity"][frac]["ssl_frozen"]["f1"].append(
                        ssl_frozen_metrics.get("f1_macro", 0)
                    )
        
        # Save results
        results_file = self.results_dir / "label_scarcity_benchmark.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to {results_file}")
        self._print_summary(results)
        
        return results
    
    def _print_summary(self, results):
        """Print summary statistics."""
        print("\n" + "="*80)
        print("LABEL SCARCITY BENCHMARK SUMMARY")
        print("="*80)
        
        for frac in results["label_fractions"]:
            print(f"\n[{frac:.1%} Labeled Data]")
            
            for method in ["supervised", "ssl_finetuned", "ssl_frozen"]:
                if method in results["label_scarcity"][frac]:
                    aurocs = results["label_scarcity"][frac][method]["auroc"]
                    f1s = results["label_scarcity"][frac][method]["f1"]
                    
                    if aurocs:
                        mean_auroc = np.mean(aurocs)
                        std_auroc = np.std(aurocs)
                        mean_f1 = np.mean(f1s)
                        std_f1 = np.std(f1s)
                        
                        print(f"  {method:20} AUROC: {mean_auroc:.4f}±{std_auroc:.4f} | "
                              f"F1: {mean_f1:.4f}±{std_f1:.4f}")


def main():
    data_root = Path("data/PTB-XL")
    checkpoint_dir = Path("checkpoints")
    results_dir = Path("label_scarcity_results")
    
    benchmark = LabelScarcityBenchmark(data_root, checkpoint_dir, results_dir)
    benchmark.run_label_scarcity_benchmark(
        label_fractions=[0.05, 0.1, 0.25, 1.0],  # Start with these for quick testing
        seeds=[42, 52, 62],
        epochs=30
    )


if __name__ == "__main__":
    main()
