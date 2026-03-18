from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ssrl_ecg.data.mitbih import MITBIHDataset
from ssrl_ecg.models.cnn import ECGClassifier, ECGEncoder1DCNN
from ssrl_ecg.utils import choose_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transfer model (pretrained or supervised) to MIT-BIH evaluation.")
    parser.add_argument("--mitbih-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--use-ssl", action="store_true", help="Load encoder from SSL checkpoint")
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--finetune-epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def evaluate_binary(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict:
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
    y_prob_arr = np.concatenate(y_prob, axis=0).squeeze()

    from sklearn.metrics import roc_auc_score, f1_score

    auc = roc_auc_score(y_true_arr, y_prob_arr)
    f1 = f1_score(y_true_arr, (y_prob_arr >= 0.5).astype(int))
    return {"auroc": float(auc), "f1": float(f1)}


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = choose_device()

    ds = MITBIHDataset(args.mitbih_root, signal_length=1000)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Create encoder with 12 channels to match PTB-XL training format
    # (MIT-BIH signals are replicated across 12 channels for transfer learning consistency)
    encoder = ECGEncoder1DCNN(in_ch=12, width=64)
    ckpt = torch.load(args.checkpoint, map_location="cpu")

    # Handle both pure SSL checkpoints (with "encoder" key) and fine-tuned checkpoints (with "model" key)
    if "encoder" in ckpt:
        encoder.load_state_dict(ckpt["encoder"])
    elif "model" in ckpt:
        full_model_state = ckpt["model"]
        encoder_state = {k.replace("encoder.", ""): v for k, v in full_model_state.items() if k.startswith("encoder.")}
        if encoder_state:  # Successfully extracted encoder
            encoder.load_state_dict(encoder_state)
        else:
            raise ValueError("Could not extract encoder from checkpoint. Ensure it's a valid model checkpoint.")
    else:
        raise ValueError(f"Unknown checkpoint format. Expected 'encoder' or 'model' key, got: {list(ckpt.keys())}")

    model = ECGClassifier(encoder=encoder, n_classes=1).to(device)

    if args.freeze_encoder and not args.use_ssl:
        for p in model.encoder.parameters():
            p.requires_grad = False

    metrics = evaluate_binary(model, loader, device)
    print({"transfer_eval": metrics})


if __name__ == "__main__":
    main()
