from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ssrl_ecg.data.ptbxl import DEFAULT_DATA_ROOT, PTBXLRecordDataset, load_ptbxl_metadata, make_default_splits
from ssrl_ecg.models.cnn import ECGClassifier, ECGEncoder1DCNN
from ssrl_ecg.utils import choose_device, multilabel_metrics, set_seed


class CorruptedWrapper(torch.utils.data.Dataset):
    def __init__(self, base_ds: PTBXLRecordDataset, noise_std: float = 0.0, mask_ratio: float = 0.0):
        self.base_ds = base_ds
        self.noise_std = noise_std
        self.mask_ratio = mask_ratio

    def __len__(self) -> int:
        return len(self.base_ds)

    def __getitem__(self, idx: int):
        x, y = self.base_ds[idx]
        if self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std
        if self.mask_ratio > 0:
            t = x.shape[-1]
            mask_len = max(1, int(t * self.mask_ratio))
            start = np.random.randint(0, max(1, t - mask_len))
            x[:, start : start + mask_len] = 0.0
        return x, y


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PTB-XL model under clean/noisy/masked conditions.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--device", choices=("auto", "mps", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--signal-length", type=int, default=1000)
    parser.add_argument("--noise-std", type=float, default=0.0)
    parser.add_argument("--mask-ratio", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.noise_std < 0 or not 0 <= args.mask_ratio <= 1:
        raise ValueError("Invalid corruption parameters")
    set_seed(args.seed)
    device = choose_device(args.device)

    db_df, labels = load_ptbxl_metadata(args.data_root)
    splits = make_default_splits(db_df)

    indices = splits.test_idx[labels[splits.test_idx].sum(1)>0]
    if args.smoke:
        indices = indices[:128]
    base_ds = PTBXLRecordDataset(args.data_root, db_df, labels, indices,
                               use_high_resolution=args.signal_length == 5000, signal_length=args.signal_length)
    test_ds = CorruptedWrapper(base_ds, noise_std=args.noise_std, mask_ratio=args.mask_ratio)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    from ssrl_ecg.models.resnet1d import ResNetEncoder
    from ssrl_ecg.models.clocs import ReferenceCLOCSEncoder
    state = ckpt["model"]
    if "encoder.embedding.weight" in state:
        encoder = ReferenceCLOCSEncoder()
        if args.signal_length != 5000:
            raise ValueError("Reference CLOCS evaluation requires --signal-length 5000")
    elif "encoder.features.7.0.conv1.weight" in state:
        encoder = ResNetEncoder()
    else:
        encoder = ECGEncoder1DCNN(in_ch=12, width=64)
    model = ECGClassifier(encoder=encoder, n_classes=labels.shape[1]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    y_true = []
    y_prob = []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            logits = model(x)
            prob = torch.sigmoid(logits).cpu().numpy()
            y_prob.append(prob)
            y_true.append(y.numpy())

    y_true_arr = np.concatenate(y_true, axis=0)
    y_prob_arr = np.concatenate(y_prob, axis=0)
    metrics = multilabel_metrics(y_true_arr, y_prob_arr)
    print(metrics)


if __name__ == "__main__":
    main()
