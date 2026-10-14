import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler


def create_balanced_dataloader(
    dataset,
    labels,
    batch_size=64,
    strategy="standard",
    shuffle=True,
    num_workers=0,
    generator=None,
):
    y = torch.as_tensor(labels, dtype=torch.float32, device="cpu")
    if strategy not in ("standard", "oversample", "stratified"):
        raise ValueError(f"Unknown balancing strategy {strategy}")
    sampler = None
    if strategy in ("oversample", "stratified"):
        # Weighted record sampling for multilabel data does not guarantee exactly
        # equal class counts in each mini-batch. Positive labels can co-occur.
        weights = (y / y.sum(0).clamp_min(1)).sum(1) / y.sum(1).clamp_min(1)
        if not torch.any(weights > 0):
            raise ValueError("Cannot oversample a dataset without positive labels")
        sampler = WeightedRandomSampler(
            weights.double(), len(dataset), replacement=True, generator=generator
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        generator=generator,
    )


def report_class_imbalance(labels, name="Dataset"):
    counts = np.asarray(labels).sum(axis=0)
    print(f"{name} positive class counts: {counts.astype(int).tolist()}")
