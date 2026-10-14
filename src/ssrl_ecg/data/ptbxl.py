from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import wfdb
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from torch.utils.data import Dataset

DIAGNOSTIC_CLASSES = ("NORM", "CD", "HYP", "MI", "STTC")
DEFAULT_DATA_ROOT = (
    Path(__file__).resolve().parents[3]
    / "Data"
    / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
)


def load_ptbxl_metadata(data_root: Path):
    root = Path(data_root)
    df = pd.read_csv(root / "ptbxl_database.csv")
    statements = pd.read_csv(root / "scp_statements.csv", index_col=0)
    mapping = statements.loc[statements.diagnostic == 1, "diagnostic_class"].to_dict()
    labels = np.zeros((len(df), len(DIAGNOSTIC_CLASSES)), dtype=np.float32)
    for i, codes in enumerate(df.scp_codes):
        classes = {mapping[c] for c in ast.literal_eval(codes) if c in mapping}
        labels[i] = [c in classes for c in DIAGNOSTIC_CLASSES]
    # Keep all records in the label-free SSL pool. The runner explicitly excludes
    # records with no diagnostic superclass from supervised train/val/test.
    return df.reset_index(drop=True), labels


@dataclass
class PTBXLSplits:
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray


def make_default_splits(db_df):
    folds = db_df.strat_fold.to_numpy()
    splits = PTBXLSplits(
        np.flatnonzero(folds <= 8),
        np.flatnonzero(folds == 9),
        np.flatnonzero(folds == 10),
    )
    patient_sets = [
        set(db_df.iloc[idx].patient_id)
        for idx in (splits.train_idx, splits.val_idx, splits.test_idx)
    ]
    if any(patient_sets[i] & patient_sets[j] for i, j in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("Patient leakage between train, validation and test folds")
    return splits


def sample_labelled_indices(train_idx, labels, label_fraction, seed):
    if not 0 < label_fraction <= 1:
        raise ValueError("label_fraction must be in (0, 1]")
    idx = np.asarray(train_idx, dtype=np.int64)
    idx = idx[labels[idx].sum(axis=1) > 0]
    if label_fraction == 1:
        return idx.copy()
    n = max(1, round(len(idx) * label_fraction))
    if n < labels.shape[1]:
        raise ValueError("Labelled subset is too small to cover all classes")
    sampler = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=n, random_state=seed
    )
    _, selected = next(sampler.split(np.zeros((len(idx), 1)), labels[idx]))
    # Iterative stratification preserves marginal label counts but can return
    # a different number of records when labels co-occur. Enforce the declared
    # label budget by greedily minimising marginal-count distortion.
    chosen = np.zeros(len(idx), dtype=bool)
    chosen[selected] = True
    target = labels[idx].sum(0) * (n / len(idx))
    current = labels[idx[chosen]].sum(0)
    rng = np.random.default_rng(seed)
    while chosen.sum() != n:
        adding = chosen.sum() < n
        candidates = np.flatnonzero(~chosen if adding else chosen)
        next_counts = current + (1 if adding else -1) * labels[idx[candidates]]
        scores = ((next_counts - target) ** 2 / np.maximum(target, 1)).sum(1)
        if not adding:
            scores[(next_counts <= 0).any(1)] = np.inf
        best = np.flatnonzero(np.isclose(scores, scores.min(), rtol=0, atol=1e-12))
        if not np.isfinite(scores[best[0]]):
            raise ValueError("Cannot enforce label budget while retaining all classes")
        candidate = candidates[rng.choice(best)]
        chosen[candidate] = adding
        current += (1 if adding else -1) * labels[idx[candidate]]
    selected = np.flatnonzero(chosen)
    result = np.sort(idx[selected])
    if np.any(labels[result].sum(axis=0) == 0):
        raise ValueError("Labelled subset lacks a diagnostic class")
    return result


class PTBXLRecordDataset(Dataset):
    def __init__(
        self,
        data_root,
        db_df,
        labels,
        indices,
        use_high_resolution=False,
        signal_length=1000,
        return_labels=True,
        signals=None,
    ):
        self.data_root = Path(data_root)
        self.db_df, self.labels = db_df, labels
        self.indices = np.asarray(indices, dtype=np.int64)
        self.signal_length = signal_length
        self.return_labels = return_labels
        self.sampling_rate = 500 if use_high_resolution else 100
        self.filename_column = "filename_hr" if use_high_resolution else "filename_lr"
        self.signals = signals
        if signal_length != self.sampling_rate * 10:
            raise ValueError(
                "Use the complete ten-second recording: 1000 @ 100 Hz or 5000 @ 500 Hz"
            )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        row_idx = self.indices[index]
        if self.signals is None:
            row = self.db_df.iloc[row_idx]
            signal, fields = wfdb.rdsamp(
                str(self.data_root / row[self.filename_column])
            )
            if fields["fs"] != self.sampling_rate or signal.shape != (
                self.signal_length,
                12,
            ):
                raise ValueError(f"Invalid waveform shape/rate for ECG {row.ecg_id}")
            x = signal.T.astype(np.float32)
            if not np.isfinite(x).all():
                raise ValueError(f"Non-finite waveform for ECG {row.ecg_id}")
            x = (x - x.mean(axis=-1, keepdims=True)) / np.maximum(
                x.std(axis=-1, keepdims=True), 1e-6
            )
        else:
            x = np.array(self.signals[row_idx], copy=True)
        tensor = torch.from_numpy(x)
        return (
            (tensor, torch.from_numpy(self.labels[row_idx].copy()))
            if self.return_labels
            else tensor
        )
