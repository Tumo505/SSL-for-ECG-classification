"""Subject-separated MIT-BIH binary annotation-window transfer dataset.

Ten-second nonoverlapping windows are positive if an accepted ectopic beat
occurs. Paced/unknown-beat windows are excluded. This window task is distinct
from AAMI beat classification. PhysioNet identifies records 201 and 202 as
one subject: https://physionet.org/content/mitdb/1.0.0/.
"""
from pathlib import Path
import numpy as np
import torch
import wfdb
from scipy.signal import resample_poly
from torch.utils.data import Dataset

NORMAL = frozenset(("N", "L", "R", "e", "j"))
ABNORMAL = frozenset(("A", "a", "J", "S", "V", "E", "F"))
UNKNOWN = frozenset(("/", "f", "Q"))


def patient_id(record):
    return "201_202" if str(record) in ("201", "202") else str(record)


def record_splits(records, seed=42):
    patients = sorted({patient_id(r) for r in records})
    if len(patients) < 5:
        raise ValueError("At least five subjects are required for transfer partitions")
    patients = np.random.default_rng(seed).permutation(patients)
    n_test = n_val = max(1, int(.2*len(patients)))
    groups = dict(test=set(patients[:n_test]), validation=set(patients[n_test:n_test+n_val]),
                  train=set(patients[n_test+n_val:]))
    return {name:[r for r in records if patient_id(r) in group] for name,group in groups.items()}


class MITBIHDataset(Dataset):
    def __init__(self, data_root, records, signal_length=1000, max_windows_per_record=None):
        if signal_length != 1000:
            raise ValueError("Transfer protocol uses ten seconds resampled to 100 Hz")
        self.signals, self.labels, self.record_ids = [], [], []
        root = Path(data_root)
        for name in records:
            for suffix in (".hea", ".dat", ".atr"):
                if not (root/(str(name)+suffix)).exists():
                    raise FileNotFoundError(root/(str(name)+suffix))
            record = wfdb.rdrecord(str(root/str(name)))
            annotation = wfdb.rdann(str(root/str(name)), "atr")
            if record.fs != 360 or not np.isfinite(record.p_signal).all():
                raise ValueError(f"Invalid MIT-BIH waveform {name}")
            wave = resample_poly(record.p_signal[:,0], 5, 18)
            symbols = np.asarray(annotation.symbol)
            stored = 0
            for start in range(0,len(wave)-signal_length+1,signal_length):
                original_start = start*360/100
                selected = symbols[(annotation.sample>=original_start)&(annotation.sample<original_start+3600)]
                if any(s in UNKNOWN for s in selected) or not any(s in NORMAL|ABNORMAL for s in selected):
                    continue
                signal = wave[start:start+signal_length].astype(np.float32)
                signal = (signal-signal.mean())/max(float(signal.std()),1e-6)
                self.signals.append(signal)
                self.labels.append(float(any(s in ABNORMAL for s in selected)))
                self.record_ids.append(str(name))
                stored += 1
                if max_windows_per_record and stored >= max_windows_per_record:
                    break
        if not self.signals:
            raise ValueError("No eligible MIT-BIH annotation windows")

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, index):
        # Repeat one measured lead for the pretrained twelve-channel interface.
        return torch.from_numpy(np.tile(self.signals[index],(12,1))), torch.tensor([self.labels[index]])
