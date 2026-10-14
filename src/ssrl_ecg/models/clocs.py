"""CLOCS CMSC adaptation: temporal views and patient-specific cross-view NCE.

Kiyasseh et al., ICML 2021, sections 4.2–4.3:
https://proceedings.mlr.press/v139/kiyasseh21a.html
Uses the study's shared 12-lead backbone and projection head, not the original
single-lead architecture. No additional waveform perturbations are applied.
"""

import torch
from torch import nn
from torch.nn import functional as F


class PatientNCELoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1, z2, patient_ids):
        if len(z1) < 2 or self.temperature <= 0:
            raise ValueError("Patient NCE requires N >= 2 and positive temperature")
        logits = F.normalize(z1, dim=1) @ F.normalize(z2, dim=1).T / self.temperature
        positives = patient_ids[:, None] == patient_ids[None, :]
        # Stable equivalent of the author's prepare_miscellaneous.py:
        # two diagonal terms and, when present, upper/lower patient terms.
        row_logp = F.log_softmax(logits, dim=1)
        col_logp = F.log_softmax(logits, dim=0)
        losses = [-row_logp.diagonal().mean(), -col_logp.diagonal().mean()]
        upper = torch.triu(positives, diagonal=1)
        lower = torch.tril(positives, diagonal=-1)
        if upper.any():
            losses.append(-row_logp[upper].mean())
        if lower.any():
            losses.append(-col_logp[lower].mean())
        return sum(losses) / len(losses)


def temporal_views(x):
    if x.shape[-1] % 2:
        raise ValueError("CMSC requires equal adjacent temporal segments")
    return x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]


class ReferenceCLOCSEncoder(nn.Module):
    """Author CMSC architecture: lead II, 2500 samples, three CNN blocks.

    https://github.com/danikiyasseh/CLOCS/blob/master/prepare_network.py
    Native 500 Hz input is required. Downstream records use their first 5 s;
    pretraining uses both adjacent 5 s views. The label task and data split
    remain those of this study, rather than the published downstream tasks.
    """

    def __init__(self):
        super().__init__()
        layers = []
        for incoming, outgoing in ((1, 4), (4, 16), (16, 32)):
            layers.extend([
                nn.Conv1d(incoming, outgoing, 7, stride=3),
                nn.BatchNorm1d(outgoing), nn.ReLU(), nn.MaxPool1d(2),
                nn.Dropout(0.1),
            ])
        self.features = nn.Sequential(*layers)
        self.embedding = nn.Linear(320, 256)
        self.out_channels = 256

    def forward(self, x):
        if x.shape[-1] not in (2500, 5000) or x.shape[1] != 12:
            raise ValueError("Reference CLOCS requires native 500 Hz PTB-XL input")
        return self.embedding(self.features(x[:, 1:2, :2500]).flatten(1)).unsqueeze(-1)


class ReferenceCLOCSModel(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, x):
        z = self.encoder(x).squeeze(-1)
        return z, z
