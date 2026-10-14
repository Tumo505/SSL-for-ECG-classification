"""Versioned, label-free ECG augmentation policies shared by SimCLR and BYOL.

Transforms run on CPU before transfer to MPS. Both policies share jitter and
scaling. Domain policy adds seven independently gated transforms. Mixed views
share source partners and mixing coefficients to retain positive-pair identity.
No disease labels or held-out waveforms are used.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.signal import butter, sosfiltfilt
from torch.nn import functional as F

DOMAIN_TRANSFORMS = (
    "frequency_warp",
    "mixup",
    "bandpass",
    "cutmix",
    "motion",
    "channel_noise",
    "temporal_dropout",
)
PROBABILITIES = dict(zip(DOMAIN_TRANSFORMS, (0.5, 0.4, 0.3, 0.25, 0.2, 0.6, 0.3)))
POLICY_VERSION = "ecg_v2_100hz"


class ECGAugmentations:
    def __init__(
        self,
        signal_length=1000,
        sampling_rate=100,
        prob_strong=None,
        policy="domain",
        exclude=None,
        seed=42,
    ):
        if sampling_rate not in (100, 500) or signal_length != sampling_rate * 10:
            raise ValueError("Policy requires a full ten-second 100 Hz or 500 Hz ECG")
        if policy not in ("domain", "generic"):
            raise ValueError(f"Unknown policy {policy}")
        if exclude is not None and (
            policy != "domain" or exclude not in DOMAIN_TRANSFORMS
        ):
            raise ValueError(f"Invalid domain exclusion {exclude}")
        self.signal_length, self.sampling_rate = signal_length, sampling_rate
        self.policy, self.exclude = policy, exclude
        self.generator = torch.Generator().manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.last_applied = {}

    def specification(self):
        return {
            "version": POLICY_VERSION,
            "policy": self.policy,
            "exclude": self.exclude,
            "sampling_rate": self.sampling_rate,
            "length": self.signal_length,
            "common": {
                "jitter_std": 0.03,
                "jitter_p": 0.9,
                "scale_range": [0.9, 1.1],
                "scale_p": 0.8,
            },
            "probabilities": (
                {k: (0 if k == self.exclude else p) for k, p in PROBABILITIES.items()}
                if self.policy == "domain"
                else {"temporal_mask": 0.3}
            ),
            "domain_parameters": {
                "warp_fraction": 0.05,
                "warp_interpolation": "piecewise_linear",
                "mixup_beta": [1, 1],
                "anchor_weight_min": 0.5,
                "partners": "batch_derangement_shared_between_views",
                "bandpass_hz": {
                    "low": [0.5, 1.5],
                    "high": [30, 40] if self.sampling_rate == 100 else [40, 60],
                },
                "filter": "third_order_butterworth_zero_phase",
                "cutmix_fraction": [0.1, 0.3],
                "motion_hz": [0.5, 2],
                "motion_amplitude": [0.01, 0.05],
                "motion_units": "per_record_normalized_lead_std_not_mV",
                "channel_noise_std_fraction": [0.005, 0.02],
                "temporal_dropout_fraction": [0.05, 0.2],
            },
            "generic_parameters": {"temporal_mask_fraction": [0.05, 0.2], "fill": 0},
            "label_use": False,
        }

    def _rand(self, shape):
        return torch.rand(shape, generator=self.generator)

    def _mask(self, name, batch, probability):
        mask = self._rand((batch,)) < probability
        self.last_applied[name] = self.last_applied.get(name, 0) + int(mask.sum())
        return mask

    def __call__(self, x):
        if x.device.type != "cpu":
            raise ValueError(
                "Augment on CPU before transferring views to the training device"
            )
        single = x.ndim == 2
        if single:
            x = x.unsqueeze(0)
        if (
            x.ndim != 3
            or x.shape[-1] != self.signal_length
            or not torch.isfinite(x).all()
        ):
            raise ValueError("Expected finite [batch,leads,time] full-length waveforms")
        self.last_applied = {}
        b = len(x)
        # Derangement avoids self mixing. Partner/coefficient are shared by views.
        order = torch.randperm(b, generator=self.generator)
        partners = torch.empty_like(order)
        partners[order] = order.roll(1)
        lam = torch.tensor(self.rng.beta(1, 1, size=b), dtype=x.dtype)
        lam = torch.maximum(lam, 1 - lam)[:, None, None]
        views = [self._view(x.clone(), x, partners, lam) for _ in range(2)]
        return tuple(v.squeeze(0) if single else v for v in views)

    def _view(self, x, source, partners, lam):
        b, c, t = x.shape
        jitter = self._mask("jitter", b, 0.9)[:, None, None]
        x = (
            x
            + torch.randn(x.shape, generator=self.generator, dtype=x.dtype)
            * 0.03
            * jitter
        )
        scale_mask = self._mask("scaling", b, 0.8)[:, None, None]
        x = x * torch.where(scale_mask, 0.9 + 0.2 * self._rand((b, 1, 1)), 1.0)
        if self.policy == "generic":
            for i in torch.where(self._mask("temporal_mask", b, 0.3))[0].tolist():
                length = int(t * self.rng.uniform(0.05, 0.2))
                start = int(self.rng.integers(0, t - length + 1))
                x[i, :, start : start + length] = 0
            return x
        for name in DOMAIN_TRANSFORMS:
            if name == self.exclude:
                continue
            mask = self._mask(name, b, PROBABILITIES[name])
            if name in ("mixup", "cutmix") and b < 2:
                continue
            if name == "frequency_warp":
                # Ordered anchors ensure a monotone shared-across-leads time map.
                anchors = torch.linspace(0, t - 1, 5).expand(b, -1).clone()
                anchors[:, 1:-1] += (self._rand((b, 3)) * 2 - 1) * (0.05 * t)
                mapping = F.interpolate(
                    anchors[:, None], size=t, mode="linear", align_corners=True
                ).squeeze(1)
                left = mapping.floor().long().clamp(0, t - 1)
                right = (left + 1).clamp(0, t - 1)
                weight = (mapping - left)[:, None]
                warped = (
                    x.gather(2, left[:, None].expand(-1, c, -1)) * (1 - weight)
                    + x.gather(2, right[:, None].expand(-1, c, -1)) * weight
                )
                x = torch.where(mask[:, None, None], warped, x)
            elif name == "mixup":
                mixed = lam * x + (1 - lam) * source[partners]
                x = torch.where(mask[:, None, None], mixed, x)
            elif name == "bandpass":
                for i in torch.where(mask)[0].tolist():
                    low = self.rng.uniform(0.5, 1.5)
                    high = (
                        self.rng.uniform(30, 40)
                        if self.sampling_rate == 100
                        else self.rng.uniform(40, 60)
                    )
                    sos = butter(
                        3,
                        [low, high],
                        btype="bandpass",
                        fs=self.sampling_rate,
                        output="sos",
                    )
                    filtered = sosfiltfilt(sos, x[i].numpy(), axis=-1)
                    x[i] = torch.from_numpy(filtered.astype(np.float32).copy())
            elif name == "cutmix":
                for i in torch.where(mask)[0].tolist():
                    length = int(t * self.rng.uniform(0.1, 0.3))
                    start = int(self.rng.integers(1, t - length))
                    segment = source[partners[i], :, start : start + length]
                    # Taper both boundaries rather than inserting discontinuities.
                    taper = torch.ones(length)
                    edge = min(10, length // 2)
                    taper[:edge] = torch.linspace(0, 1, edge)
                    taper[-edge:] = torch.linspace(1, 0, edge)
                    x[i, :, start : start + length] = (1 - taper) * x[
                        i, :, start : start + length
                    ] + taper * segment
            elif name == "motion":
                time = torch.arange(t, dtype=x.dtype) / self.sampling_rate
                freq = 0.5 + 1.5 * self._rand((b, 1, 1))
                amp = 0.01 + 0.04 * self._rand((b, c, 1))
                phase = self._rand((b, 1, 1)) * (2 * torch.pi)
                x += (
                    mask[:, None, None]
                    * amp
                    * torch.sin(2 * torch.pi * freq * time + phase)
                )
            elif name == "channel_noise":
                std = x.std(-1, keepdim=True).clamp_min(1e-6)
                level = 0.005 + 0.015 * self._rand((b, c, 1))
                x += (
                    mask[:, None, None]
                    * level
                    * std
                    * torch.randn(x.shape, generator=self.generator)
                )
            elif name == "temporal_dropout":
                for i in torch.where(mask)[0].tolist():
                    length = int(t * self.rng.uniform(0.05, 0.2))
                    start = int(self.rng.integers(1, t - length))
                    w = torch.linspace(0, 1, length + 2)[1:-1]
                    x[i, :, start : start + length] = (
                        x[i, :, start - 1, None] * (1 - w)
                        + x[i, :, start + length, None] * w
                    )
        if not torch.isfinite(x).all():
            raise FloatingPointError("Augmentation produced non-finite values")
        return x


def make_augmentations(policy="domain", exclude=None, sampling_rate=100, seed=42):
    return ECGAugmentations(
        sampling_rate * 10, sampling_rate, policy=policy, exclude=exclude, seed=seed
    )


ContrastiveAugmentationPipeline = ECGAugmentations
