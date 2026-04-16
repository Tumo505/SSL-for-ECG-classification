"""Advanced augmentations for ECG signals, optimized for medical domain."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ECGAugmentations:
    """Advanced augmentations for ECG SSL pre-training.
    
    Includes both weak (always applied) and strong (probabilistic) augmentations.
    All augmentations are domain-aware for cardiac signals.
    """
    
    def __init__(self, signal_length=5000, sampling_rate=500, prob_strong=0.8):
        """
        Args:
            signal_length: Length of ECG signal
            sampling_rate: Sampling rate in Hz (PTB-XL = 500 Hz)
            prob_strong: Probability of applying strong augmentations
        """
        self.signal_length = signal_length
        self.sampling_rate = sampling_rate
        self.prob_strong = prob_strong
    
    def __call__(self, x):
        """Apply augmentations to create two views for contrastive learning.
        
        Args:
            x: [channels, time] or [batch, channels, time] tensor
            
        Returns:
            x1, x2: Two augmented views (same shape as input)
        """
        # Handle both single samples (2D) and batches (3D)
        is_2d = len(x.shape) == 2
        if is_2d:
            x = x.unsqueeze(0)  # Add batch dimension: [channels, time] -> [1, channels, time]
        
        x1 = self._apply_augmentations(x.clone())
        x2 = self._apply_augmentations(x.clone())
        
        if is_2d:
            x1 = x1.squeeze(0)  # Remove batch dimension
            x2 = x2.squeeze(0)
        
        return x1, x2
    
    def _apply_augmentations(self, x):
        """Apply augmentation pipeline: weak -> strong."""
        # WEAK (always safe, small perturbations)
        x = self._weak_jitter(x)              # Gaussian noise (~2-5%)
        x = self._weak_scaling(x)             # Amplitude scaling (~5-10%)
        x = self._augment_channel_noise(x)    # Per-channel noise
        
        # STRONG (larger transformations, probabilistic)
        if torch.rand(1).item() < self.prob_strong:
            # Temporal transformations
            x = self._time_warp(x)            # Frequency domain stretching
            x = self._augment_time_shift(x)   # Temporal shift (realistic)
            x = self._augment_dropout(x)      # Temporal dropout (masking)
            
            # Frequency domain
            x = self._augment_bandpass_variation(x)  # Device-specific filtering
            
            # Segment operations
            x = self._augment_segment_cropping(x)    # Missing data simulation
            
            # Mixing augmentations
            x = self._augment_mixup(x, alpha=0.1)    # Mix with other samples
            x = self._augment_cutmix(x, cutmix_prob=0.3)  # Segment mixing
            
            # Realistic artifacts
            x = self._augment_motion_artifacts(x)    # Motion artifact simulation
            x = self._augment_channel_shift(x)       # Per-channel variation
        
        return torch.clamp(x, -10, 10)  # Prevent exploding values
    
    # ==================== WEAK AUGMENTATIONS ====================
    
    def _weak_jitter(self, x, std=0.03):
        """Add Gaussian noise (2-5% of signal std).
        
        Simulates: Sensor noise, electrical interference
        """
        if torch.rand(1).item() < 0.9:
            noise = torch.randn_like(x) * std
            x = x + noise
        return x
    
    def _weak_scaling(self, x, scale_range=0.15):
        """Scale amplitude globally (±7.5% typical).
        
        Simulates: Recording gain variations across sessions
        """
        if torch.rand(1).item() < 0.8:
            scale = 1.0 + torch.empty(1).uniform_(-scale_range, scale_range).item()
            x = x * scale
        return x
    
    # ==================== STRONG AUGMENTATIONS ====================
    
    def _time_warp(self, x, num_points=3):
        """Non-linear time warping using simple resampling.
        
        Simulates: Temporal distortions, heart rate variation
        Creates smooth frequency-domain stretching via warped indices.
        
        Args:
            x: [batch, channels, time]
            num_points: Number of random control points (suggested: 3-5)
        """
        batch, channels, time = x.shape
        device = x.device
        
        if torch.rand(1).item() < 0.5:
            # Simple approach: create warped index mapping
            # Start and end indices are fixed
            indices = [0]
            
            # Add random intermediate waypoints
            for _ in range(num_points - 1):
                idx = np.random.randint(time // num_points, time - time // num_points)
                indices.append(idx)
            indices.append(time - 1)
            indices = sorted(set(indices))
            
            # Create continuous mapping with some elasticity
            old_indices = np.linspace(0, time - 1, len(indices), dtype=np.float32)
            new_indices = old_indices.copy()
            
            # Add random jitter to intermediate points (not endpoints)
            for i in range(1, len(new_indices) - 1):
                jitter = np.random.randn() * (time * 0.05)  # ±5% jitter
                new_indices[i] = np.clip(new_indices[i] + jitter, old_indices[i-1], old_indices[i+1])
            
            # Interpolate between waypoints
            full_mapping = np.interp(np.arange(time), old_indices, new_indices)
            
            # Apply warping via indexing
            warped = torch.zeros_like(x)
            for b in range(batch):
                for c in range(channels):
                    # Round to nearest index for gathering
                    indices_rounded = np.round(full_mapping).astype(np.int64)
                    indices_rounded = np.clip(indices_rounded, 0, time - 1)
                    indices_tensor = torch.from_numpy(indices_rounded).to(device)
                    warped[b, c] = x[b, c, indices_tensor]
            
            return warped
        
        return x
    
    def _augment_time_shift(self, x, max_shift_ratio=0.1):
        """Shift signal in time (circular, maintains length).
        
        Simulates: Recording start delay, heart rate variability
        
        Args:
            x: [batch, channels, time]
            max_shift_ratio: Max shift as fraction of signal length (~5-10%)
        """
        batch, channels, length = x.shape
        
        if torch.rand(1).item() < 0.7:
            max_shift = int(length * max_shift_ratio)
            shift = torch.randint(-max_shift, max_shift + 1, (1,)).item()
            
            if shift > 0:
                # Shift right: [zeros... | original_left_part]
                x = torch.cat([torch.zeros(batch, channels, shift, device=x.device), 
                              x[:, :, :length - shift]], dim=2)
            elif shift < 0:
                # Shift left: [original_right_part | zeros...]
                x = torch.cat([x[:, :, -shift:], 
                              torch.zeros(batch, channels, -shift, device=x.device)], dim=2)
        
        return x
    
    def _augment_channel_shift(self, x):
        """Apply different shifts/scales per channel.
        
        Simulates: Electrode placement variation, inter-lead differences
        ECG has 12 leads, each can vary independently.
        """
        batch, channels, length = x.shape
        
        if torch.rand(1).item() < 0.6 and channels > 1:
            for c in range(channels):
                # Per-channel scaling
                scale = 1.0 + torch.randn(1).item() * 0.1
                # Per-channel shift
                shift = torch.randn(1).item() * 0.05
                x[:, c, :] = x[:, c, :] * scale + shift
        
        return x
    
    def _augment_dropout(self, x, dropout_ratio=0.1):
        """Temporal dropout: mask out contiguous segments.
        
        Simulates: Signal interruptions, missing data, sensor failures
        Unlike standard dropout, maintains temporal coherence.
        
        Args:
            x: [batch, channels, time]
            dropout_ratio: Fraction of time to mask (10-20%)
        """
        batch, channels, length = x.shape
        
        if torch.rand(1).item() < 0.5:
            num_segments = max(1, int(length * dropout_ratio / 50))  # ~50-sample segments
            segment_length = int(length * dropout_ratio / num_segments)
            
            for _ in range(num_segments):
                start_idx = torch.randint(0, length - segment_length + 1, (1,)).item()
                # Mask by zeroing out segment (or replace with mean)
                mask_value = x[:, :, :].mean()
                x[:, :, start_idx:start_idx + segment_length] = mask_value
        
        return x
    
    def _augment_frequency_filter(self, x):
        """Variable frequency filtering simulating electrode/signal variation.
        
        Simulates: Low-pass filter variations, baseline wander differences
        (Alternative to time-warp for frequency-domain effects)
        """
        batch, channels, length = x.shape
        
        if torch.rand(1).item() < 0.3:
            # Simulate bandpass variation (0.5-100 Hz nominal)
            # Apply weak lowpass
            kernel_size = torch.randint(3, 8, (1,)).item() * 2 + 1
            x_filt = F.avg_pool1d(x.view(batch * channels, 1, length), 
                                  kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
            x = x_filt.view(batch, channels, length)
        
        return x
    
    # ==================== ADVANCED AUGMENTATIONS (NEW) ====================
    
    def _augment_mixup(self, x, alpha=0.2):
        """Mixup augmentation (medical-safe version).
        
        Simulates: Average of multiple ECG readings (patient variability)
        Unlike image domain, ECG mixup is clinically plausible.
        
        Args:
            x: [batch, channels, time]
            alpha: Beta distribution parameter (lower = less mixed)
        """
        batch, channels, length = x.shape
        
        if torch.rand(1).item() < 0.4 and batch > 1:
            # Random permutation
            idx = torch.randperm(batch)
            x_mixed = x.clone()
            
            # Sample mixing coefficient from Beta distribution
            lam = torch.distributions.Beta(alpha, alpha).sample().item()
            
            # Mix: lam * x + (1-lam) * x_shuffled
            x_mixed = lam * x_mixed + (1 - lam) * x[idx]
            x = x_mixed
        
        return x
    
    def _augment_cutmix(self, x, cutmix_prob=0.5):
        """CutMix for ECG: segment mixing.
        
        Simulates: Stitching recordings or electrode switching
        Replaces a segment with corresponding segment from another sample.
        
        Args:
            x: [batch, channels, time]
            cutmix_prob: Probability of CutMix
        """
        batch, channels, length = x.shape
        
        if torch.rand(1).item() < cutmix_prob and batch > 1:
            idx = torch.randperm(batch)
            
            # Random segment to replace
            segment_start = torch.randint(0, length // 2, (1,)).item()
            segment_length = torch.randint(length // 10, length // 3, (1,)).item()
            segment_end = min(segment_start + segment_length, length)
            
            # Replace segment from random sample
            x[:, :, segment_start:segment_end] = x[idx, :, segment_start:segment_end]
        
        return x
    
    def _augment_bandpass_variation(self, x):
        """Bandpass filter with random corner frequencies.
        
        Simulates: Different recording devices with different frequency responses
        ECG normal bandwidth: 0.5-100 Hz, but varies by device (0.05-250 Hz possible)
        """
        batch, channels, length = x.shape
        device = x.device
        
        if torch.rand(1).item() < 0.5:
            # Random corner frequencies (Hz)
            lowcut = torch.randint(0, 50, (1,)).item() * 0.1  # 0-5 Hz
            highcut = torch.randint(50, 250, (1,)).item()      # 50-250 Hz
            
            if lowcut >= highcut:
                lowcut, highcut = 0.1, 100.0
            
            # Simple frequency domain filtering (FFT)
            x_fft = torch.fft.rfft(x, dim=-1)
            
            # Create frequency mask (triangular window)
            freqs = torch.fft.rfftfreq(length, d=1.0/self.sampling_rate, device=device)
            mask = torch.ones_like(freqs)
            mask[freqs < lowcut] = torch.linspace(0, 1, (freqs < lowcut).sum()).to(device)
            mask[freqs > highcut] = torch.linspace(1, 0, (freqs > highcut).sum()).to(device)
            
            x_fft = x_fft * mask.unsqueeze(0).unsqueeze(0)
            x = torch.fft.irfft(x_fft, n=length)[:, :, :length]
        
        return x
    
    def _augment_segment_cropping(self, x, crop_ratio_range=(0.1, 0.3)):
        """Segment cropping: remove contiguous regions and interpolate.
        
        Simulates: Missing data, signal dropout, electrode contact loss
        Maintains temporal continuity by interpolation.
        
        Args:
            x: [batch, channels, time]
            crop_ratio_range: Fraction of signal to remove (10-30%)
        """
        batch, channels, length = x.shape
        
        if torch.rand(1).item() < 0.5:
            crop_ratio = torch.empty(1).uniform_(crop_ratio_range[0], crop_ratio_range[1]).item()
            crop_length = int(length * crop_ratio)
            
            for b in range(batch):
                # Random start position
                start_idx = torch.randint(0, max(1, length - crop_length), (1,)).item()
                end_idx = min(start_idx + crop_length, length)
                
                # Simple linear interpolation across gap
                if start_idx > 0 and end_idx < length:
                    left_val = x[b, :, start_idx - 1]
                    right_val = x[b, :, end_idx]
                    
                    # Create interpolated values
                    t = torch.linspace(0, 1, end_idx - start_idx + 2)[1:-1].to(x.device)
                    interp_vals = left_val.unsqueeze(-1) * (1 - t) + right_val.unsqueeze(-1) * t
                    x[b, :, start_idx:end_idx] = interp_vals
        
        return x
    
    def _augment_channel_noise(self, x):
        """Per-channel noise simulation.
        
        Simulates: Different noise levels on each of 12 ECG leads
        Each lead can have different signal quality.
        
        Args:
            x: [batch, channels, time]
        """
        batch, channels, length = x.shape
        
        if torch.rand(1).item() < 0.6 and channels > 1:
            for c in range(channels):
                # Per-channel noise level (0.5-2% of channel std)
                noise_level = torch.empty(1).uniform_(0.005, 0.02).item()
                channel_std = x[:, c, :].std()
                
                noise = torch.randn(batch, length, device=x.device) * noise_level * channel_std
                x[:, c, :] = x[:, c, :] + noise
        
        return x
    
    def _augment_motion_artifacts(self, x, artifact_duration_ratio=0.05):
        """Motion artifact simulation.
        
        Simulates: Patient movement, electrode motion, baseline wander
        Adds realistic motion artifact patterns (low-frequency + high baseline shift).
        
        Args:
            x: [batch, channels, time]
            artifact_duration_ratio: Duration as fraction of signal (5% typical)
        """
        batch, channels, length = x.shape
        device = x.device
        
        if torch.rand(1).item() < 0.5:
            num_artifacts = torch.randint(1, 4, (1,)).item()  # 1-3 artifacts
            
            for b in range(batch):
                for _ in range(num_artifacts):
                    # Artifact location and duration
                    artifact_start = torch.randint(0, length - int(length * artifact_duration_ratio), (1,)).item()
                    artifact_length = int(length * artifact_duration_ratio)
                    artifact_end = artifact_start + artifact_length
                    
                    # Motion artifact: low-freq baseline shift + high-freq noise burst
                    t = torch.linspace(0, 1, artifact_length, device=device)
                    
                    # Low-frequency baseline shift (0.5-2 Hz)
                    freq = torch.empty(1).uniform_(0.5, 2.0).item()
                    baseline_shift = torch.sin(2 * np.pi * freq * t) * torch.empty(1).uniform_(0.5, 2.0).item()
                    
                    # High-frequency noise burst
                    noise_burst = torch.randn(artifact_length, device=device) * 0.3
                    
                    # Combined artifact
                    artifact = baseline_shift + noise_burst
                    
                    for c in range(channels):
                        x[b, c, artifact_start:artifact_end] += artifact
        
        return x


class ContrastiveAugmentationPipeline:
    """Pipeline for creating augmented pairs for contrastive learning."""
    
    def __init__(self, signal_length=5000, sampling_rate=500, prob_strong=0.8):
        self.augment = ECGAugmentations(signal_length, sampling_rate, prob_strong)
    
    def __call__(self, x):
        """Create two views from same sample.
        
        Args:
            x: ECG signal [batch, channels, time]
            
        Returns:
            (view1, view2): Two augmented versions
        """
        return self.augment(x)


# ======================== TESTING ========================

if __name__ == "__main__":
    # Example usage
    batch_size = 4
    channels = 12  # PTB-XL: 12-lead ECG
    signal_length = 5000
    
    x = torch.randn(batch_size, channels, signal_length)
    
    aug = ECGAugmentations(signal_length=signal_length, prob_strong=0.8)
    x1, x2 = aug(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Augmented view 1: {x1.shape}")
    print(f"Augmented view 2: {x2.shape}")
    print(f"Views differ: {not torch.allclose(x1, x2)}")
    print(f"Signal range: [{x1.min():.3f}, {x1.max():.3f}]")
