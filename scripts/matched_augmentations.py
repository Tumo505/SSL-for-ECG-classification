"""Generic control with domain-matched gate counts and per-view RMS distance.

Reference views supply magnitudes, never labels or diagnostic predictions.
The seven generic slots are not asserted to have ECG physiological fidelity.
"""
import torch
from torch.nn import functional as F
from ssrl_ecg.augmentations import ECGAugmentations, DOMAIN_TRANSFORMS


class MatchedGeneric(ECGAugmentations):
    def __init__(self, sampling_rate=100, seed=42):
        super().__init__(sampling_rate*10, sampling_rate, policy='domain', seed=seed)
        self.recorded = []
        self.recording = False
        self.match_diagnostics = {}

    def _mask(self, name, batch, probability):
        mask = super()._mask(name, batch, probability)
        if self.recording:
            self.recorded.append((name, mask.clone()))
        return mask

    def specification(self):
        return dict(version='generic_gate_rms_v1', policy='generic_matched',
                    reference=super().specification(),
                    slots=['noise','scaling','circular_shift','zero_mask','moving_average','linear_trend','segment_permutation'],
                    matching='identical per-record gates and per-view RMS distance from the source',
                    limitation='RMS and gate matching do not match spectral or morphological distortion',
                    label_use=False, extra_compute='domain reference views generated to obtain magnitudes')

    def __call__(self, x):
        if x.ndim != 3 or len(x) < 2:
            raise ValueError('Matched control requires batched CPU waveforms with at least two records')
        self.recorded = []
        self.recording = True
        reference = super().__call__(x)
        self.recording = False
        all_gates = self.recorded
        outputs, errors = [], []
        for view, target in enumerate(reference):
            gates = dict(all_gates[view*9:(view+1)*9])
            z = x.clone()
            b,c,t = x.shape
            z += gates['jitter'][:,None,None]*torch.randn(z.shape,generator=self.generator)*.03
            z *= torch.where(gates['scaling'][:,None,None],.9+.2*self._rand((b,1,1)),1.)
            for slot,name in enumerate(DOMAIN_TRANSFORMS):
                mask = gates[name]
                if slot == 0:
                    changed = z + .1*torch.randn(z.shape,generator=self.generator)
                elif slot == 1:
                    changed = z*(.8+.4*self._rand((b,1,1)))
                elif slot == 2:
                    changed = torch.stack([torch.roll(z[i],int(self.rng.integers(-50,51)),-1) for i in range(b)])
                elif slot == 3:
                    changed = z.clone()
                    for i in range(b):
                        length=int(t*self.rng.uniform(.05,.2)); start=int(self.rng.integers(0,t-length+1))
                        changed[i,:,start:start+length]=0
                elif slot == 4:
                    changed = F.avg_pool1d(F.pad(z,(4,4),mode='replicate'),9,stride=1)
                elif slot == 5:
                    changed = z + (.1*(2*self._rand((b,c,1))-1))*torch.linspace(-1,1,t)
                else:
                    changed = z.clone()
                    for i in range(b):
                        order=torch.randperm(5,generator=self.generator)
                        changed[i]=z[i].reshape(c,5,t//5)[:,order].reshape(c,t)
                z=torch.where(mask[:,None,None],changed,z)
            delta=z-x
            target_norm=(target-x).square().mean((1,2),keepdim=True).sqrt()
            norm=delta.square().mean((1,2),keepdim=True).sqrt()
            if (norm < 1e-12).any():
                delta=torch.where(norm<1e-12,torch.randn(delta.shape,generator=self.generator),delta)
                norm=delta.square().mean((1,2),keepdim=True).sqrt()
            output=x+delta*(target_norm/norm)
            if not torch.isfinite(output).all():
                raise FloatingPointError('Nonfinite matched augmentation')
            errors.append(float(((output-x).square().mean((1,2)).sqrt()-target_norm.flatten()).abs().max()))
            outputs.append(output)
        self.match_diagnostics=dict(max_absolute_rms_error=max(errors),gates_equal=True)
        return tuple(outputs)
