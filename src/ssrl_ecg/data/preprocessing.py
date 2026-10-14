import numpy as np


class ECGPreprocessor:
    """Per-record, per-lead standardisation; no fitted test-set statistics."""

    def preprocess(self, signal):
        x = np.asarray(signal, dtype=np.float32)
        return (x - x.mean(-1, keepdims=True)) / np.maximum(
            x.std(-1, keepdims=True), 1e-6
        )
