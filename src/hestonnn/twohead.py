"""Two-headed network: one shared encoder -> Heston params (Head A) + short-end correction (Head B).

Multi-task hypothesis: training the encoder on BOTH the calibration-surrogate task (M2.1) and the
gap-closer task (M2.2) yields a richer market-state representation and better generalization on the
gap-closer than the standalone (which peaked at 60% on test).

Reuses the exact, already-validated output parameterizations:
  Head A -> (kappa, sigma_v, rho) squashed into Heston-valid bands (from surrogate.py).
  Head B -> per-short-bucket (d0, d1, d2) in the rescaled z-basis (from gap_closer.py) that made the
            curvature trainable.
"""
from __future__ import annotations
import numpy as np

try:
    import torch
    import torch.nn as nn
except Exception:
    torch = None
    nn = object

from .surrogate import PARAM_BANDS, TARGETS          # (kappa, sigma_v, rho) bands
from .gap_closer import N_BUCKETS                     # short tenor buckets


class TwoHead(nn.Module):
    def __init__(self, n_features, hidden=64, dropout=0.05):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_features, hidden), nn.Tanh(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.Tanh(), nn.Dropout(dropout),
        )
        self.head_params = nn.Linear(hidden, len(TARGETS))
        self.head_corr = nn.Linear(hidden, N_BUCKETS * 3)
        lo = torch.tensor([PARAM_BANDS[t][0] for t in TARGETS])
        hi = torch.tensor([PARAM_BANDS[t][1] for t in TARGETS])
        self.register_buffer("plo", lo)
        self.register_buffer("pspan", hi - lo)

    def forward(self, feats):
        h = self.encoder(feats)
        # Head A: Heston params in valid bands
        params = self.plo + self.pspan * torch.sigmoid(self.head_params(h))     # (B, 3)
        # Head B: per-bucket correction coeffs (z-space scaling that trains — see gap_closer)
        raw = self.head_corr(h).view(-1, N_BUCKETS, 3)
        d0 = raw[..., 0] * 0.05
        d1 = raw[..., 1] * 0.10
        d2 = (torch.nn.functional.softplus(raw[..., 2]) * 0.30).clamp(max=2.0)
        corr = torch.stack([d0, d1, d2], dim=-1)                                 # (B, N_BUCKETS, 3)
        return params, corr
