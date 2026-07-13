"""M2.1 deep-calibration surrogate: features -> (kappa, sigma_v, rho). PyTorch.

Key design choice — CONSTRAINED OUTPUTS. The net emits three unbounded numbers; we squash each into
its Heston-valid band via  lo + (hi-lo)*sigmoid(raw). The network therefore cannot output an invalid
parameter (kappa<=0, |rho|>=1, etc.) no matter what it learns. Domain knowledge lives in the
architecture, not just the loss.

Small + regularized on purpose (~340 samples): 2 hidden layers, tanh (smooth; also needed for the
no-arb autograd in Route B), dropout + weight decay + early stopping.
"""
from __future__ import annotations
import numpy as np

try:
    import torch
    import torch.nn as nn
except Exception:                       # torch optional at import time (sandbox has no torch)
    torch = None
    nn = object

# Heston-valid output bands (match the DE calibrator's BOUNDS in calibrate.py)
PARAM_BANDS = {"kappa": (0.1, 20.0), "sigma_v": (0.05, 2.5), "rho": (-0.95, 0.10)}
TARGETS = ["kappa", "sigma_v", "rho"]


class Surrogate(nn.Module):
    def __init__(self, n_features, hidden=64, dropout=0.15):
        super().__init__()
        lo = torch.tensor([PARAM_BANDS[t][0] for t in TARGETS])
        hi = torch.tensor([PARAM_BANDS[t][1] for t in TARGETS])
        self.register_buffer("lo", lo)
        self.register_buffer("span", hi - lo)
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden), nn.Tanh(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.Tanh(), nn.Dropout(dropout),
            nn.Linear(hidden, len(TARGETS)),
        )

    def forward(self, x):
        raw = self.net(x)
        return self.lo + self.span * torch.sigmoid(raw)   # -> (kappa, sigma_v, rho) in valid bands


def balanced_mse(pred, target, weight, target_var):
    """Per-target-variance-normalized, per-sample-weighted MSE.

    Dividing by target_var puts kappa (0-20) and rho (-1-0) on equal footing; `weight` (1/RMSE)
    emphasizes well-fit dates.
    """
    se = (pred - target) ** 2 / target_var            # (N, 3)
    return (se.mean(1) * weight).mean()
