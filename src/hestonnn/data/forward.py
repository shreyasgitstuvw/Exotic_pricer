"""Chain-implied forward and discount factor via put-call parity.

For European options:  C(K) - P(K) = df * (F - K),  which is LINEAR in K with
slope = -df and intercept = df * F. A robust line fit across liquid strikes recovers both the
discount factor df and the forward F straight from the option chain — no external rate curve or
dividend assumption needed (D3-A). Implied r from short-dated PCP is noisy, so we clamp it to a
sane band and primarily trust F.
"""
from __future__ import annotations
import numpy as np

_R_BAND = (-0.02, 0.15)  # plausible annualized rate band for sanity clamping


def implied_forward_df(strikes, call_px, put_px, T, weights=None):
    """Return (F, df, r, n_used). Uses a weighted least-squares fit of (C-P) vs K.

    strikes, call_px, put_px: 1D arrays for a single expiry snapshot.
    T: year-fraction (for reporting r only).
    """
    K = np.asarray(strikes, float)
    y = np.asarray(call_px, float) - np.asarray(put_px, float)
    good = np.isfinite(K) & np.isfinite(y)
    if weights is None:
        weights = np.ones_like(K)
    w = np.asarray(weights, float)
    good &= np.isfinite(w) & (w > 0)
    K, y, w = K[good], y[good], w[good]
    if K.size < 3:
        return np.nan, np.nan, np.nan, K.size

    # weighted linear fit y = a*K + b  ->  slope a = -df, intercept b = df*F
    W = np.sqrt(w)
    A = np.column_stack([K * W, W])
    coef, *_ = np.linalg.lstsq(A, y * W, rcond=None)
    a, b = coef
    df = -a
    if not np.isfinite(df) or df <= 0:
        return np.nan, np.nan, np.nan, K.size
    F = b / df
    r = np.nan
    if T and T > 0:
        r = -np.log(np.clip(df, 1e-6, 1.0)) / T
        r = float(np.clip(r, *_R_BAND))
        df = float(np.exp(-r * T))  # reconcile df with clamped r
    return float(F), float(df), r, int(K.size)
