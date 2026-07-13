"""Surface -> fixed-size feature vector for the M2 calibration surrogate.

A neural net needs a fixed-length input, but each day's surface has a variable number of strikes and
tenors. We summarize the surface into a fixed, *interpretable* fingerprint by exploiting the fact that
each Heston parameter governs one geometric feature of the smile:

    ATM level  (a)  <->  v0 / theta      per tenor, IV(x) = a + b*x + c*x^2,  x = log(K/F)
    skew       (b)  <->  rho             (dIV/dx at the money)
    curvature  (c)  <->  sigma_v         (convexity of the smile)

Features per date = [a, b, c] for each maturity bucket + the ATM term-structure slope. Handing the
network these quantities (instead of raw prices) is what lets a small net learn from ~340 examples:
the hard part (turning prices into shape) is done here, analytically.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

# maturity buckets in calendar days (short -> long). One (level, skew, curv) triple per bucket.
BUCKETS = [(14, 30), (30, 60), (60, 120), (120, 250), (250, 1000)]
BUCKET_LABELS = ["b1_2w1m", "b2_1_2m", "b3_2_4m", "b4_4_8m", "b5_8m_plus"]
FEATURE_NAMES = ([f"{lbl}_{k}" for lbl in BUCKET_LABELS for k in ("lvl", "skew", "curv")]
                 + ["term_slope", "atm_level"])


def _fit_smile(x, iv, w=None):
    """Weighted quadratic fit IV = a + b*x + c*x^2. Returns (a, b, c) = (level, skew, curvature).

    a is evaluated at x=0 (at-the-money) = the intercept. Needs >=3 points; else lower-order fallback.
    """
    x = np.asarray(x, float); iv = np.asarray(iv, float)
    good = np.isfinite(x) & np.isfinite(iv)
    x, iv = x[good], iv[good]
    if w is None:
        w = np.ones_like(x)
    else:
        w = np.asarray(w, float)[good]
    n = len(x)
    if n == 0:
        return np.nan, np.nan, np.nan
    if n == 1:
        return float(iv[0]), 0.0, 0.0
    deg = 2 if n >= 3 else 1
    W = np.sqrt(np.clip(w, 1e-9, None))
    A = np.vander(x, deg + 1)          # columns: [x^deg ... x 1]
    coef, *_ = np.linalg.lstsq(A * W[:, None], iv * W, rcond=None)
    c = coef[0] if deg == 2 else 0.0   # x^2 coeff
    b = coef[-2]                       # x^1 coeff
    a = coef[-1]                       # intercept = ATM
    return float(a), float(b), float(c)


def surface_features(surf: pd.DataFrame) -> dict:
    """Compute the fixed feature dict for ONE date's surface.

    Expects columns: tte_yr, log_m, iv (one date). Missing buckets are back/forward-filled from the
    nearest present bucket so the vector is always full-length.
    """
    feats = {}
    bucket_atm = {}
    for (lo, hi), lbl in zip(BUCKETS, BUCKET_LABELS):
        sub = surf[(surf["tte_yr"] * 365 >= lo) & (surf["tte_yr"] * 365 < hi)]
        if len(sub):
            # weight toward the ATM region so level/skew are pinned by liquid strikes
            wgt = np.exp(-((sub["log_m"].to_numpy()) ** 2) / (2 * 0.1 ** 2))
            a, b, c = _fit_smile(sub["log_m"], sub["iv"], wgt)
        else:
            a = b = c = np.nan
        feats[f"{lbl}_lvl"], feats[f"{lbl}_skew"], feats[f"{lbl}_curv"] = a, b, c
        bucket_atm[lbl] = a

    # fill empty buckets from nearest present neighbour (short<->long continuity)
    present = [l for l in BUCKET_LABELS if np.isfinite(bucket_atm[l])]
    if present:
        for i, lbl in enumerate(BUCKET_LABELS):
            if not np.isfinite(bucket_atm[lbl]):
                near = min(present, key=lambda p: abs(BUCKET_LABELS.index(p) - i))
                for k in ("lvl", "skew", "curv"):
                    feats[f"{lbl}_{k}"] = feats[f"{near}_{k}"]
                bucket_atm[lbl] = bucket_atm[near]

    atm_short = bucket_atm[BUCKET_LABELS[0]]
    atm_long = bucket_atm[BUCKET_LABELS[-1]]
    feats["term_slope"] = (atm_long - atm_short) if np.isfinite(atm_short) and np.isfinite(atm_long) else 0.0
    feats["atm_level"] = atm_short if np.isfinite(atm_short) else np.nan
    return feats


# --- short-end features (M2.2): the observed weekly smile, which the >=14d surface doesn't determine ---
SHORT_BUCKETS = [(1, 7), (7, 14)]
SHORT_LABELS = ["s1_lt7d", "s2_7_14d"]
SHORT_FEATURE_NAMES = ([f"{l}_{k}" for l in SHORT_LABELS for k in ("lvl", "skew", "curv")]
                       + ["days_to_expiry"])


def short_end_features(surf: pd.DataFrame) -> dict:
    """Level/skew/curvature of the observed <14d smile per short bucket, + days to nearest expiry.

    These are what make Route B a market-conditioned no-arb parameterization (industry use): the desk
    observes the weeklies, and the layer represents them arbitrage-free. Empty buckets are filled from
    the neighbour so the vector is always full-length.
    """
    feats, atm = {}, {}
    for (lo, hi), lbl in zip(SHORT_BUCKETS, SHORT_LABELS):
        sub = surf[(surf["tte_yr"] * 365 >= lo) & (surf["tte_yr"] * 365 < hi)]
        if len(sub):
            w = np.exp(-((sub["log_m"].to_numpy()) ** 2) / (2 * 0.1 ** 2))
            a, b, c = _fit_smile(sub["log_m"], sub["iv"], w)
        else:
            a = b = c = np.nan
        feats[f"{lbl}_lvl"], feats[f"{lbl}_skew"], feats[f"{lbl}_curv"] = a, b, c
        atm[lbl] = a
    present = [l for l in SHORT_LABELS if np.isfinite(atm[l])]
    if present:
        for lbl in SHORT_LABELS:
            if not np.isfinite(atm[lbl]):
                near = present[0]
                for k in ("lvl", "skew", "curv"):
                    feats[f"{lbl}_{k}"] = feats[f"{near}_{k}"]
    feats["days_to_expiry"] = float(surf["tte_yr"].min() * 365) if len(surf) else np.nan
    return feats


def features_frame(surfaces: pd.DataFrame) -> pd.DataFrame:
    """Feature matrix (one row per date) from a multi-date surface frame (needs a 'date' column)."""
    rows = []
    for d, g in surfaces.groupby("date"):
        f = surface_features(g)
        f["date"] = d
        rows.append(f)
    df = pd.DataFrame(rows).set_index("date").sort_index()
    return df[FEATURE_NAMES]
