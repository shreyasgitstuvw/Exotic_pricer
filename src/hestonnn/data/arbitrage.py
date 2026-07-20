"""Static no-arbitrage checks WITHIN a single-expiry smile.

Two flags on the call-price curve C(K) at fixed maturity:
  * vertical:  C(K) must be non-increasing in K            (dC/dK in [-df, 0])
  * butterfly: C(K) must be convex in K                    (second difference >= 0)
Calendar-spread arbitrage needs multiple maturities and is deferred to the bhavcopy layer
(D3-A) where a real term structure exists.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def flag_smile(df: pd.DataFrame, price_col="call_px", strike_col="K", df_col="df") -> pd.DataFrame:
    """Add boolean columns arb_vertical, arb_butterfly, arb_any. Expects strikes ascending."""
    d = df.sort_values(strike_col).reset_index(drop=True).copy()
    K = d[strike_col].to_numpy(float)
    C = d[price_col].to_numpy(float)
    dfac = float(d[df_col].iloc[0]) if df_col in d and len(d) else 1.0

    n = len(d)
    vert = np.zeros(n, bool)
    fly = np.zeros(n, bool)

    # vertical: slope between neighbours must lie in [-df, 0] (small tolerance)
    tol = 1e-6 * max(1.0, np.nanmax(np.abs(C)) if n else 1.0)
    for i in range(n - 1):
        dK = K[i + 1] - K[i]
        if dK <= 0:
            continue
        slope = (C[i + 1] - C[i]) / dK
        if slope > tol or slope < -dfac - tol:
            vert[i] = vert[i + 1] = True

    # butterfly: convexity via second difference of C in K
    for i in range(1, n - 1):
        h1 = K[i] - K[i - 1]
        h2 = K[i + 1] - K[i]
        if h1 <= 0 or h2 <= 0:
            continue
        second = (C[i - 1] / (h1 * (h1 + h2))
                  - C[i] / (h1 * h2)
                  + C[i + 1] / (h2 * (h1 + h2))) * 2.0
        if second < -tol:
            fly[i] = True

    d["arb_vertical"] = vert
    d["arb_butterfly"] = fly
    d["arb_any"] = vert | fly
    return d
