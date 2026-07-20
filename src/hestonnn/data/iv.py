"""Black-76 pricing and robust implied-vol inversion (options on the forward).

NIFTY options are European on the index; we price against the chain-implied forward F (see
forward.py), so Black-76 is the natural model. IV convention here MUST be reconciled with the frozen
Heston engine's own inverter when the engine is wired in (D1 open item) — keep this the single
source of truth until then.
"""
from __future__ import annotations
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

SQRT_EPS = 1e-12


def black76_price(F, K, T, sigma, df=1.0, call=True):
    """Undiscounted-forward Black-76 price times discount factor df.

    F: forward, K: strike, T: year-fraction, sigma: vol, df: discount factor exp(-rT).
    """
    F = np.asarray(F, float); K = np.asarray(K, float)
    T = np.asarray(T, float); sigma = np.asarray(sigma, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        vsqrt = sigma * np.sqrt(np.maximum(T, 0.0))
        d1 = (np.log(F / K) + 0.5 * vsqrt**2) / np.where(vsqrt > SQRT_EPS, vsqrt, np.nan)
        d2 = d1 - vsqrt
        call_px = df * (F * norm.cdf(d1) - K * norm.cdf(d2))
        put_px = df * (K * norm.cdf(-d2) - F * norm.cdf(-d1))
    px = np.where(call, call_px, put_px)
    # sigma -> 0 limit = discounted intrinsic on the forward
    intrinsic = df * np.where(call, np.maximum(F - K, 0.0), np.maximum(K - F, 0.0))
    px = np.where(vsqrt > SQRT_EPS, px, intrinsic)
    return px


def black76_vega(F, K, T, sigma, df=1.0):
    F = np.asarray(F, float); K = np.asarray(K, float)
    T = np.asarray(T, float); sigma = np.asarray(sigma, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        vsqrt = sigma * np.sqrt(np.maximum(T, 0.0))
        d1 = (np.log(F / K) + 0.5 * vsqrt**2) / np.where(vsqrt > SQRT_EPS, vsqrt, np.nan)
        vega = df * F * norm.pdf(d1) * np.sqrt(np.maximum(T, 0.0))
    return np.where(vsqrt > SQRT_EPS, vega, 0.0)


def implied_vol_one(price, F, K, T, df=1.0, call=True, lo=1e-4, hi=5.0):
    """Invert a single quote for Black-76 IV. Returns np.nan if no arbitrage-free root exists."""
    if not np.isfinite(price) or price <= 0 or T <= 0:
        return np.nan
    intrinsic = df * (max(F - K, 0.0) if call else max(K - F, 0.0))
    upper = df * (F if call else K)  # no-arb price ceiling
    if price <= intrinsic + 1e-8 or price >= upper - 1e-10:
        return np.nan  # below intrinsic or above ceiling -> not invertible

    def f(sig):
        return float(black76_price(F, K, T, sig, df, call)) - price

    try:
        if f(lo) * f(hi) > 0:
            return np.nan
        return brentq(f, lo, hi, xtol=1e-8, maxiter=100)
    except (ValueError, RuntimeError):
        return np.nan


def implied_vol(price, F, K, T, df=1.0, call=True, newton_iter=60):
    """Vectorized IV inversion via Newton-Raphson across the whole smile, brentq fallback.

    Newton runs on all strikes simultaneously (numpy), which is ~100x faster than a per-quote
    brentq loop; any quote that fails to converge falls back to the robust bracketed solver.
    """
    price = np.atleast_1d(np.asarray(price, float))
    K = np.broadcast_to(np.asarray(K, float), price.shape).astype(float)
    call = np.broadcast_to(np.asarray(call), price.shape)
    F = float(F); T = float(T); df = float(df)

    # invertibility mask: strictly between discounted intrinsic and no-arb ceiling
    intrinsic = df * np.where(call, np.maximum(F - K, 0.0), np.maximum(K - F, 0.0))
    ceil = df * np.where(call, F, K)
    valid = np.isfinite(price) & (price > intrinsic + 1e-8) & (price < ceil - 1e-10) & (T > 0)

    sig = np.full(price.shape, 0.3)  # init
    for _ in range(newton_iter):
        model = black76_price(F, K, T, sig, df, call)
        vega = black76_vega(F, K, T, sig, df)
        step = np.where(vega > 1e-8, (model - price) / vega, 0.0)
        step = np.clip(step, -0.5, 0.5)  # damp
        sig = np.clip(sig - step, 1e-4, 5.0)
    resid = np.abs(black76_price(F, K, T, sig, df, call) - price)
    conv = valid & (resid < 1e-4 * np.maximum(1.0, price))

    out = np.where(conv, sig, np.nan)
    # brentq fallback for the few valid-but-unconverged quotes
    bad = valid & ~conv
    for i in np.flatnonzero(bad):
        out.flat[i] = implied_vol_one(price.flat[i], F, K.flat[i], T, df, bool(call.flat[i]))
    out[~valid] = np.nan
    return out
