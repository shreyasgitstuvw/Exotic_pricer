"""Full Heston calibration to an assembled surface — vega-weighted IV RMSE (BENCHMARKS ceiling).

This is the M1 "full optimizer" baseline (DE -> L-BFGS-B), the accuracy ceiling the M2 NN surrogate
is later measured against. Calibrates all five params; D2's (kappa, sigma_v, rho) NN split applies to
the M2 surrogate, not to this reference fit.

Objective avoids re-inverting model IV each eval: price residual scaled by market vega approximates
the IV residual, i.e. minimize  sqrt(mean( ((C_model - C_mkt)/vega_mkt)^2 ))  in vol units.
Uses heston_ref (reference pricer) until the frozen engine is pinned via engine.py.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import differential_evolution, minimize

from .heston_ref import HParams, heston_call
from .data.iv import black76_vega

# (kappa, theta, sigma_v, rho, v0)
BOUNDS = [(0.1, 20.0), (1e-3, 1.00), (0.05, 2.5), (-0.95, 0.10), (1e-3, 1.00)]  # v0,theta->100% vol (COVID)


def _prep(surface):
    """Group surface quotes by expiry into per-tenor arrays (F, df, T, K, mkt_call, vega)."""
    groups = []
    for e, sub in surface.groupby("expiry"):
        F = float(sub["F"].iloc[0]); df = float(sub["df"].iloc[0]); T = float(sub["tte_yr"].iloc[0])
        K = sub["K"].to_numpy(float)
        mkt = sub["call_px"].to_numpy(float)
        iv = sub["iv"].to_numpy(float)
        vega = black76_vega(F, K, T, iv, df)
        vega = np.maximum(vega, 1e-4 * F)          # floor so deep wings don't blow up the ratio
        groups.append((F, df, T, K, mkt, vega))
    return groups


def objective(x, groups, feller_w=0.0):
    p = HParams(*x)
    se = 0.0; n = 0
    for F, df, T, K, mkt, vega in groups:
        model = heston_call(F, K, T, p, df)
        r = (model - mkt) / vega
        se += float(np.sum(r * r)); n += len(K)
    rmse = np.sqrt(se / max(n, 1))
    if feller_w:                                    # soft Feller penalty (2*k*theta >= sigma_v^2)
        rmse += feller_w * max(0.0, p.sigma_v ** 2 - 2 * p.kappa * p.theta)
    return rmse


def calibrate(surface, feller_w=0.0, seed=0, de_maxiter=40, popsize=15, polish=True):
    """Return (HParams, stats). stats has iv_rmse (vol units), iv_rmse_bps, feller_ok, n_quotes."""
    groups = _prep(surface)
    res = differential_evolution(objective, BOUNDS, args=(groups, feller_w), seed=seed,
                                 maxiter=de_maxiter, popsize=popsize, tol=1e-7,
                                 mutation=(0.5, 1.0), recombination=0.7, polish=False)
    x = res.x
    if polish:
        r2 = minimize(objective, x, args=(groups, feller_w), bounds=BOUNDS, method="L-BFGS-B")
        if r2.fun <= res.fun:
            x = r2.x
    p = HParams(*x)
    rmse = objective(x, groups, 0.0)
    stats = {"iv_rmse": rmse, "iv_rmse_bps": rmse * 1e4,
             "feller_ok": p.feller_ok(), "n_quotes": int(sum(len(g[3]) for g in groups)),
             "n_tenors": len(groups)}
    return p, stats


# ---- constrained single-smile fit (short-end instability proxy) -------------------------------
# One short-tenor smile can't identify all five Heston params, but with (kappa, theta) fixed the
# level/skew/curvature of the smile pin (v0, rho, sigma_v) reasonably. Used by M1's short-end proxy.
_C_BOUNDS = [(0.05, 2.5), (-0.95, 0.10), (1e-3, 1.00)]   # sigma_v, rho, v0 (COVID-wide)


def calibrate_constrained(smile, kappa=2.0, theta=0.04, starts=2, seed=0):
    """Fit (sigma_v, rho, v0) to a single-tenor smile with kappa, theta fixed. Returns (HParams, rmse_bps)."""
    from .heston_ref import HParams
    F = float(smile["F"].iloc[0]); df = float(smile["df"].iloc[0]); T = float(smile["tte_yr"].iloc[0])
    K = smile["K"].to_numpy(float); mkt = smile["call_px"].to_numpy(float)
    iv = smile["iv"].to_numpy(float)
    vega = np.maximum(black76_vega(F, K, T, iv, df), 1e-4 * F)

    def obj(z):
        p = HParams(kappa, theta, z[0], z[1], z[2])
        r = (heston_call(F, K, T, p, df) - mkt) / vega
        return float(np.sqrt(np.mean(r * r)))

    rng = np.random.default_rng(seed)
    best_x, best_f = None, np.inf
    inits = [np.array([0.6, -0.5, max(iv[np.argmin(np.abs(K - F))] ** 2, 1e-3)])]
    for _ in range(max(0, starts - 1)):
        inits.append(np.array([rng.uniform(0.2, 1.5), rng.uniform(-0.9, -0.1),
                               rng.uniform(0.005, 0.08)]))
    for x0 in inits:
        r = minimize(obj, x0, bounds=_C_BOUNDS, method="L-BFGS-B")
        if r.fun < best_f:
            best_f, best_x = r.fun, r.x
    from .heston_ref import HParams as HP
    return HP(kappa, theta, best_x[0], best_x[1], best_x[2]), best_f * 1e4


def calibrate_warm(surface, x0, jitter=1, seed=0):
    """Local re-calibration from a warm start x0 (previous date's params). Fast: L-BFGS-B + a jittered
    restart. Use for sequential M1 time series where params evolve slowly day to day."""
    groups = _prep(surface)
    rng = np.random.default_rng(seed)
    cands = [np.asarray(x0, float)]
    for _ in range(max(0, jitter)):
        j = np.asarray(x0, float) * (1 + rng.uniform(-0.2, 0.2, size=5))
        cands.append(np.clip(j, [b[0] for b in BOUNDS], [b[1] for b in BOUNDS]))
    best_x, best_f = None, np.inf
    for c in cands:
        r = minimize(objective, c, args=(groups, 0.0), bounds=BOUNDS, method="L-BFGS-B")
        if r.fun < best_f:
            best_f, best_x = r.fun, r.x
    p = HParams(*best_x)
    return p, {"iv_rmse_bps": best_f * 1e4, "feller_ok": p.feller_ok(),
               "n_quotes": int(sum(len(g[3]) for g in groups)), "n_tenors": len(groups)}
