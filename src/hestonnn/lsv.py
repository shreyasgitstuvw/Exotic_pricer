"""Heston stochastic-local-volatility (SLV) via the Guyon-Henry-Labordere particle method.

Model (forward measure, forward F a martingale):
    dF/F = L(F,t) sqrt(v) dW1
    dv   = kappa(theta - v) dt + sigma_v sqrt(v) dW2,   dW1 dW2 = rho dt
The leverage L is calibrated so the model reproduces the market vanilla smile:
    L(F,t)^2 = sigma_LV(F,t)^2 / E[v_t | F_t = F]
E[v|F] is estimated on the fly from the particle cloud (binning), so ONE forward simulation both
builds L and yields the paths used to price path-dependent exotics.

Reduces to: pure Heston when L==1; pure local-vol when v is frozen. Validated by the Gyongy identity
(SLV reproduces the same vanillas as the local-vol model it was calibrated to).
"""
from __future__ import annotations
import numpy as np


def _corr_normals(rng, n, rho):
    z1 = rng.standard_normal(n)
    z2 = rho * z1 + np.sqrt(max(1 - rho ** 2, 0.0)) * rng.standard_normal(n)
    return z1, z2


def local_vol_mc(sigma_lv, F0, T, n_steps=100, n_paths=40000, seed=0, barrier=None):
    """Pure local-vol MC: dF/F = sigma_lv(F,t) dW. Returns terminal F (and knock-out survival mask)."""
    dt = T / n_steps; rng = np.random.default_rng(seed)
    F = np.full(n_paths, F0); alive = np.ones(n_paths, bool)
    for j in range(n_steps):
        s = np.clip(sigma_lv(F, j * dt), 1e-4, 5.0)
        F = F * np.exp(-0.5 * s ** 2 * dt + s * np.sqrt(dt) * rng.standard_normal(n_paths))
        if barrier is not None:
            alive &= _barrier_ok(F, barrier)
    return F, alive


def particle_slv(params, sigma_lv, F0, T, n_steps=100, n_paths=40000, n_bins=60, seed=0,
                 barrier=None, l_clip=(0.1, 5.0)):
    """Heston-SLV particle method. params=(kappa,theta,sigma_v,rho,v0). sigma_lv(F,t)->local vol.
    Returns terminal F, knock-out survival mask, and the leverage samples over time (diagnostic)."""
    kappa, theta, sigma_v, rho, v0 = params
    dt = T / n_steps; rng = np.random.default_rng(seed)
    F = np.full(n_paths, F0); v = np.full(n_paths, v0); alive = np.ones(n_paths, bool)
    Lhist = []
    for j in range(n_steps):
        t = j * dt
        # E[v | F] by binning F into quantile bins, mean v per bin
        bins = np.quantile(F, np.linspace(0, 1, n_bins + 1))
        bins[0] -= 1e-6; bins[-1] += 1e-6
        idx = np.clip(np.digitize(F, bins) - 1, 0, n_bins - 1)
        vpos = np.maximum(v, 0.0)
        # E[v|bin] vectorized: bin sums / bin counts, then scatter back
        sums = np.bincount(idx, weights=vpos, minlength=n_bins)
        cnts = np.bincount(idx, minlength=n_bins)
        binmean = sums / np.maximum(cnts, 1)
        Ev = binmean[idx]
        L = np.clip(sigma_lv(F, t) / np.sqrt(np.maximum(Ev, 1e-8)), *l_clip)
        Lhist.append(L.mean())
        z1, z2 = _corr_normals(rng, n_paths, rho)
        F = F * np.exp(-0.5 * (L ** 2) * vpos * dt + L * np.sqrt(vpos * dt) * z1)
        v = v + kappa * (theta - vpos) * dt + sigma_v * np.sqrt(vpos * dt) * z2
        if barrier is not None:
            alive &= _barrier_ok(F, barrier)
    return F, alive, np.array(Lhist)


def heston_mc(params, F0, T, n_steps=100, n_paths=40000, seed=0, barrier=None):
    """Pure Heston MC (L==1), for the exotic comparison baseline."""
    return particle_slv(params, lambda F, t: np.sqrt(params[4]) * 0 + 1.0, F0, T, n_steps, n_paths,
                        seed=seed, barrier=barrier, l_clip=(1.0, 1.0))[:2]


def _barrier_ok(F, barrier):
    kind, level = barrier
    return (F < level) if kind == "up" else (F > level)


def vanilla_smile_from_terminal(FT, F0, T, df, ks):
    """MC vanilla IV smile at log-moneyness points ks (call for k>=0, put for k<0)."""
    from .data.iv import implied_vol
    ivs = []
    for k in ks:
        K = F0 * np.exp(k); call = k >= 0
        px = df * (np.maximum(FT - K, 0).mean() if call else np.maximum(K - FT, 0).mean())
        ivs.append(float(implied_vol(np.array([px]), F0, np.array([K]), T, df=df,
                                     call=np.array([call]))[0]))
    return np.array(ivs)
