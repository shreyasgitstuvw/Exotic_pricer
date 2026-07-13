"""Reference Heston (1993) pricer — NumPy, semi-analytical.

STATUS: P2 reference implementation. The FROZEN engine (engine.py) remains ground truth; this exists
to (a) unblock M1 calibration now and (b) be cross-checked against the engine once pinned — they must
agree to ~1e-4 in price. Do not treat this as the deliverable pricer.

Method: Heston characteristic function in the stable "little trap" form (Albrecher et al.), integrated
with Lewis's (2000) single-integral call formula on the forward, evaluated on a fixed Gauss-Legendre
grid and vectorized across strikes. Prices European options on the forward F (martingale measure), so
put-call parity and discounting are exact by construction.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class HParams:
    kappa: float
    theta: float
    sigma_v: float
    rho: float
    v0: float

    def feller_ok(self) -> bool:
        return 2 * self.kappa * self.theta >= self.sigma_v ** 2


def cf_log_forward(u, p: HParams, T):
    """Characteristic function of ln(S_T/F) under Heston (drift-free, 'trap' form). u may be complex array."""
    u = np.asarray(u, complex)
    a = p.kappa * p.theta
    sig2 = p.sigma_v ** 2
    beta = p.kappa - p.rho * p.sigma_v * 1j * u
    d = np.sqrt(beta ** 2 + sig2 * (1j * u + u ** 2))
    # g2 = (beta - d)/(beta + d)  -> stable "trap" branch
    g = (beta - d) / (beta + d)
    edt = np.exp(-d * T)
    D = ((beta - d) / sig2) * (1.0 - edt) / (1.0 - g * edt)
    C = (a / sig2) * ((beta - d) * T - 2.0 * np.log((1.0 - g * edt) / (1.0 - g)))
    return np.exp(C + D * p.v0)


# Fixed trapezoid grid on (0, umax). A dense uniform grid resolves the integrand's peak near u=0
# (which sparse Gauss-Legendre misses); N=2000, umax=200 gives <5e-4 price error, ~0.5 bps IV.
_N_QUAD, _UMAX = 2000, 200.0
_U = np.linspace(1e-8, _UMAX, _N_QUAD)
_W = np.full(_N_QUAD, _U[1] - _U[0]); _W[0] *= 0.5; _W[-1] *= 0.5


def heston_call(F, K, T, p: HParams, df=1.0):
    """European call price(s) on forward F. K may be array. Lewis single-integral form."""
    K = np.atleast_1d(np.asarray(K, float))
    F = float(F); T = float(T); df = float(df)
    k = np.log(F / K)                              # log-moneyness, shape (nK,)
    u, w = _U, _W                                   # (nu,)
    phi = cf_log_forward(u - 0.5j, p, T)           # CF at u - i/2, shape (nu,)
    # integrand for each strike: Re[ e^{i u k} phi ] / (u^2 + 1/4)
    ex = np.exp(1j * np.outer(k, u))               # (nK, nu)
    integ = (ex * phi[None, :]).real / (u[None, :] ** 2 + 0.25)
    integral = integ @ w                           # (nK,)
    call = df * (F - np.sqrt(F * K) / np.pi * integral)
    # clip to no-arb bounds to kill quadrature noise at extreme strikes
    lo = df * np.maximum(F - K, 0.0); hi = df * F
    return np.clip(call, lo, hi)


def heston_put(F, K, T, p: HParams, df=1.0):
    c = heston_call(F, K, T, p, df)
    return c - df * (F - np.asarray(K, float))     # put-call parity
