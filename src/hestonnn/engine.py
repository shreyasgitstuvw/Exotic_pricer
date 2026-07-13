"""Adapter seam to the FROZEN Heston engine (github.com/shreyasgitstuvw/Heston-engine).

Design rule (DECISIONS D-note): P2 never edits the engine. It is a READ-ONLY dependency, pinned to a
commit SHA in pyproject.toml:
    "heston-engine @ git+https://github.com/shreyasgitstuvw/Heston-engine@<PINNED_SHA>"

This module is the ONLY place P2 touches it, so the surface between the two repos is explicit and
swappable. Everything downstream (calibration harness, M2 surrogate targets) calls these adapters,
never the engine directly.

Expected engine interface (map to the real function names when you pin it):
    price_surface(params, forwards, strikes, ttes, df) -> IV or price grid   # CF + Carr-Madan FFT
    calibrate(surface, weights) -> HestonParams (kappa, theta, sigma_v, rho, v0)  # DE -> L-BFGS-B
Until the engine is installed, `require_engine()` raises with the pin instructions so nothing
silently no-ops.
"""
from __future__ import annotations
from dataclasses import dataclass

try:                                   # the pinned engine package, if installed
    import heston_engine as _engine    # noqa: F401
    HAVE_ENGINE = True
except Exception:
    _engine = None
    HAVE_ENGINE = False


@dataclass
class HestonParams:
    kappa: float
    theta: float
    sigma_v: float
    rho: float
    v0: float

    def feller_ok(self) -> bool:
        return 2 * self.kappa * self.theta >= self.sigma_v ** 2


_PIN_MSG = (
    "Frozen Heston engine not importable. Pin & install it (never edit it):\n"
    '  pip install "heston-engine @ git+https://github.com/shreyasgitstuvw/'
    'Heston-engine@<PINNED_SHA>"\n'
    "Then map price_surface()/calibrate() below to the engine's actual entry points."
)


def require_engine():
    if not HAVE_ENGINE:
        raise ImportError(_PIN_MSG)
    return _engine


def calibrate(surface, weights=None) -> HestonParams:   # pragma: no cover - needs pinned engine
    """Calibrate frozen Heston to one assembled surface. Wire to the engine's DE->L-BFGS-B routine."""
    eng = require_engine()
    raise NotImplementedError(
        "Map to the engine's calibration entry point once pinned. Surface schema is the output of "
        "data/surface_bhav.build_surface (cols: K, m, iv, tte_yr, F, df, expiry)."
    )


def price_surface(params: HestonParams, forwards, strikes, ttes, df=1.0):  # pragma: no cover
    """Reprice a maturity x moneyness grid under given params via the engine's CF/FFT pricer."""
    eng = require_engine()
    raise NotImplementedError("Map to the engine's CF + Carr-Madan pricer once pinned.")
