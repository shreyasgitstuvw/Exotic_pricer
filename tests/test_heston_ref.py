import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hestonnn.heston_ref import HParams, heston_call, heston_put
from hestonnn.data.iv import black76_price

F, T, df = 24000.0, 0.5, 0.98


def test_collapses_to_black76_when_volofvol_zero():
    sig = 0.2
    p = HParams(2.0, sig**2, 1e-4, 0.0, sig**2)
    K = np.array([21000, 24000, 27000.0])
    hc = heston_call(F, K, T, p, df)
    bs = np.array([black76_price(F, k, T, sig, df, True) for k in K])
    assert np.max(np.abs(hc - bs)) < 5e-3            # ~sub-bp IV


def test_put_call_parity_exact():
    p = HParams(2.0, 0.04, 0.5, -0.7, 0.045)
    K = np.array([20000, 24000, 28000.0])
    c = heston_call(F, K, T, p, df); pu = heston_put(F, K, T, p, df)
    assert np.max(np.abs((c - pu) - df * (F - K))) < 1e-8


def test_monotone_and_convex_in_strike():
    p = HParams(2.0, 0.04, 0.5, -0.7, 0.045)
    K = np.linspace(20000, 28000, 9)
    c = heston_call(F, K, T, p, df)
    assert np.all(np.diff(c) < 0)
    assert np.all(np.diff(c, 2) > -1e-6)


def test_negative_rho_makes_downside_skew():
    K = np.array([22000.0, 26000.0])
    down = heston_call(F, K, T, HParams(2, 0.04, 0.6, -0.8, 0.04), df)
    flat = heston_call(F, K, T, HParams(2, 0.04, 0.6,  0.0, 0.04), df)
    # negative rho lifts OTM puts (low strike calls via parity) relative to symmetric
    assert down[0] - flat[0] > 0
