import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hestonnn.data.iv import black76_price, implied_vol_one, black76_vega


def test_price_iv_roundtrip():
    F, T, df = 20000.0, 20 / 365, 0.99
    for K in [18000, 19000, 20000, 21000, 22000]:
        for sig in [0.08, 0.15, 0.30, 0.60]:
            for call in (True, False):
                px = float(black76_price(F, K, T, sig, df, call))
                iv = implied_vol_one(px, F, K, T, df, call)
                assert abs(iv - sig) < 1e-4, (K, sig, call, iv)


def test_put_call_parity():
    F, K, T, df, sig = 20000.0, 20500.0, 0.05, 0.98, 0.2
    c = float(black76_price(F, K, T, sig, df, True))
    p = float(black76_price(F, K, T, sig, df, False))
    assert abs((c - p) - df * (F - K)) < 1e-6


def test_below_intrinsic_returns_nan():
    F, K, T, df = 20000.0, 19000.0, 0.05, 0.99
    intrinsic = df * (F - K)
    assert np.isnan(implied_vol_one(intrinsic * 0.9, F, K, T, df, True))


def test_vega_positive():
    v = black76_vega(20000, 20000, 0.05, 0.2, 0.99)
    assert v > 0
