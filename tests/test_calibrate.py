import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hestonnn.heston_ref import HParams, heston_call
from hestonnn.data.iv import implied_vol
from hestonnn import calibrate as cal


def _synth(true):
    rows = []
    for T, F in [(0.1, 24000), (0.5, 24200), (1.0, 24500)]:
        df = np.exp(-0.06 * T)
        Ks = np.round(np.linspace(0.9, 1.1, 9) * F / 50) * 50
        c = heston_call(F, Ks, T, true, df)
        iv = implied_vol(c, F, Ks, T, df=df, call=(Ks >= F))
        for k, cc, v in zip(Ks, c, iv):
            if np.isfinite(v):
                rows.append(dict(expiry=f"T{T}", tte_yr=T, F=F, df=df, K=k, call_px=cc, iv=v))
    return pd.DataFrame(rows)


def test_roundtrip_recovers_well_identified_params():
    true = HParams(2.5, 0.045, 0.55, -0.65, 0.035)
    surf = _synth(true)
    p, st = cal.calibrate(surf, de_maxiter=15, popsize=12, seed=1)
    assert st["iv_rmse_bps"] < 15                     # tight fit on self-generated data
    # rho, v0, theta are well-identified; kappa is the known sloppy direction (not asserted tightly)
    assert abs(p.rho - true.rho) < 0.08
    assert abs(p.v0 - true.v0) < 0.010
    assert abs(p.theta - true.theta) < 0.010
