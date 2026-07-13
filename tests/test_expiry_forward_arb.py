import sys
from pathlib import Path
from datetime import date
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hestonnn.data import expiry as expmod
from hestonnn.data.forward import implied_forward_df
from hestonnn.data.iv import black76_price
from hestonnn.data.arbitrage import flag_smile


def test_expiry_detection_and_shift():
    # synthetic time values: collapse on day idx 3 (a "Thu") and idx 7 (shifted "Wed")
    days = [date(2021, 2, 8 + i) for i in range(9)]
    tv = pd.DataFrame({"dt": days,
                       "atm_time_value": [200, 150, 90, 3, 210, 120, 40, 2, 190]})
    emap = expmod.detect(tv, threshold=15.0)
    assert emap.expiries == [date(2021, 2, 11), date(2021, 2, 15)]
    # a day before the first expiry maps to it with positive tte
    assert emap.front_expiry(date(2021, 2, 8)) == date(2021, 2, 11)
    assert emap.tte(date(2021, 2, 8)) > 0


def test_forward_recovery_from_pcp():
    F_true, df_true, T = 20000.0, 0.995, 0.05
    Ks = np.arange(19000, 21000, 100.0)
    C = np.array([black76_price(F_true, k, T, 0.2, df_true, True) for k in Ks])
    P = np.array([black76_price(F_true, k, T, 0.2, df_true, False) for k in Ks])
    F, df, r, n = implied_forward_df(Ks, C, P, T)
    assert abs(F - F_true) < 1e-3
    assert abs(df - df_true) < 1e-4


def test_butterfly_flag_catches_concavity():
    # convex (arb-free) call curve vs a dinted one
    K = np.array([100, 110, 120, 130, 140.0])
    C_ok = np.array([42.0, 33.0, 25.0, 18.0, 12.0])       # decreasing & convex
    d_ok = flag_smile(pd.DataFrame({"K": K, "call_px": C_ok, "df": 1.0}))
    assert not d_ok["arb_any"].any()
    C_bad = C_ok.copy(); C_bad[2] = 8.0                   # dip -> concave kink
    d_bad = flag_smile(pd.DataFrame({"K": K, "call_px": C_bad, "df": 1.0}))
    assert d_bad["arb_any"].any()
