"""M3 (v1) — the exotic-pricing payoff: how much does Heston's short-end error move exotic prices?

Digital (cash-or-nothing) options are the cleanest first case: a digital call paying 1 if S_T > K is
worth exactly  -dC/dK  (the negative slope of the call-price curve) — so it is entirely determined by
the smile's SKEW at K, which is precisely what Heston gets wrong at the short end. We price digitals
under (a) the raw Heston short smile and (b) the corrected smile (Heston + fitted short-end residual,
the deployable smoother), and report the mispricing.

    python experiments/m3_exotics.py --data-dir data/m2b
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hestonnn.data.iv import black76_price

# report digitals at these moneyness points (log-strike vs forward)
TARGETS = {"5% OTM put": -0.05, "2% OTM put": -0.02, "ATM": 0.0,
           "2% OTM call": 0.02, "5% OTM call": 0.05}


def smile_quad(x, iv):
    """Quadratic IV smile fit; returns a callable sig(xq)."""
    if len(x) > 2:
        c = np.polyfit(x, iv, 2)
    elif len(x) == 2:
        c = np.r_[0.0, np.polyfit(x, iv, 1)]
    else:
        c = np.r_[0.0, 0.0, iv.mean()]
    return lambda xq: np.polyval(c, xq)


def digital_call(F, T, df, sig_fn, xq, h=1e-3):
    """Digital call (pays 1 if S_T>K) = -dC/dK, via central difference on the smile-implied call price."""
    def C(x):
        K = F * np.exp(x)
        return black76_price(F, K, T, sig_fn(x), df, True)
    Kq = F * np.exp(xq)
    dK = F * np.exp(xq) * h                      # dK for the bump in x
    return -(C(xq + h) - C(xq - h)) / (2 * dK)   # -dC/dK


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/m2b")
    ap.add_argument("--split", default="test")
    args = ap.parse_args()
    Q = pd.read_parquet(Path(args.data_dir) / "quotes.parquet")
    if args.split != "all":
        Q = Q[Q["split"] == args.split]

    rows = []
    for d, g in Q.groupby("date"):
        for T, gt in g.groupby("tte_yr"):
            gt = gt.sort_values("K")
            x = gt["log_m"].to_numpy()
            if len(x) < 4 or x.min() > -0.03 or x.max() < 0.03:
                continue
            F = float(gt["F"].iloc[0]); dfc = float(gt["df"].iloc[0]); T = float(T)
            sig_h = smile_quad(x, gt["heston_iv"].to_numpy())
            sig_c = smile_quad(x, gt["market_iv"].to_numpy())   # corrected ~ observed market
            for name, xq in TARGETS.items():
                if xq < x.min() or xq > x.max():
                    continue
                dh = digital_call(F, T, dfc, sig_h, xq)
                dc = digital_call(F, T, dfc, sig_c, xq)
                rows.append(dict(date=d, tte_d=T * 365, point=name,
                                 dig_heston=dh, dig_corrected=dc, miss=dh - dc))
    R = pd.DataFrame(rows)
    print(f"M3 digitals — {args.split} split, {R['date'].nunique()} dates, {len(R)} quotes\n")
    print("Heston vs corrected-smile digital price (payoff=1). 'miss' = Heston - corrected:")
    print(f"{'strike':14}{'dig(corrected)':>15}{'|miss| median':>16}{'|miss| p90':>13}{'as % of price':>15}")
    for name in TARGETS:
        s = R[R["point"] == name]
        if s.empty:
            continue
        med_abs = s["miss"].abs().median()
        p90 = s["miss"].abs().quantile(0.9)
        pct = (s["miss"].abs() / s["dig_corrected"].abs().clip(lower=1e-4)).median() * 100
        print(f"{name:14}{s['dig_corrected'].median():15.3f}{med_abs:16.3f}{p90:13.3f}{pct:14.0f}%")
    print("\nread: |miss| is the digital mispricing in payoff units (1 = full notional). "
          "Heston's shallow short-end skew systematically mis-prices OTM digitals.")


if __name__ == "__main__":
    main()
