"""M3 (v2) — synthetic short-dated variance swap / VIX: another TERMINAL-distribution exotic priced
by static replication over the smile (CBOE-VIX style), so it needs no dynamics — the corrected smile
suffices, exactly like the digitals. Heston's shallow short-end smile biases the fair variance vs the
market-consistent (corrected) smile. This is the same short-end error, expressed as a vol-index bias.

Fair variance (discrete VIX replication):
    sigma^2 = (2/T) sum_i (dK_i / K_i^2) e^{rT} Q(K_i)  -  (1/T) (F/K0 - 1)^2
with Q = OTM option price (put for K<F, call for K>F). Reported as sqrt -> annualized vol points.

    python experiments/m3_varswap.py --data-dir data/m2b
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hestonnn.data.iv import black76_price


def fair_vol(F, T, df, K, iv):
    """CBOE-style fair variance from a smile (K ascending), returned as annualized vol (sqrt)."""
    K = np.asarray(K, float); iv = np.asarray(iv, float)
    r = -np.log(df) / T if df > 0 and T > 0 else 0.0
    Q = np.where(K >= F, black76_price(F, K, T, iv, df, True),
                 black76_price(F, K, T, iv, df, False))          # OTM option prices
    dK = np.gradient(K)
    K0 = K[K <= F].max() if (K <= F).any() else K.min()          # first strike below forward
    var = (2.0 / T) * np.sum(dK / K ** 2 * np.exp(r * T) * Q) - (1.0 / T) * (F / K0 - 1.0) ** 2
    return np.sqrt(max(var, 1e-8))


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
            if len(gt) < 5:
                continue
            F = float(gt["F"].iloc[0]); dfc = float(gt["df"].iloc[0]); T = float(T)
            K = gt["K"].to_numpy()
            vh = fair_vol(F, T, dfc, K, gt["heston_iv"].to_numpy())
            vc = fair_vol(F, T, dfc, K, gt["market_iv"].to_numpy())     # corrected ~ market
            rows.append(dict(date=d, tte_d=T * 365, vix_heston=vh, vix_corrected=vc,
                             miss_volpts=(vh - vc) * 100))
    R = pd.DataFrame(rows)
    print(f"M3 variance-swap / synthetic VIX — {args.split} split, {len(R)} tenor-dates\n")
    print(f"  fair vol (corrected) median : {R['vix_corrected'].median()*100:.1f}%")
    print(f"  Heston bias (vh - vc)       : median {R['miss_volpts'].median():+.2f} vol pts | "
          f"mean {R['miss_volpts'].mean():+.2f} | p90 |{R['miss_volpts'].abs().quantile(.9):.2f}|")
    print(f"  Heston bias as %% of level   : {(R['miss_volpts']/(R['vix_corrected']*100)).abs().median()*100:.1f}%")
    print("\nread: static Heston's short-end smile systematically mis-states the short-dated fair "
          "variance (the tradeable vol level) vs the market-consistent smile.")


if __name__ == "__main__":
    main()
