"""Oracle diagnostic for Route B: how much gap can each correction FORM remove, at best?

For each date we fit the residual (market_IV - heston_IV) with the best-possible correction and
measure the gap reduction. This is the *ceiling* for a given form (in-sample, no network), so it
isolates 'is the form expressive enough' from 'can the net predict it'. Compare to the trained net.

  single-quad   : one d0+d1x+d2x^2 for the whole date (our current model's form)  -> its ceiling
  per-tenor quad: separate quadratic per short tenor                              -> value of tenor-dep
  per-tenor cubic: separate cubic per short tenor                                 -> value of higher order

    python experiments/oracle_gap.py --data-dir data/m2b
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd


def _resid_after(x, r, deg):
    """Residual after subtracting the best-fit degree-`deg` polynomial (in-sample)."""
    if len(x) > deg:
        return r - np.polyval(np.polyfit(x, r, deg), x)
    return r


def _reduction(Q, mode):
    reds = []
    for d, g in Q.groupby("date"):
        resid = (g["market_iv"] - g["heston_iv"]).to_numpy()
        gap_before = np.sqrt(np.mean(resid ** 2))
        if mode == "single_quad":                         # one quadratic for the whole date
            after = _resid_after(g["log_m"].to_numpy(), resid, 2)
        else:                                             # per-tenor quadratic or cubic
            deg = 3 if "cubic" in mode else 2
            parts = [_resid_after(gt["log_m"].to_numpy(),
                                  (gt["market_iv"] - gt["heston_iv"]).to_numpy(), deg)
                     for _, gt in g.groupby("tte_yr")]
            after = np.concatenate(parts)
        gap_after = np.sqrt(np.mean(after ** 2))
        reds.append(1 - gap_after / gap_before)
    return np.array(reds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/m2b")
    ap.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    args = ap.parse_args()
    Q = pd.read_parquet(Path(args.data_dir) / "quotes.parquet")
    if args.split != "all" and "split" in Q.columns:
        Q = Q[Q["split"] == args.split]
    Q = Q.reset_index(drop=True)
    print(f"oracle on split='{args.split}': {Q['date'].nunique()} dates, {len(Q)} quotes\n")
    print(f"{'form':16}{'median reduction':>18}{'mean':>8}")
    for mode in ["single_quad", "per_tenor_quad", "per_tenor_cubic"]:
        r = _reduction(Q, mode)
        print(f"{mode:16}{np.median(r)*100:16.0f}%{r.mean()*100:7.0f}%")
    print("\nread: if single_quad ~ the net's 16% -> form is the limit -> add tenor-dependence.")
    print("      if single_quad >> 16% -> net can't predict from features -> add short-end features.")
    print("      if even per_tenor_cubic is low -> quadratic family too weak -> grid / LSV.")


if __name__ == "__main__":
    main()
