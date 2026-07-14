"""Deployable short-end smoother: fit the OBSERVED residual (market_IV - heston_IV) per tenor with a
quadratic, then project to no-arb. No learning, no train/test gap — each day is fit independently,
exactly as a desk fits today's smile. This is the right tool when the short end is observed (it is).

The decomposition (diagnose_ceiling.py) showed the neural net is feature-limited to ~60%, while the
per-tenor oracle is 87% — and that 87% is *available at deployment* because the residual is observed.
This script measures the deployable smoother (should ≈ 87% before projection, a bit less after).

    python experiments/direct_smoother.py --data-dir data/m2b
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hestonnn.gap_closer import project_arbfree
from hestonnn.data.arbitrage import flag_smile
from hestonnn.data.iv import black76_price, implied_vol


def run(Q, split, project):
    Qs = Q[Q["split"] == split] if "split" in Q.columns else Q
    reds, viol, n = [], 0, 0
    for d, g in Qs.groupby("date"):
        gb, ga, dv = [], [], False
        for T, gt in g.groupby("tte_yr"):
            gt = gt.sort_values("K")
            x = gt["log_m"].to_numpy(); r = (gt["market_iv"] - gt["heston_iv"]).to_numpy()
            hiv = gt["heston_iv"].to_numpy(); miv = gt["market_iv"].to_numpy()
            coef = np.polyfit(x, r, 2) if len(x) > 2 else (
                np.r_[0.0, np.polyfit(x, r, 1)] if len(x) == 2 else np.r_[0.0, 0.0, r.mean()])
            corr_iv = hiv + np.polyval(coef, x)
            if project:
                F = float(gt["F"].iloc[0]); dfc = float(gt["df"].iloc[0]); T = float(T)
                K = gt["K"].to_numpy()
                cpx = project_arbfree(K, black76_price(F, K, T, corr_iv, dfc, True), dfc, F)
                piv = implied_vol(cpx, F, K, T, df=dfc, call=np.ones(len(K), bool))
                corr_iv = np.where(np.isfinite(piv), piv, corr_iv)
                if flag_smile(pd.DataFrame({"K": K, "call_px": cpx, "df": dfc}))["arb_any"].any():
                    dv = True
            gb.append(miv - hiv); ga.append(miv - corr_iv)
        gb = np.concatenate(gb); ga = np.concatenate(ga)
        reds.append(1 - np.sqrt(np.mean(ga ** 2)) / np.sqrt(np.mean(gb ** 2))); viol += int(dv); n += 1
    tag = "with projection" if project else "raw fit"
    print(f"  {split:5} ({tag:15}): gap reduction median {np.median(reds)*100:.0f}%  arb {viol}/{n}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/m2b")
    args = ap.parse_args()
    Q = pd.read_parquet(Path(args.data_dir) / "quotes.parquet")
    print("Direct per-tenor smoother (fit observed residual + project). No training.")
    for split in ["train", "val", "test"]:
        run(Q, split, project=False)
        run(Q, split, project=True)
    print("\ncompare: neural net (M2.2) test = 60%; per-tenor oracle = 87%.")


if __name__ == "__main__":
    main()
