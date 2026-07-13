"""Full diagnosis of the Route-B (gap-closer) pipeline. No guessing — measure each layer.

Sections:
  1. DATA     — residual scale/shape, per-date structure, NaNs, moneyness range.
  2. FORM     — oracle ceilings (best quadratic per date / per tenor) reconfirmed.
  3. NEEDED   — what curvature (d2) the residual actually demands, in RAW x and SCALED z.
  4. COND     — minimal reproduction: fit ONE date's residual by gradient descent in raw-x vs
                scaled-z. If raw-x stalls and scaled-z converges, the basis conditioning is the bug.
  5. MODEL    — load the trained net, dump the (d0,d1,d2) it actually outputs vs what's NEEDED.

    python experiments/diagnose_m2b.py --data-dir data/m2b
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

X_SCALE = 0.1   # typical |log-moneyness|; the proposed rescale z = x / X_SCALE


def section(t): print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/m2b")
    args = ap.parse_args()
    Q = pd.read_parquet(Path(args.data_dir) / "quotes.parquet")
    F = pd.read_parquet(Path(args.data_dir) / "features.parquet")
    Q["resid"] = Q["market_iv"] - Q["heston_iv"]

    section("1. DATA")
    print(f"quotes {len(Q)} | dates {Q['date'].nunique()} | median quotes/date {Q.groupby('date').size().median():.0f}")
    print(f"log_m range: {Q['log_m'].min():.3f} .. {Q['log_m'].max():.3f}  (|x| p90 = {Q['log_m'].abs().quantile(.9):.3f})")
    print(f"residual (market-heston) vol-pts: mean {Q['resid'].mean()*100:+.2f}  std {Q['resid'].std()*100:.2f}")
    print(f"feature NaNs: {int(F.drop(columns=['date','split']).isna().sum().sum())} | quote NaNs: {int(Q[['market_iv','heston_iv','log_m']].isna().sum().sum())}")
    Q["mny"] = pd.cut(Q["log_m"], [-1, -0.05, -0.02, 0.02, 0.05, 1], labels=["deep put", "put", "ATM", "call", "deep call"])
    print("residual by moneyness (vol pts):")
    print((Q.groupby("mny", observed=True)["resid"].mean() * 100).round(2).to_string())

    section("2. FORM — oracle ceilings (test split)")
    Qt = Q[Q["split"] == "test"] if "split" in Q else Q
    for mode, deg, per_tenor in [("single_quad", 2, False), ("per_tenor_quad", 2, True), ("per_tenor_cubic", 3, True)]:
        reds = []
        for d, g in Qt.groupby("date"):
            r0 = g["resid"].to_numpy(); gb = np.sqrt(np.mean(r0 ** 2))
            if per_tenor:
                after = np.concatenate([_ra(gt["log_m"].to_numpy(), gt["resid"].to_numpy(), deg)
                                        for _, gt in g.groupby("tte_yr")])
            else:
                after = _ra(g["log_m"].to_numpy(), r0, deg)
            reds.append(1 - np.sqrt(np.mean(after ** 2)) / gb)
        print(f"  {mode:16} median {np.median(reds)*100:.0f}%")

    section("3. NEEDED curvature d2 (per tenor, best quadratic)")
    d2_raw, d2_scaled = [], []
    for d, g in Q.groupby("date"):
        for _, gt in g.groupby("tte_yr"):
            x = gt["log_m"].to_numpy(); r = gt["resid"].to_numpy()
            if len(x) > 2:
                d2_raw.append(np.polyfit(x, r, 2)[0])
                d2_scaled.append(np.polyfit(x / X_SCALE, r, 2)[0])
    d2_raw, d2_scaled = np.array(d2_raw), np.array(d2_scaled)
    print(f"  needed d2 in RAW x   : median {np.median(np.abs(d2_raw)):.2f}  (coeff must reach this)")
    print(f"  needed d2 in SCALED z: median {np.median(np.abs(d2_scaled)):.4f}  (z=x/{X_SCALE})")
    print(f"  -> raw-x needs d2 ~{np.median(np.abs(d2_raw)):.0f}; but x^2~{X_SCALE**2:.3f}, so its gradient is ~{X_SCALE**2:.2f}x d0's")

    section("4. CONDITIONING — gradient-descent fit of ONE date, raw-x vs scaled-z")
    g = Q[Q["date"] == Q["date"].unique()[len(Q['date'].unique()) // 2]]
    x = g["log_m"].to_numpy(); r = g["resid"].to_numpy()
    for label, xx in [("raw-x  ", x), ("scaled-z", x / X_SCALE)]:
        c = np.zeros(3); lr = 0.01
        for _ in range(5000):
            pred = c[0] + c[1] * xx + c[2] * xx ** 2
            grad = np.array([np.mean(2 * (pred - r)),
                             np.mean(2 * (pred - r) * xx),
                             np.mean(2 * (pred - r) * xx ** 2)])
            c -= lr * grad
        rmse = np.sqrt(np.mean((r - (c[0] + c[1] * xx + c[2] * xx ** 2)) ** 2))
        red = 1 - rmse / np.sqrt(np.mean(r ** 2))
        print(f"  {label}: after 5000 GD steps  reduction {red*100:5.0f}%   d2={c[2]:.3f}")
    print("  (identical LR/steps; if raw-x lags scaled-z, conditioning is confirmed the bug)")

    section("5. MODEL — what the trained net actually outputs")
    try:
        import torch
        from hestonnn.gap_closer import GapCloser, bucket_of
        feat_cols = [c for c in F.columns if c not in ("date", "split")]
        m = GapCloser(len(feat_cols)); m.load_state_dict(torch.load(Path(args.data_dir) / "gap_closer.pt")); m.eval()
        X = torch.tensor(F[feat_cols].to_numpy(float), dtype=torch.float32)
        with torch.no_grad():
            co = m(X).numpy()   # (N, 2, 3)
        print(f"  net output d2 (curvature): bucket0 median {np.median(co[:,0,2]):.2f}, bucket1 median {np.median(co[:,1,2]):.2f}")
        print(f"  net output d1 (skew)     : bucket0 median {np.median(co[:,0,1]):+.2f}")
        print(f"  net output d0 (level)    : bucket0 median {np.median(co[:,0,0]):+.3f}")
        print(f"  --> NEEDED d2 ~{np.median(np.abs(d2_raw)):.0f}. If net d2 << that, curvature never trained (conditioning).")
    except Exception as e:
        print(f"  (skipped model load: {e})")


def _ra(x, r, deg):
    return r - np.polyval(np.polyfit(x, r, deg), x) if len(x) > deg else r


if __name__ == "__main__":
    main()
