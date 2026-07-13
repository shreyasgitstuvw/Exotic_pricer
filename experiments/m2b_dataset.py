"""Assemble the Route-B (short-end gap-closer) dataset.

Residual-learning setup: for each date we take the >=14d-calibrated Heston backbone (from M1), price
the <14d options with it, and record the RESIDUAL  d = market_IV - heston_IV  at every short-dated
strike. The network will learn d as a smooth function of log-moneyness, conditioned on the same
surface features M2.1 used. Heston does the bulk; the net learns only its short-end error.

Outputs:
  data/m2b/features.parquet  — one row/date: 17 features (normalized on train) + split
  data/m2b/quotes.parquet    — one row/short-strike: date, tte_yr, log_m, market_iv, heston_iv, vega
  data/m2b/norm.npz          — feature mean/std (train), for inference

Validation: the pre-correction gap (RMSE of residuals per date) should match M1's short_gap (~165 bps).

    python experiments/m2b_dataset.py --params data/m1/params_timeseries_bhavcopy.csv --cache-dir bhavcopy
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hestonnn.config import Config
from hestonnn.data import bhavcopy as bc, surface_bhav as sb
from hestonnn.data.iv import black76_vega, implied_vol
from hestonnn.heston_ref import HParams, heston_call
from hestonnn.features import (surface_features, short_end_features,
                               FEATURE_NAMES, SHORT_FEATURE_NAMES)

CRASH_RMSE = 200.0
M2B_FEATURES = FEATURE_NAMES + SHORT_FEATURE_NAMES   # long-end + observed short-end smile
# NOTE: India VIX was evaluated as a feature but REJECTED — the series ends 2025-11-11, so the 2026
# test split has zero coverage and VIX adds no test-time signal. `load_vix_eod` kept for future use
# once the VIX series is extended past the test period.


def load_vix_eod(path):
    """India VIX end-of-day close per date -> {date: vix}. (Not currently used; see note above.)"""
    try:
        v = pd.read_csv(path)
    except Exception:
        return {}
    v["d"] = pd.to_datetime(v["date"], format="%d/%m/%y", errors="coerce").dt.date
    v["hm"] = v["time"].astype(str).str.slice(0, 5)
    eod = v[v["hm"] <= "15:30"].groupby("d")["close"].last()
    return {d: float(x) for d, x in eod.items()}


def build(params_csv, cache_dir, config_path, min_oi, n_sd, min_tte_days, val_start, test_start,
          limit=None):
    cfg = Config.load(config_path)
    P = pd.read_csv(params_csv, parse_dates=["date"]).sort_values("date")
    P = P[P["rmse_bps"] <= CRASH_RMSE].reset_index(drop=True)
    if limit:
        P = P.iloc[:limit]
    floor = min_tte_days / 365.0
    feat_rows, quote_rows = [], []
    for _, r in P.iterrows():
        f = Path(cache_dir) / f"bhavcopy_FO_{pd.to_datetime(r['date']):%Y%m%d}.csv.zip"
        if not f.exists():
            continue
        try:
            day = bc.load_bhavcopy_day(f)
            surf, _ = sb.build_surface(day, bc.usable_expiries(day, min_oi), cfg.surf)
            clean = sb.calibration_quotes(surf, n_sd=max(n_sd, 3.0))   # wider band to keep short skew
        except Exception:
            continue
        long_ = clean[clean["tte_yr"] >= floor]        # backbone side (features)
        short = clean[(clean["tte_yr"] < floor) & (clean["tte_yr"] > 1.5 / 365)]  # target side, skip 0-1DTE
        if long_["expiry"].nunique() < 3 or len(short) < 5:
            continue
        feat_long = surface_features(long_)
        if not np.isfinite(list(feat_long.values())).all():
            continue
        feat = {**feat_long, **short_end_features(short)}   # add observed short-end smile
        p = HParams(r["kappa"], r["theta"], r["sigma_v"], r["rho"], r["v0"])
        # Heston backbone IV at each short strike (extrapolate the >=14d fit into <14d)
        for e, s in short.groupby("expiry"):
            F, dfc, T = float(s["F"].iloc[0]), float(s["df"].iloc[0]), float(s["tte_yr"].iloc[0])
            K = s["K"].to_numpy(float)
            hiv = implied_vol(heston_call(F, K, T, p, dfc), F, K, T, df=dfc, call=np.ones(len(K), bool))
            miv = s["iv"].to_numpy(float)
            vega = black76_vega(F, K, T, miv, dfc)
            for k in range(len(K)):
                if np.isfinite(hiv[k]) and np.isfinite(miv[k]):
                    quote_rows.append(dict(date=r["date"], tte_yr=T, F=F, df=dfc, K=float(K[k]),
                                           log_m=float(s["log_m"].iloc[k]),
                                           market_iv=miv[k], heston_iv=hiv[k], vega=float(vega[k])))
        feat_rows.append({"date": r["date"], **feat})

    F = pd.DataFrame(feat_rows)
    Q = pd.DataFrame(quote_rows)
    Q = Q[Q["date"].isin(F["date"])]                    # keep quotes whose date has features
    # time split
    def split_of(d):
        d = pd.to_datetime(d)
        return "test" if d >= pd.to_datetime(test_start) else ("val" if d >= pd.to_datetime(val_start) else "train")
    F["split"] = F["date"].map(split_of); Q["split"] = Q["date"].map(split_of)
    return F, Q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default="data/m1/params_timeseries_bhavcopy.csv")
    ap.add_argument("--cache-dir", default="bhavcopy")
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--out-dir", default="data/m2b")
    ap.add_argument("--min-oi-strikes", type=int, default=7)
    ap.add_argument("--n-sd", type=float, default=1.5)
    ap.add_argument("--min-tte-days", type=float, default=14)
    ap.add_argument("--val-start", default="2025-01-01")
    ap.add_argument("--test-start", default="2026-01-01")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    F, Q = build(args.params, args.cache_dir, args.config, args.min_oi_strikes, args.n_sd,
                 args.min_tte_days, args.val_start, args.test_start, args.limit)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    # impute any residual NaN (e.g. a missing short bucket) with TRAIN median, then normalize on TRAIN
    tr = F[F.split == "train"]
    med = tr[M2B_FEATURES].median()
    F[M2B_FEATURES] = F[M2B_FEATURES].fillna(med)
    tr = F[F.split == "train"]
    mean = tr[M2B_FEATURES].mean().to_numpy(); std = tr[M2B_FEATURES].std().to_numpy(); std[std < 1e-8] = 1
    F[M2B_FEATURES] = (F[M2B_FEATURES].to_numpy() - mean) / std
    F.to_parquet(out / "features.parquet"); Q.to_parquet(out / "quotes.parquet")
    np.savez(out / "norm.npz", mean=mean, std=std, feat_names=np.array(M2B_FEATURES))

    # validation: pre-correction gap = residual RMSE per date (should ~match M1 short_gap ~165 bps)
    Q["resid_bps"] = (Q["market_iv"] - Q["heston_iv"]) * 1e4
    per_date = Q.groupby("date").apply(lambda g: np.sqrt(np.mean(g["resid_bps"] ** 2)), include_groups=False)
    print(f"dates: {len(F)} (train {sum(F.split=='train')} / val {sum(F.split=='val')} / test {sum(F.split=='test')})")
    print(f"short-end quotes: {len(Q)} | median {Q.groupby('date').size().median():.0f}/date")
    print(f"PRE-correction gap (residual RMSE): median {per_date.median():.0f} bps  (M1 short_gap ~165)")
    print(f"saved -> {out}/features.parquet, quotes.parquet, norm.npz")


if __name__ == "__main__":
    main()
