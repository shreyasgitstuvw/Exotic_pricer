"""Assemble the M2.1 surrogate dataset: features X, targets y, weights, time-based splits.

For each date in the M1 params series we rebuild the SAME >=14d arb-clean surface the DE calibrator
saw (so features and targets are consistent), extract the fixed feature vector, and pair it with the
calibrated (kappa, sigma_v, rho). Quality filtering drops the COVID-crash outliers and weights the
rest by 1/RMSE. Standardization stats are fit on TRAIN ONLY (no leakage), then applied to all splits.

Output: data/m2/dataset.npz  (Xtr,ytr,wtr, Xva,yva,wva, Xte,yte,wte, feat_mean, feat_std, ...).

    python experiments/m2_dataset.py --config configs/data.yaml --cache-dir bhavcopy \
        --params data/m1/params_timeseries_bhavcopy.csv --out-dir data/m2
"""
from __future__ import annotations
import argparse, glob, sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hestonnn.config import Config
from hestonnn.data import bhavcopy as bc, surface_bhav as sb
from hestonnn.features import surface_features, FEATURE_NAMES

TARGETS = ["kappa", "sigma_v", "rho"]      # D2: the NN predicts these three
CRASH_RMSE = 200.0                          # above this = unfittable (COVID crash), excluded


def _date_to_file(cache_dir, d):
    return Path(cache_dir) / f"bhavcopy_FO_{pd.to_datetime(d):%Y%m%d}.csv.zip"


def build(params_csv, cache_dir, config_path, min_oi, n_sd, min_tte_days, limit=None):
    cfg = Config.load(config_path)
    P = pd.read_csv(params_csv, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    P = P[P["rmse_bps"] <= CRASH_RMSE]                    # drop crash outliers
    if limit:
        P = P.iloc[:limit]
    floor = min_tte_days / 365.0
    rows, meta = [], []
    for _, r in P.iterrows():
        f = _date_to_file(cache_dir, r["date"])
        if not f.exists():
            continue
        try:
            day = bc.load_bhavcopy_day(f)
            surf, _ = sb.build_surface(day, bc.usable_expiries(day, min_oi), cfg.surf)
            clean = sb.calibration_quotes(surf, n_sd=n_sd)
            clean = clean[clean["tte_yr"] >= floor]       # same >=14d set the params were fit to
            if clean["expiry"].nunique() < 3:
                continue
            feat = surface_features(clean)
        except Exception:
            continue
        rows.append({**feat, **{t: r[t] for t in TARGETS},
                     "date": r["date"], "rmse_bps": r["rmse_bps"]})
    D = pd.DataFrame(rows).dropna(subset=FEATURE_NAMES + TARGETS).reset_index(drop=True)
    return D


def split_and_pack(D, val_start, test_start, out_dir):
    D = D.sort_values("date").reset_index(drop=True)
    d = pd.to_datetime(D["date"])
    tr = D[d < val_start]; va = D[(d >= val_start) & (d < test_start)]; te = D[d >= test_start]

    def XyW(part):
        X = part[FEATURE_NAMES].to_numpy(float)
        y = part[TARGETS].to_numpy(float)
        w = 1.0 / np.clip(part["rmse_bps"].to_numpy(float), 20, None)   # down-weight poor fits
        w = w / w.mean() if len(w) else w
        return X, y, w

    Xtr, ytr, wtr = XyW(tr)
    # standardization stats from TRAIN ONLY (no leakage)
    mean = Xtr.mean(0); std = Xtr.std(0); std[std < 1e-8] = 1.0
    norm = lambda X: (X - mean) / std
    Xva, yva, wva = XyW(va); Xte, yte, wte = XyW(te)

    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "dataset.npz",
             Xtr=norm(Xtr), ytr=ytr, wtr=wtr,
             Xva=norm(Xva) if len(Xva) else Xva, yva=yva, wva=wva,
             Xte=norm(Xte) if len(Xte) else Xte, yte=yte, wte=wte,
             feat_mean=mean, feat_std=std,
             feat_names=np.array(FEATURE_NAMES), target_names=np.array(TARGETS),
             dates_tr=tr["date"].astype(str).to_numpy(),
             dates_va=va["date"].astype(str).to_numpy(),
             dates_te=te["date"].astype(str).to_numpy())
    return len(tr), len(va), len(te), out / "dataset.npz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--cache-dir", default="bhavcopy")
    ap.add_argument("--params", default="data/m1/params_timeseries_bhavcopy.csv")
    ap.add_argument("--out-dir", default="data/m2")
    ap.add_argument("--min-oi-strikes", type=int, default=7)
    ap.add_argument("--n-sd", type=float, default=1.5)
    ap.add_argument("--min-tte-days", type=float, default=14)
    ap.add_argument("--val-start", default="2025-01-01")
    ap.add_argument("--test-start", default="2026-01-01")
    ap.add_argument("--limit", type=int, default=None, help="debug: cap #dates")
    args = ap.parse_args()

    D = build(args.params, args.cache_dir, args.config, args.min_oi_strikes, args.n_sd,
              args.min_tte_days, args.limit)
    D.to_csv(Path(args.out_dir).parent / "m2" / "dataset_table.csv", index=False) if False else None
    ntr, nva, nte, path = split_and_pack(D, args.val_start, args.test_start, args.out_dir)
    print(f"assembled {len(D)} usable dates -> train {ntr} / val {nva} / test {nte}")
    print(f"features {len(FEATURE_NAMES)} | targets {TARGETS}")
    print(f"saved -> {path}")


if __name__ == "__main__":
    main()
