"""Sampled bhavcopy -> multi-tenor surfaces + manifest (D3-A, full-surface layer).

Pipeline: date schedule -> fetch (or use cached) bhavcopy -> parse (dual-format) -> assemble a
maturity x moneyness surface per date -> write a per-(date,tenor) manifest and stacked surfaces.

Fetching only happens for dates not already cached; pass --no-fetch to run purely on cached files
(what we do in the sandbox, since NSE isn't reachable here).

Example (your machine):
  python experiments/build_bhavcopy_surfaces.py --config configs/data.yaml \
    --start 2024-08-01 --end 2026-07-10 --weekly-weekday 2 --cache-dir bhavcopy --out-dir data/bhav
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hestonnn.config import Config
from hestonnn.data import bhavcopy as bc, surface_bhav as sb, sampling
from hestonnn.data.fetch_bhavcopy import fetch_one, cache_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--start"); ap.add_argument("--end")
    ap.add_argument("--dates", nargs="*", help="explicit YYYY-MM-DD dates (overrides schedule)")
    ap.add_argument("--weekly-weekday", type=int, default=2)
    ap.add_argument("--monthly-day", type=int, default=None)
    ap.add_argument("--symbol", default="NIFTY")
    ap.add_argument("--cache-dir", default="bhavcopy")
    ap.add_argument("--out-dir", default="data/bhav")
    ap.add_argument("--no-fetch", action="store_true", help="use only cached files (no network)")
    ap.add_argument("--min-oi-strikes", type=int, default=7)
    args = ap.parse_args()

    cfg = Config.load(args.config)
    if args.dates:
        dates = [pd.to_datetime(d).date() for d in args.dates]
    else:
        dates = sampling.schedule(args.start, args.end, args.weekly_weekday, args.monthly_day)
    print(f"[bhav] {len(dates)} scheduled dates")

    out_dir = Path(args.out_dir); (out_dir / "surfaces").mkdir(parents=True, exist_ok=True)
    tenor_rows, surfaces = [], []
    for d in dates:
        p = cache_path(d, args.cache_dir)
        if not p.exists() and not args.no_fetch:
            try:
                p = fetch_one(d, cache_dir=args.cache_dir)
            except Exception as e:
                print(f"  {d}  fetch-failed: {e}"); continue
        if not p or not Path(p).exists():
            continue
        try:
            day = bc.load_bhavcopy_day(p, symbol=args.symbol)
            exps = bc.usable_expiries(day, args.min_oi_strikes)
            surf, tenor = sb.build_surface(day, exps, cfg.surf)
        except Exception as e:
            print(f"  {d}  parse/build-failed: {e}"); continue
        tenor["date"] = d
        tenor_rows.append(tenor)
        if len(surf):
            surf["date"] = d
            surfaces.append(surf)
        ok = tenor["ok"].sum()
        print(f"  {d}  tenors_ok={ok}  maturity<= {tenor.loc[tenor.ok,'tte_yr'].max()*365:.0f}d")

    if not tenor_rows:
        print("[bhav] no surfaces built (no cached files?).")
        return
    T = pd.concat(tenor_rows, ignore_index=True)
    T.to_csv(out_dir / "tenor_manifest.csv", index=False)
    if surfaces:
        pd.concat(surfaces, ignore_index=True).to_parquet(out_dir / "surfaces/bhav_surfaces.parquet")

    okT = T[T.ok]
    print("\n===== bhavcopy full-surface layer =====")
    print(f"dates with a surface : {okT['date'].nunique()}")
    print(f"tenor-smiles built    : {len(okT)}")
    print(f"median tenors/date    : {okT.groupby('date').size().median():.0f}")
    print(f"maturity reach (median max): {okT.groupby('date').tte_yr.max().median()*365:.0f} days")
    print(f"mean intra-smile arb  : {okT.arb_frac.mean()*100:.2f}%")
    print(f"manifest -> {out_dir/'tenor_manifest.csv'}")


if __name__ == "__main__":
    main()
