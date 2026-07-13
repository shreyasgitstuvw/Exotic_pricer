#!/usr/bin/env python3
"""Probe cached bhavcopy files for term-structure VOLUME — answers 'is 2020-2023 usable?'.

For each *.zip in --cache-dir, reports: detected format (old/new), NIFTY option rows, distinct
expiries, usable tenors (>=min-oi strikes carrying OI), max usable TTE, and ATM level. Run this on a
small old-era test pull before committing to a full backfill.

    python fetch_bhavcopy_data.py --start 2020-03-01 --end 2020-03-31 --cache-dir bhavcopy_test
    python experiments/probe_bhavcopy.py --cache-dir bhavcopy_test
"""
from __future__ import annotations
import argparse, glob, sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hestonnn.data import bhavcopy as bc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="bhavcopy")
    ap.add_argument("--min-oi", type=int, default=7)
    ap.add_argument("--symbol", default="NIFTY")
    args = ap.parse_args()

    files = sorted(glob.glob(str(Path(args.cache_dir) / "*.zip")))
    if not files:
        sys.exit(f"no .zip files in {Path(args.cache_dir).resolve()} — pull a test set first.")
    rows = []
    for f in files:
        rec = {"file": Path(f).name, "fmt": "?", "opt_rows": 0, "expiries": 0,
               "usable_tenors": 0, "max_tte_d": 0, "atm": None, "note": ""}
        try:
            import zipfile, io
            with zipfile.ZipFile(f) as z:
                name = [n for n in z.namelist() if n.lower().endswith(".csv")][0]
                cols = pd.read_csv(io.BytesIO(z.open(name).read()), nrows=0).columns
            rec["fmt"] = bc.detect_format(cols)
            day = bc.load_bhavcopy_day(f, symbol=args.symbol)
            rec["opt_rows"] = len(day)
            rec["expiries"] = day["expiry"].nunique()
            use = bc.usable_expiries(day, args.min_oi)
            rec["usable_tenors"] = len(use)
            if use:
                td = pd.to_datetime(day["date"].iloc[0])
                rec["max_tte_d"] = int((pd.to_datetime(max(use)) - td).days)
            spot = day["underlying"].dropna()
            rec["atm"] = round(float(spot.median()), 0) if len(spot) else None
        except Exception as e:
            rec["note"] = f"{type(e).__name__}: {str(e)[:50]}"
        rows.append(rec)

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    ok = df[df["usable_tenors"] >= 4]
    print(f"\nfiles with >=4 usable tenors: {len(ok)}/{len(df)}")
    if len(ok):
        print(f"median usable tenors: {ok['usable_tenors'].median():.0f} | "
              f"median max TTE: {ok['max_tte_d'].median():.0f} days "
              f"({ok['max_tte_d'].median()/365:.1f}y)")
        print(f"formats seen: {df['fmt'].value_counts().to_dict()}")
        print("VERDICT: term structure present — backfill is worth it." if ok["max_tte_d"].median() >= 180
              else "VERDICT: tenors present but short — check max TTE before relying on it for M3.")
    else:
        print("VERDICT: little/no usable term structure — old archives may be thin or unreachable.")


if __name__ == "__main__":
    main()
