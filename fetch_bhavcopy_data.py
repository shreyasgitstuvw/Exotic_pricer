#!/usr/bin/env python3
"""RUN THIS ON YOUR MACHINE to download the bhavcopy files P2 needs.

    cd research/p2-heston-nn
    pip install requests            # one-time
    python fetch_bhavcopy_data.py                 # recommended first pull (clean UDiFF era)
    python fetch_bhavcopy_data.py --start 2021-01-01   # full history (mixed format, also handled)
    python fetch_bhavcopy_data.py --dry-run       # just list dates/URLs, download nothing

What it does: builds a sampled trading-date schedule (weekly + monthly anchors), then downloads each
day's NSE F&O bhavcopy into ./bhavcopy/ (cookie-primed, retries, caches, skips weekends/holidays via
404). Re-running only fetches what's missing. Afterwards, build surfaces with:

    python experiments/build_bhavcopy_surfaces.py --config configs/data.yaml \
        --start <same start> --end <same end> --cache-dir bhavcopy --out-dir data/bhav

Then the real M1:  python experiments/run_m1_instability.py --source bhavcopy ...
"""
from __future__ import annotations
import argparse, sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from hestonnn.data import sampling
from hestonnn.data.fetch_bhavcopy import fetch_many, bhavcopy_url, cache_path


def main():
    ap = argparse.ArgumentParser(description="Download NSE F&O bhavcopy for P2.")
    ap.add_argument("--start", default="2024-07-08", help="YYYY-MM-DD (default = UDiFF cutover)")
    ap.add_argument("--end", default=date.today().isoformat(), help="YYYY-MM-DD (default today)")
    ap.add_argument("--weekly-weekday", type=int, default=2, help="0=Mon..4=Fri (default Wed)")
    ap.add_argument("--monthly-day", type=int, default=15, help="also anchor one date/month near this")
    ap.add_argument("--cache-dir", default="bhavcopy")
    ap.add_argument("--pause", type=float, default=1.5, help="seconds between requests (be polite)")
    ap.add_argument("--dry-run", action="store_true", help="list dates + URLs, download nothing")
    args = ap.parse_args()

    dates = sampling.schedule(args.start, args.end,
                              weekly_weekday=args.weekly_weekday, monthly_day=args.monthly_day)
    print(f"schedule: {len(dates)} dates  {dates[0]} .. {dates[-1]}")
    print(f"target dir: {Path(args.cache_dir).resolve()}")

    if args.dry_run:
        for d in dates[:5] + (["..."] if len(dates) > 8 else []) + dates[-3:]:
            if d == "...":
                print("  ...")
            else:
                print(f"  {d}  ->  {bhavcopy_url(d)}")
        print(f"\n(dry-run) would download {len(dates)} files. Drop --dry-run to fetch.")
        return

    try:
        import requests  # noqa: F401
    except ImportError:
        sys.exit("Missing dependency: run  pip install requests")

    got = fetch_many(dates, cache_dir=args.cache_dir, pause=args.pause)
    n = sum(1 for v in got.values() if v)
    print(f"\nDone: {n}/{len(dates)} files in {Path(args.cache_dir).resolve()}")
    print("Next: python experiments/build_bhavcopy_surfaces.py --config configs/data.yaml "
          f"--start {args.start} --end {args.end} --cache-dir {args.cache_dir} --out-dir data/bhav")


if __name__ == "__main__":
    main()
