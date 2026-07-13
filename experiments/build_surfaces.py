"""M0 entry point: 1-min front-contract strip -> clean EOD short-end smiles + manifest.

Run (sandbox):
  python experiments/build_surfaces.py --config configs/data.yaml \
      --data-root /sessions/.../mnt/Quant --years 2021 --out-dir data

Every number is produced here and written to disk; nothing is hand-edited (BENCHMARKS rule).
Results are committed alongside this script.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hestonnn.config import Config
from hestonnn.data import loader, expiry as expmod, surface as surfmod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--data-root", default=None, help="override data_root (e.g. sandbox mount)")
    ap.add_argument("--years", nargs="*", type=int, default=None, help="subset of years to build")
    ap.add_argument("--out-dir", default=None, help="override output dir")
    ap.add_argument("--limit-weeks", type=int, default=None, help="debug: cap number of week files")
    args = ap.parse_args()

    cfg = Config.load(args.config, data_root=args.data_root)
    cstart, cend = cfg.close_window

    week_files = loader.list_week_files(cfg.src("options_history"))
    if args.years:
        yrs = {str(y) for y in args.years}
        week_files = [f for f in week_files if f.parts[-3] in yrs]
    if args.limit_weeks:
        week_files = week_files[: args.limit_weeks]
    print(f"[build] {len(week_files)} week files")

    frames = [loader.load_week(f) for f in week_files]

    # 1) empirical expiry map across the loaded span
    tv = expmod.day_time_values(frames, cstart, cend)
    emap = expmod.detect(tv, cfg.tv_threshold)
    print(f"[build] {len(tv)} trading days | {len(emap.expiries)} detected expiries")

    # 2) per-date smiles
    out_dir = Path(args.out_dir) if args.out_dir else Path(".")
    surf_dir = out_dir / "surfaces"
    surf_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    all_quotes = []
    for df in frames:
        for d, day in df.groupby("dt"):
            if pd.isna(d) or d not in emap.per_day.index:
                continue
            tte = emap.tte(d)
            res = surfmod.build_smile(day, d, tte, cfg.surf, cstart, cend)
            manifest_rows.append(res.summary)
            if res.ok:
                all_quotes.append(res.quotes)

    manifest = pd.DataFrame(manifest_rows).sort_values("date")
    man_path = out_dir / "manifest.csv"
    manifest.to_csv(man_path, index=False)

    if all_quotes:
        quotes = pd.concat(all_quotes, ignore_index=True)
        quotes.to_parquet(surf_dir / "short_end_smiles.parquet", index=False)

    ok = manifest["ok"].sum()
    arb = manifest.loc[manifest["ok"], "arb_frac"]
    print("\n===== M0 short-end layer =====")
    print(f"clean daily smiles : {ok}  (gate: >=250)")
    print(f"median clean strikes/smile: {manifest.loc[manifest['ok'],'n_clean'].median():.0f}")
    print(f"tte range (days)   : {manifest['tte_yr'].min()*365:.1f} .. {manifest['tte_yr'].max()*365:.1f}")
    if len(arb):
        print(f"arb-violation frac : mean {arb.mean()*100:.2f}%  p95 {arb.quantile(.95)*100:.2f}%  (gate: <2%)")
    print(f"manifest -> {man_path}")
    rej = manifest.loc[~manifest["ok"], "reject"].value_counts()
    if len(rej):
        print("rejections:\n" + rej.to_string())


if __name__ == "__main__":
    main()
