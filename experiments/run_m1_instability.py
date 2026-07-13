"""M1 — the instability exhibit: calibrate over a date set -> parameter time series + money-plot.

Two sources:
  --source bhavcopy  : FULL Heston (5-param) per date from assembled bhavcopy surfaces (the real M1).
                       Independent thorough DE per date (trustworthy; parallelizable). Resumable —
                       appends to params_timeseries_bhavcopy.csv and skips dates already done, so it
                       can run in chunks or be re-run safely.
  --source short-end : SHORT-END PROXY — constrained Heston (kappa,theta fixed; sigma_v,rho,v0 free)
                       on the 1-min front-contract smiles. A runnable signal without bhavcopy.

Recommended full run (on your machine, unmetered):
    python experiments/run_m1_instability.py --source bhavcopy --cache-dir bhavcopy \
        --config configs/data.yaml --out-dir data/m1
Outputs: params_timeseries_<src>.csv, jump_stats_<src>.txt (median + p95 |dParam|), money_plot_<src>.png
"""
from __future__ import annotations
import argparse, glob, sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hestonnn import calibrate as cal


# ----------------------------------------------------------------------------- short-end proxy
def _short_end_series(parquet, every, kappa, theta, starts):
    S = pd.read_parquet(parquet)
    dates = sorted(S["date"].unique())[::every]
    rows = []
    for d in dates:
        sm = S[S["date"] == d]
        if len(sm) < 5:
            continue
        p, rb = cal.calibrate_constrained(sm, kappa=kappa, theta=theta, starts=starts)
        atm = float(sm["iv"].iloc[(sm["K"] - sm["F"].iloc[0]).abs().values.argmin()])
        rows.append(dict(date=pd.to_datetime(d), tte_d=float(sm["tte_yr"].iloc[0]) * 365,
                         atm_iv=atm, sigma_v=p.sigma_v, rho=p.rho, v0=p.v0, rmse_bps=rb))
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


# ----------------------------------------------------------------------------- full-Heston bhavcopy
def _bhavcopy_series(cache_dir, config_path, out, max_new, min_oi, n_sd, every, de_maxiter, popsize,
                     min_tte_days=14):
    from hestonnn.config import Config
    from hestonnn.data import bhavcopy as bc, surface_bhav as sb
    from hestonnn.heston_ref import heston_call
    from hestonnn.data.iv import implied_vol
    cfg = Config.load(config_path)
    files = sorted(glob.glob(str(Path(cache_dir) / "*.zip")))[::max(1, every)]
    csv = out / "params_timeseries_bhavcopy.csv"
    done, prev = set(), pd.DataFrame()
    if csv.exists():
        prev = pd.read_csv(csv, parse_dates=["date"])
        if "short_gap_bps" not in prev.columns:      # older-schema run -> archive & start fresh
            archive = csv.with_name("params_timeseries_bhavcopy.old.csv")
            csv.replace(archive)
            print(f"  [note] archived old-schema results -> {archive.name}; recalibrating fresh")
            prev, done = pd.DataFrame(), set()
        else:
            done = set(prev["date"].dt.date.astype(str))
    rows, new = [], 0
    for f in files:
        try:
            day = bc.load_bhavcopy_day(f)
        except Exception:
            continue
        d = str(day["date"].iloc[0])
        if d in done:
            continue
        if max_new and new >= max_new:
            break
        surf, _ = sb.build_surface(day, bc.usable_expiries(day, min_oi), cfg.surf)
        clean = sb.calibration_quotes(surf, n_sd=n_sd)
        if clean.empty or clean["expiry"].nunique() < 3:
            continue
        clean = (clean.assign(dd=(clean["m"] - 1).abs())
                       .sort_values(["expiry", "dd"])
                       .groupby("expiry").head(13).drop(columns="dd").reset_index(drop=True))
        # Heston is calibrated to the tenors it CAN fit (>= min_tte_days); the short end it cannot
        # reach is measured separately as the "Heston gap" — the M2 target. (Finding: forcing short
        # tenors in extremizes sigma_v/kappa and corrupts the whole fit.)
        floor = min_tte_days / 365.0
        fit_set = clean[clean["tte_yr"] >= floor]
        short_set = clean[clean["tte_yr"] < floor]
        if fit_set["expiry"].nunique() < 3:
            continue
        p, st = cal.calibrate(fit_set, de_maxiter=de_maxiter, popsize=popsize, seed=1)
        short_gap = np.nan
        if len(short_set):
            errs = []
            for e, s in short_set.groupby("expiry"):
                F = s["F"].iloc[0]; dfc = s["df"].iloc[0]; T = s["tte_yr"].iloc[0]
                K = s["K"].to_numpy()
                miv = implied_vol(heston_call(F, K, T, p, dfc), F, K, T, df=dfc,
                                  call=np.ones(len(K), bool))
                errs.append(miv - s["iv"].to_numpy())
            e = np.concatenate(errs); e = e[np.isfinite(e)]
            short_gap = float(np.sqrt(np.mean(e ** 2)) * 1e4) if len(e) else np.nan
        row = dict(date=pd.to_datetime(day["date"].iloc[0]), atm_iv=float(clean["iv"].median()),
                   kappa=p.kappa, theta=p.theta, sigma_v=p.sigma_v, rho=p.rho, v0=p.v0,
                   rmse_bps=st["iv_rmse_bps"], short_gap_bps=short_gap,
                   feller_ok=st["feller_ok"], n_tenors=st["n_tenors"])
        rows.append(row)
        pd.DataFrame([row]).to_csv(csv, mode="a", header=not csv.exists(), index=False)
        new += 1
        print(f"  {day['date'].iloc[0]}  kappa={p.kappa:5.2f} sig_v={p.sigma_v:4.2f} "
              f"rho={p.rho:+.2f} v0={p.v0:.4f} rmse={st['iv_rmse_bps']:3.0f}bps "
              f"short_gap={short_gap:4.0f}bps", flush=True)
    allrows = pd.concat([prev, pd.DataFrame(rows)], ignore_index=True) if len(prev) else pd.DataFrame(rows)
    return allrows.drop_duplicates("date").sort_values("date").reset_index(drop=True)


# ----------------------------------------------------------------------------- shared
def _jump_stats(ts, cols):
    return {c: (float(ts[c].diff().abs().median()), float(ts[c].diff().abs().quantile(0.95)))
            for c in cols}


def _money_plot(ts, cols, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(cols)
    fig, ax = plt.subplots(n + 1, 1, figsize=(11, 2.0 * (n + 1)))
    for i, c in enumerate(cols):
        ax[i].plot(ts["date"], ts[c], lw=0.9, color="#1f4e79")
        if "tte_d" in ts:
            near = ts["tte_d"] <= 2
            ax[i].scatter(ts["date"][near], ts[c][near], s=14, color="#c00000", zorder=3,
                          label="<=2d to expiry")
            if i == 0:
                ax[i].legend(loc="upper right", fontsize=8)
        ax[i].set_ylabel(c); ax[i].grid(alpha=0.25)
    jd = ts["atm_iv"].diff().abs().dropna() * 100
    ax[-1].hist(jd, bins=30, color="#7f7f7f")
    ax[-1].set_xlabel("|change in ATM IV between samples| (vol pts)"); ax[-1].set_ylabel("count")
    ax[-1].grid(alpha=0.25)
    fig.suptitle(title, fontsize=12); fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(path, dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["short-end", "bhavcopy"], default="short-end")
    ap.add_argument("--parquet", default="data/surfaces/short_end_smiles.parquet")
    ap.add_argument("--cache-dir", default="bhavcopy")
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--every", type=int, default=1, help="use every Nth date/file")
    ap.add_argument("--max-new", type=int, default=0, help="process at most N new dates (chunking)")
    ap.add_argument("--min-oi-strikes", type=int, default=7)
    ap.add_argument("--n-sd", type=float, default=1.5)
    ap.add_argument("--de-maxiter", type=int, default=24, help="DE iterations/date (bhavcopy quality)")
    ap.add_argument("--popsize", type=int, default=15)
    ap.add_argument("--min-tte-days", type=float, default=14,
                    help="calibrate Heston to tenors >= this; shorter = the 'Heston gap' (M2 target)")
    ap.add_argument("--kappa", type=float, default=2.0, help="short-end: fixed kappa")
    ap.add_argument("--theta", type=float, default=0.03, help="short-end: fixed theta")
    ap.add_argument("--starts", type=int, default=1, help="short-end: restarts")
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    if args.source == "short-end":
        ev = args.every if args.every > 1 else 8
        ts = _short_end_series(args.parquet, ev, args.kappa, args.theta, args.starts)
        cols = ["atm_iv", "sigma_v", "rho", "v0"]
        title = f"NIFTY short-end constrained-Heston instability (kappa={args.kappa}, theta={args.theta} fixed)"
        tag = "shortend"
    else:
        ts = _bhavcopy_series(out=out, cache_dir=args.cache_dir, config_path=args.config,
                              max_new=args.max_new, min_oi=args.min_oi_strikes, n_sd=args.n_sd,
                              every=args.every, de_maxiter=args.de_maxiter, popsize=args.popsize,
                              min_tte_days=args.min_tte_days)
        cols = ["atm_iv", "kappa", "sigma_v", "rho", "v0", "short_gap_bps"]
        title = "NIFTY Heston instability (fit to >=14d) + short-end Heston gap (M2 target)"
        tag = "bhavcopy"

    if ts.empty:
        print("no dates processed."); return
    cols = [c for c in cols if c in ts.columns]        # robust to older/missing columns
    ts.to_csv(out / f"params_timeseries_{tag}.csv", index=False)
    js = _jump_stats(ts, cols)
    lines = [f"M1 instability — {args.source} — {len(ts)} dates "
             f"({ts['date'].min().date()}..{ts['date'].max().date()})", "",
             f"{'param':10}{'median|dP|':>14}{'p95|dP|':>12}"]
    for c in cols:
        m, p95 = js[c]; lines.append(f"{c:10}{m:14.4f}{p95:12.4f}")
    lines.append(f"\nfit quality: median IV RMSE {ts['rmse_bps'].median():.0f} bps")
    (out / f"jump_stats_{tag}.txt").write_text("\n".join(lines))
    _money_plot(ts, cols, out / f"money_plot_{tag}.png", title)
    print("\n".join(lines))
    print(f"\nplot   -> {out/f'money_plot_{tag}.png'}\nseries -> {out/f'params_timeseries_{tag}.csv'}")


if __name__ == "__main__":
    main()
