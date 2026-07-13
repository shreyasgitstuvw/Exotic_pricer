"""Train the M2.1 calibration surrogate and evaluate it against the Route-A gate.

Two evaluations:
  (1) parameter space  — MAE / correlation of predicted vs DE-calibrated (kappa, sigma_v, rho).
  (2) PRICE space (the gate) — reprice each test surface with the surrogate's (kappa,sigma_v,rho)
      plus M1's (theta,v0), and compare IV RMSE to the full DE optimizer. Gate: surrogate within
      25 bps of the optimizer. We grade on prices, not params (kappa is the sloppy direction).

    python experiments/train_surrogate.py --data data/m2/dataset.npz --out-dir data/m2 \
        --params data/m1/params_timeseries_bhavcopy.csv --cache-dir bhavcopy   # (repricing eval)
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch
from hestonnn.surrogate import Surrogate, balanced_mse, TARGETS


def train(data, epochs=800, lr=1e-3, wd=1e-4, patience=80, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    Xtr = torch.tensor(data["Xtr"], dtype=torch.float32)
    ytr = torch.tensor(data["ytr"], dtype=torch.float32)
    wtr = torch.tensor(data["wtr"], dtype=torch.float32)
    Xva = torch.tensor(data["Xva"], dtype=torch.float32)
    yva = torch.tensor(data["yva"], dtype=torch.float32)
    wva = torch.tensor(data["wva"], dtype=torch.float32)
    tvar = torch.tensor(data["ytr"].var(0) + 1e-9, dtype=torch.float32)   # per-target variance

    model = Surrogate(Xtr.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    best, best_state, wait = float("inf"), None, 0
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        loss = balanced_mse(model(Xtr), ytr, wtr, tvar)
        loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vloss = balanced_mse(model(Xva), yva, wva, tvar).item() if len(Xva) else loss.item()
        if vloss < best - 1e-5:
            best, best_state, wait = vloss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state:
        model.load_state_dict(best_state)
    return model, best, ep


def param_eval(model, X, y, tag):
    if not len(X):
        print(f"  [{tag}] empty"); return
    with torch.no_grad():
        pred = model(torch.tensor(X, dtype=torch.float32)).numpy()
    print(f"  [{tag}] n={len(X)}")
    for i, t in enumerate(TARGETS):
        mae = np.mean(np.abs(pred[:, i] - y[:, i]))
        cc = np.corrcoef(pred[:, i], y[:, i])[0, 1] if len(X) > 2 else np.nan
        print(f"      {t:8} MAE {mae:6.3f}  corr {cc:+.2f}")
    return pred


def reprice_gate(model, data, params_csv, cache_dir, config_path):
    """The Route-A gate: reprice test surfaces with surrogate params vs DE params -> IV RMSE."""
    import pandas as pd
    from hestonnn.config import Config
    from hestonnn.data import bhavcopy as bc, surface_bhav as sb
    from hestonnn.heston_ref import HParams, heston_call
    from hestonnn.data.iv import implied_vol
    cfg = Config.load(config_path)
    P = pd.read_csv(params_csv, parse_dates=["date"]).set_index("date")
    dates_te = [str(x) for x in data["dates_te"]]
    with torch.no_grad():
        pred = model(torch.tensor(data["Xte"], dtype=torch.float32)).numpy()
    sur_rmse, de_rmse = [], []
    for i, ds in enumerate(dates_te):
        f = Path(cache_dir) / f"bhavcopy_FO_{pd.to_datetime(ds):%Y%m%d}.csv.zip"
        if not f.exists():
            continue
        row = P.loc[pd.to_datetime(ds)]
        day = bc.load_bhavcopy_day(f)
        surf, _ = sb.build_surface(day, bc.usable_expiries(day, 7), cfg.surf)
        clean = sb.calibration_quotes(surf, n_sd=1.5)
        clean = clean[clean["tte_yr"] >= 14 / 365]
        if clean["expiry"].nunique() < 3:
            continue
        for label, (k, sv, rho) in {
            "sur": (pred[i, 0], pred[i, 1], pred[i, 2]),
            "de": (row["kappa"], row["sigma_v"], row["rho"]),
        }.items():
            p = HParams(k, row["theta"], sv, rho, row["v0"])
            errs = []
            for e, s in clean.groupby("expiry"):
                F, dfc, T = s["F"].iloc[0], s["df"].iloc[0], s["tte_yr"].iloc[0]
                K = s["K"].to_numpy()
                miv = implied_vol(heston_call(F, K, T, p, dfc), F, K, T, df=dfc,
                                  call=np.ones(len(K), bool))
                errs.append(miv - s["iv"].to_numpy())
            r = np.concatenate(errs); r = r[np.isfinite(r)]
            (sur_rmse if label == "sur" else de_rmse).append(np.sqrt(np.mean(r ** 2)) * 1e4)
    if sur_rmse:
        gap = np.median(np.array(sur_rmse) - np.array(de_rmse))
        print(f"\n=== Route-A GATE (test repricing) ===")
        print(f"  surrogate IV RMSE median {np.median(sur_rmse):.0f} bps | DE median {np.median(de_rmse):.0f} bps")
        print(f"  median gap (surrogate - DE): {gap:.0f} bps   GATE <25 bps: {'PASS' if gap < 25 else 'FAIL'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/m2/dataset.npz")
    ap.add_argument("--out-dir", default="data/m2")
    ap.add_argument("--params", default=None, help="M1 params csv (enables repricing gate)")
    ap.add_argument("--cache-dir", default="bhavcopy")
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--epochs", type=int, default=800)
    args = ap.parse_args()

    data = np.load(args.data, allow_pickle=True)
    model, vloss, ep = train(data, epochs=args.epochs)
    print(f"trained {ep+1} epochs | best val balanced-MSE {vloss:.4f}")
    param_eval(model, data["Xtr"], data["ytr"], "train")
    param_eval(model, data["Xva"], data["yva"], "val")
    param_eval(model, data["Xte"], data["yte"], "test")
    torch.save(model.state_dict(), Path(args.out_dir) / "surrogate.pt")
    print(f"saved -> {Path(args.out_dir)/'surrogate.pt'}")
    # speed: one forward pass over all test dates vs the DE baseline (~10 s/date)
    if len(data["Xte"]):
        Xte = torch.tensor(data["Xte"], dtype=torch.float32)
        t0 = time.time()
        with torch.no_grad():
            _ = model(Xte)
        dt = (time.time() - t0) / len(Xte) * 1e6
        print(f"surrogate speed: {dt:.0f} microsec/date  vs DE ~10 s/date  (~{10e6/max(dt,1):.0f}x faster)")
    if args.params:
        reprice_gate(model, data, args.params, args.cache_dir, args.config)


if __name__ == "__main__":
    main()
