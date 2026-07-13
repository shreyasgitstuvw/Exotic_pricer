"""Train the Route-B parametric gap-closer and evaluate the M2.2 gate.

Loss per date = vega-weighted (corrected_IV - market_IV)^2  +  lambda_bf * butterfly_penalty
                +  lambda_sm * curvature_smoothness.
Gate (held-out test): median short-end gap reduced >= 50%  AND  0 arbitrage violations.

    python experiments/train_gap_closer.py --data-dir data/m2b --out-dir data/m2b
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch
from hestonnn.gap_closer import (GapCloser, correction, butterfly_penalty, bucket_of, X_SCALE,
                                 project_arbfree)
from hestonnn.data.arbitrage import flag_smile
from hestonnn.data.iv import black76_price, implied_vol

FEAT_COLS = None    # set from the features file (long-end + short-end columns)


def pack(data_dir):
    """date -> {feats, tenors:[{F,df,T,K,x,miv,hiv,vega}]}, grouped by split."""
    global FEAT_COLS
    F = pd.read_parquet(Path(data_dir) / "features.parquet")
    Q = pd.read_parquet(Path(data_dir) / "quotes.parquet")
    FEAT_COLS = [c for c in F.columns if c not in ("date", "split")]
    feats_by_date = {r["date"]: torch.tensor(r[FEAT_COLS].to_numpy(float), dtype=torch.float32)
                     for _, r in F.iterrows()}
    split_by_date = dict(zip(F["date"], F["split"]))
    days = {"train": [], "val": [], "test": []}
    for d, g in Q.groupby("date"):
        if d not in feats_by_date:
            continue
        tenors = []
        for T, gt in g.groupby("tte_yr"):
            gt = gt.sort_values("K")
            tenors.append(dict(
                F=float(gt["F"].iloc[0]), df=float(gt["df"].iloc[0]), T=float(T),
                K=torch.tensor(gt["K"].to_numpy(float), dtype=torch.float32),
                x=torch.tensor(gt["log_m"].to_numpy(float), dtype=torch.float32),
                miv=torch.tensor(gt["market_iv"].to_numpy(float), dtype=torch.float32),
                hiv=torch.tensor(gt["heston_iv"].to_numpy(float), dtype=torch.float32),
                vega=torch.tensor(gt["vega"].to_numpy(float), dtype=torch.float32)))
        days[split_by_date[d]].append((feats_by_date[d], tenors, d))
    return days


def date_loss(model, feats, tenors, lam_bf=0.0):
    # NOTE: fit loss is IV^2 (~1e-4); the butterfly penalty is ~1e-2. They are on very different
    # scales, so lam_bf must be SMALL (start 0 to confirm the fit works; then ~0.02 to nudge no-arb).
    coeffs = model(feats.unsqueeze(0))[0]                 # (N_BUCKETS, 3)
    fit, bf, n = 0.0, 0.0, 0
    for t in tenors:
        corr = t["hiv"] + correction(coeffs[bucket_of(t["T"])], t["x"])
        # UNWEIGHTED IV MSE: the gap lives in the wings, so every strike counts equally
        # (matches the gate metric). Vega-weighting would ignore exactly the wings we must fix.
        fit = fit + ((corr - t["miv"]) ** 2).sum()
        n += len(t["x"])
        bf = bf + butterfly_penalty(torch.tensor(t["F"]), t["K"], torch.tensor(t["T"]),
                                    corr, torch.tensor(t["df"]))
    fit = fit / max(n, 1)
    # No penalty on curvature magnitude: the fix REQUIRES large d2. Butterfly alone guards no-arb.
    return fit + lam_bf * bf, fit


def train(days, epochs=2000, lr=2e-3, wd=1e-5, patience=200, batch=32, seed=0, lam_bf=0.0):
    torch.manual_seed(seed); np.random.seed(seed)
    rng = np.random.default_rng(seed)
    model = GapCloser(len(FEAT_COLS))
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    tr = list(days["train"])
    best, best_state, wait = float("inf"), None, 0
    for ep in range(epochs):
        model.train()
        order = rng.permutation(len(tr))
        for i in range(0, len(tr), batch):        # MINIBATCH: ~8 steps/epoch, not 1
            idx = order[i:i + batch]
            opt.zero_grad()
            loss = 0.0
            for j in idx:
                loss = loss + date_loss(model, tr[j][0], tr[j][1], lam_bf)[0]
            (loss / len(idx)).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pool = days["val"] if days["val"] else tr
            vf = float(np.mean([date_loss(model, f, t)[1].item() for f, t, _ in pool]))
        if vf < best - 1e-7:
            best, best_state, wait = vf, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state:
        model.load_state_dict(best_state)
    return model, ep


def gate(model, days, split="test"):
    model.eval()
    red, viol_dates = [], 0
    with torch.no_grad():
        for feats, tenors, d in days[split]:
            coeffs = model(feats.unsqueeze(0))[0].numpy()      # (N_BUCKETS, 3)
            gb, ga, day_viol = [], [], False
            for t in tenors:
                x = t["x"].numpy(); z = x / X_SCALE
                cb = coeffs[bucket_of(t["T"])]
                corr_iv = t["hiv"].numpy() + (cb[0] + cb[1] * z + cb[2] * z ** 2)
                K = t["K"].numpy(); F = float(t["F"]); dfc = float(t["df"]); T = float(t["T"])
                # deployable output = corrected smile projected to the nearest arb-free one
                cpx = project_arbfree(K, black76_price(F, K, T, corr_iv, dfc, True), dfc, F)
                proj_iv = implied_vol(cpx, F, K, T, df=dfc, call=np.ones(len(K), bool))
                proj_iv = np.where(np.isfinite(proj_iv), proj_iv, corr_iv)
                gb.append(t["miv"].numpy() - t["hiv"].numpy())
                ga.append(t["miv"].numpy() - proj_iv)
                if flag_smile(pd.DataFrame({"K": K, "call_px": cpx, "df": dfc}))["arb_any"].any():
                    day_viol = True
            gb = np.concatenate(gb); ga = np.concatenate(ga)
            gap_before = np.sqrt(np.mean(gb ** 2)); gap_after = np.sqrt(np.mean(ga ** 2))
            red.append(1 - gap_after / gap_before)
            viol_dates += int(day_viol)
    red = np.array(red)
    print(f"\n=== M2.2 GATE ({split}, n={len(red)}) ===")
    print(f"  short-end gap reduction: median {np.median(red)*100:.0f}%  (mean {red.mean()*100:.0f}%)")
    print(f"  dates with an arb violation: {viol_dates}/{len(red)}")
    ok = np.median(red) >= 0.50 and viol_dates == 0
    print(f"  GATE (>=50% reduction & 0 arb): {'PASS' if ok else 'FAIL'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/m2b")
    ap.add_argument("--out-dir", default="data/m2b")
    ap.add_argument("--epochs", type=int, default=2000)   # early stopping ends it well before this
    ap.add_argument("--lam-bf", type=float, default=0.0,
                    help="butterfly (no-arb) penalty weight. 0 to confirm fit; ~0.02 to nudge no-arb.")
    args = ap.parse_args()
    days = pack(args.data_dir)
    print(f"dates -> train {len(days['train'])} / val {len(days['val'])} / test {len(days['test'])}")
    model, ep = train(days, epochs=args.epochs, lam_bf=args.lam_bf)
    print(f"trained {ep+1} epochs")
    torch.save(model.state_dict(), Path(args.out_dir) / "gap_closer.pt")
    gate(model, days, "train")     # diagnostic: can it even fit what it sees?
    if days["val"]:
        gate(model, days, "val")
    gate(model, days, "test")


if __name__ == "__main__":
    main()
