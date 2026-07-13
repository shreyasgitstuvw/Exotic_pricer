"""Train the two-headed net (shared encoder + params head + correction head) and compare Head B's
short-end gap reduction to the standalone gap-closer (60% test).

Combined loss (both terms normalized to O(1) so neither dominates):
    L = w_a * balanced_param_MSE(Head A)  +  w_b * (correction_fit_MSE / pre-correction_var)(Head B)

Eval: Head B gap reduction + arb violations (with the deterministic projection, same as M2.2) and
Head A parameter correlation. If Head B beats 60% at 0 arb, multi-task helped.

    python experiments/train_twohead.py --data-dir data/m2b --params data/m1/params_timeseries_bhavcopy.csv
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch
from hestonnn.twohead import TwoHead
from hestonnn.surrogate import TARGETS
from hestonnn.gap_closer import (correction, butterfly_penalty, bucket_of, X_SCALE, project_arbfree)
from hestonnn.data.arbitrage import flag_smile
from hestonnn.data.iv import black76_price, implied_vol

FEAT_COLS = None


def pack(data_dir, params_csv):
    global FEAT_COLS
    F = pd.read_parquet(Path(data_dir) / "features.parquet")
    Q = pd.read_parquet(Path(data_dir) / "quotes.parquet")
    P = pd.read_csv(params_csv, parse_dates=["date"]).set_index("date")[TARGETS]
    FEAT_COLS = [c for c in F.columns if c not in ("date", "split")]
    feats_by_date = {r["date"]: torch.tensor(r[FEAT_COLS].to_numpy(float), dtype=torch.float32)
                     for _, r in F.iterrows()}
    split_by_date = dict(zip(F["date"], F["split"]))
    param_by_date = {}
    for d in F["date"]:
        try:
            param_by_date[d] = torch.tensor(P.loc[pd.Timestamp(d)].to_numpy(float), dtype=torch.float32)
        except Exception:
            param_by_date[d] = None
    days = {"train": [], "val": [], "test": []}
    for d, g in Q.groupby("date"):
        if d not in feats_by_date or param_by_date.get(d) is None:
            continue
        tenors = []
        for T, gt in g.groupby("tte_yr"):
            gt = gt.sort_values("K")
            tenors.append(dict(F=float(gt["F"].iloc[0]), df=float(gt["df"].iloc[0]), T=float(T),
                               K=torch.tensor(gt["K"].to_numpy(float), dtype=torch.float32),
                               x=torch.tensor(gt["log_m"].to_numpy(float), dtype=torch.float32),
                               miv=torch.tensor(gt["market_iv"].to_numpy(float), dtype=torch.float32),
                               hiv=torch.tensor(gt["heston_iv"].to_numpy(float), dtype=torch.float32)))
        days[split_by_date[d]].append((feats_by_date[d], param_by_date[d], tenors, d))
    # normalizers from TRAIN
    ptr = torch.stack([p for _, p, _, _ in days["train"]])
    param_var = ptr.var(0) + 1e-9
    resid_list = [(t["miv"] - t["hiv"]).numpy() for _, _, tn, _ in days["train"] for t in tn]
    resid_var = float(np.mean(np.concatenate(resid_list) ** 2)) + 1e-12
    return days, param_var, resid_var


def date_loss(model, feats, param_tgt, tenors, param_var, resid_var, w_a, w_b):
    params, corr = model(feats.unsqueeze(0))
    params, corr = params[0], corr[0]
    la = (((params - param_tgt) ** 2) / param_var).mean()
    fit, n = 0.0, 0
    for t in tenors:
        cor = t["hiv"] + correction(corr[bucket_of(t["T"])], t["x"])
        fit = fit + ((cor - t["miv"]) ** 2).sum(); n += len(t["x"])
    lb = (fit / max(n, 1)) / resid_var
    return w_a * la + w_b * lb, la.item(), (fit / max(n, 1)).item()


def train(days, param_var, resid_var, w_a, w_b, epochs=2000, lr=2e-3, wd=1e-5, patience=200,
          batch=32, seed=0):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    model = TwoHead(len(FEAT_COLS))
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    tr = list(days["train"]); best, best_state, wait = float("inf"), None, 0
    for ep in range(epochs):
        model.train(); order = rng.permutation(len(tr))
        for i in range(0, len(tr), batch):
            opt.zero_grad(); loss = 0.0
            for j in order[i:i + batch]:
                f, p, tn, _ = tr[j]
                loss = loss + date_loss(model, f, p, tn, param_var, resid_var, w_a, w_b)[0]
            (loss / len(order[i:i + batch])).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pool = days["val"] if days["val"] else tr
            vl = float(np.mean([date_loss(model, f, p, tn, param_var, resid_var, w_a, w_b)[0].item()
                                for f, p, tn, _ in pool]))
        if vl < best - 1e-7:
            best, best_state, wait = vl, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state:
        model.load_state_dict(best_state)
    return model, ep


def gate(model, days, split):
    model.eval(); red, viol = [], 0
    pcorr = {t: ([], []) for t in TARGETS}
    with torch.no_grad():
        for feats, ptgt, tenors, d in days[split]:
            params, corr = model(feats.unsqueeze(0))
            params = params[0].numpy(); coeffs = corr[0].numpy()
            for i, t in enumerate(TARGETS):
                pcorr[t][0].append(params[i]); pcorr[t][1].append(float(ptgt[i]))
            gb, ga, dv = [], [], False
            for t in tenors:
                x = t["x"].numpy(); z = x / X_SCALE; cb = coeffs[bucket_of(t["T"])]
                civ = t["hiv"].numpy() + (cb[0] + cb[1] * z + cb[2] * z ** 2)
                K = t["K"].numpy(); F = float(t["F"]); dfc = float(t["df"]); T = float(t["T"])
                cpx = project_arbfree(K, black76_price(F, K, T, civ, dfc, True), dfc, F)
                piv = implied_vol(cpx, F, K, T, df=dfc, call=np.ones(len(K), bool))
                piv = np.where(np.isfinite(piv), piv, civ)
                gb.append(t["miv"].numpy() - t["hiv"].numpy()); ga.append(t["miv"].numpy() - piv)
                if flag_smile(pd.DataFrame({"K": K, "call_px": cpx, "df": dfc}))["arb_any"].any():
                    dv = True
            gb = np.concatenate(gb); ga = np.concatenate(ga)
            red.append(1 - np.sqrt(np.mean(ga ** 2)) / np.sqrt(np.mean(gb ** 2))); viol += int(dv)
    red = np.array(red)
    print(f"\n=== TWO-HEAD ({split}, n={len(red)}) ===")
    print(f"  Head B gap reduction: median {np.median(red)*100:.0f}%  (standalone was 60% test)")
    print(f"  arb violations: {viol}/{len(red)}")
    ca = [np.corrcoef(pcorr[t][0], pcorr[t][1])[0, 1] for t in TARGETS] if len(red) > 2 else [np.nan]*3
    print(f"  Head A param corr: " + ", ".join(f"{t} {c:+.2f}" for t, c in zip(TARGETS, ca)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/m2b")
    ap.add_argument("--params", default="data/m1/params_timeseries_bhavcopy.csv")
    ap.add_argument("--out-dir", default="data/m2b")
    ap.add_argument("--w-a", type=float, default=1.0, help="Head A (params) loss weight")
    ap.add_argument("--w-b", type=float, default=1.0, help="Head B (correction) loss weight")
    ap.add_argument("--epochs", type=int, default=2000)
    args = ap.parse_args()
    days, param_var, resid_var = pack(args.data_dir, args.params)
    print(f"dates -> train {len(days['train'])} / val {len(days['val'])} / test {len(days['test'])}")
    model, ep = train(days, param_var, resid_var, args.w_a, args.w_b, epochs=args.epochs)
    print(f"trained {ep+1} epochs")
    torch.save(model.state_dict(), Path(args.out_dir) / "twohead.pt")
    gate(model, days, "train");
    if days["val"]:
        gate(model, days, "val")
    gate(model, days, "test")


if __name__ == "__main__":
    main()
