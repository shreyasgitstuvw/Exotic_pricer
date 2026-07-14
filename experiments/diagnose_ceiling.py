"""Decompose the M2.2 gap between the network (60% test) and the per-tenor oracle (87%).

Ladder of ceilings (all on the TEST split, so comparable to the net's 60%):
  A. per-tenor oracle    : best quadratic per expiry (in-sample fit)             -> 87% reference
  B. per-bucket oracle   : best quadratic per 2 tenor buckets (the net's form)   -> A-B = bucketing loss
  C. features->coeffs     : predict B's coefficients from the 24 features (ridge  -> B-C = feature-
                            fit on train, applied to test), + a nonlinear variant     sufficiency loss
  net (from M2.2)        : 60%                                                    -> C-net = model/opt loss
Train-vs-test of C also isolates generalization.

    python experiments/diagnose_ceiling.py --data-dir data/m2b
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd

FEAT_EXCL = ("date", "split")


def best_quad(x, r):
    """Best a+b*x+c*x^2 fit; returns coeffs [c,b,a] (np.polyfit order). Residual after."""
    if len(x) > 2:
        return np.polyfit(x, r, 2)
    if len(x) == 2:
        b = np.polyfit(x, r, 1); return np.array([0.0, b[0], b[1]])
    return np.array([0.0, 0.0, r.mean() if len(r) else 0.0])


def date_reduction(groups, coeffs_by_key=None):
    """groups: list of (x, r, key). If coeffs_by_key given, apply those; else in-sample best_quad.
    Returns 1 - RMSE_after/RMSE_before over the pooled quotes of one date."""
    rb, ra = [], []
    for x, r, key in groups:
        rb.append(r)
        c = coeffs_by_key[key] if coeffs_by_key is not None else best_quad(x, r)
        ra.append(r - np.polyval(c, x))
    rb = np.concatenate(rb); ra = np.concatenate(ra)
    return 1 - np.sqrt(np.mean(ra ** 2)) / np.sqrt(np.mean(rb ** 2))


def ridge(X, Y, lam=1.0):
    Xa = np.hstack([X, np.ones((len(X), 1))])
    W = np.linalg.solve(Xa.T @ Xa + lam * np.eye(Xa.shape[1]), Xa.T @ Y)
    return lambda Xn: np.hstack([Xn, np.ones((len(Xn), 1))]) @ W


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/m2b")
    args = ap.parse_args()
    Q = pd.read_parquet(Path(args.data_dir) / "quotes.parquet")
    F = pd.read_parquet(Path(args.data_dir) / "features.parquet")
    feats = [c for c in F.columns if c not in FEAT_EXCL]
    Q["resid"] = Q["market_iv"] - Q["heston_iv"]
    Q["bucket"] = (Q["tte_yr"] >= 7 / 365).astype(int)
    feat_by_date = {r["date"]: r[feats].to_numpy(float) for _, r in F.iterrows()}
    split_by_date = dict(zip(F["date"], F["split"]))

    # per-date grouped quotes, keyed by tenor (for A) and by bucket (for B)
    def reductions(keycol):
        out = {"train": [], "val": [], "test": []}
        for d, g in Q.groupby("date"):
            if d not in split_by_date:
                continue
            groups = [(gt["log_m"].to_numpy(), gt["resid"].to_numpy(), k)
                      for k, gt in g.groupby(keycol)]
            out[split_by_date[d]].append(date_reduction(groups))
        return out

    A = reductions("tte_yr"); B = reductions("bucket")
    print("A. per-tenor oracle  (test): %2.0f%%" % (np.median(A["test"]) * 100))
    print("B. per-bucket oracle (test): %2.0f%%   [A-B bucketing loss = %2.0f pts]"
          % (np.median(B["test"]) * 100, (np.median(A["test"]) - np.median(B["test"])) * 100))

    # C. features -> per-bucket coeffs, ridge fit on TRAIN, applied to TEST (and train, for gen gap)
    samples = {0: {"Xtr": [], "Ytr": []}, 1: {"Xtr": [], "Ytr": []}}
    per_date = {}   # date -> list of (x, r, bucket)
    for d, g in Q.groupby("date"):
        if d not in split_by_date:
            continue
        rows = []
        for b, gt in g.groupby("bucket"):
            x = gt["log_m"].to_numpy(); r = gt["resid"].to_numpy()
            rows.append((x, r, b))
            if split_by_date[d] == "train":
                samples[b]["Xtr"].append(feat_by_date[d]); samples[b]["Ytr"].append(best_quad(x, r))
        per_date[d] = rows
    predictors = {}
    for b in (0, 1):
        X = np.array(samples[b]["Xtr"]); Y = np.array(samples[b]["Ytr"])
        predictors[b] = ridge(X, Y, lam=1.0)
    def C_reduction(split):
        reds = []
        for d, rows in per_date.items():
            if split_by_date[d] != split:
                continue
            cbk = {b: predictors[b](feat_by_date[d][None, :])[0] for (_, _, b) in rows}
            groups = [(x, r, b) for (x, r, b) in rows]
            reds.append(date_reduction(groups, cbk))
        return np.median(reds) * 100
    print("C. features->coeffs ridge (train): %2.0f%%   (test): %2.0f%%   [B-C feature/lin loss, C-train vs C-test = generalization]"
          % (C_reduction("train"), C_reduction("test")))

    # optional nonlinear predictor (does nonlinearity beat ridge? -> feature-limit vs model-form)
    try:
        from sklearn.ensemble import GradientBoostingRegressor
        preds_nl = {}
        for b in (0, 1):
            X = np.array(samples[b]["Xtr"]); Y = np.array(samples[b]["Ytr"])
            preds_nl[b] = [GradientBoostingRegressor(max_depth=2, n_estimators=150).fit(X, Y[:, j]) for j in range(3)]
        reds = []
        for d, rows in per_date.items():
            if split_by_date[d] != "test":
                continue
            cbk = {b: np.array([preds_nl[b][j].predict(feat_by_date[d][None, :])[0] for j in range(3)]) for (_, _, b) in rows}
            reds.append(date_reduction([(x, r, b) for (x, r, b) in rows], cbk))
        print("D. features->coeffs nonlinear (test): %2.0f%%   [D-C = nonlinearity gain; if D~=net, features are the wall]"
              % (np.median(reds) * 100))
    except Exception as e:
        print(f"D. (sklearn not available: {e})")

    print("\nnet (M2.2 standalone) test: 60%.  Read which loss dominates -> that's the lever.")


if __name__ == "__main__":
    main()
