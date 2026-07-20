"""Assemble a maturity x moneyness surface from one bhavcopy day (D3-A).

Groups canonical bhavcopy rows by expiry and runs the SAME per-smile machinery used for the
short-end layer (put-call-parity forward, Black-76 IV recompute, intra-smile arb flags), then
stacks the tenors into one surface with a per-tenor summary. Calendar-arb across tenors is checked
on the total-variance (w = iv^2 * T) monotonicity at matched moneyness.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from .forward import implied_forward_df
from .iv import implied_vol
from .arbitrage import flag_smile


def _smile_from_expiry(sub: pd.DataFrame, tte_yr: float, cfg_surf: dict):
    """sub: canonical rows for a single (date, expiry). Returns (quotes_df, summary)."""
    s = cfg_surf
    liq = sub[sub["oi"] > 0] if "oi" in sub.columns and (sub["oi"] > 0).any() else sub
    piv = liq.pivot_table(index="K", columns="right", values="px", aggfunc="last")
    piv = piv.dropna(subset=[c for c in ["CE", "PE"] if c in piv.columns])
    if not {"CE", "PE"}.issubset(piv.columns) or len(piv) < 3:
        return None, {"tte_yr": tte_yr, "ok": False, "reject": "no_ce_pe_pairs"}
    K = piv.index.to_numpy(float)
    C = piv["CE"].to_numpy(float); P = piv["PE"].to_numpy(float)

    # forward from PCP, ATM-weighted
    spot = float(np.median(K))
    w = 1.0 / (1.0 + ((K - spot) / spot * 20) ** 2)
    F, df, r, n = implied_forward_df(K, C, P, tte_yr, weights=w)
    summ = {"tte_yr": tte_yr, "F": F, "df": df, "r": r, "n_clean": 0,
            "atm_iv": np.nan, "arb_frac": np.nan, "ok": False, "reject": ""}
    if not np.isfinite(F):
        summ["reject"] = "forward_fit_failed"; return None, summ

    q = pd.DataFrame({"K": K})
    q["m"] = q["K"] / F
    lo, hi = s["moneyness_clip"]
    q = q[(q["m"] >= lo) & (q["m"] <= hi)].copy()
    q["use_call"] = q["K"] >= F
    pxmap = {k: (c, p) for k, c, p in zip(K, C, P)}
    q["mkt_px"] = [pxmap[k][0] if uc else pxmap[k][1] for k, uc in zip(q["K"], q["use_call"])]
    q = q[q["mkt_px"] >= s["min_price"]]
    if q.empty:
        summ["reject"] = "no_quotes_after_clip"; return None, summ
    q["iv"] = implied_vol(q["mkt_px"].to_numpy(), F, q["K"].to_numpy(), tte_yr,
                          df=df, call=q["use_call"].to_numpy())
    lob, hib = s["iv_bounds"]
    q = q[np.isfinite(q["iv"]) & (q["iv"] >= lob) & (q["iv"] <= hib)]
    if len(q) < s["min_strikes"]:
        summ["reject"] = "too_few_clean_strikes_%d" % len(q); return None, summ

    q["call_px"] = np.where(q["use_call"], q["mkt_px"], q["mkt_px"] + df * (F - q["K"]))
    q["df"] = df
    q = flag_smile(q, price_col="call_px", strike_col="K", df_col="df")
    q["tte_yr"] = tte_yr
    q["log_m"] = np.log(q["m"])
    q["F"] = F
    q["w_totvar"] = q["iv"] ** 2 * tte_yr
    iatm = (q["K"] - F).abs().values.argmin()
    summ.update(n_clean=len(q), atm_iv=float(q["iv"].iloc[iatm]),
                arb_frac=float(q["arb_any"].mean()), ok=True)
    return q.reset_index(drop=True), summ


def build_surface(day: pd.DataFrame, expiries, cfg_surf: dict):
    """day: canonical bhavcopy frame. expiries: list to include. Returns (surface_df, tenor_summary)."""
    smiles, rows = [], []
    for e in expiries:
        sub = day[day["expiry"] == e]
        tte = float(sub["tte_yr"].iloc[0])
        q, summ = _smile_from_expiry(sub, tte, cfg_surf)
        summ["expiry"] = e
        rows.append(summ)
        if q is not None:
            q["expiry"] = e
            smiles.append(q)
    surface = pd.concat(smiles, ignore_index=True) if smiles else pd.DataFrame()
    tenor = pd.DataFrame(rows).sort_values("tte_yr").reset_index(drop=True)

    # calendar-arb flag: ATM total variance must be non-decreasing in T
    ok = tenor[tenor["ok"]].sort_values("tte_yr").copy()
    ok["atm_totvar"] = ok["atm_iv"] ** 2 * ok["tte_yr"]
    ok["calendar_arb"] = ok["atm_totvar"].diff() < -1e-8
    tenor = tenor.merge(ok[["expiry", "atm_totvar", "calendar_arb"]], on="expiry", how="left")
    return surface, tenor


def calibration_quotes(surface, n_sd: float = 1.5, min_per_tenor: int = 5):
    """Tenor-adaptive cleanup for calibration: keep strikes within ATM +/- n_sd * atm_iv * sqrt(T)
    (a fixed moneyness band is too wide for short tenors, too tight for long ones), drop
    arb-flagged quotes, and require a minimum strike count per tenor. Returns a filtered copy.
    """
    import numpy as np
    keep = []
    for e, sub in surface.groupby("expiry"):
        s = sub[~sub["arb_any"]].copy()
        if s.empty:
            continue
        T = float(s["tte_yr"].iloc[0])
        iatm = (s["m"] - 1.0).abs().values.argmin()
        atm_iv = float(s["iv"].iloc[iatm])
        band = max(n_sd * atm_iv * np.sqrt(max(T, 1e-6)), 0.02)   # min 2% log-moneyness
        s = s[s["log_m"].abs() <= band]
        if len(s) >= min_per_tenor:
            keep.append(s)
    import pandas as pd
    return pd.concat(keep, ignore_index=True) if keep else surface.iloc[0:0]
