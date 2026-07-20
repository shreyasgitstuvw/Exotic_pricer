"""Assemble one clean EOD short-end smile per trading date from the 1-min front-contract strip.

Pipeline per date:
  1. take the close-window (D1) marks, median across minutes per strike (robust to single-print noise)
  2. imply forward F and discount factor df from put-call parity across liquid strikes (forward.py)
  3. recompute IV per strike from the OTM side (calls for K>=F, puts for K<F) via Black-76 (iv.py)
  4. clip by moneyness / price / IV bounds; flag intra-smile arbitrage (arbitrage.py)

Output: a tidy DataFrame of clean quotes plus a one-row summary for the manifest.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

from .loader import close_window_slice
from .forward import implied_forward_df
from .iv import implied_vol
from .arbitrage import flag_smile


@dataclass
class SmileResult:
    quotes: pd.DataFrame
    summary: dict
    ok: bool


def _close_marks(day: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    win = close_window_slice(day, start, end)
    if win.empty:
        return win
    g = win.groupby("Strike").agg(
        CE_Close=("CE_Close", "median"),
        PE_Close=("PE_Close", "median"),
        CE_Volume=("CE_Volume", "sum"),
        PE_Volume=("PE_Volume", "sum"),
        Spot=("Spot", "median"),
    ).reset_index()
    return g


def build_smile(day, date, tte_yr, cfg_surf, close_start, close_end):
    s = cfg_surf
    summ = {"date": date, "tte_yr": tte_yr, "n_raw": 0, "n_clean": 0,
            "F": np.nan, "df": np.nan, "r": np.nan, "atm_iv": np.nan,
            "arb_frac": np.nan, "ok": False, "reject": ""}

    marks = _close_marks(day, close_start, close_end)
    if marks.empty or not np.isfinite(tte_yr) or tte_yr <= 0:
        summ["reject"] = "no_close_marks_or_tte"
        return SmileResult(marks, summ, False)
    if tte_yr * 365.0 < s.get("min_tte_days", 0.0):
        summ["reject"] = "expiry_day_0dte"
        return SmileResult(marks.iloc[0:0], summ, False)
    summ["n_raw"] = len(marks)

    liq = marks[(marks["CE_Close"] >= s["min_price"]) & (marks["PE_Close"] >= s["min_price"])].copy()
    spot = float(marks["Spot"].median())
    liq["w"] = 1.0 / (1.0 + ((liq["Strike"] - spot) / spot * 20) ** 2)
    F, df, r, n_used = implied_forward_df(liq["Strike"], liq["CE_Close"], liq["PE_Close"],
                                          tte_yr, weights=liq["w"])
    if not np.isfinite(F):
        summ["reject"] = "forward_fit_failed"
        return SmileResult(marks.iloc[0:0], summ, False)
    summ["F"], summ["df"], summ["r"] = F, df, r

    lo, hi = s["moneyness_clip"]
    q = marks.copy()
    q["K"] = q["Strike"].astype(float)
    q["m"] = q["K"] / F
    q = q[(q["m"] >= lo) & (q["m"] <= hi)]
    q["use_call"] = q["K"] >= F
    q["mkt_px"] = np.where(q["use_call"], q["CE_Close"], q["PE_Close"])
    q = q[q["mkt_px"] >= s["min_price"]]
    if q.empty:
        summ["reject"] = "no_quotes_after_clip"
        return SmileResult(q, summ, False)

    q["iv"] = implied_vol(q["mkt_px"].to_numpy(), F, q["K"].to_numpy(), tte_yr,
                          df=df, call=q["use_call"].to_numpy())
    lob, hib = s["iv_bounds"]
    q = q[np.isfinite(q["iv"]) & (q["iv"] >= lob) & (q["iv"] <= hib)]
    if len(q) < s["min_strikes"]:
        summ["reject"] = "too_few_clean_strikes_%d" % len(q)
        return SmileResult(q, summ, False)

    q["call_px"] = np.where(q["use_call"], q["mkt_px"], q["mkt_px"] + df * (F - q["K"]))
    q["df"] = df
    q = flag_smile(q, price_col="call_px", strike_col="K", df_col="df")

    q["date"] = date
    q["tte_yr"] = tte_yr
    q["log_m"] = np.log(q["m"])
    q["F"] = F
    summ["n_clean"] = len(q)
    summ["arb_frac"] = float(q["arb_any"].mean())
    iatm = (q["K"] - F).abs().values.argmin()
    summ["atm_iv"] = float(q["iv"].iloc[iatm])
    summ["ok"] = True
    cols = ["date", "tte_yr", "K", "m", "log_m", "iv", "call_px", "F", "df",
            "use_call", "arb_vertical", "arb_butterfly", "arb_any"]
    return SmileResult(q[cols].reset_index(drop=True), summ, True)
