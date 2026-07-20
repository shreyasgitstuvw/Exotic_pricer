"""Empirical expiry detection (D3).

There is no expiry column, and the expiry weekday shifted at least once in-sample (e.g. an
observed Wed expiry where Thursday was the norm), so expiry MUST be inferred from the data, not
assumed from the calendar. Method: a trading day is an expiry if the front at-the-money straddle's
*time value* collapses to ~0 at the close. From the set of expiry dates we map every trading date to
its front expiry and a time-to-expiry (year fraction).
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252.0
CALENDAR_DAYS_PER_YEAR = 365.0


@dataclass
class ExpiryMap:
    expiries: list          # sorted list of datetime.date that are expiry days
    per_day: pd.DataFrame   # index=date, cols: front_expiry, tte_cal, tte_yr

    def tte(self, d):
        return float(self.per_day.loc[d, "tte_yr"])

    def front_expiry(self, d):
        return self.per_day.loc[d, "front_expiry"]


def _atm_time_value(day_close: pd.DataFrame) -> float:
    """Time value of the ATM straddle at the close snapshot of one trading day."""
    spot = float(day_close["Spot"].iloc[0])
    i = (day_close["Strike"] - spot).abs().values.argmin()
    row = day_close.iloc[i]
    k = float(row["Strike"])
    ce, pe = float(row["CE_Close"]), float(row["PE_Close"])
    intrinsic = max(spot - k, 0.0) + max(k - spot, 0.0)
    return (ce + pe) - intrinsic


def day_time_values(week_frames, close_start, close_end) -> pd.DataFrame:
    """One row per trading date: its ATM straddle time value at the close window's last minute."""
    from .loader import close_window_slice
    rows = []
    for df in week_frames:
        for d, day in df.groupby("dt"):
            if pd.isna(d):
                continue
            win = close_window_slice(day, close_start, close_end)
            if win.empty:
                continue
            last_t = win["time"].max()
            snap = win[win["time"] == last_t]
            rows.append((d, _atm_time_value(snap)))
    tv = pd.DataFrame(rows, columns=["dt", "atm_time_value"]).drop_duplicates("dt")
    return tv.sort_values("dt").reset_index(drop=True)


def detect(tv: pd.DataFrame, threshold: float, day_count="calendar") -> ExpiryMap:
    """Given per-day ATM time values, flag expiries and build the date->front-expiry map."""
    tv = tv.sort_values("dt").reset_index(drop=True)
    exp = sorted(tv.loc[tv["atm_time_value"] < threshold, "dt"].tolist())
    dpy = CALENDAR_DAYS_PER_YEAR if day_count == "calendar" else TRADING_DAYS_PER_YEAR

    fronts, ttes_cal, ttes_yr = [], [], []
    exp_arr = np.array(exp)
    for d in tv["dt"]:
        future = exp_arr[exp_arr >= d]
        if len(future) == 0:
            fronts.append(pd.NaT); ttes_cal.append(np.nan); ttes_yr.append(np.nan); continue
        fe = future[0]
        ncal = (fe - d).days
        fronts.append(fe)
        ttes_cal.append(ncal)
        # floor tte at a fraction of a day so expiry-day options aren't T=0 exactly
        ttes_yr.append(max(ncal, 0.25) / dpy)
    per = pd.DataFrame(
        {"front_expiry": fronts, "tte_cal": ttes_cal, "tte_yr": ttes_yr},
        index=tv["dt"].values,
    )
    return ExpiryMap(expiries=exp, per_day=per)
