"""NSE EOD F&O bhavcopy ingestion -> multi-expiry surface (D3 Option A).

Handles BOTH schemas via header auto-detection (crosswalk from NSE's format readme):
  new UDiFF: TradDt, FinInstrmTp, TckrSymb, XpryDt, StrkPric, OptnTp, ClsPric, SttlmPric,
             UndrlygPric, OpnIntrst, TtlTradgVol
  old      : TIMESTAMP, INSTRUMENT, SYMBOL, EXPIRY_DT, STRIKE_PR, OPTION_TYP, CLOSE, SETTLE_PR,
             OPEN_INT, CONTRACTS  (underlying not present -> implied later from PCP)

A single bhavcopy day already contains the full listed term structure (NIFTY: ~4d to ~5y), so one
file per date yields a real maturity x moneyness surface. Settlement price preferred; close as
fallback. Output is one canonical tidy frame that flows through the SAME forward/iv/arbitrage
machinery, grouped by expiry.
"""
from __future__ import annotations
from pathlib import Path
import zipfile
import io
import numpy as np
import pandas as pd

# canonical column -> (new-format header, old-format header)
_CROSSWALK = {
    "trad_dt":     ("TradDt", "TIMESTAMP"),
    "instr_tp":    ("FinInstrmTp", "INSTRUMENT"),
    "symbol":      ("TckrSymb", "SYMBOL"),
    "expiry":      ("XpryDt", "EXPIRY_DT"),
    "K":           ("StrkPric", "STRIKE_PR"),
    "right":       ("OptnTp", "OPTION_TYP"),
    "close_px":    ("ClsPric", "CLOSE"),
    "settle_px":   ("SttlmPric", "SETTLE_PR"),
    "underlying":  ("UndrlygPric", None),
    "oi":          ("OpnIntrst", "OPEN_INT"),
    "volume":      ("TtlTradgVol", "CONTRACTS"),
}
# instrument-type tokens that mean "index option" in each schema
_INDEX_OPTION_TOKENS = {"IDO", "OPTIDX"}


def _read_any(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if p.suffix == ".zip":
        with zipfile.ZipFile(p) as z:
            name = [n for n in z.namelist() if n.lower().endswith(".csv")][0]
            with z.open(name) as f:
                return pd.read_csv(io.BytesIO(f.read()))
    return pd.read_csv(p)


def detect_format(cols) -> str:
    cols = set(cols)
    if "FinInstrmTp" in cols or "TckrSymb" in cols:
        return "new"
    if "INSTRUMENT" in cols or "SYMBOL" in cols:
        return "old"
    raise ValueError(f"Unrecognized bhavcopy schema; columns={sorted(cols)[:12]}...")


def load_bhavcopy_day(path: str | Path, symbol: str = "NIFTY") -> pd.DataFrame:
    """Parse one bhavcopy file into canonical tidy option rows for `symbol`.

    Returns columns: [date, expiry, tte_yr, K, right, px, oi, volume, underlying].
    `px` is settlement where available, else close. TTE in calendar-day year fraction.
    """
    raw = _read_any(path)
    fmt = detect_format(raw.columns)
    idx = 0 if fmt == "new" else 1

    def col(canon):
        header = _CROSSWALK[canon][idx]
        return raw[header] if header and header in raw.columns else None

    out = pd.DataFrame()
    for canon in ["trad_dt", "instr_tp", "symbol", "expiry", "K", "right",
                  "close_px", "settle_px", "underlying", "oi", "volume"]:
        s = col(canon)
        out[canon] = s if s is not None else np.nan

    # filter to index options for the requested symbol
    out["symbol"] = out["symbol"].astype(str).str.strip()
    out["instr_tp"] = out["instr_tp"].astype(str).str.strip()
    out["right"] = out["right"].astype(str).str.strip().str.upper()
    m = (out["symbol"] == symbol) & (out["instr_tp"].isin(_INDEX_OPTION_TOKENS)) \
        & (out["right"].isin(["CE", "PE"]))
    out = out[m].copy()
    if out.empty:
        raise ValueError(f"No {symbol} index options found (format={fmt}).")

    out["date"] = pd.to_datetime(out["trad_dt"], errors="coerce").dt.date
    out["expiry"] = pd.to_datetime(out["expiry"], errors="coerce").dt.date
    out["K"] = pd.to_numeric(out["K"], errors="coerce")
    for c in ["close_px", "settle_px", "oi", "volume", "underlying"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    # settlement preferred, close fallback
    out["px"] = out["settle_px"].where(out["settle_px"] > 0, out["close_px"])

    td = pd.to_datetime(out["date"].iloc[0])
    out["tte_yr"] = (pd.to_datetime(out["expiry"]) - td).dt.days.clip(lower=0) / 365.0

    keep = ["date", "expiry", "tte_yr", "K", "right", "px", "oi", "volume", "underlying"]
    return out[keep].dropna(subset=["expiry", "K", "px", "right"]).reset_index(drop=True)


def usable_expiries(day: pd.DataFrame, min_strikes_oi: int = 7) -> list:
    """Expiries with at least `min_strikes_oi` strikes carrying open interest (liquidity gate)."""
    g = day[day["oi"] > 0].groupby("expiry")["K"].nunique()
    return sorted(g[g >= min_strikes_oi].index.tolist())
