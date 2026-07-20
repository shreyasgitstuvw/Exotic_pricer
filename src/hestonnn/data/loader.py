"""Discovery + loading of the raw 1-min option strips, futures, and VIX."""
from __future__ import annotations
from pathlib import Path
import glob
import pandas as pd

OPT_COLS = ["date", "time", "Strike", "Spot",
            "CE_Close", "PE_Close", "CE_Volume", "PE_Volume", "CE_OI", "PE_OI"]


def list_week_files(history_dir: str | Path) -> list[Path]:
    files = sorted(glob.glob(str(Path(history_dir) / "*" / "*" / "*.parquet")))
    return [Path(f) for f in files]


def load_week(path: str | Path, columns=OPT_COLS) -> pd.DataFrame:
    df = pd.read_parquet(path, columns=columns)
    # normalize date -> datetime.date; time stays 'HH:MM' string
    df["dt"] = pd.to_datetime(df["date"], format="%d-%m-%Y", errors="coerce").dt.date
    return df


def close_window_slice(day: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Rows whose 'time' (HH:MM) falls in [start, end] inclusive."""
    t = day["time"].astype(str)
    return day[(t >= start) & (t <= end)]


def load_futures(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["ts"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
    df["dt"] = df["ts"].dt.tz_convert("Asia/Kolkata").dt.date
    df["hm"] = df["ts"].dt.tz_convert("Asia/Kolkata").dt.strftime("%H:%M")
    return df


def load_vix(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["dt"] = pd.to_datetime(df["date"], format="%d/%m/%y", errors="coerce").dt.date
    df["hm"] = df["time"].astype(str).str.slice(0, 5)
    return df
