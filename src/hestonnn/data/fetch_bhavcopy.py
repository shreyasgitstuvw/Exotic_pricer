"""NSE F&O bhavcopy downloader — RUNS ON THE USER'S MACHINE (not the sandbox).

NSE serves EOD F&O bhavcopy from public archives. Two eras (my parser handles both schemas):
  * UDiFF (>= 2024-07-08):
      https://nsearchives.nseindia.com/content/fo/BhavCopy_NSE_FO_0_0_0_YYYYMMDD_F_0000.csv.zip
  * legacy (< 2024-07-08):
      https://archives.nseindia.com/content/historical/DERIVATIVES/YYYY/MON/foDDMONYYYYbhav.csv.zip

NSE blocks bare programmatic hits, so we (1) prime cookies by GETting the homepage with a
browser-like User-Agent, then (2) request the archive with a Referer. Results are cached to disk;
re-runs skip files already present. A 404 means a non-trading day (weekend/holiday) — skipped, not
an error. Be polite: small pause between requests, modest retries with backoff.

Usage (on your machine, where `requests` is installed and your network reaches NSE):
    from hestonnn.data.fetch_bhavcopy import fetch_many
    from hestonnn.data.sampling import weekly_dates
    dates = weekly_dates("2024-08-01", "2026-07-10", weekday=2)   # Wednesdays
    paths = fetch_many(dates, cache_dir="bhavcopy")
"""
from __future__ import annotations
from pathlib import Path
from datetime import date, datetime, timedelta
import time

UDIFF_CUTOVER = date(2024, 7, 8)
_HOME = "https://www.nseindia.com"
_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/125.0 Safari/537.36")


def _as_date(d) -> date:
    if isinstance(d, date):
        return d
    return datetime.strptime(str(d)[:10], "%Y-%m-%d").date()


def bhavcopy_url(d) -> str:
    """Archive URL for a given trading date, choosing the era automatically."""
    d = _as_date(d)
    if d >= UDIFF_CUTOVER:
        return ("https://nsearchives.nseindia.com/content/fo/"
                f"BhavCopy_NSE_FO_0_0_0_{d:%Y%m%d}_F_0000.csv.zip")
    mon = d.strftime("%b").upper()
    fname = f"fo{d.strftime('%d%b%Y').upper()}bhav.csv.zip"
    return f"https://archives.nseindia.com/content/historical/DERIVATIVES/{d:%Y}/{mon}/{fname}"


def cache_path(d, cache_dir) -> Path:
    d = _as_date(d)
    return Path(cache_dir) / f"bhavcopy_FO_{d:%Y%m%d}.csv.zip"


def make_session():
    """A cookie-primed requests session. Imported lazily so the module loads without `requests`."""
    import requests
    s = requests.Session()
    s.headers.update({
        "User-Agent": _UA,
        "Accept": "*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": f"{_HOME}/all-reports",
        "Connection": "keep-alive",
    })
    try:
        s.get(_HOME, timeout=10)                       # prime cookies
        s.get(f"{_HOME}/all-reports", timeout=10)
    except Exception:
        pass
    return s


def fetch_one(d, cache_dir="bhavcopy", session=None, retries=3, pause=1.5, timeout=30):
    """Download one date. Returns local Path, or None if the date isn't available (404).

    Skips download if a non-empty cached file already exists.
    """
    d = _as_date(d)
    dest = cache_path(d, cache_dir)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 100:
        return dest
    s = session or make_session()
    url = bhavcopy_url(d)
    last = None
    for i in range(retries):
        try:
            r = s.get(url, timeout=timeout)
        except Exception as e:                          # network hiccup -> back off
            last = e; time.sleep(pause * (i + 1)); continue
        if r.status_code == 200 and r.content[:2] == b"PK":
            dest.write_bytes(r.content)
            return dest
        if r.status_code == 404:
            return None                                 # non-trading day / not yet published
        last = f"HTTP {r.status_code}"
        time.sleep(pause * (i + 1))
    raise RuntimeError(f"failed to fetch {url}: {last}")


def fetch_many(dates, cache_dir="bhavcopy", pause=1.5, progress=True):
    """Download a list of dates politely, reusing one primed session. Returns {date: path|None}."""
    s = make_session()
    out = {}
    for d in dates:
        d = _as_date(d)
        try:
            p = fetch_one(d, cache_dir=cache_dir, session=s, pause=pause)
        except RuntimeError as e:
            p = None
            if progress:
                print(f"  [warn] {d}: {e}")
        out[d] = p
        if progress:
            tag = "cached/ok" if p else "skip(404)"
            print(f"  {d}  {tag}")
        time.sleep(pause)                               # rate-limit courtesy
    got = sum(1 for v in out.values() if v)
    if progress:
        print(f"[fetch] {got}/{len(out)} dates available")
    return out
