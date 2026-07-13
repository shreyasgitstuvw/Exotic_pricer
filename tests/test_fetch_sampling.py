import sys
from pathlib import Path
from datetime import date
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hestonnn.data.fetch_bhavcopy import bhavcopy_url, cache_path, UDIFF_CUTOVER
from hestonnn.data import sampling


def test_url_udiff_era():
    u = bhavcopy_url("2026-07-10")
    assert u == ("https://nsearchives.nseindia.com/content/fo/"
                 "BhavCopy_NSE_FO_0_0_0_20260710_F_0000.csv.zip")


def test_url_legacy_era():
    u = bhavcopy_url("2021-02-11")
    assert "archives.nseindia.com/content/historical/DERIVATIVES/2021/FEB/" in u
    assert u.endswith("fo11FEB2021bhav.csv.zip")


def test_cutover_boundary():
    assert bhavcopy_url(UDIFF_CUTOVER).startswith("https://nsearchives")
    before = date(2024, 7, 5)
    assert "historical" in bhavcopy_url(before)


def test_cache_path_naming():
    assert cache_path("2026-07-10", "bc").name == "bhavcopy_FO_20260710.csv.zip"


def test_weekly_sampler_weekday_and_bounds():
    ds = sampling.weekly_dates("2025-01-01", "2025-03-31", weekday=2)  # Wednesdays
    assert all(d.weekday() == 2 for d in ds)
    assert ds[0] >= date(2025, 1, 1) and ds[-1] <= date(2025, 3, 31)


def test_schedule_dedup_no_weekends():
    s = sampling.schedule("2025-01-01", "2025-06-30", weekly_weekday=2,
                          monthly_day=15, events=["2025-02-01"], event_pad=1)
    assert s == sorted(set(s))                     # deduped + sorted
    assert all(d.weekday() < 5 for d in s)         # no weekends
