"""Date schedules for the bhavcopy surface set.

M1/M2 do NOT need every trading day — a sampled set (a weekly anchor + optional monthly anchor +
event windows) captures the parameter dynamics while keeping the number of files to source small.
Weekends are skipped; NSE holidays are left to the downloader (a 404 just means 'not a trading day').
"""
from __future__ import annotations
from datetime import date, datetime, timedelta


def _d(x) -> date:
    if isinstance(x, date):
        return x
    return datetime.strptime(str(x)[:10], "%Y-%m-%d").date()


def _business(d: date) -> date:
    """Nudge weekends to the preceding Friday."""
    while d.weekday() >= 5:      # 5=Sat, 6=Sun
        d -= timedelta(days=1)
    return d


def weekly_dates(start, end, weekday: int = 2) -> list:
    """One date per week on the given weekday (0=Mon..4=Fri; default Wed)."""
    start, end = _d(start), _d(end)
    d = start + timedelta((weekday - start.weekday()) % 7)
    out = []
    while d <= end:
        out.append(d)
        d += timedelta(days=7)
    return out


def monthly_dates(start, end, day: int = 15) -> list:
    """One date per month near `day`, nudged off weekends."""
    start, end = _d(start), _d(end)
    out, y, m = [], start.year, start.month
    while date(y, m, 1) <= end:
        try:
            cand = _business(date(y, m, min(day, 28)))
        except ValueError:
            cand = _business(date(y, m, 28))
        if start <= cand <= end:
            out.append(cand)
        m += 1
        if m > 12:
            m = 1; y += 1
    return out


def event_windows(events, pad_before: int = 2, pad_after: int = 2) -> list:
    """Daily dates within +/- pad of each event date (e.g., budget, RBI policy, expiry-week spikes)."""
    out = []
    for e in events:
        e = _d(e)
        for k in range(-pad_before, pad_after + 1):
            out.append(_business(e + timedelta(days=k)))
    return out


def schedule(start, end, weekly_weekday: int = 2, monthly_day: int | None = None,
             events=None, event_pad=2) -> list:
    """Union of a weekly anchor + optional monthly anchor + optional event windows, deduped/sorted."""
    dates = set(weekly_dates(start, end, weekly_weekday))
    if monthly_day is not None:
        dates |= set(monthly_dates(start, end, monthly_day))
    if events:
        dates |= set(event_windows(events, event_pad, event_pad))
    return sorted(d for d in dates if d.weekday() < 5)
