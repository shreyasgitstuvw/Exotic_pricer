# `data/` — provenance

This directory is **not committed** (see repo `.gitignore`: `data/`, `*.parquet`, `*.npz`, `*.pt`,
`HISTORY_NIFTY/`, `bhavcopy/`, `*.csv`). Nothing here ships in the public repo. This file documents
where it comes from and how to regenerate it, so a reader (or an interviewer) can reproduce every
number in `README.md` / `PROGRESS.md` from source.

## Sources

| Input | What it is | How it's obtained | License / redistribution |
|---|---|---|---|
| NSE EOD F&O bhavcopy | Official end-of-day derivatives bhavcopy (all NIFTY strikes/expiries/tenors) | Auto-downloaded from NSE's public archive by `fetch_bhavcopy_data.py` (dual-format: legacy + new UDiFF) | NSE publishes bhavcopy files publicly on nseindia.com for EOD reporting. **CONFIRM before any public claim**: re-check NSE's current terms-of-use page for redistribution limits — this project only ever *derives* aggregates (manifest, parameter series, plots) and never redistributes raw bhavcopy files, but confirm that's sufficient. |
| `HISTORY_NIFTY/` | NIFTY option chain, 1-minute resolution, 2021–2026 (weekly parquet, ~375 intraday stamps/day) | **CONFIRM: where did this come from?** If it's Orion Quant desk data (employer-derived) it must **never** be described as self-collected, and per `FRAMEWORK.md` §5 the public repo must ship bhavcopy-only with `HISTORY_NIFTY` excluded — which `.gitignore` already does today (verified: not tracked in git history). If it's a paid vendor feed, name the vendor here instead. |
| `NIFTY50_FUTURES.csv` | 1-minute futures OHLCV, 2019– | Same source as `HISTORY_NIFTY` — **same CONFIRM applies.** | — |
| `India_VIX.csv` | 1-minute India VIX, 2017– | Same source as `HISTORY_NIFTY` — **same CONFIRM applies.** Not used as a model feature (rejected D-note: series ends 2025-11, no test coverage). | — |

**Until the CONFIRM lines above are resolved, do not state a specific data-acquisition story in an
interview beyond "NSE public bhavcopy for EOD surfaces; the 1-minute option-chain history comes from
[source TBC]."** The governance rule this file exists to satisfy (`PROGRESS.md` gap #4) is precisely
that this can't stay unanswered before anything ships further in public.

## What's actually in this folder (regenerated, not hand-edited)

| Path | Produced by | Rows / size | SHA-256 (first 16 hex) |
|---|---|---|---|
| `manifest.csv` | `experiments/run_m1_instability.py` (M0 hygiene scan) | 1,238 dates scanned, **1,016 pass hygiene** (`ok=True`); range 2021-02-08 → 2026-02-11 | `84e562486c2f8715` |
| `surfaces/short_end_smiles.parquet` | surface assembly (M0/M1) | 796 KB | `79914c955015604c` |
| `m1/`, `m2/`, `m2b/` | `run_m1_instability.py`, `m2_dataset.py`, `m2b_dataset.py` | see `REGISTRY.md` for per-run row counts once reconciled (open item, tracked in `PROGRESS.md`) | regenerate, don't hand-check |

**Note on the "1,016" vs "355" discrepancy:** `manifest.csv` (M1's per-date hygiene scan) shows 1,016
clean dates; `PROGRESS.md`'s M0 gate line currently reads "355 surfaces 2020–26." These are tracked
as separate pipeline stages (raw-date hygiene vs. final multi-tenor surface count) and the exact
relationship between them is an open reconciliation item already flagged in `PROGRESS.md` /
`REGISTRY.md` — don't quote both numbers as interchangeable until that's closed.

## Reproducing this folder from scratch

```bash
python fetch_bhavcopy_data.py --start 2020-01-01 --end 2026-07-05   # NSE bhavcopy, public, ~290 files
python experiments/run_m1_instability.py --source bhavcopy --cache-dir bhavcopy \
    --config configs/data.yaml --out-dir data/m1 --min-tte-days 14
```

Checksums above are for the exact files this document was generated against
(2026-07-20). Re-run `sha256sum data/manifest.csv data/surfaces/short_end_smiles.parquet` after any
regeneration and update this table — treat a checksum mismatch with no matching commit message as a
signal something upstream changed silently.
