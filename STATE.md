# STATE — P2 Heston-NN
Updated: Jul 13, 2026 (supervision + documentation session)

## Milestone: M1 — AHEAD of schedule (executing in W1; plan said W3–4)
M0 short-end gate: **PASS** (1,016 clean smiles 2021–2026, 1.00% arb-flagged vs <2% bar, tests green).
M0 full-surface layer: built + validated; 113 real bhavcopy files downloaded (2024-07→2026-07).

## What works now
- Data layer: 1-min strip → EOD short-end smiles (config-driven, empirical expiry detection, PCP forwards, own IV inversion, butterfly/vertical/calendar arb checks) + bhavcopy multi-tenor surface (both NSE schemas auto-detected) + NSE auto-downloader with holiday handling.
- `heston_ref.py` (validated CF pricer, ~0.5 bps vs Black-76 limit) + `calibrate.py` (vega-weighted DE→L-BFGS-B, round-trip validated).
- M1 core finding (D6): naive full-surface calibration bound-slams 39% of dates — single Heston structurally cannot fit NIFTY's <14d skew. Amended protocol (fit ≥14d, measure `short_gap_bps`): median 34 bps fit, 0 bound-slams, **short-end Heston gap median 207 bps** on 17-date validation subset. That gap is M2's target.

## Exact next actions
1. **Shreyas, local (~25 min):** full 113-date M1 run — `python experiments/run_m1_instability.py --source bhavcopy --cache-dir bhavcopy --config configs/data.yaml --out-dir data/m1` (≥14d default; resumable). Then EXP-001b gets its final verdict.
2. **Provenance (blocker before anything goes public):** write `data/README.md` documenting HISTORY_NIFTY's source/license. If it's employer-derived, the public repo ships bhavcopy-only (FRAMEWORK §5).
3. **D7 decision needed:** define the test lockbox for the two-layer reality (proposal: bhavcopy dates 2026-04-10→07-10 barred from all M2 training/model-selection; descriptive M1 exempt).
4. Pin Heston-engine SHA in `engine.py`; reconcile IV conventions (README open item).
5. Freeze EXP-002 spec (incl. gap-closure metric) BEFORE the first line of NN training code.
6. Decide P2's public home (fold into Heston-engine vs new repo) before the W4 post links out.

## Open questions
- M2 gap-closure metric: e.g., median short-end gap reduction ≥50% with belly degradation ≤5 bps — to be frozen in EXP-002 spec.
- Long-tenor (>1y) smile thinness: per-bucket liquidity selection before those tenors are calibration-grade.

## Content hooks (Sunday review reads this)
- "39% of my Heston calibrations were slamming into bounds — the optimizer wasn't broken, the model was": pull post #2 ("Your Heston calibration is lying to you") forward — the numbers exist NOW.
- 207 bps: "single Heston mis-prices NIFTY's <2-week skew by ~2 vol points" — tweet-sized, sourced from data/m1/.
- Expiry-shift caught again in P2's own data (Wed 10-Mar-2021) — corroborates post #1.
