# Heston + Neural Components for NIFTY Exotic Pricing

Static Heston parameters are a solvability compromise, not a market fact. This project measures — on
six years of real NSE data — **how wrong the resulting short-dated NIFTY option prices are, and
where**, then builds an arbitrage-free correction and shows its consequences for exotic pricing.

It deliberately spans the full stack a desk uses: classical calibration, a neural surrogate, a
learned no-arb correction, and a stochastic-local-volatility (SLV) engine — and, importantly, it
documents where the neural approach won and where a simpler classical method won.

---

## Headline results (all on held-out / converged footing)

| stage | what it does | result |
|---|---|---|
| **M0** data | auto-downloaded, arb-checked NIFTY surfaces, 2020–2026 | 355 clean daily surfaces; dual-format bhavcopy parser |
| **M1** instability | daily full-Heston calibration; measure the short-end miss | params unstable (κ p95 |Δ|≈13); **short-end gap ≈165 bps**, regime-dependent (43→310 bps by quarter) |
| **M2.1** surrogate | NN: surface features → (κ, σ_v, ρ) | reprices within **17 bps** of the full optimiser, **~10⁵–10⁶× faster** |
| **M2.2** gap-closer | Heston backbone + short-end correction + arb-projection | **deployable smoother closes 87%** of the gap (165→~25 bps), **0 arbitrage**; the neural predictor tops out at 60% (kept as a predictability study) |
| **M3** exotics | price exotics: raw Heston vs corrected model | digitals mis-priced up to **15%** OTM; synthetic VIX biased; **SLV** (Gyöngy-validated, 11 bps) shows Heston **over-prices a weekly barrier by ~9%** |

---

## The idea, in one paragraph

Calibrate Heston to the liquid ≥2-week NIFTY surface — where it fits well (~30 bps) — and it
systematically mis-prices the sub-2-week smile, because Heston's skew decays too fast toward expiry.
That miss (~165 bps of IV, up to ~600 in stress) is real, regime-dependent, and it propagates into
any exotic priced off the short end. We correct the short-end smile (arbitrage-free), then quantify
the exotic-pricing consequences — digitals, a synthetic VIX, and, via a stochastic-local-vol model, a
path-dependent barrier.

---

## Pipeline

- **M0 — data.** `HISTORY_NIFTY` 1-min strips + NSE EOD F&O **bhavcopy** (auto-downloaded), parsed by a
  dual-format reader (new UDiFF + legacy). One bhavcopy day yields the full listed term structure;
  surfaces are built at a fixed EOD stamp, IV recomputed via put-call-parity forward + Black-76, and
  arbitrage-flagged. Empirical expiry detection (the NIFTY expiry weekday shifted in-sample).
- **M1 — instability exhibit.** Full Heston (DE→L-BFGS-B, vega-weighted IV RMSE) calibrated to the
  ≥14-day surface each date; the sub-14-day miss ("short-end gap") measured separately. Produces the
  parameter time series and the money-plot. *No neural network here — classical calibration.*
- **M2 — neural components.**
  - **M2.1 surrogate**: a small constrained MLP maps 17 surface features → (κ, σ_v, ρ); calibration
    becomes one forward pass, graded on *repricing* (not parameter error — κ is the sloppy direction).
  - **M2.2 gap-closer**: the deliverable is a **direct per-tenor smoother** — fit the observed
    short-end residual (market − Heston) and project to the arbitrage-free cone (87%). A neural
    predictor of that correction was built too (60%) and kept as a *predictability study*; a
    two-headed unification was tested and **rejected** (negative transfer). A ceiling decomposition
    proved the features cap any predictor near 60% and that direct observation bypasses the wall.
- **M3 — exotic-pricing payoff.**
  - *Terminal* payoffs (digitals, variance swap / synthetic VIX) priced directly from the smile.
  - *Path-dependent* payoffs need dynamics → **Heston-SLV** via the particle method (leverage
    L(F,t)=σ_Dupire/√E[v|F]); validated by the Gyöngy identity, then MC a weekly barrier: Heston vs SLV.

---

## Repo layout

```
src/hestonnn/
  config.py                 YAML-driven config
  data/                     market-data layer
    bhavcopy.py             dual-format NSE bhavcopy parser
    fetch_bhavcopy.py       NSE downloader (cookie-primed, both eras)
    sampling.py             trading-date schedules
    surface_bhav.py         multi-tenor surface assembly + hygiene
    surface.py, expiry.py, forward.py, iv.py, arbitrage.py, loader.py
  heston_ref.py             reference Heston CF pricer (Lewis integral)
  calibrate.py              full + constrained Heston calibration
  features.py               surface -> feature vectors (long-end + short-end)
  surrogate.py              M2.1 calibration-surrogate net
  gap_closer.py             M2.2 correction net + arb-projection (project_arbfree)
  twohead.py                two-headed net (tested/rejected)
  lsv.py                    Heston-SLV particle method + MC pricers
  engine.py                 adapter seam to the frozen Heston engine (pin instructions)
experiments/                runnable entry points (see "How to run")
fetch_bhavcopy_data.py      one-shot data downloader (run on your machine)
configs/data.yaml           paths, close window, hygiene thresholds
tests/                      pytest (IV round-trip, PCP, arb flags, calibrator round-trip, ...)
docs: PLAN.md BENCHMARKS.md M2_PLAN.md DECISIONS.md M1_FINDINGS.md M2_FINDINGS.md M3_FINDINGS.md STATUS.md
```

---

## Setup

```
pip install -e .            # numpy, pandas, pyarrow, scipy, pyyaml
pip install torch requests  # torch for M2 training; requests for the NSE downloader
pytest -q                   # sanity: pricer/IV/arb/calibrator checks
```

Point `configs/data.yaml:data_root` at the folder holding `HISTORY_NIFTY/`, `NIFTY50_FUTURES.csv`,
`India_VIX.csv`.

---

## How to run (end to end)

```bash
# 0. data — download NSE bhavcopy (your machine; needs `requests`)
python fetch_bhavcopy_data.py --start 2020-01-01 --end 2026-07-05   # ~290 weekly files into ./bhavcopy/

# 1. M1 — Heston instability + short-end gap (parameter time series + money-plot)
python experiments/run_m1_instability.py --source bhavcopy --cache-dir bhavcopy \
    --config configs/data.yaml --out-dir data/m1 --min-tte-days 14

# 2. M2.1 — calibration surrogate
python experiments/m2_dataset.py --params data/m1/params_timeseries_bhavcopy.csv --cache-dir bhavcopy
python experiments/train_surrogate.py --data data/m2/dataset.npz --params data/m1/params_timeseries_bhavcopy.csv --cache-dir bhavcopy

# 3. M2.2 — short-end correction (deployable smoother + the neural study)
python experiments/m2b_dataset.py   --params data/m1/params_timeseries_bhavcopy.csv --cache-dir bhavcopy
python experiments/direct_smoother.py --data-dir data/m2b        # the 87% deployable layer
python experiments/train_gap_closer.py --data-dir data/m2b       # the neural predictor (60%) study
python experiments/diagnose_ceiling.py --data-dir data/m2b       # the 60-vs-87 decomposition

# 4. M3 — exotic mispricing
python experiments/m3_exotics.py   --data-dir data/m2b           # digitals
python experiments/m3_varswap.py   --data-dir data/m2b           # synthetic VIX / variance swap
python experiments/m3_barrier_lsv.py --paths 120000 --steps 1500 # LSV barrier (calib must PASS <50 bps)
```

---

## Key findings & honest limitations

- **Heston's short end is a separate regime.** The correction is only ~16% predictable from the
  long-end surface and ~60% from market-state features — the NIFTY weekly smile carries largely
  independent information (event / expiry-pinning risk). The deployable layer therefore *observes* the
  short end and fits it (87%) rather than predicting it.
- **When not to use ML.** A ceiling decomposition showed the neural predictor was already near the
  feature ceiling; a deterministic per-tenor fit + arb-projection reached the 87% oracle. The network
  is retained as a scientific study, not shipped.
- **Documented negatives:** COVID-crash dates excluded (Heston can't fit an inverted crash term
  structure); India VIX rejected as a feature (series ends 2025-11, no test coverage); two-headed
  unification rejected (negative transfer).
- **Data caveat:** `HISTORY_NIFTY` is front-contract only (≤22 days); the maturity axis comes from
  bhavcopy. The M3 LSV barrier uses a synthetic corrected smile by default — swap in the real
  M2-corrected smiles (densified with Heston tenors) for per-date barrier mispricing.
- **Not claimed:** no live-trading or "beats-the-market" claim; the correction is a market-consistent
  no-arb layer for pricing, and pre-registered gates (see `BENCHMARKS.md`) govern every result.

See `STATUS.md` for the one-page summary and `DECISIONS.md` for the pre-registered decision log.
