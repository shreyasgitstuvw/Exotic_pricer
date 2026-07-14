# STATUS — P2: Heston + Neural Components for NIFTY Exotic Pricing

**One line.** On 6 years of real NSE data, measure how badly static Heston mis-prices the short-dated
NIFTY smile, correct it arbitrage-free, and quantify the exotic-pricing consequences — spanning
classical calibration, a neural surrogate, a learned no-arb correction, and a stochastic-local-vol
engine.

**Overall status: M0–M3 complete.** All results on held-out (time-split) or MC-converged footing;
every gate pre-registered in `BENCHMARKS.md`.

## Scorecard

| stage | status | result |
|---|---|---|
| M0 — data pipeline | ✅ | 355 clean daily surfaces 2020–2026; dual-format bhavcopy + auto-downloader; PCP forward + arb checks |
| M1 — Heston instability | ✅ | params unstable (κ p95 |Δ|≈13, ρ −0.87→−0.12); **short-end gap ≈165 bps**, regime-dependent |
| M2.1 — calibration surrogate | ✅ PASS | reprices within **17 bps** of DE optimiser (gate <25), **~10⁵–10⁶× faster** |
| M2.2 — short-end gap-closer | ✅ PASS | **direct smoother closes 87%** (165→~25 bps), **0 arb**; neural predictor 60% (predictability study) |
| M3 — terminal exotics | ✅ | digitals mis-priced up to **15%** OTM; synthetic VIX biased |
| M3 — LSV barrier | ✅ | Heston-SLV **validated (Gyöngy 11 bps)**; calib converges to 33 bps; Heston **over-prices weekly barrier ~9%** |
| Optimisation passes | ✅ (rejected) | two-headed unification (negative transfer); India VIX feature (no test coverage) — both tested, documented |

## Architecture (industry-shaped, three layers)
1. **Heston SDE backbone** — calibrated to ≥14-day surface; arbitrage-free by construction.
2. **Short-end correction** — deployable: fit the observed residual per tenor (87%). Studied: a neural
   predictor of it (60%). The corrected smile feeds pricing.
3. **Arb-projection / dynamics** — corrected smiles projected to the no-arb cone; for path-dependent
   payoffs, a Heston-SLV (particle method) supplies dynamics and MC-prices the barrier.

## Findings that make it credible
- The short-end miss is a **partly independent regime** (~16% predictable from the long end).
- **Knowing when not to use ML**: decomposition proved a deterministic fit beats the network for
  deployment; the network is kept as a study, not shipped.
- **Defended negatives**: COVID-crash exclusion, VIX-feature rejection, two-headed rejection — all
  measured, not hand-waved.

## Open / optional
- Point the LSV at real M2-corrected smiles (default uses a synthetic target) for per-date barrier
  mispricing across the sample — mechanical; pipeline proven.
- Version control the repo (`git init`).

## Map
Design: `PLAN.md`, `BENCHMARKS.md`, `M2_PLAN.md` · Results: `M1_FINDINGS.md`, `M2_FINDINGS.md`,
`M3_FINDINGS.md` · Decision log: `DECISIONS.md` · How to run: `README.md`.
