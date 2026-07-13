# P2 — Heston + Neural Components for NIFTY Exotic Pricing (flagship status)

**One line:** static Heston parameters aren't a market fact; this project measures how wrong the
resulting short-dated NIFTY option prices are, and supplies a neural, arbitrage-free correction —
built and validated on 6 years of real NSE data across five volatility regimes.

## Pipeline (all gates pre-registered, all results on held-out 2026 data)

| stage | what | headline result |
|---|---|---|
| **M0** data | 1-min NIFTY strips + dual-format NSE bhavcopy (2020–2026), auto-downloaded, arb-checked surfaces | 355 clean daily surfaces; forward/df from put-call parity |
| **M1** instability | full Heston calibrated daily (≥14d), short-end miss measured | params unstable (κ p95 |Δ|=13); **short-end gap ~165 bps**, regime-dependent (43→310 bps by quarter) |
| **M2.1** surrogate | MLP maps surface features → (κ,σ_v,ρ) — calibration as one forward pass | reprices within **17 bps** of the full optimizer, **~300,000× faster** |
| **M2.2** gap-closer | Heston backbone + neural short-end correction + deterministic arb-projection | closes **60%** of the short-end gap on test, **0 arbitrage violations** |

## The M2.2 architecture (industry-shaped)
1. **Heston SDE backbone** (≥14d calibration) — arbitrage-free by construction.
2. **Neural correction** — MLP(long-end + observed short-end smile features) → per-tenor
   (level, skew, curvature) shift; residual learning in a rescaled moneyness basis.
3. **Arb-projection** — corrected call prices projected onto the exact no-arb cone (delta-bounded,
   spacing-aware convex) via a small QP → guaranteed tradeable smile.

## Honest findings (both stand — a defended pair)
- **Deliverable**: a market-conditioned, no-arb neural calibration layer closing ~60% of the gap
  Heston structurally can't reach — deployable for exotic pricing.
- **Finding**: the short-end correction is only ~16% predictable from the *long-end* surface alone —
  the NIFTY weekly smile is a partly independent regime (event/pinning risk). Route B works because
  it *observes* the short end (industry-correct), not because it extrapolates it.
- **Negatives kept, not hidden**: COVID-crash days excluded (Heston can't fit an inverted crash term
  structure); India VIX rejected as a feature (series ends 2025-11, no test coverage).

## What the debugging taught (documented in M2_FINDINGS.md)
Five failed gates, each diagnosed to a specific measured cause — basis conditioning (curvature
gradient ~100× too weak), loss-scale traps, wrong arbitrage conditions — not guessed. The diagnosis
method (minimal reproductions, oracle ceilings, per-layer instrumentation) is itself a result.

## Docs
`PLAN.md` · `BENCHMARKS.md` · `M2_PLAN.md` (design) · `M1_FINDINGS.md` · `M2_FINDINGS.md` (results) ·
`DECISIONS.md` (D1–D8 pre-registered decision log) · `README.md` (how to run).

## Next (open)
- **M3** — price short-dated exotics (digitals, weekly barriers, Asians) under static Heston vs the
  M2-corrected smile; quantify the mispricing. The payoff that turns the gap into rupees.
- **Polish levers** (optional, measured): two-headed unification of M2.1+M2.2; realized-vol features;
  LSV leverage function for rigorous path-dependent pricing.
