# P2 — M2 Plan: neural components under no-arbitrage constraints

Pre-registered, Orion-style. Builds on M1 (`M1_FINDINGS.md`). Locked metrics in `BENCHMARKS.md`.
Reviewed before any code is written.

## What M1 hands to M2
- A ≥14d Heston backbone per date (κ, σ_v, ρ, v0; θ analytic) fitting the belly/long end to ~33 bps.
- A **measured short-end gap**: median ~165 bps, state-dependent (2025Q1 310 → 2025Q4 43), fat-tailed
  to >600 bps on event days. This is what M2 must close.
- The finding that the gap is **not a constant offset** → a market-state-conditioned correction is
  the right tool; a static local-vol bump is not.

## Two routes, and the sequence
The earlier architecture note (two-headed / cascading SLV-NN) is the destination. We build toward it:

**Route A — deep-calibration surrogate (supporting result).** NN: (surface features, VIX term) →
(κ, σ_v, ρ). Turns nightly DE calibration into a forward pass. Benchmark: reproduce the 113-date DE
fit within tolerance at ~1000× speed. Low-dimensional output (3 numbers) → trainable on our small
sample. *Ships first as the warm-up; it also gives M2 a differentiable param map.*

**Route B — the short-end gap-closer (the on-thesis result).** NN: (surface features, VIX, regime)
→ a short-end IV correction Δ(K, T) applied on top of the ≥14d Heston backbone, trained to close the
measured gap under no-arbitrage penalties. This is the milestone the writeup is built around, because
M1 proved the gap is real, large, and state-dependent.

Recommendation: **A then B**, with B as the flagship. B reuses A's encoder (shared representation).

## Data — inputs, target, splits, and the size problem
**Inputs (market-state features per date):** the ≥14d surface encoded compactly — ATM level + term
slope + skew + curvature per tenor bucket (or PCA of the IV grid), plus India VIX level and the ATM
term-structure slope. ~15–30 features.

**Targets:** Route A → the DE-calibrated (κ, σ_v, ρ). Route B → the market <14d IV smile (the thing
Heston misses); loss is model-minus-market IV on the short end.

**Splits (BENCHMARKS rule — by TIME, never shuffled):** train = 2024-07→2025-12, test = final
3 months untouched until the gate. No surface crosses the boundary.

**⚠️ The #1 risk — sample size.** 113 weekly dates is small for a NN. Two mitigations, in order:
1. **Re-pull bhavcopy DAILY** (drop the weekly sampler): 2024-07→2026-07 ≈ ~500 trading days. Same
   downloader, no `--every` gaps. ~500 files. This is the single biggest lever and I recommend it as
   **M2.0**. Cost: one overnight download + a ~2-3h calibration pass to rebuild the M1 backbone daily.
2. Keep networks tiny (Route A: 2×64 MLP; Route B: a small CNN/MLP over the strike grid), heavy
   weight-decay + dropout, early-stop on the time-based val fold. Report with error bars over seeds.
If even daily data proves too thin for B, we fall back to Route A only + an honest "gap is real but
under-powered to learn from N dates" note — a defended negative (BENCHMARKS kill-criteria style).

## Architecture (PyTorch — required for the no-arb autograd)
- Shared encoder: features → hidden (2× hidden, tanh/softplus so outputs are C² for arb penalties).
- Head A: → (κ, σ_v, ρ), domain-constrained (softplus / tanh), Feller as a penalty.
- Head B: → short-end IV correction on the (K, T<14d) grid, added to the backbone Heston IV.
- The NN **never outputs a price** — it outputs params (A) or an IV correction feeding the SDE (B).

## Loss & no-arbitrage
- Route A: vega-weighted IV RMSE of the surrogate params repricing the ≥14d surface + Feller penalty.
- Route B: vega-weighted IV RMSE on the <14d smile + **butterfly** (∂²C/∂K² ≥ 0) and **calendar**
  (total-variance monotonic in T) penalties + smoothness (TV) on the correction surface.
- Seeds fixed and logged; every figure states path counts / sample sizes (BENCHMARKS).

## Milestones & pre-registered gates
- **M2.0 (data):** daily bhavcopy pulled + daily ≥14d Heston backbone rebuilt. Gate: ≥400 clean
  dates, backbone median RMSE ≤ 40 bps (consistent with the weekly run's 33).
- **M2.1 (Route A):** surrogate → (κ, σ_v, ρ). **Gate:** surrogate params reprice the held-out ≥14d
  surface within **25 bps** IV RMSE of the full DE optimizer; 0 Feller/arb violations on test.
- **M2.2 (Route B):** short-end gap-closer. **Gate:** cut the held-out short-end gap by **≥50%**
  (≈165→≤80 bps median) with **0** butterfly/calendar violations on the test set. Report the
  gap-reduction distribution, not just the mean.
- **M2.3 (writeup):** the gap-closing result + the speed/accuracy frontier from A.

## Kill criteria (honest-null, pre-registered)
- If Route B can't beat a **25%** gap reduction on val by end of the milestone → ship A + the M1 gap
  characterization as the honest result, and post the negative: "the short-end gap is real and large
  but not learnable from N NIFTY dates without more data/features." A defended negative ships.
- No-arb violations on test are disqualifying regardless of RMSE — the whole pitch is a no-arb SDE.

## Non-goals
No pricing by the NN, ever. No exotics here (that's M3). No claim the NN "beats the market" — it
supplies a correction inside a constrained model. No leakage: test months untouched until the gate.

## Decision needed before M2.0
Daily bhavcopy re-pull (recommended) vs. proceed on the 113 weekly dates with tiny nets + error bars.
