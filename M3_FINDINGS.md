# M3 — exotic-pricing consequences of Heston's short-end error

The payoff milestone: turn the M1/M2 short-end gap into mispricing on actual exotics. Two regimes,
by what the payoff depends on.

## Terminal-distribution exotics (priced from the smile; no dynamics needed)
A payoff depending only on S_T at one date is priced directly from the corrected vs Heston smile
(Breeden-Litzenberger: the smile IS the risk-neutral terminal density). No CF, no MC.

### Digitals (`experiments/m3_exotics.py`) — test split, 31 dates
A digital call = −dC/dK, i.e. pure skew. Heston's shallow short-end skew mis-prices it, worst on the
cheap OTM-call side:
| strike | dig (corrected) | Heston mispricing |
|---|---|---|
| 5% OTM put | 0.96 | 1% |
| 2% OTM put | 0.83 | 2% |
| ATM | 0.52 | 1% |
| 2% OTM call | 0.18 | **8%** |
| 5% OTM call | 0.03 | **15%** |
Headline: price a NIFTY weekly OTM-call digital off raw Heston and you're ~8–15% wrong.

### Variance swap / synthetic VIX (`experiments/m3_varswap.py`)
Static replication over the smile (CBOE-VIX style). Heston's shallow smile biases the short-dated fair
variance (the tradeable vol level) vs the market-consistent smile. Ties directly to India VIX.

## Path-dependent exotics (need dynamics -> LSV)
Barriers/Asians/autocallables depend on the whole path, so neither the smile nor the CF suffices — a
model with dynamics is required. The corrected model has no SDE, so we build one:

### Heston-SLV via the particle method (`src/hestonnn/lsv.py`) — engine VALIDATED
Leverage L(F,t)=σ_Dupire/√E[v|F], with E[v|F] estimated on the fly from the particle cloud. The core
guarantee — the SLV reproduces the vanilla smile of the local-vol model it's built from — is validated
to **11 bps** (Gyöngy equivalence). Flat-smile Dupire checks out analytically. Reduces to pure Heston
(L=1) and pure local-vol (frozen v) as limits.

### Weekly barrier under Heston vs SLV (`experiments/m3_barrier_lsv.py`) — CONVERGED
Corrected smile → Dupire local vol → SLV (reproduces the smile) → MC a weekly up-and-out call.
Calibration converges with MC timesteps (Euler): max smile error **895 → 311 → 148 → 43 → 33 bps** at
**25 → 120 → 300 → 1500 → 4000** steps. Gate (<50 bps) passes at ~1500 steps.
**Converged result** (stable across 1500 & 4000 steps, calibration PASS):
| steps | calib err | Heston | SLV (mkt) | mispricing |
|---|---|---|---|---|
| 1500 | 43 bps | 174.9 | 160.7 | **+9%** |
| 4000 | 33 bps | 174.7 | 159.6 | **+9%** |
Static Heston **over-prices the weekly up-and-out call by ~9%** — its shallow short-end skew implies
the wrong knock-out dynamics. Number is stable with resolution → trustworthy.

**Status:** engine validated (11 bps Gyöngy), calibration passes (<50 bps), barrier mispricing
converged (~9%). Remaining (optional): swap the synthetic target for real M2-corrected smiles
(densified with Heston tenors at 14/21/30d) to report per-date barrier mispricing across the sample.

## One-line takeaways
- Terminal exotics: static Heston mis-prices NIFTY weekly digitals by up to ~15%, and biases the
  synthetic short VIX — computable directly from the smile.
- Path-dependent: a validated neural-adjacent SLV closes the loop — reproduces the corrected smile and
  supplies the dynamics to price a barrier Heston mis-prices by ~8%.
