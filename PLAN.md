# P2 — Heston with Neural Components for Exotic Pricing
Status: M0 starts W2 (Jul 20) · Flagship project · Extends github.com/shreyasgitstuvw/Heston-engine
> Operating rules: FRAMEWORK.md (roles, session protocol, experiment lifecycle, data governance). Session memory: STATE.md / LOG.md / LESSONS.md / experiments/REGISTRY.md.

## Thesis
Static Heston parameters (κ, σ_v, ρ) are a solvability compromise, not a market fact. Replace the compromised components with neural functions — while keeping the SDE skeleton and no-arbitrage structure — and show the exotic-pricing consequences on real NIFTY data. The deliverable is not "NN beats Heston"; it is a quantified answer to "how wrong are static-parameter exotic prices, and where."

## Why this wins interviews
Hybrid model-plus-NN is how desks actually do it (LSV leverage learning, deep calibration). You can defend every layer: the Heston part is your existing engine; the NN part is constrained by financial physics; the data is the market you already research daily at Orion.

## Milestones (pre-registered, Orion-style)

### M0 (W2): Data + scaffold
- NIFTY option chain → clean daily vanilla surface snapshots (mid IVs, maturity × moneyness grid), 12+ months.
- Data provenance check: use your own/public NSE data, not Orion desk data, unless you confirm it's cleared. NSE EOD bhavcopy fallback (you've done this in NSE Analytics).
- Repo restructure: `heston_engine` as installable core; new `neural/` and `exotics/` modules; pytest harness.
- **Gate:** 250+ clean surface snapshots, arb-violations flagged <2% of quotes.

### M1 (W3–4): The instability exhibit (motivation)
- Daily Heston calibration (your DE→L-BFGS-B, vega-weighted IV error) across the full period.
- Deliverables: time series of (κ, θ, σ_v, ρ, v0); rolling repricing RMSE; regime scatter (params vs VIX level/term slope); the money-plot: parameter jumps around expiry weeks/events.
- **Gate (pre-registered):** report median + p95 day-over-day parameter change. If params were actually stable (median |Δρ| < 0.02 etc.), the project pivots honestly to "Heston is fine for NIFTY vanillas — the exotic question remains" and M2 tests exotics only. Publish either way.

### M2 (W5–7): Neural components under no-arb constraints
Route A (primary): **deep calibration surrogate** — NN: (surface features, VIX term structure) → (κ, σ_v, ρ) with smoothness + Feller penalties; nightly recalibration becomes a forward pass. Benchmark: surrogate params reprice vanillas within X bps of full optimizer at ~1000× speed.
Route B (stretch, the industry route): **LSV leverage function** — Heston-SLV: learn leverage L(S,t) matching the vanilla surface (particle method / Kolmogorov PDE target), NN regression for L.
- **Gate:** Route A surrogate within 25 bps IV RMSE of full calibration on held-out 3 months; Feller/no-arb violations = 0 on test set.

### M3 (W8–9): Price the exotics — the payoff
- Instruments: NIFTY autocallable (quarterly obs, KI put barrier), down-and-out call, Asian call. MC pricing (reuse QE scheme + antithetic + control variates) under (a) static-Heston, (b) M2 dynamic-param/LSV model.
- Deliverables: price gaps by instrument/moneyness/tenor; delta-hedging error simulation under both models on realized paths; "where static Heston lies" heatmap.
- **Gate:** hedging-error variance comparison with CIs (bootstrap, like your Merton work); honest nulls reported.

### M4 (W10): Writeup
- README overhaul + 8–12 page writeup (arXiv-style, not submitted) + 3 posts already extracted along the way.

## Non-goals (scope control)
No American exotics (early exercise ≠ this project), no multi-asset worst-of (that's P1's dimension story), no live trading claims, no "beats the market" language anywhere.

## References to anchor against
Horvath–Muguruza–Tomas, *Deep Learning Volatility* (calibration surrogates); Guyon & Henry-Labordère (particle LSV); Cuchiero et al. (neural SDEs); Gatheral *The Volatility Surface*; Austing *Smile Pricing Explained* (reading track W1–6).

## Interview lines this project buys you
- "I measured how unstable Heston params are on NIFTY across 2024–26 — here's the p95 daily move and what it does to a 2-year autocallable price."
- "My NN never prices; it only supplies parameters/leverage inside a no-arb SDE — extrapolation collapses back to Heston, which is the regulatory-friendly architecture."
