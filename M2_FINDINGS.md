# M2 — neural components (results)

## M2.1 — deep-calibration surrogate: **PASS**

Trained a small constrained MLP (17 surface features → κ, σ_v, ρ) on the M1 answer key.
Split by time: train 258 (2020–2024) · val 57 (2025) · test 31 (2026). COVID-crash dates excluded,
poor fits down-weighted by 1/RMSE. Standardization stats from train only.

### Route-A gate (pre-registered in BENCHMARKS): reprice within 25 bps of the full optimizer
| metric | value |
|---|---|
| surrogate repricing IV RMSE (test median) | 66 bps |
| full DE optimizer IV RMSE (test median) | 43 bps |
| **gap (surrogate − DE)** | **17 bps → PASS (<25)** |
| speed | ~sub-µs/date vs ~10 s/date DE → ~10⁶× |

### What the numbers teach
- **Parameter accuracy ≠ price accuracy.** Test correlations: ρ 0.83, σ_v 0.74, **κ 0.39** (weak,
  MAE ~3.3). κ is the sloppy direction — the surface barely constrains it — yet the surrogate still
  reprices within 17 bps, because a wrong κ costs almost nothing in price. We trained on params but
  graded on prices, and that is exactly why it passes.
- **Mild, controlled overfitting.** Train corr (0.77–0.83) > val (0.31–0.61); early stopping halted
  on best-val, not best-train. Expected at N≈258; the time-split holdout is what makes it visible.
- **The surrogate is a differentiable, instant calibrator** — the encoder Route B reuses.

### Honest caveats / levers if we want it tighter
- Test period (2026) includes a vol spike, so the DE ceiling itself is 43 bps (vs ~30 in calm).
- To sharpen: add India VIX + realized-vol as inputs; more dates (daily cadence); or the
  calibration-consistent loss (backprop through the pricer) instead of param-MSE.

Artifacts: `data/m2/dataset.npz`, `data/m2/surrogate.pt`, `experiments/{m2_dataset,train_surrogate}.py`.

## M2.2 — short-end gap-closer (Route B): **PASS**

Three-layer architecture, each doing one job (this is how production vol engines are built):
1. **Heston SDE backbone** (≥14d calibration) — arbitrage-free by construction.
2. **Neural correction** — MLP(24 features: long-end surface + observed short-end smile + days-to-
   expiry) → per-short-tenor (level, skew, curvature) shift, added to the Heston short-end IV.
   Residual learning: the net learns only Heston's *error*, in a rescaled basis z=x/0.1.
3. **Deterministic arb-projection** — the corrected call-price smile is projected onto the exact
   no-arb cone (delta-bounded, spacing-aware convex) via a small QP. Guarantees a tradeable smile.

### Gate (held-out, time-split): PASS
| split | gap reduction (median) | arb violations |
|---|---|---|
| train (2020–24) | 64% | 0 |
| val (2025) | 60% | 0 |
| **test (2026)** | **60%** | **0** |

Short-end gap ~165 bps → ~60% closed → ~66 bps residual, with a guaranteed arbitrage-free smile.

### What it took (debugging lessons — real, each measured not guessed)
- **Basis conditioning** (the big one): correcting in raw log-moneyness x (~±0.1) makes the x²
  gradient ~100× too weak, so the curvature coefficient never trained (stuck 18%). Rescaling to
  z=x/0.1 unlocked it (→52%). Diagnosed with a minimal GD reproduction.
- **Loss-scale traps**: an early butterfly penalty (~1e-2) dwarfed the IV² fit (~1e-4) and froze the
  model; later the same penalty at lam_bf=0.05 was negligible. Soft penalties can't guarantee 0
  violations — hence the projection.
- **Wrong arb conditions**: first projection ignored the −df slope bound and used spacing-blind
  convexity → made things worse; the correct cone (matching flag_smile) stress-tested 95→0.
- Underfitting (train≈val below oracle) fixed by less dropout + wider curvature range.

### Honest framing (both claims stand)
- **Deliverable**: a market-conditioned, no-arb neural calibration layer that closes ~60% of the
  short-end gap Heston can't reach — deployable for M3 exotic pricing.
- **Finding** (separate, from the earlier oracle): the short-end correction is only ~16% predictable
  from the *long-end* surface alone — the NIFTY weekly smile is a partly independent regime. Route B
  works because it *observes* the short end (industry-correct), not because it predicts it.
- Ceiling: per-tenor oracle is 87% in-sample; 60% out-of-sample is the generalization gap (levers:
  India VIX feature, more data, or the LSV route).

Artifacts: `experiments/{m2b_dataset,train_gap_closer,diagnose_m2b}.py`, `src/hestonnn/gap_closer.py`,
`data/m2b/gap_closer.pt`.
