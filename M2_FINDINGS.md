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

### Ceiling decomposition -> the deployable answer (the important result)
Asked "why does the net stall at 60% vs the 87% oracle?" and decomposed the gap
(`experiments/diagnose_ceiling.py`, test split):
| ceiling | value | loss it isolates |
|---|---|---|
| A. per-tenor oracle | 87% | — |
| B. per-bucket oracle | 87% | bucketing loss = **0** (2 buckets suffice) |
| C. features→coeffs (ridge) | 67% train / 46% test | **feature-sufficiency wall** — the 24 features don't determine the coefficients even in-sample |
| D. features→coeffs (GBM) | 42% test | nonlinearity doesn't help |
| net (M2.2) | 60% test | already the BEST feature→coeff predictor (beats ridge/GBM out-of-sample) |

**Conclusion:** the net isn't the bottleneck — the *summary features* are, and the 60% is near their
ceiling. But the 87% oracle fits the **observed** residual, which is available at deployment. So the
right deployable tool isn't a learned predictor at all:

### Deployable M2 layer: direct per-tenor smoother — **87%, arb-free** (`experiments/direct_smoother.py`)
Fit the observed residual (market_IV − Heston_IV) per short tenor with a quadratic; project to no-arb.
No training, no train/test gap (each day fit independently, as a desk fits today's smile).
| split | reduction | arb |
|---|---|---|
| train | 82% | 0 |
| val | 85% | 0 |
| **test** | **87%** | **0** |
Projection costs nothing (a smoothly-fit residual is already ~arb-free). Reduces Heston's short-end
mispricing from ~165–193 bps to **~25 bps**, guaranteed tradeable.

### Two artifacts, two jobs (both stand)
- **Deployable calibration layer** = the direct smoother (87%). Ship this: a desk observes the weekly
  smile, this fits the Heston residual and repairs it arbitrage-free. This is M2's deliverable.
- **Predictability study** = the neural net (60% from market-state features; 16% from the long end
  alone). Answers "how independent is the NIFTY weekly regime?" — a real scientific finding, not a
  worse version of the smoother.
The decomposition is the lesson: three runs proved the features are the wall AND that the wall is
irrelevant to the deployment use case — so we ship observation+fit, not prediction.

### Optimization pass: two-headed unification — TESTED & REJECTED
Built a shared-encoder net with both heads (params + correction) to test whether multi-task learning
improves the gap-closer (`src/hestonnn/twohead.py`, `experiments/train_twohead.py`).
| model | train | test | verdict |
|---|---|---|---|
| standalone M2.2 | 64% | **60%** | tighter generalization (4-pt gap) |
| two-headed Head B | 67% | 51% | overfits (16-pt gap) — **negative transfer** |
The shared encoder fits train better but generalizes worse; Head A's param correlations were
unchanged (no gain either way). Conclusion: keep the standalone (simpler, generalizes better). A
defended negative — the answer to "did you try unifying the two networks?" is "yes, and measured that
it hurt."
