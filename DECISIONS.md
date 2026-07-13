# P2 — Decision Record (pre-registered, Orion-style)

Append-only log of design decisions locked before the work they govern. Each entry states what we
decided, why, and what it commits us to. Do not silently reverse an entry — supersede it with a new
one that references it.

---

## D1 — Calibration snapshot cadence: **EOD daily surfaces** (not intraday)

**Status:** Decided · 2026-07-12 · owner: Shreyas · governs M0–M2
**Guiding preference (stated):** precision and accuracy of outputs over computational feasibility.

### Context — the data we actually hold
- `HISTORY_NIFTY/` — NIFTY option chain at **1-minute** resolution, 2021 → 2026 (262 weekly
  parquet files, ~375 intraday stamps per trading day). Columns: `Strike, Spot, CE/PE_{Open,High,
  Low,Close}, CE/PE_IV, CE/PE_Volume, CE/PE_OI`.
- `NIFTY50_FUTURES.csv` — 1-minute futures OHLCV (2019 →), source of the per-expiry **forward**.
- `India_VIX.csv` — 1-minute India VIX (2017 →), a market-state conditioning feature.

The 1-minute grain makes intraday cadence *possible*, so the choice is deliberate, not forced.

### Decision
Build **one calibration surface per trading day, per expiry**, sampled at a **fixed daily close
window** (median of the 15:25–15:29 minute marks, to avoid closing-auction noise and single-print
outliers). Intraday resolution is **not** used to build calibration surfaces.

### Reasoning
The stated preference — precision over compute — points *toward* EOD, not away from it, for three
reasons:

1. **Calibration needs cross-sectional synchronicity, and that is where intraday loses.** Heston/SLV
   calibration fits the whole smile (all strikes × maturities) *as one instantaneous cross-section*.
   Away-from-ATM NIFTY strikes trade thinly intraday; their quotes update non-synchronously and go
   stale between prints. A 30-minute "surface" is a smear of quotes from different instants and
   market states. The temporal resolution gained is paid for in cross-sectional quote noise — and
   the wings, which identify σ_v and ρ, are exactly the least reliable part intraday. The provider
   IV column confirms this: illiquid strikes carry `IV = 0.00` (no reliable mark). EOD closing
   marks are the cleanest synchronized cross-section the day produces.

2. **The signal we are trying to measure precisely is daily.** P2's headline (M1) is *day-over-day*
   parameter instability — median and p95 daily |Δκ|, |Δρ|, etc. That is a daily-frequency question
   by construction. Calibrating every 30 minutes folds intraday microstructure noise into the very
   quantity we want to measure cleanly, inflating the instability metric with noise rather than
   sharpening it. EOD maximizes precision *of the output that matters*.

3. **Volume confirms rather than fights the choice.** EOD over ~5 years ≈ 1,200 trading days of
   surfaces — comfortably past the M0 gate (250+ clean snapshots) with room for a strict time-based
   test split. Intraday would yield ~450k surfaces, but the added variance is largely noise and it
   ~375×'s the M0 cleaning burden for negative marginal signal.

### Consequences (what this commits M0 to)
- Surface builder emits **one surface per (date, expiry)** from the 15:25–15:29 close window.
- **Recompute IV ourselves** from `CE/PE_Close` against our own forward (from futures) and rate
  curve — do not trust the provider `*_IV` column (contains 0.00s and unknown conventions). This
  keeps the surface consistent with the frozen engine's own IV definition.
- Term structure is **assembled across expiry files**: a given calendar day's surface stacks the
  near weekly expiry (this week's file) with the next weeklies and the monthly, to span the
  maturity axis. (Verify the expiry/file mapping in M0 — see open items.)
- M0 gate unchanged: 250+ clean daily surfaces, arb-violations flagged < 2% of quotes.

### Where intraday data *is* used (not discarded)
- **M3 delta-hedging simulation** — higher-frequency futures/spot lets us rebalance realistically
  and tightens the hedging-error estimate. Temporal resolution genuinely improves accuracy there.
- **VIX / futures conditioning features** — sampled at the same daily close stamp as the surface.
- A single clean intraday robustness check (e.g., a mid-session 12:00 surface) may be run in M1 to
  confirm parameter estimates are not an artifact of the close — optional, not the primary path.

### Open verification items (resolve during M0, before locking the surface builder)
- ~~Confirm whether each weekly parquet holds a single expiry or multiple.~~ **RESOLVED 2026-07-12:**
  each `(date, time)` snapshot is a **single-expiry** strike strip (front contract only). See D3.
- Confirm the daily last-stamp (expected ~15:29 — verified present) and choose the close window
  empirically from liquidity.
- Rate curve source for discounting (T-bill / OIS proxy) — forward comes from futures per expiry.

---

## D2 — NN parameter set: **(κ, σ_v, ρ)**, not all five

**Status:** Decided · 2026-07-12 · owner: Shreyas · governs M2 (Route A surrogate)

### Decision
The neural head outputs the three **shape** parameters (κ, σ_v, ρ). The two **level** parameters
(v0, θ) are handled classically/analytically (pinned by the ATM level and, where available, the
term structure), not emitted by the NN.

### Reasoning
κ, σ_v, ρ are the poorly-identified, day-to-day-unstable parameters that govern skew and smile
curvature — the ones worth learning as a function of market state. v0 and θ are near-observable from
the ATM level/term structure, so learning them adds noise, not signal. Smaller output head =
less overfitting on ~1k surfaces and a cleaner interpretation of what the network is actually doing.

### Consequences
- Route A head has 3 outputs, domain-constrained: κ,σ_v > 0 (softplus), ρ ∈ (−1,1) (tanh).
- v0 fixed from ATM variance per snapshot; θ from a classical fit / prior (revisit under D3, since
  θ identifiability depends on having a term structure — which the current data lacks).

---

## D3 — Maturity coverage constraint (front-contract-only data)

**Status:** DECIDED (Option A) · 2026-07-12 · owner: Shreyas · governs M0 data layer + M3 scope

### Finding (evidence, 2021 full-year scan)
Every `(date, time)` snapshot contains a **single expiry** — the front contract — as a ~±5% strike
strip (21–25 strikes). Detected expiry days: 42 in 2021; expiry spacing median 7 days, **max 22**.
So the **only maturities ever present are ~0–22 calendar days** (mostly ≤7). There are **no
long-dated options** in the dataset. Empirical-expiry detection also caught a shift (e.g. Wed
10-Mar-2021 instead of Thursday), confirming expiry must be computed, not assumed.

### Why this blocks the plan as written
- **Full Heston is not identifiable from a single short smile.** κ and θ live in the *term
  structure* of variance; with maturities ≤22 days there is almost no term-structure information to
  pin them. Daily 5-param calibration would be ill-posed.
- **M3 autocallables (2-yr, quarterly obs) are impossible from this data** — they need vol out to
  years; we have ≤22 days.

### Options (decision needed before M0 builds)
- **A — Add EOD multi-expiry data (NSE bhavcopy).** Restores the full maturity axis (all listed
  expiries, out to ~years) at EOD — consistent with D1. The 1-min strip becomes the intraday /
  short-end / hedging layer. Keeps the flagship autocallable scope.
- **B — Reframe P2 to the data's strength.** High-frequency short-dated smile dynamics: constrained
  short-Heston (calibrate σ_v, ρ, v0; κ,θ fixed by prior), instability measured at 1-min resolution,
  exotics limited to short-tenor (weekly barriers / digitals / Asians). Drops autocallables.
- **C — Hybrid.** Reframe calibration to short-dated (B), pull bhavcopy only where long maturities
  are required for M3 (A).

### Decision (A)
Two data layers, both EOD:
- **Long-end / full surface — NSE bhavcopy (all listed expiries).** Primary calibration surface for
  M1/M2 and the source of long-dated vol for M3 autocallables. *User action: source cleared EOD
  F&O bhavcopy (udiff) files.* Pipeline exposes a `bhavcopy` ingestion module (built as an interface
  now, populated when files land).
- **Short-end / high-res — the 1-min `HISTORY_NIFTY` strip.** Front-contract smiles at minute
  resolution → the M3 delta-hedging simulation and an optional short-end smile-instability study.
  Built and runnable now.

### Consequences
- M0 gate splits: (i) short-end layer — 250+ clean daily smiles from the 1-min strip (buildable now);
  (ii) full-surface layer — maturity×moneyness surfaces from bhavcopy (pending user data).
- Forward + discount factor implied from the chain via put-call parity (self-contained, no external
  rate curve needed); futures used as cross-check.
- θ, κ identifiability (D2) restored once the bhavcopy term structure is in.

### Update 2026-07-12 — bhavcopy validated; BankNifty rejected
User raised bhavcopy sourcing pain (format inconsistency) and proposed BankNifty futures instead.
Resolved after inspecting a sample UDiFF file + NSE's format readme:
- **BankNifty rejected.** Futures carry no strikes/smile — cannot yield an IV surface at all. And
  BankNifty *options* in the sample bhavcopy top out at ~354 days (monthly, no weeklies) vs NIFTY's
  1,810 days. Strictly worse for calibration/exotics.
- **One NIFTY bhavcopy file = full term structure.** The 2026-07-10 file has 18 NIFTY expiries;
  those with real liquidity (≥7 strikes carrying OI) span 4→~900 days (~2.5y) — enough for full
  Heston identification AND a 2-yr autocallable.
- **Format inconsistency solved.** The "old vs new UDiFF" split is documented in the user's readme
  crosswalk. Built `data/bhavcopy.py` with header auto-detection mapping BOTH schemas to one
  canonical frame (settlement-preferred, close fallback, zip/csv). `data/surface_bhav.py` assembles
  the multi-tenor surface via the SAME forward/iv/arbitrage machinery + calendar-arb check.
  Validated on the sample: clean tenors (4/11/18/46/172d) at ~10–12% ATM IV, cross-checking the
  1-min short-end. Long tail (>1y) is thin — needs per-bucket liquidity selection + wider moneyness
  band (M1 hygiene tuning).
- **Sourcing burden reduced:** M1/M2 need a *sampled* date set (e.g., weekly/monthly + event
  windows), not every trading day — cutting the number of bhavcopy files to source, most of which
  are recent (clean UDiFF) anyway.

---
```
## D4 — Reference Heston pricer to unblock calibration (frozen engine still ground truth)

**Status:** Decided · 2026-07-12 · owner: Shreyas · governs M1 calibration until engine pinned

### Context
The frozen engine (github.com/shreyasgitstuvw/Heston-engine) is the intended pricer/calibrator but
isn't installable in this environment yet. M1 (the instability exhibit) needs a working calibrator now.

### Decision
Ship `heston_ref.py` — a NumPy Heston-1993 CF pricer (stable "trap" CF + Lewis single-integral,
fixed N=2000 trapezoid grid, vectorized over strikes) — as P2's **reference** pricer, plus
`calibrate.py` (vega-weighted IV RMSE, DE→L-BFGS-B). The frozen engine stays ground truth: when
pinned via `engine.py`, the reference must agree with it to ~1e-4 in price, and the pipeline swaps to
the engine. `heston_ref` then remains as the differentiable-twin basis for the M2 surrogate.

### Validation
Pricer: put-call parity exact; σ_v→0 collapses to Black-76 to <5e-4 (≈0.5 bps IV); vs adaptive
quadrature <5e-4. Calibrator round-trip: recovers ρ, v0, θ to 3–4 dp and σ_v to ~0.02 on synthetic
data at ~2.7 bps; κ left as the known sloppy direction. (21 tests pass.)

### First real fit (NIFTY 2026-07-10 bhavcopy, tenor-adaptive surface)
κ=11.37, θ=0.0181, σ_v=0.775, ρ=−0.411, v0=0.0119; **vega-weighted IV RMSE 34 bps**. Belly
(11–46d) fits 12–22 bps; 4d strains at 77 bps — the short-dated-skew limit that motivates M2; long
tail thin. Feller violated (typical for equity-index Heston). Saved: `data/first_fit_20260710.txt`.

### Consequences
- M1 can proceed now: calibrate across the sampled bhavcopy dates → parameter time series.
- Surface hygiene added: `surface_bhav.calibration_quotes()` uses a √T-scaled moneyness band
  (fixed band is too wide short, too tight long) + arb drop + min-strikes.
- Diagnostic note: always invert model **call** prices as calls (a K<F put-convention inversion of a
  call price produces garbage IV — caught and fixed during this fit).

---

## D5 — Full-Heston M1 wired on real bhavcopy; full run is a local-machine job

**Status:** Decided · 2026-07-13 · owner: Shreyas · governs M1 execution

### Context
Downloader pulled 113 real UDiFF bhavcopy files (2024-07 → 2026-07, weekly+monthly; 404s = NSE
holidays). `run_m1_instability.py --source bhavcopy` assembles the surface + full 5-param calibration
per date, resumable (appends per date, skips done). Validated on the real files.

### Findings
- Per-date fit quality depends on optimizer effort: a THOROUGH DE (maxiter≈24, popsize≈15) gives
  sane params (e.g. 2024-09-13: κ=4.8, σ_v=0.56, ρ=−0.58, 51 bps). A cheap DE, or warm-start
  chaining, undercooks some dates into the bounds (σ_v→2, κ→15) — optimizer noise, not market signal.
  → **Default is thorough independent DE per date; warm-start chaining removed.**
- Compute: ~10–15 s/date thorough. 110 dates ≈ 20–25 min. The sandbox's 45 s/call cap makes the full
  sweep impractical here, so **the full M1 run is documented as a local command** (like the
  downloader) and produces `data/m1/params_timeseries_bhavcopy.csv` + money-plot.

### Consequences
- M1 flagship result (5-param instability time series + day-over-day jump stats) is one command away
  on the user's machine; the short-end proxy remains for a quick look.
- Some dates genuinely strain single-Heston (short-dated skew) — that misfit is on-thesis and feeds
  M2; distinguish it from optimizer noise by requiring the thorough DE before reading instability.

---

## D6 — The M1 result: Heston can't fit NIFTY's short end; fit ≥14d, measure the gap

**Status:** Decided · 2026-07-13 · owner: Shreyas · the core M1 finding

### Finding (from the real 113-date run)
Naive full-surface calibration bound-slammed on 39% of dates (σ_v→2, κ→15, RMSE 100–196 bps). It is
**not** an optimizer failure: dropping the sub-14-day tenors makes those exact dates fit cleanly
(196→47 bps, κ→6.4, σ_v→0.86). Single Heston **structurally cannot reproduce NIFTY's short-dated
skew**, and forcing those tenors in extremizes the params, corrupting the whole calibration. This is
the project thesis, shown empirically.

### Decision
Calibrate Heston to tenors **≥ 14 days** (`--min-tte-days 14`); measure Heston's IV RMSE on the
excluded short tenors as the **short-end Heston gap** — the quantity M2's neural layer must close.
Recorded per date as `short_gap_bps`.

### Result (17-date validation subset, improved pipeline)
Fit RMSE median **34 bps** (was 77), **0 bound-slams** (was 39%). Short-end Heston gap median
**207 bps** (range 26–331) — this is M2's target. Parameters still show genuine instability
(σ_v median |Δ|≈0.35, ρ≈0.15, κ weakly identified) — now trustworthy, not optimizer noise.
Artifacts: `data/m1/money_plot_bhavcopy_v2.png`, `params_timeseries_bhavcopy_v2.csv`.

### Consequences
- Re-run the full set: `run_m1_instability.py --source bhavcopy --min-tte-days 14 ...` (default is 14).
- M2 is now precisely framed: learn the ~200 bps short-end correction Heston can't reach (the LSV
  leverage / neural local-vol layer), with the ≥14d Heston fit as the backbone.
- Interview line, now with a number: "single Heston mis-prices NIFTY's <2-week skew by ~2 vol points
  on average; my NN supplies exactly that correction inside a no-arb SDE."

---

## D7 — Backfill to 2020 (old-format bhavcopy) + COVID-aware bounds

**Status:** Decided · 2026-07-13 · owner: Shreyas · governs M2.0 data

### What happened
Legacy NSE archives are still served: pulled 266/280 weekly old-format files 2020-01→2024-07 into the
same `bhavcopy/` folder (now ~379 files, 2020→2026). Real old files validated — **88%** carry ≥4
usable tenors (median 13, term structure to 2–3.8y). ~12% thin (a mid-May pattern in NSE's archives),
auto-skipped by the driver (<3 tenors). The old-format parser works on real data (was only unit-tested
before).

### Bounds fix (required)
COVID pushed vols to ~80% (v0≈0.6); the old calibration capped v0/θ at 0.25. Widened to
**v0,θ ∈ (1e-3, 1.0)**, σ_v→2.5, κ→20 in `calibrate.py`. Normal dates are unaffected (they sit far
from the bounds); the widening only matters for 2020–2022 stress. 21 tests still pass.

### Regime heterogeneity (a feature, not a bug)
Fit quality varies by regime: 2022–2026 ~30–40 bps; 2020–2021 stressed at 70–160 bps with ρ pegging
−0.95 (Heston cannot make skew steeper than ρ→−1) and the ~6 March–April-2020 crash days fitting
poorly (inverted term structure). This is on-thesis — more Heston inadequacy to motivate M2 — and
gives the training set real regime diversity. M2 will carry `rmse_bps` as a per-date quality weight/
feature and may down-weight or flag the crash days.

### Consequences
- Re-run M1 (incremental — keeps the 113 done, appends ~200 old-era dates): full 2020→2026 backbone +
  gap, ~40–60 min. Satisfies M2.0's ≥400-date intent at weekly cadence (~300 usable).
- Keep `rmse_bps` in the series; M2 uses it for quality weighting. Consider excluding the ~6 COVID
  crash days from training (record, don't silently drop).

---

## D8 — Route B parameterization: quadratic parametric correction first; LSV as escalation

**Status:** Decided · 2026-07-13 · owner: Shreyas · governs M2.2

M2.1 passed (surrogate reprices test within 17 bps of DE, <25 gate). For the short-end gap-closer we
build the **parametric quadratic correction** first: NN(features) → (level, skew, curvature) shift per
short-tenor bucket; corrected_IV = Heston-backbone_IV + δ0 + δ1·x + δ2·x² (x=log-moneyness). Rationale:
the M1 gap is structurally a skew/curvature deficit, so a quadratic spans the right space; 6–9 outputs
are trainable on 258 train dates; no-arb (butterfly/calendar) is analytic on a low-order total-variance
curve; reuses the M2.1 encoder. Grid and true **LSV leverage L(S,t)** (particle-method / Fokker-Planck
target) are documented escalations if the quadratic plateaus below the ≥50% gap-reduction gate. User
prefers LSV as the eventual destination; we escalate only on measured need.

---

Template for future entries:
## D2 — <short title>
**Status:** Proposed | Decided | Superseded-by-Dn · date · owner · governs <milestones>
### Context
### Decision
### Reasoning
### Consequences
```
