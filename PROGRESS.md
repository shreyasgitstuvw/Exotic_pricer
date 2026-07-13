# P2 PROGRESS — Supervision Report #1
2026-07-13 (manual; auto-generated weekly by `p2-supervisor` from now on) · Roadmap week: W1

## Verdict: AHEAD of schedule, one governance blocker
W1 was planned as infrastructure week; P2 has instead passed the M0 short-end gate AND substantially executed M1 — roughly two weeks ahead. The core thesis is already empirically demonstrated (D6).

## Gates
| Gate | Bar | Status |
|---|---|---|
| M0 short-end | ≥250 clean smiles, arb <2%, tests green | **PASS** — 1,016 smiles, 1.00%, 6 test modules green |
| M0 full-surface | maturity×moneyness surfaces from bhavcopy | **Built + validated**; 113 files cached |
| M1 instability | median + p95 daily param drift | **Partial PASS** (17-date subset, amended protocol per D6): 34 bps median fit, 0 bound-slams, gap 207 bps — full 113-date run pending on Shreyas's machine |
| M2 / M3 / M4 | — | pending; M2 now precisely framed: close a ~207 bps short-end gap |

## Compliance audit (vs FRAMEWORK.md)
**Green:** D1–D6 decision discipline is exemplary — D6 especially: a protocol amendment after data contact, justified as a structural identifiability fix with evidence both ways (196→47 bps on the same dates), which is exactly the bug-fix-vs-result-chasing line done right. Numbers live in disk artifacts; config-driven code; honest per-tenor RMSE reporting; expiry detection ported rather than assumed.

**Gaps found and status:**
1. Registry lagged the code by two sessions (EXP-001 ran unregistered) — **fixed today**; rule tightened: registry line before first run, descriptive studies included.
2. STATE.md was stale from the framework session — **fixed today**; FRAMEWORK §2.5 (close = update STATE) needs honoring every session.
3. **`data/README.md` provenance missing — the one blocker.** HISTORY_NIFTY's source/license must be documented before any of this goes public. If employer-derived, public repo ships bhavcopy-only.
4. Lockbox undefined for the two-layer data reality — needs **D7** (proposal in STATE.md).
5. Engine seam unpinned (`heston_ref` standing in) — pin Heston-engine SHA + reconcile IV conventions before M2.

## Risks
- M2 target honesty: gap range is wide (26–331 bps) — the gap-closure metric must be frozen in EXP-002's spec before training, or M2 becomes movable-goalposts territory.
- Repo home decision (extend Heston-engine vs new repo) blocks the W4 post's public links; D-entry needed.
- Full M1 run is a local job; if it slips past W2, the "ahead" cushion erodes.

## This week's required actions (Shreyas)
1. Run full M1 locally (command in STATE.md §next-actions) → EXP-001b final verdict.
2. Write the provenance note (unblocks going public).
3. Make D7 (lockbox) and repo-home decisions.
4. Don't let P2's momentum starve outreach: the W1 sends still matter more than an extra M1 date.

## Content extracted this period
39%-bound-slam story (post #2 can move up to W2–3), 207 bps headline number, expiry-shift corroboration for post #1.
