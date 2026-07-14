"""M3 (v3) — path-dependent exotic under LSV: the payoff that needs dynamics.

Pipeline: corrected short-end smiles -> Dupire local vol sigma_LV(F,t) -> Heston-SLV calibrated by the
particle method (reproduces the corrected smile) -> Monte-Carlo a weekly up-and-out barrier under
(a) pure Heston (its too-shallow short smile) and (b) the SLV (market-consistent). The price gap is
the barrier mispricing caused by Heston's short-end error — a path-dependent consequence the terminal
(digital / var-swap) methods cannot capture.

Runs on synthetic smiles by default (self-validating); point it at real corrected smiles later.

    python experiments/m3_barrier_lsv.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hestonnn.lsv import particle_slv, vanilla_smile_from_terminal
from hestonnn.data.iv import black76_price, implied_vol


def dupire_local_vol(tenors):
    """tenors: list of (T, poly) where poly gives IV as np.polyval(poly, k), k=log(K/F).
    Returns sigma_LV(F, t, F0) via the Gatheral total-variance formula. Fully vectorized over F."""
    Ts = np.array([T for T, _ in tenors])
    ders = [(p, np.polyder(p), np.polyder(p, 2)) for _, p in tenors]

    def sigma_lv(F, t, F0):
        k = np.log(np.maximum(F, 1e-8) / F0)                 # (n,)
        i = int(np.clip(np.searchsorted(Ts, t), 1, len(Ts) - 1))   # scalar tenor bracket for this t
        T0, T1 = Ts[i - 1], Ts[i]; frac = (t - T0) / max(T1 - T0, 1e-9)
        (p0, d0, dd0), (p1, d1, dd1) = ders[i - 1], ders[i]
        iv0 = np.polyval(p0, k); iv1 = np.polyval(p1, k)     # (n,)
        w0 = iv0 ** 2 * T0; w1 = iv1 ** 2 * T1               # total variance at each tenor
        w = np.maximum((1 - frac) * w0 + frac * w1, 1e-8)
        dwdt = (w1 - w0) / max(T1 - T0, 1e-9)
        dk0 = np.polyval(d0, k); dk1 = np.polyval(d1, k)
        d2k0 = np.polyval(dd0, k); d2k1 = np.polyval(dd1, k)
        dwk = (1 - frac) * 2 * iv0 * dk0 * T0 + frac * 2 * iv1 * dk1 * T1
        d2wk = ((1 - frac) * (2 * dk0 ** 2 + 2 * iv0 * d2k0) * T0
                + frac * (2 * dk1 ** 2 + 2 * iv1 * d2k1) * T1)
        denom = (1 - k / w * dwk
                 + 0.25 * (-0.25 - 1 / w + k ** 2 / w ** 2) * dwk ** 2
                 + 0.5 * d2wk)
        return np.sqrt(np.clip(dwdt / np.maximum(denom, 1e-3), 1e-4, 25.0))
    return sigma_lv


def price_barrier(FT, alive, F0, K, df, call=True):
    payoff = np.where(alive, np.maximum(FT - K, 0) if call else np.maximum(K - FT, 0), 0.0)
    return df * payoff.mean()


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", type=int, default=60000)
    ap.add_argument("--steps", type=int, default=80)
    args = ap.parse_args()
    F0, df = 24000.0, 0.999
    heston = (3.0, 0.02, 0.7, -0.55, 0.018)                 # kappa,theta,sigma_v,rho,v0
    # DENSE term structure (fix for Dupire ∂w/∂T): corrected-style short smiles + more tenors so the
    # time-derivative is smooth. iv = c*k^2 + b*k + a; skew flattens & level rises gently with T.
    # (On real data: use the M2 corrected smiles at the short tenors + Heston smiles at 14/21/30d.)
    def smile(T):
        r = T * 365 / 11.0
        return np.array([6.0 - 2.0 * r, -0.9 + 0.4 * r, 0.13 + 0.02 * np.sqrt(r)])
    tenors = [(d / 365, smile(d / 365)) for d in [4, 7, 11, 14, 21]]
    lv = dupire_local_vol(tenors)
    T = 11 / 365

    # calibration check: does the SLV reproduce the target 11d smile?
    FT, alive, Lh = particle_slv(heston, lambda F, t: lv(F, t, F0), F0, T,
                                 n_steps=args.steps, n_paths=args.paths, seed=1)
    ks = np.array([-0.05, -0.02, 0.0, 0.02, 0.05])
    iv_slv = vanilla_smile_from_terminal(FT, F0, T, df, ks) * 100
    iv_tgt = np.polyval(smile(11 / 365), ks) * 100
    print("SLV calibration check (11d smile) — must reproduce the target before the barrier is trusted:")
    print(f"{'logK':>7}{'target IV':>11}{'SLV IV':>9}{'diff bps':>10}")
    for k, a, b in zip(ks, iv_tgt, iv_slv):
        print(f"{k:7.2f}{a:10.1f}%{b:8.1f}%{(b-a)*100:9.0f}")
    maxbps = np.max(np.abs(iv_slv - iv_tgt)) * 100
    print(f"mean leverage L: {Lh.mean():.2f} | max calib error {maxbps:.0f} bps "
          f"({'PASS <50' if maxbps < 50 else 'CHECK'})\n")

    # weekly up-and-out barrier call, KO at +4%
    barrier = ("up", F0 * 1.04); K = F0 * 1.00
    FT_h, alive_h, _ = particle_slv(heston, lambda F, t: 1.0 + 0 * F, F0, T, n_steps=args.steps,
                                    n_paths=args.paths, seed=2, barrier=barrier, l_clip=(1.0, 1.0))
    FT_s, alive_s, _ = particle_slv(heston, lambda F, t: lv(F, t, F0), F0, T, n_steps=args.steps,
                                    n_paths=args.paths, seed=2, barrier=barrier)
    ph = price_barrier(FT_h, alive_h, F0, K, df)
    ps = price_barrier(FT_s, alive_s, F0, K, df)
    print(f"weekly up-and-out call (KO +4%, strike ATM):")
    print(f"  pure Heston : {ph:8.2f}")
    print(f"  SLV (mkt)   : {ps:8.2f}")
    print(f"  mispricing  : {ph - ps:+8.2f}  ({(ph/ps - 1)*100:+.0f}% vs market-consistent)")
    print("\nHeston's shallow short-end skew misprices the barrier; the SLV reproduces the market "
          "smile AND has the dynamics to price the knock-out.")


if __name__ == "__main__":
    main()
