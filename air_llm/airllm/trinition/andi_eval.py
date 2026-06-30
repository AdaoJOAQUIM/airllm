"""
Real-benchmark evaluation on the AnDi (Anomalous Diffusion) Challenge data.

This is the gate from NATURE_PATH.md: does the fractional operator hold up on the
*community-standard* anomalous-diffusion benchmark, not just our own synthetic
fGn? It uses the official `andi_datasets` generator (Muñoz-Gil et al., Nature
Communications 2021) to produce trajectories from the five canonical diffusion
models (ATTM, CTRW, fBM, LW, SBM), and evaluates AnDi **Task 1**: inferring the
anomalous exponent `alpha` from a single trajectory. The metric is the
challenge's own: mean absolute error (MAE) on `alpha`.

We compare two *parameter-light, interpretable* estimators:

  * fractional  -- fit a single fractional-difference order `d` to the increments
    by one-step prediction; `alpha_hat = 2(d + 1/2) = 2d + 1` (exact for fBM).
  * TA-MSD      -- the classical time-averaged mean-squared-displacement log-log
    slope, the standard physics baseline.

Honest scope:
  * The fractional estimator is calibrated to fBM/fGn memory. We therefore expect
    it to be strong on fBM (and partly SBM) and *biased* on CTRW/ATTM/LW, whose
    anomalous diffusion comes from waiting-times / step-length statistics, not
    Gaussian long-memory. Reporting that per-model gap honestly IS the result:
    it says exactly which physics the fractional algebra matches.
  * This is NOT a claim of beating the AnDi state of the art. The challenge
    winners are trained deep nets (convolutional/recurrent) and achieve markedly
    lower MAE; reproducing them needs a GPU training stack. This module
    establishes the interpretable-baseline performance and the per-model regime
    map -- the prerequisite an interpretability-focused paper builds on.

Requires `andi_datasets` and numpy. Run:

    python -m airllm.trinition.andi_eval --quick
    python -m airllm.trinition.andi_eval
"""

from __future__ import annotations

import argparse
from typing import Dict, List, Tuple

import numpy as np

from .anomalous_diffusion import frac_diff_coeffs


# ---------------------------------------------------------------------------
# Data (official AnDi generator)
# ---------------------------------------------------------------------------

ANDI_MODELS = ["attm", "ctrw", "fbm", "lw", "sbm"]

# Each model only admits anomalous exponents in a physical range:
# ATTM/CTRW are subdiffusive (alpha<=1), Levy walks superdiffusive (alpha>1),
# fBM/SBM span (0, 2). Evaluate each on its valid range.
VALID_EXPONENTS = {
    "attm": [0.4, 0.6, 0.8, 1.0],
    "ctrw": [0.4, 0.6, 0.8, 0.95],
    "fbm": [0.4, 0.7, 1.0, 1.3, 1.6],
    "lw": [1.1, 1.4, 1.7],
    "sbm": [0.4, 0.7, 1.0, 1.3, 1.6],
}
VALID_EXPONENTS_QUICK = {
    "attm": [0.5, 0.9],
    "ctrw": [0.5, 0.9],
    "fbm": [0.5, 1.0, 1.5],
    "lw": [1.3, 1.7],
    "sbm": [0.5, 1.0, 1.5],
}


def generate_andi(model_idx: int, exponents: List[float], n_per: int,
                  T: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return positions (N, T) and true alpha (N,) for one diffusion model."""
    from andi_datasets.datasets_theory import datasets_theory
    np.random.seed(seed)
    dt = datasets_theory()
    rows = np.asarray(dt.create_dataset(T=T, N_models=n_per, exponents=exponents,
                                        models=[model_idx]))
    alpha = rows[:, 1].astype(float)
    pos = rows[:, 2:].astype(float)
    return pos, alpha


def _normalize_increments(pos: np.ndarray) -> np.ndarray:
    inc = np.diff(pos, axis=1)
    inc = inc - inc.mean(axis=1, keepdims=True)
    inc = inc / (inc.std(axis=1, keepdims=True) + 1e-9)
    return inc


# ---------------------------------------------------------------------------
# Estimator 1: fractional order -> anomalous exponent (fBM-calibrated).
# ---------------------------------------------------------------------------

def estimate_alpha_fractional(pos: np.ndarray, memory: int = 32,
                              grid: int = 49) -> np.ndarray:
    """Per-trajectory: fit d by one-step prediction MSE on normalized increments;
    return alpha_hat = 2d + 1 clipped to [0, 2]."""
    inc = _normalize_increments(pos)
    n, Tm1 = inc.shape
    ds = np.linspace(-0.49, 0.49, grid)
    best_mse = np.full(n, np.inf)
    best_d = np.zeros(n)
    for d in ds:
        pi = frac_diff_coeffs(d, memory)[1:]              # k = 1..memory
        sse = np.zeros(n)
        cnt = 0
        for t in range(memory, Tm1):
            past = inc[:, t - memory:t][:, ::-1]           # lags 1..memory
            pred = -(past @ pi)
            sse += (inc[:, t] - pred) ** 2
            cnt += 1
        mse = sse / max(cnt, 1)
        upd = mse < best_mse
        best_mse[upd] = mse[upd]
        best_d[upd] = d
    return np.clip(2.0 * best_d + 1.0, 0.0, 2.0)


# ---------------------------------------------------------------------------
# Estimator 2: TA-MSD log-log slope (classical physics baseline).
# ---------------------------------------------------------------------------

def estimate_alpha_tamsd(pos: np.ndarray, tau_max: int = None) -> np.ndarray:
    """Per-trajectory anomalous exponent from the time-averaged MSD slope."""
    n, T = pos.shape
    if tau_max is None:
        tau_max = max(4, T // 4)
    taus = np.arange(1, tau_max + 1)
    logt = np.log(taus)
    logt = logt - logt.mean()
    denom = (logt ** 2).sum()
    out = np.zeros(n)
    for i in range(n):
        r = pos[i]
        msd = np.array([np.mean((r[tau:] - r[:-tau]) ** 2) + 1e-12 for tau in taus])
        y = np.log(msd)
        slope = (logt * (y - y.mean())).sum() / denom
        out[i] = slope
    return np.clip(out, 0.0, 2.0)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _mae(a, b):
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


def run(exponents: Dict[str, List[float]], n_per: int, T: int,
        seed: int) -> Dict[str, dict]:
    results = {}
    for mi, name in enumerate(ANDI_MODELS):
        try:
            pos, alpha = generate_andi(mi, exponents[name], n_per, T, seed + mi)
        except Exception as exc:                           # some models reject some alphas
            results[name] = {"error": str(exc)[:60]}
            continue
        a_frac = estimate_alpha_fractional(pos)
        a_msd = estimate_alpha_tamsd(pos)
        results[name] = {
            "n": len(alpha),
            "mae_frac": _mae(a_frac, alpha),
            "mae_msd": _mae(a_msd, alpha),
        }
    return results


def main(argv=None) -> str:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    T = 128 if not args.quick else 64
    n_per = 120 if not args.quick else 40
    exponents = VALID_EXPONENTS_QUICK if args.quick else VALID_EXPONENTS

    res = run(exponents, n_per, T, args.seed)

    out = []
    out.append("=" * 64)
    out.append(" AnDi Challenge Task 1: anomalous-exponent inference")
    out.append(" Official andi_datasets generator; metric = MAE on alpha")
    out.append("=" * 64)
    out.append("")
    out.append(f"  T={T}, trajectories/model = n_per x #valid-exponents "
               f"(n_per={n_per})")
    out.append("")
    out.append(f"  {'model':<8}{'n':>6}{'MAE fractional[1p]':>20}{'MAE TA-MSD':>14}")
    out.append("  " + "-" * 46)
    maf, mam = [], []
    for name in ANDI_MODELS:
        r = res[name]
        if "error" in r:
            out.append(f"  {name:<8}{'--':>6}{'(skipped: ' + r['error'] + ')':>30}")
            continue
        out.append(f"  {name:<8}{r['n']:>6}{r['mae_frac']:>20.3f}{r['mae_msd']:>14.3f}")
        maf.append(r["mae_frac"])
        mam.append(r["mae_msd"])
    out.append("  " + "-" * 46)
    out.append(f"  {'MEAN':<8}{'':>6}{np.mean(maf):>20.3f}{np.mean(mam):>14.3f}")
    out.append("")
    out.append("Honest read-out:")
    out.append("  * fBM is the fractional estimator's home: a single order d should")
    out.append("    give the lowest MAE there. SBM partly. CTRW/ATTM/LW arise from")
    out.append("    waiting-time / step statistics, not Gaussian long memory, so the")
    out.append("    fBM-calibrated order is mis-specified -- a higher, HONEST MAE that")
    out.append("    maps exactly which physics the fractional algebra matches.")
    out.append("  * Neither estimator here beats the AnDi deep-learning winners,")
    out.append("    which need a trained GPU model. This sets the interpretable-")
    out.append("    baseline and the regime map -- the prerequisite for the")
    out.append("    interpretability/efficiency argument in NATURE_PATH.md.")
    report = "\n".join(out)
    print(report)
    return report


if __name__ == "__main__":
    main()
