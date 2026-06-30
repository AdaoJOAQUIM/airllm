"""
The decisive test for the only positive signal in the whole study: does a
fractional-order memory operator hold up against a real long-memory baseline
(a diagonal state-space model, the core of S4/S4D) -- not just a naive AR window?

This is the experiment that gates publishability. If a learnable fractional order
(O(1) memory parameters) matches or beats a state-space model that needs O(state)
parameters *on data with the right kind of memory*, that is a real,
parameter-efficiency result. If it does not, we learn that cheaply.

To avoid rigging the terrain, we test on TWO memory regimes:

  * "power"  -- targets with polynomial (long-memory) decay. Fractional
    integration produces exactly power-law kernels, so this favors the
    fractional operator.
  * "exp"    -- targets with exponential / geometric decay. A diagonal SSM
    represents these exactly, so this favors the state-space model.

A fractional operator that wins on "power" and loses on "exp" is the honest,
publishable result: it locates *where* fractional memory is the right inductive
bias, rather than claiming universal superiority.

numpy-only; isolated from inference. Run:

    python -m airllm.trinition.benchmark_longmemory --quick
    python -m airllm.trinition.benchmark_longmemory
"""

from __future__ import annotations

import argparse
from typing import Dict, List, Tuple

import numpy as np

from .benchmark import ridge_fit, ridge_predict, _mse, readout_params
from .learn_algebra import _adam_step


# ---------------------------------------------------------------------------
# Data: long-memory regression. x is white noise; y_T is a functional of the
# whole history with either power-law or exponential memory.
# ---------------------------------------------------------------------------

def _power_kernel(T: int, q: float) -> np.ndarray:
    """Polynomial-decay memory kernel ~ k^{q-1} (fractional-integration shape)."""
    k = np.arange(T)
    w = (k + 1.0) ** (q - 1.0)
    return w / np.linalg.norm(w)


def _exp_kernel(T: int, rho: float) -> np.ndarray:
    """Exponential-decay memory kernel ~ rho^k (state-space shape)."""
    k = np.arange(T)
    w = rho ** k
    return w / np.linalg.norm(w)


def make_dataset(n: int, T: int, regime: str, rng: np.random.Generator,
                 noise: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """Return X (n, T) white-noise sequences and y (n, 1) = <kernel, reversed x>."""
    X = rng.normal(size=(n, T))
    if regime == "power":
        ker = _power_kernel(T, q=0.4)
    elif regime == "exp":
        ker = _exp_kernel(T, rho=0.9)
    else:
        raise ValueError(regime)
    # y_T depends on the past: most recent sample weighted by ker[0].
    y = (X[:, ::-1] @ ker).reshape(-1, 1)
    y = y + noise * rng.normal(size=y.shape)
    return X, y


# ---------------------------------------------------------------------------
# Model 1: fractional-order memory with a LEARNABLE order q (O(1) memory params).
# ---------------------------------------------------------------------------

def _gl_weights_grad(order: float, n: int):
    """Return GL weights w_k for the given order and their derivative dw/d(order).
    w_0 = 1; w_k = w_{k-1} (k-1-order)/k."""
    w = np.empty(n + 1)
    dw = np.empty(n + 1)
    w[0], dw[0] = 1.0, 0.0
    for k in range(1, n + 1):
        factor = (k - 1.0 - order) / k
        w[k] = w[k - 1] * factor
        dw[k] = dw[k - 1] * factor + w[k - 1] * (-1.0 / k)
    return w, dw


def frac_feature_and_grad(X: np.ndarray, q: float, memory: int):
    """Fractional-integration feature of order q over the last `memory` samples,
    and its derivative w.r.t. q. Returns (feat (N,), dfeat_dq (N,))."""
    order = -q                                   # fractional integration
    w, dw_dorder = _gl_weights_grad(order, memory - 1)
    dw_dq = -dw_dorder                           # chain rule: d/dq, order=-q
    window = X[:, -memory:][:, ::-1]             # align k=0 with most recent
    feat = window @ w
    dfeat = window @ dw_dq
    return feat, dfeat


def train_fractional(Xtr, ytr, Xte, yte, memory, epochs, lr, seed=0):
    """Learn q + a 2-feature linear readout [frac, last] -> y. ~4 params total,
    of which the memory mechanism is a SINGLE parameter q."""
    rng = np.random.default_rng(seed)
    params = {
        "qraw": np.array([0.0]),                 # q = sigmoid(qraw) in (0,1)
        "w": rng.normal(scale=0.1, size=(2, 1)),
        "b": np.zeros((1, 1)),
    }
    state = {k: (np.zeros_like(v), np.zeros_like(v)) for k, v in params.items()}
    n = Xtr.shape[0]

    def featurize(X, q):
        feat, dfeat = frac_feature_and_grad(X, q, memory)
        last = X[:, -1]
        F = np.stack([feat, last], axis=1)       # (N, 2)
        return F, dfeat

    for ep in range(1, epochs + 1):
        q = 1.0 / (1.0 + np.exp(-params["qraw"][0]))
        F, dfeat = featurize(Xtr, q)
        pred = F @ params["w"] + params["b"]
        diff = pred - ytr
        dpred = (2.0 / n) * diff
        gw = F.T @ dpred
        gb = dpred.sum(axis=0, keepdims=True)
        # gradient to q through the fractional feature (first readout channel).
        dF0 = dpred @ params["w"].T              # (N, 2)
        dq_feat = float((dF0[:, 0] * dfeat).sum())
        dq = dq_feat * q * (1.0 - q)             # sigmoid derivative
        grads = {"qraw": np.array([dq]), "w": gw, "b": gb}
        _adam_step(params, grads, state, lr, ep)

    q = 1.0 / (1.0 + np.exp(-params["qraw"][0]))
    Fte, _ = featurize(Xte, q)
    pred = Fte @ params["w"] + params["b"]
    n_params = 1 + 2 + 1                          # q + readout(2) + bias
    return _mse(pred, yte), n_params, q


# ---------------------------------------------------------------------------
# Model 2: diagonal linear state-space model (the S4/S4D core), learned by BPTT.
# h_t = a (.) h_{t-1} + b x_t ;  y = c . h_T + d.   ~3*state + 1 params.
# ---------------------------------------------------------------------------

def train_ssm(Xtr, ytr, Xte, yte, state_dim, epochs, lr, seed=0):
    rng = np.random.default_rng(seed)
    params = {
        # a in (0,1) via sigmoid(araw) for stable decay.
        "araw": rng.normal(scale=0.5, size=(state_dim,)),
        "b": rng.normal(scale=0.3, size=(state_dim,)),
        "c": rng.normal(scale=0.3, size=(state_dim,)),
        "d": np.zeros(1),
    }
    state = {k: (np.zeros_like(v), np.zeros_like(v)) for k, v in params.items()}
    n, T = Xtr.shape

    def forward(X, a):
        h = np.zeros((X.shape[0], state_dim))
        hs = []
        for t in range(T):
            h = h * a + np.outer(X[:, t], params["b"])
            hs.append(h)
        return h, hs

    for ep in range(1, epochs + 1):
        a = 1.0 / (1.0 + np.exp(-params["araw"]))
        hT, hs = forward(Xtr, a)
        pred = hT @ params["c"][:, None] + params["d"]
        diff = pred - ytr                        # (n,1)
        dpred = (2.0 / n) * diff                 # (n,1)
        gc = (hT * dpred).sum(axis=0)
        gd = dpred.sum(axis=0)
        # BPTT for a and b.
        dh = dpred @ params["c"][None, :]        # (n, state)
        ga = np.zeros(state_dim)
        gb = np.zeros(state_dim)
        for t in reversed(range(T)):
            h_prev = hs[t - 1] if t > 0 else np.zeros((n, state_dim))
            ga += (dh * h_prev).sum(axis=0)
            gb += (dh * Xtr[:, t][:, None]).sum(axis=0)
            dh = dh * a                          # propagate through h_t = a*h_{t-1}+...
        araw = params["araw"]
        sig = 1.0 / (1.0 + np.exp(-araw))
        grads = {
            "araw": ga * sig * (1.0 - sig),
            "b": gb,
            "c": gc,
            "d": gd,
        }
        gn = np.sqrt(sum(float(np.sum(g * g)) for g in grads.values()))
        if gn > 5.0:
            for k in grads:
                grads[k] *= 5.0 / gn
        _adam_step(params, grads, state, lr, ep)

    a = 1.0 / (1.0 + np.exp(-params["araw"]))
    hT, _ = forward(Xte, a)
    pred = hT @ params["c"][:, None] + params["d"]
    n_params = 3 * state_dim + 1
    return _mse(pred, yte), n_params


# ---------------------------------------------------------------------------
# Model 3: linear window (AR) baseline -- closed-form ridge.
# ---------------------------------------------------------------------------

def eval_window(Xtr, ytr, Xte, yte, k):
    Ftr, Fte = Xtr[:, -k:], Xte[:, -k:]
    W = ridge_fit(Ftr, ytr)
    return _mse(ridge_predict(Fte, W), yte), readout_params(k, 1)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _row(label, params, mse, width=12):
    return f"  {label:<34}{params:>6}{mse:>{width}.4f}"


def run_regime(regime, rng, n_train, n_test, T, epochs):
    Xtr, ytr = make_dataset(n_train, T, regime, rng)
    Xte, yte = make_dataset(n_test, T, regime, rng)
    lines = [f"Regime '{regime}'  (T={T}, train={n_train})",
             f"  {'model':<34}{'#par':>6}{'test MSE':>12}"]

    mse_f, p_f, q = train_fractional(Xtr, ytr, Xte, yte, memory=T,
                                     epochs=epochs, lr=2e-2, seed=0)
    lines.append(_row(f"fractional (learned q={q:.3f})", p_f, mse_f))

    best = ("fractional", mse_f, p_f)
    for sd in (2, 4, 8):
        mse_s, p_s = train_ssm(Xtr, ytr, Xte, yte, state_dim=sd,
                               epochs=epochs, lr=2e-2, seed=0)
        lines.append(_row(f"diagonal SSM (state={sd})", p_s, mse_s))
        if mse_s < best[1]:
            best = (f"SSM(state={sd})", mse_s, p_s)
    for k in (2, 8):
        mse_w, p_w = eval_window(Xtr, ytr, Xte, yte, k)
        lines.append(_row(f"linear window AR(k={k})", p_w, mse_w))
    return lines, mse_f, p_f, best


def main(argv=None) -> str:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    n_train = 1500 if args.quick else 4000
    n_test = 1500
    T = 48 if args.quick else 64
    epochs = 250 if args.quick else 600

    out = []
    out.append("=" * 60)
    out.append(" Fractional memory vs. state-space model (S4 core)")
    out.append(" Equal-ish parameter budgets; two memory regimes.")
    out.append("=" * 60)
    out.append("")

    summary = []
    for regime in ("power", "exp"):
        lines, mse_f, p_f, best = run_regime(regime, rng, n_train, n_test, T, epochs)
        out.extend(lines)
        out.append(f"  -> best: {best[0]} (MSE {best[1]:.4f}, {best[2]} params)")
        out.append("")
        summary.append((regime, mse_f, p_f, best))

    out.append("-" * 60)
    out.append("Honest read-out:")
    out.append("  * 'power' regime rewards polynomial long memory -> the single-")
    out.append("    parameter fractional order should be competitive with, or beat,")
    out.append("    an SSM that spends many parameters to approximate it.")
    out.append("  * 'exp' regime is the SSM's home turf -> it should win there.")
    out.append("  * A fractional win on 'power' at far fewer params is the real,")
    out.append("    bounded, publishable result. A loss on 'exp' is expected and")
    out.append("    must be reported, not hidden.")
    report = "\n".join(out)
    print(report)
    return report


if __name__ == "__main__":
    main()
