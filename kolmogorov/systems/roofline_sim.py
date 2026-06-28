"""Roofline simulator for Project KOLMOGOROV (plan §5 / §11.1).

Analytically tests sub-claim **C2 (systems)**: is it faster to *recompute* a
layer's weights from a small resident generator than to *reload* them from the
slowest storage tier?

Crossover condition (plan §2.3), per layer:

    t_reload    = bytes_per_layer / BW_slow
    t_recompute = flops_per_layer / (C_gpu * utilization)
    recompute wins  <=>  flops_per_layer < bytes_per_layer * (C_gpu * u / BW_slow)

The right-hand factor  K = C_gpu * u / BW_slow  is the **FLOP budget per byte**
of produced weight before hitting the wall. A rank-`r` generator that emits a
`d x d` weight block from two `d x r` factors costs ~`2*d*d*r` FLOPs to produce
`d*d` weights, i.e. ~`2*r` FLOPs per weight element. With fp16 weights (2 bytes)
that is ~`r` FLOPs/byte, so recompute wins while `r < K`.

This module is pure-numpy, no GPU, no downloads. It is the cheapest gate in the
plan: it tells us, per hardware profile, the maximum generator rank that still
beats reload — a hard go/no-go input before any model is touched.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


# ---------------------------------------------------------------------------
# Hardware / model profiles
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HardwareProfile:
    name: str
    bw_slow_bytes_per_s: float   # bandwidth of the tier weights stream from
    gpu_flops_per_s: float       # peak compute of the device doing recompute
    utilization: float = 0.30    # realistic sustained fraction of peak

    @property
    def flop_budget_per_byte(self) -> float:
        """K = C_gpu * u / BW_slow : FLOPs we may spend per byte produced."""
        return self.gpu_flops_per_s * self.utilization / self.bw_slow_bytes_per_s


@dataclass(frozen=True)
class ModelProfile:
    name: str
    n_params: int
    bitwidth: int = 16           # storage bitwidth of the weights being streamed
    n_layers: int = 1            # used only for per-token aggregation

    @property
    def bytes_total(self) -> float:
        return self.n_params * self.bitwidth / 8.0


# Representative storage tiers (order-of-magnitude, documented assumptions).
HARDWARE = {
    "nvme_gen4": HardwareProfile("NVMe Gen4 SSD", bw_slow_bytes_per_s=5e9,
                                 gpu_flops_per_s=1.0e14),   # ~100 TFLOP/s fp16 class
    "nvme_gen5": HardwareProfile("NVMe Gen5 SSD", bw_slow_bytes_per_s=12e9,
                                 gpu_flops_per_s=1.0e14),
    "sata_ssd":  HardwareProfile("SATA SSD",      bw_slow_bytes_per_s=0.5e9,
                                 gpu_flops_per_s=1.0e14),
    "dram":      HardwareProfile("CPU DRAM",      bw_slow_bytes_per_s=50e9,
                                 gpu_flops_per_s=1.0e14),
    "pcie5":     HardwareProfile("PCIe5 host->GPU", bw_slow_bytes_per_s=63e9,
                                 gpu_flops_per_s=1.0e14),
}

MODELS = {
    "llama3_8b":   ModelProfile("Llama-3 8B",   n_params=8_000_000_000,  n_layers=32),
    "llama2_70b":  ModelProfile("Llama-2 70B",  n_params=70_000_000_000, n_layers=80),
    "llama31_405b":ModelProfile("Llama-3.1 405B",n_params=405_000_000_000,n_layers=126),
}


# ---------------------------------------------------------------------------
# Core analytical model
# ---------------------------------------------------------------------------

@dataclass
class CrossoverResult:
    hardware: str
    model: str
    flop_budget_per_byte: float
    max_generator_rank: float        # largest rank r where recompute still wins
    reload_time_s: float             # full-model reload (one forward, batch=1)
    note: str = ""


def generator_flops_per_weight(rank: int) -> float:
    """FLOPs to *produce* one weight element from a rank-`r` low-rank generator.

    Ŵ = A @ B with A:(d,r), B:(r,d) -> producing d*d entries costs 2*d*d*r MACs,
    i.e. 2*r FLOPs per produced element. Independent of d, which is what makes
    the budget analysis clean.
    """
    return 2.0 * rank


def analyse(hardware_key: str, model_key: str,
            assumed_rank: int | None = None) -> CrossoverResult:
    hw = HARDWARE[hardware_key]
    m = MODELS[model_key]

    bytes_per_weight = m.bitwidth / 8.0
    # recompute wins while flops_per_weight < K * bytes_per_weight
    budget_flops_per_weight = hw.flop_budget_per_byte * bytes_per_weight
    # flops_per_weight = 2*r  =>  r_max = budget / 2
    r_max = budget_flops_per_weight / 2.0

    reload_time = m.bytes_total / hw.bw_slow_bytes_per_s

    note = ""
    if assumed_rank is not None:
        wins = generator_flops_per_weight(assumed_rank) < budget_flops_per_weight
        note = (f"rank={assumed_rank} -> "
                f"{'RECOMPUTE WINS' if wins else 'reload wins'}")

    return CrossoverResult(
        hardware=hw.name, model=m.name,
        flop_budget_per_byte=hw.flop_budget_per_byte,
        max_generator_rank=r_max,
        reload_time_s=reload_time,
        note=note,
    )


def monte_carlo_crossover(hardware_key: str, model_key: str,
                          n: int = 20000, seed: int = 0) -> dict:
    """Propagate uncertainty on utilization and bandwidth (plan §5).

    Returns the distribution of the max viable generator rank, so a go/no-go
    decision accounts for hardware variance rather than a single point estimate.
    """
    rng = np.random.default_rng(seed)
    hw = HARDWARE[hardware_key]
    m = MODELS[model_key]
    bytes_per_weight = m.bitwidth / 8.0

    # utilization ~ uniform[0.15, 0.45]; bandwidth ~ +-30% lognormal-ish jitter
    u = rng.uniform(0.15, 0.45, size=n)
    bw = hw.bw_slow_bytes_per_s * np.exp(rng.normal(0.0, 0.30, size=n))
    k = hw.gpu_flops_per_s * u / bw
    r_max = k * bytes_per_weight / 2.0

    return {
        "hardware": hw.name,
        "model": m.name,
        "r_max_p05": float(np.percentile(r_max, 5)),
        "r_max_p50": float(np.percentile(r_max, 50)),
        "r_max_p95": float(np.percentile(r_max, 95)),
    }


def print_report() -> None:
    print("=" * 74)
    print("Project KOLMOGOROV - Roofline crossover (C2 gate)")
    print("recompute beats reload while generator rank r < r_max")
    print("=" * 74)
    print(f"{'hardware':<18}{'budget FLOP/byte':>18}{'r_max':>12}"
          f"{'  (MC p05..p95)':>22}")
    print("-" * 74)
    for hk in ("sata_ssd", "nvme_gen4", "nvme_gen5", "pcie5", "dram"):
        res = analyse(hk, "llama2_70b")
        mc = monte_carlo_crossover(hk, "llama2_70b")
        print(f"{res.hardware:<18}{res.flop_budget_per_byte:>18,.0f}"
              f"{res.max_generator_rank:>12,.0f}"
              f"   {mc['r_max_p05']:>7,.0f}..{mc['r_max_p95']:<10,.0f}")
    print("-" * 74)
    print("Reading: even on a slow SATA SSD, any low-rank generator with rank")
    print("below r_max produces weights faster than reloading them. The whole")
    print("question of C2 reduces to: 'does a viable generator have rank < r_max?'")
    print("which C1 (the ML experiments) must answer.")
    print("=" * 74)


if __name__ == "__main__":
    print_report()
