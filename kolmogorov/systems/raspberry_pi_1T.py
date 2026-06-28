"""Can a 1-trillion-parameter model run on a single Raspberry Pi, offline?

A grounded feasibility calculator, using the same roofline logic as the rest of
the project (Theorem 4: per-token boundary traffic is the floor). It separates
the two distinct questions the dream conflates:

  (1) CAN it run at all (store + produce correct tokens)?  -> a memory question.
  (2) Can it run at USABLE speed (interactive)?            -> a bandwidth question.

All hardware numbers are documented, conservative order-of-magnitude assumptions
for a Raspberry Pi 5 + attached USB3 SSD. Pure arithmetic, no GPU, no downloads.
"""
from __future__ import annotations

from dataclasses import dataclass

# --- Raspberry Pi 5 class, documented assumptions ---------------------------
RAM_GB = 8.0
STORAGE = {
    "microSD":   0.09,   # GB/s, typical UHS-I card
    "USB3_SSD":  0.40,   # GB/s, USB3.0 SATA SSD on a Pi (real-world)
    "USB3_NVMe": 0.90,   # GB/s, best case NVMe-over-USB / Pi5 PCIe HAT
}
N_PARAMS = 1_000_000_000_000   # 1 trillion


@dataclass
class Scenario:
    name: str
    bits: float            # storage bit-width per weight
    active_fraction: float # 1.0 = dense; <1 = MoE / contextual sparsity

    def bytes_per_token(self) -> float:
        return N_PARAMS * self.active_fraction * self.bits / 8.0


SCENARIOS = [
    Scenario("dense fp16",            16,   1.00),
    Scenario("dense 4-bit",            4,   1.00),
    Scenario("dense 2-bit (BitNet)",   2,   1.00),
    Scenario("MoE 4-bit  (5% active)", 4,   0.05),
    Scenario("MoE 2-bit  (5% active)", 2,   0.05),
    Scenario("MoE 1.58b (3% active)",  1.58, 0.03),
    Scenario("+gen weights (H_eps/n~0.3%, 2b)", 2, 0.003),  # speculative C1 target
]


def humantime(s: float) -> str:
    if s < 1:    return f"{s*1000:.0f} ms"
    if s < 90:   return f"{s:.1f} s"
    if s < 5400: return f"{s/60:.1f} min"
    return f"{s/3600:.1f} h"


def report():
    print("=" * 78)
    print("1 TRILLION PARAMETERS ON A RASPBERRY PI — feasibility (per-token time)")
    print(f"RAM={RAM_GB:.0f}GB  |  N={N_PARAMS:,} params  |  Theorem 4 floor: stream"
          " ~active*bits per token")
    print("=" * 78)
    for tier, bw in STORAGE.items():
        print(f"\n storage tier: {tier}  ({bw:.2f} GB/s)")
        print(f"   {'scenario':<32}{'stream/token':>13}{'time/token':>12}")
        print("   " + "-" * 57)
        for sc in SCENARIOS:
            gb = sc.bytes_per_token() / 1e9
            t = gb / bw
            print(f"   {sc.name:<32}{gb:>10.1f}GB{humantime(t):>12}")

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    print("""\
 (1) CAN it run at all?  YES. Store a 2-bit 1T model (~250GB) on a USB SSD,
     stream it layer-by-layer (sub-layer if needed) through 8GB RAM — exactly
     AirLLM's mechanism. The Pi WILL produce correct tokens. Memory is not the
     blocker; you never hold the whole model resident.

 (2) At usable speed?  Depends entirely on bytes-streamed-per-token:
       - DENSE 1T: minutes-to-hours per token. Physics (SSD bandwidth) forbids
         interactive chat. Fine ONLY for offline/batch jobs.
       - MoE (sparse activation) + 2-bit: ~10-60 s/token on a Pi. Slow, but a
         REAL offline assistant for non-chat tasks.
       - The dream's speed hinges on cutting bytes/token toward the Theorem-4
         floor H_eps. Every lever we studied does exactly that:
           MoE/sparsity (active_fraction) x extreme quant (bits) x weight
           generation (H_eps/n). The last row shows the target regime.

 Honest bottom line: '1T offline on a weak Pi' is ALREADY TRUE for correctness
 and storage. The whole game is bytes-per-token, and that is precisely the
 quantity our theory bounds and our experiments shrink.""")
    print("=" * 78)


if __name__ == "__main__":
    report()
