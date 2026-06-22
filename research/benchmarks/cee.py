"""Cognitive Efficiency Engine — scoring core.

Implements the metrics defined in ../cognitive-efficiency-engine.md. This module
does NOT run agents. It *measures* them: it consumes run logs (JSONL of TaskRun
records) and produces the CEQ report plus the ablation (marginal + cumulative).

Design rule (same discipline as Phase 1): measurement is independent of the system
measured, and total token cost `T` includes ALL overhead (memory reads, planner
deliberation, validation round-trips) — never just the "final" tokens.

Headline metric — Cognitive Efficiency Quotient:

        K * Q          complexity actually mastered, at proven quality
  CEQ = -------   =   --------------------------------------------------
        T / 1000                    per kilotoken

A capability's MULTIPLIER = CEQ(config) / CEQ(baseline). > 1 is the only admissible
evidence that an addition buys intelligence per unit of compute. <= 1 means the
graft costs more than it returns — drop it, however fashionable.

Run the built-in self-test (no external data needed):

    python3 cee.py --selftest

Score a real/synthetic log:

    python3 cee.py --runs sample_runs.synthetic.jsonl --baseline C0
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from statistics import mean
from typing import Iterable


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #
@dataclass
class TaskRun:
    """One measured run of one task under one configuration.

    All fields are observable from an instrumented session. `complexity` (K) and
    `human_baseline_s` (H_b) come from the FROZEN task spec, never self-reported by
    the system under test.
    """

    task_id: str
    config: str
    tokens_in: int
    tokens_out: int
    useful_decisions: int      # D_u
    total_decisions: int       # D_t
    actions_succeeded: int     # A_s
    actions_attempted: int     # A_a
    solution_quality: float    # Q in [0,1]
    complexity: float          # K in [0,1]  (from task spec)
    human_baseline_s: float    # H_b (from task spec; SOFT metric)
    oversight_s: float         # H_o

    @property
    def tokens(self) -> int:  # T
        return self.tokens_in + self.tokens_out

    def validate(self) -> None:
        if self.tokens <= 0:
            raise ValueError(f"{self.task_id}/{self.config}: non-positive tokens")
        for name, v in (("solution_quality", self.solution_quality), ("complexity", self.complexity)):
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"{self.task_id}/{self.config}: {name}={v} out of [0,1]")
        if self.total_decisions < self.useful_decisions:
            raise ValueError(f"{self.task_id}/{self.config}: useful>total decisions")
        if self.actions_attempted < self.actions_succeeded:
            raise ValueError(f"{self.task_id}/{self.config}: succeeded>attempted actions")


# --------------------------------------------------------------------------- #
# Per-run metrics (the value chain)
# --------------------------------------------------------------------------- #
def ceq(r: TaskRun) -> float:
    """Cognitive Efficiency Quotient: complexity mastered at quality, per kilotoken."""
    return (r.complexity * r.solution_quality) / (r.tokens / 1000.0)


def decision_yield(r: TaskRun) -> float:
    return r.useful_decisions / r.total_decisions if r.total_decisions else 0.0


def frugality(r: TaskRun) -> float:
    """Useful decisions per kilotoken."""
    return r.useful_decisions / (r.tokens / 1000.0)


def action_conversion(r: TaskRun) -> float:
    return r.actions_succeeded / r.actions_attempted if r.actions_attempted else 0.0


def human_leverage(r: TaskRun) -> float:
    """SOFT metric: net human seconds saved per kilotoken. Report with a caveat."""
    net = (r.human_baseline_s * r.solution_quality) - r.oversight_s
    return net / (r.tokens / 1000.0)


# --------------------------------------------------------------------------- #
# Aggregation per configuration
# --------------------------------------------------------------------------- #
@dataclass
class ConfigReport:
    config: str
    n: int
    mean_ceq: float
    mean_quality: float
    mean_tokens: float
    decision_yield: float
    frugality: float
    action_conversion: float
    human_leverage: float  # soft

    def line(self) -> str:
        return (
            f"{self.config:>10} | n={self.n:3d} | CEQ={self.mean_ceq:7.3f} | "
            f"Q={self.mean_quality:5.3f} | T={self.mean_tokens:8.0f} | "
            f"yield={self.decision_yield:5.3f} | frugal={self.frugality:6.3f} | "
            f"act={self.action_conversion:5.3f} | hum/kT={self.human_leverage:8.1f}"
        )


def aggregate(runs: list[TaskRun]) -> ConfigReport:
    assert runs, "no runs to aggregate"
    cfg = runs[0].config
    return ConfigReport(
        config=cfg,
        n=len(runs),
        mean_ceq=mean(ceq(r) for r in runs),
        mean_quality=mean(r.solution_quality for r in runs),
        mean_tokens=mean(r.tokens for r in runs),
        decision_yield=mean(decision_yield(r) for r in runs),
        frugality=mean(frugality(r) for r in runs),
        action_conversion=mean(action_conversion(r) for r in runs),
        human_leverage=mean(human_leverage(r) for r in runs),
    )


# --------------------------------------------------------------------------- #
# Ablation
# --------------------------------------------------------------------------- #
@dataclass
class Ablation:
    config: str
    multiplier: float       # CEQ(config) / CEQ(baseline)
    verdict: str

    def line(self) -> str:
        return f"{self.config:>10} | x{self.multiplier:5.2f} | {self.verdict}"


def _verdict(mult: float) -> str:
    if mult > 1.10:
        return "MULTIPLIES (>1): keep — buys intelligence per token"
    if mult >= 0.95:
        return "NEUTRAL (~1): graft pays for itself but does not multiply"
    return "COSTS MORE THAN IT RETURNS (<1): drop, however popular"


def marginal_ablation(by_config: dict[str, ConfigReport], baseline: str) -> list[Ablation]:
    base = by_config[baseline].mean_ceq
    out = []
    for cfg, rep in by_config.items():
        if cfg == baseline:
            continue
        mult = rep.mean_ceq / base if base else float("inf")
        out.append(Ablation(cfg, mult, _verdict(mult)))
    return sorted(out, key=lambda a: a.multiplier, reverse=True)


# --------------------------------------------------------------------------- #
# IO + report
# --------------------------------------------------------------------------- #
def load_runs(path: Path) -> list[TaskRun]:
    runs: list[TaskRun] = []
    for i, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            rec = TaskRun(**json.loads(line))
            rec.validate()
        except Exception as e:  # noqa: BLE001 - want a clear, located error
            raise SystemExit(f"{path}:{i}: bad run record: {e}")
        runs.append(rec)
    return runs


def group_by_config(runs: Iterable[TaskRun]) -> dict[str, list[TaskRun]]:
    g: dict[str, list[TaskRun]] = {}
    for r in runs:
        g.setdefault(r.config, []).append(r)
    return g


def report(runs: list[TaskRun], baseline: str, synthetic: bool) -> None:
    groups = group_by_config(runs)
    if baseline not in groups:
        raise SystemExit(f"baseline config {baseline!r} not present in runs: {list(groups)}")
    by_config = {cfg: aggregate(rs) for cfg, rs in groups.items()}

    if synthetic:
        print("=" * 96)
        print("WARNING: synthetic run log. Numbers below test the ARITHMETIC, not any real system.")
        print("=" * 96)

    print("\nPer-configuration (cost T includes ALL overhead):")
    for cfg in sorted(by_config):
        print("  " + by_config[cfg].line())

    print(f"\nMarginal ablation vs baseline {baseline!r}  (multiplier = CEQ_cfg / CEQ_base):")
    for ab in marginal_ablation(by_config, baseline):
        print("  " + ab.line())

    print("\nReminder: hum/kT is the SOFT metric (human-time estimate). No conclusion "
          "should rest on it alone. See cognitive-efficiency-engine.md, threats to validity.")


# --------------------------------------------------------------------------- #
# Self-test (validates the math with no external data)
# --------------------------------------------------------------------------- #
def _selftest() -> int:
    # A baseline run and an "improved" run that doubles useful output at +25% tokens.
    base = TaskRun("t", "C0", 800, 200, useful_decisions=2, total_decisions=5,
                   actions_succeeded=1, actions_attempted=2, solution_quality=0.5,
                   complexity=0.8, human_baseline_s=600, oversight_s=120)
    better = TaskRun("t", "C2", 1000, 250, useful_decisions=4, total_decisions=5,
                     actions_succeeded=2, actions_attempted=2, solution_quality=1.0,
                     complexity=0.8, human_baseline_s=600, oversight_s=60)
    base.validate(); better.validate()

    # CEQ_base = 0.8*0.5 / (1000/1000) = 0.4
    assert abs(ceq(base) - 0.4) < 1e-9, ceq(base)
    # CEQ_better = 0.8*1.0 / (1250/1000) = 0.64
    assert abs(ceq(better) - 0.64) < 1e-9, ceq(better)
    # multiplier 0.64/0.4 = 1.6 -> MULTIPLIES
    rep = {"C0": aggregate([base]), "C2": aggregate([better])}
    abl = marginal_ablation(rep, "C0")
    assert abs(abl[0].multiplier - 1.6) < 1e-9, abl[0].multiplier
    assert "MULTIPLIES" in abl[0].verdict

    # A graft that adds tokens without quality must score <1.
    bloat = TaskRun("t", "C1", 2000, 500, useful_decisions=2, total_decisions=8,
                    actions_succeeded=1, actions_attempted=2, solution_quality=0.5,
                    complexity=0.8, human_baseline_s=600, oversight_s=120)
    rep2 = {"C0": aggregate([base]), "C1": aggregate([bloat])}
    abl2 = marginal_ablation(rep2, "C0")
    assert abl2[0].multiplier < 1.0, abl2[0].multiplier
    assert "COSTS MORE" in abl2[0].verdict

    print("selftest OK: CEQ, multiplier, and 'drop the bloated graft' all verified.")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Cognitive Efficiency Engine scorer")
    ap.add_argument("--runs", type=Path, help="JSONL of TaskRun records")
    ap.add_argument("--baseline", default="C0", help="config id to treat as baseline")
    ap.add_argument("--selftest", action="store_true", help="run arithmetic self-test and exit")
    args = ap.parse_args(argv)

    if args.selftest:
        return _selftest()
    if not args.runs:
        ap.error("provide --runs PATH or --selftest")
    runs = load_runs(args.runs)
    synthetic = "synthetic" in args.runs.name
    report(runs, args.baseline, synthetic)
    return 0


if __name__ == "__main__":
    sys.exit(main())
