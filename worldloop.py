"""WonderingQwenSol experiment 001: provenance-aware active inquiry.

No third-party packages needed for the baseline. Optional Qwen policy lives in
qwen_policy.py. The agent does not receive the world's hidden state.

Thinking is NOT established by this finite Bayesian simulator. It supplies a
transparent causal scaffold and controls for an eventual learned Qwen workspace.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Callable

# Hidden state: did the world change, did vision fail, did audio fail?
STATES = tuple((w, v, a) for w in (0, 1) for v in (0, 1) for a in (0, 1))
NAMES = ("world_changed", "vision_fault", "audio_fault")
# (which bit is measured, reading accuracy, resource cost).
# Visual/audio readings invert under a fault, while calibration measures fault.
PROBES = {
    "vision": (0.90, 1.0),
    "audio": (0.90, 1.0),
    "calibrate_vision": (0.85, 1.3),
    "calibrate_audio": (0.85, 1.3),
    "touch": (0.75, 1.6),
}


def truth_signal(state: tuple[int, int, int], probe: str) -> int:
    w, v, a = state
    return {
        "vision": w ^ v, "audio": w ^ a,
        "calibrate_vision": v, "calibrate_audio": a, "touch": w,
    }[probe]


def prior() -> dict[tuple[int, int, int], float]:
    # Designed uncertainty: one change most likely, multiple changes possible.
    mass = {0: 0.05, 1: 0.75, 2: 0.18, 3: 0.02}
    per = {0: 1, 1: 3, 2: 3, 3: 1}
    return {s: mass[sum(s)] / per[sum(s)] for s in STATES}


def entropy(weights) -> float:
    return -sum(p * math.log2(p) for p in weights if p > 0)


class Belief:
    def __init__(self, weights=None):
        self.p = dict(prior() if weights is None else weights)
        self._normalize()

    def _normalize(self):
        total = sum(self.p.values())
        if total <= 0:
            raise ValueError("posterior has zero mass")
        self.p = {k: v / total for k, v in self.p.items()}

    def copy(self) -> "Belief":
        return Belief(self.p.copy())

    def marginal(self, variable: int) -> float:
        return sum(p for s, p in self.p.items() if s[variable])

    def signature(self) -> dict[str, float]:
        return {name: round(self.marginal(i), 6) for i, name in enumerate(NAMES)}

    def likelihood(self, state, probe, observed) -> float:
        accuracy = PROBES[probe][0]
        return accuracy if truth_signal(state, probe) == observed else 1 - accuracy

    def probability_one(self, probe: str) -> float:
        return sum(p * self.likelihood(s, probe, 1) for s, p in self.p.items())

    def conditioned(self, probe: str, observed: int) -> "Belief":
        if observed not in (0, 1):
            raise ValueError("sensor reading must be 0 or 1")
        return Belief({s: p * self.likelihood(s, probe, observed)
                       for s, p in self.p.items()})

    def observe(self, probe: str, observed: int):
        self.p = self.conditioned(probe, observed).p

    def uncertainty(self, variable=None) -> float:
        if variable is None:
            return entropy(self.p.values())
        p = self.marginal(variable)
        return entropy((p, 1 - p))

    def imagination(self, probe: str, variable=None) -> dict:
        """Predict both outcomes WITHOUT committing either to the posterior."""
        p1 = self.probability_one(probe)
        before = self.uncertainty(variable)
        after = ((1 - p1) * self.conditioned(probe, 0).uncertainty(variable)
                 + p1 * self.conditioned(probe, 1).uncertainty(variable))
        return {"p_one": p1, "information_gain_bits": max(0.0, before - after)}

    def most_likely(self) -> tuple[int, int, int]:
        return max(self.p, key=self.p.get)


@dataclass
class World:
    seed: int
    state: tuple[int, int, int]

    def __post_init__(self):
        self.count = {probe: 0 for probe in PROBES}

    @classmethod
    def sample(cls, seed: int) -> "World":
        rng = random.Random(seed)
        r = rng.random()
        total = 0.0
        for state, p in prior().items():
            total += p
            if r < total:
                return cls(seed, state)
        return cls(seed, STATES[-1])

    def observe(self, probe: str) -> int:
        if probe not in PROBES:
            raise ValueError("unknown probe")
        occurrence = self.count[probe]
        self.count[probe] += 1
        # Counterfactual-consistent random potential outcomes per probe, per
        # occurrence: requesting vision never perturbs later audio noise.
        stamp = f"{self.seed}:{probe}:{occurrence}".encode("utf-8")
        r = int.from_bytes(hashlib.sha256(stamp).digest()[:8], "big") / 2**64
        target = truth_signal(self.state, probe)
        return target if r < PROBES[probe][0] else 1 - target


class Mind:
    """A persistent posterior with separate observed and imagined ledgers."""

    def __init__(self):
        self.belief = Belief()
        self.events = []
        self.imagined = []

    def see(self, probe: str, value: int):
        before = self.belief.signature()
        self.belief.observe(probe, value)
        self.events.append({
            "source": "observed", "probe": probe, "value": value,
            "before": before, "after": self.belief.signature(),
        })

    def wonder(self, probe: str, target=None):
        prediction = self.belief.imagination(probe, target)
        self.imagined.append({
            "source": "imagined", "probe": probe,
            "target": NAMES[target] if target is not None else "joint",
            "p_one": round(prediction["p_one"], 6),
            "information_gain_bits": round(prediction["information_gain_bits"], 6),
        })
        return prediction

    def next_question(self) -> int:
        """Select the most uncertain latent claim without a user-supplied goal."""
        score = [self.belief.uncertainty(i) * (1.1 if i == 0 else 1)
                 for i in range(3)]
        return max(range(3), key=lambda i: score[i])

    def active_probe(self, target: int) -> str:
        """Counterfactual expected-information choice, not a hardcoded rule."""
        # Penalize redundant, expensive measurements. Include a small incentive
        # to resolve OTHER latent variables rather than forgetting them.
        scores = {}
        for probe in PROBES:
            targeted = self.wonder(probe, target)["information_gain_bits"]
            joint = self.wonder(probe, None)["information_gain_bits"]
            scores[probe] = targeted + 0.12 * joint - 0.08 * PROBES[probe][1]
        return max(PROBES, key=lambda p: scores[p])


def run_episode(seed: int, policy="active", budget=4,
                qwen: Callable | None = None) -> dict:
    if budget < 0:
        raise ValueError("budget must be >= 0")
    if policy not in ("active", "random", "repeat", "qwen"):
        raise ValueError("unknown policy")
    if policy == "qwen" and qwen is None:
        raise ValueError("qwen policy requires a callback")
    world, mind = World.sample(seed), Mind()
    rng = random.Random(seed + 707071)
    # A first sensory report arrives unrequested. The world may have changed
    # OR a sensor may be unreliable; the agent is not told which.
    mind.see("vision", world.observe("vision"))
    steps = []
    for tick in range(budget):
        target = mind.next_question()
        if policy == "active":
            probe = mind.active_probe(target)
        elif policy == "random":
            probe = rng.choice(tuple(PROBES))
        elif policy == "repeat":
            probe = "vision"
        else:
            probe = qwen(mind, target)
            if probe not in PROBES:
                raise ValueError("Qwen selected invalid probe " + str(probe))
        # Every chosen action has a forecast stored BEFORE seeing the result.
        forecast = mind.wonder(probe, target)
        result = world.observe(probe)
        mind.see(probe, result)
        steps.append({
            "tick": tick + 1, "self_chosen_question": NAMES[target],
            "action": probe, "predicted_one": round(forecast["p_one"], 6),
            "observed": result, "source": "observed",
        })
    return {
        "seed": seed, "policy": policy,
        "hidden_truth": dict(zip(NAMES, world.state)),
        "posterior": mind.belief.signature(),
        "map": dict(zip(NAMES, mind.belief.most_likely())),
        "observations": mind.events,
        "counterfactuals": mind.imagined,
        "steps": steps,
        "probe_cost": sum(PROBES[o["probe"]][1] for o in mind.events),
    }


def metrics(records: list[dict]) -> dict:
    if not records:
        raise ValueError("empty records")
    variables = NAMES
    result = {}
    for name in variables:
        result[name + "_accuracy"] = round(sum(
            (r["posterior"][name] >= 0.5) == bool(r["hidden_truth"][name])
            for r in records) / len(records), 4)
    result["mean_brier"] = round(sum(
        (r["posterior"][name] - r["hidden_truth"][name]) ** 2
        for r in records for name in variables) / (len(records) * 3), 4)
    result["joint_map_accuracy"] = round(sum(
        all(r["map"][name] == r["hidden_truth"][name] for name in variables)
        for r in records) / len(records), 4)
    result["mean_probe_cost"] = round(sum(r["probe_cost"] for r in records) / len(records), 3)
    return result


def naive_replay(record: dict) -> dict:
    """Identical evidence, but falsely assume both channels are always sound."""
    wrong_model = Belief({(0, 0, 0): 0.5, (1, 0, 0): 0.5})
    for o in record["observations"]:
        wrong_model.observe(o["probe"], o["value"])
    replay = dict(record)
    replay["posterior"] = wrong_model.signature()
    replay["map"] = dict(zip(NAMES, wrong_model.most_likely()))
    return replay


def benchmark(start=10000, episodes=100, budget=4, qwen=None,
              policies=("active", "random", "repeat")) -> dict:
    results = {}
    for policy in policies:
        rows = [run_episode(seed, policy, budget, qwen)
                for seed in range(start, start + episodes)]
        results[policy] = {"metrics": metrics(rows), "examples": rows[:3]}
        if policy == "active":
            results["no_fault_model_same_evidence"] = {
                "metrics": metrics([naive_replay(r) for r in rows]),
                "note": "Same actions/measurements as active, but only world-change hypotheses.",
            }
    return {
        "status": "simulation, not Qwen cognitive result",
        "protocol": "All policies see paired hidden episodes and matched probe budgets.",
        "seed_start": start, "episodes": episodes, "followup_budget": budget,
        "probe_accuracy": {p: a for p, (a, _) in PROBES.items()},
        "results": results,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=10000)
    p.add_argument("--budget", type=int, default=4)
    p.add_argument("--policy", choices=("all", "active", "random", "repeat", "qwen"),
                   default="all")
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--device", default="auto")
    p.add_argument("--out", type=Path, default=Path("results/experiment001.json"))
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    if args.episodes < 1 or args.budget < 0:
        p.error("episodes must be positive and budget nonnegative")
    if args.out.exists() and not args.overwrite:
        p.error(f"{args.out} exists; use --overwrite to replace intentionally")
    qwen = None
    if args.policy == "qwen":
        from qwen_policy import QwenPolicy
        qwen = QwenPolicy(args.model, args.device)
    policies = ("active", "random", "repeat") if args.policy == "all" else (args.policy,)
    report = benchmark(args.seed, args.episodes, args.budget, qwen, policies)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(args.out.suffix + ".tmp")
    tmp.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    tmp.replace(args.out)
    print(json.dumps({k: v["metrics"] for k, v in report["results"].items()}, indent=2))
    print("Saved", args.out)


if __name__ == "__main__":
    main()
