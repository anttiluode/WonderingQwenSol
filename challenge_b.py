"""Challenge B: world changes after the first sighting.

Calibration probes expose sensor health, never the world's *new* value.
Bayes is used only by explicit reference/heuristic policies, NOT by the
learned 002B controller. This is a separately versioned benchmark.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random

from worldloop import Belief, NAMES, PROBES, STATES, entropy, metrics


class ChangedWorld:
    """The first vision reading observes baseline w=0; all later probes see
    an independently drawn final w. Sensor faults are persistent."""
    def __init__(self, seed, final_w, vision_fault, audio_fault):
        self.seed = seed
        self.state = (final_w, vision_fault, audio_fault)
        self.n = {name: 0 for name in PROBES}
        self.t = 0

    @classmethod
    def sample(cls, seed):
        r = random.Random(seed + 161711)
        return cls(seed, r.randrange(2), int(r.random() < 0.25),
                   int(r.random() < 0.25))

    def observe(self, probe):
        if probe not in PROBES:
            raise ValueError("unknown probe")
        w, v, a = self.state
        effective_w = 0 if self.t == 0 else w
        truth = {
            "vision": effective_w ^ v,
            "audio": effective_w ^ a,
            "calibrate_vision": v,
            "calibrate_audio": a,
            "touch": effective_w,
        }[probe]
        stamp = f"challenge-b:{self.seed}:{probe}:{self.n[probe]}".encode()
        noise = int.from_bytes(hashlib.sha256(stamp).digest()[:8], "big") / 2**64
        self.n[probe] += 1
        self.t += 1
        return truth if noise < PROBES[probe][0] else 1 - truth


def independent_prior():
    """At baseline w=0. Faults have independent Bernoulli(.25) priors."""
    weights = {}
    for s in STATES:
        w, v, a = s
        weights[s] = ((1 if w == 0 else 0) *
                      (0.25 if v else 0.75) *
                      (0.25 if a else 0.75))
    return Belief(weights)


def transition_after_first_vision(belief):
    """w_new ~ Bernoulli(.5) independently of past and faults."""
    sums = {(v, a): sum(p for (w1, v1, a1), p in belief.p.items()
                         if v1 == v and a1 == a)
            for v in (0, 1) for a in (0, 1)}
    return Belief({(w, v, a): 0.5 * sums[v, a] for w, v, a in STATES})


def query_target(b):
    h = [b.uncertainty(i) * (1.1 if i == 0 else 1.0) for i in range(3)]
    return max(range(3), key=lambda i: h[i])


def best_probe(b, q):
    return max(PROBES, key=lambda name: (
        b.imagination(name, q)["information_gain_bits"]
        + 0.12 * b.imagination(name, None)["information_gain_bits"]
        - 0.08 * PROBES[name][1]))


def run(seed, policy="active", budget=4):
    if policy not in {"active", "calibrate_alternate", "random", "repeat"}:
        raise ValueError("unknown policy")
    w = ChangedWorld.sample(seed)
    b = independent_prior()
    events = []
    # First observation: old world w=0, before the hidden transition.
    first = w.observe("vision")
    events.append({"probe": "vision", "value": first, "phase": "baseline"})
    b.observe("vision", first)
    b = transition_after_first_vision(b)
    random_probe = random.Random(seed + 771)
    for t in range(budget):
        if policy == "active":
            probe = best_probe(b, query_target(b))
        elif policy == "calibrate_alternate":
            probe = "calibrate_audio" if t % 2 == 0 else "calibrate_vision"
        elif policy == "repeat":
            probe = "vision"
        else:
            probe = random_probe.choice(tuple(PROBES))
        observed = w.observe(probe)
        events.append({"probe": probe, "value": observed, "phase": "after_change"})
        b.observe(probe, observed)
    p = {name: round(b.marginal(i), 6) for i, name in enumerate(NAMES)}
    guess = max(b.p, key=b.p.get)
    return {
        "seed": seed, "policy": policy,
        "hidden_truth": dict(zip(NAMES, w.state)),
        "posterior": p, "map": dict(zip(NAMES, guess)),
        "observations": events,
        "probe_cost": sum(PROBES[e["probe"]][1] for e in events),
    }


def benchmark(start=410000, episodes=200, budget=4):
    if episodes < 1 or budget < 0:
        raise ValueError("invalid size")
    all_results = {}
    for policy in ("active", "calibrate_alternate", "random", "repeat"):
        rows = [run(seed, policy, budget) for seed in range(start, start + episodes)]
        all_results[policy] = {"metrics": metrics(rows), "examples": rows[:3]}
    return {
        "status": "engineered causal challenge, not learned AI",
        "baseline": "w=0, one visual observation, then independent unobserved world change",
        "final_w": "Bernoulli(0.5) independent of both sensor faults",
        "sensor_fault_prob": 0.25,
        "seed_start": start, "episodes": episodes, "budget": budget,
        "results": all_results,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--seed", type=int, default=410000)
    p.add_argument("--budget", type=int, default=4)
    p.add_argument("--out", type=Path, default=Path("results/challenge_b.json"))
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    if args.out.exists() and not args.overwrite:
        p.error(f"{args.out} exists; use --overwrite")
    result = benchmark(args.seed, args.episodes, args.budget)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result_text = json.dumps(result, indent=2) + "\n"
    args.out.write_text(result_text, encoding="utf8")
    print(json.dumps({k: v["metrics"] for k, v in result["results"].items()}, indent=2))


if __name__ == "__main__":
    main()
