"""002B-2: probe frozen Qwen's early-layer workspace with learned q and belief.

This is an *intervention harness*, not a trained latent translator. The belief
signal is the GRU's marginal, converted into a length-matched blend of J-lens
'yes'/'no' vectors. Qwen receives NO exact Python Bayesian posterior.
Controls distinguish question, belief, random and clean interventions.

Requires an adjacent Qwenimagination checkout and its LOCAL cached V9 lens:
    python bridge_002b.py --checkpoint results/002b_a_seed42.pt --cases 2
No Qwen results are claimed until this actually runs on the user's GPU.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import torch

from learner002 import encode_observation
from learner002b import PulledInquirer, learned_episode
from worldloop import NAMES, ACTIONS

BAND = (4, 12)
DECOY_Q = "stone"
DECOY_B = "quiet"
Q_WORDS = ("world", "vision", "audio")
GOOD, BAD, RANDOM = "yes", "no", "cloud"
REQUIRED = (DECOY_Q, DECOY_B, *Q_WORDS, GOOD, BAD, RANDOM)


def get_qimag(path: Path):
    directory = path.resolve()
    if not (directory / "qimag" / "lens.py").exists():
        raise FileNotFoundError(f"{directory} is not a Qwenimagination checkout")
    sys.path.insert(0, str(directory))
    from qimag.model import load
    from qimag.lens import Lens, _hs, _replace, tap_layers
    from confirm import extend_lens
    return load, Lens, extend_lens, _hs, _replace, tap_layers


def load_controller(checkpoint: Path):
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = PulledInquirer(ckpt["hidden"], ckpt.get("slot_size", 12))
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt["challenge"]


@torch.no_grad()
def extract_state(model, history):
    """Replay sensor pairs ONLY through the trained GRU. No Python Bayes."""
    h, tension = model.empty_state(1, "cpu")
    for obs in history:
        x = torch.tensor([encode_observation(obs["probe"], obs["value"])])
        h, tension, _, scores, _, _, q, marg = model.step(x, h, tension)
    return int(q[0]), [float(v) for v in marg[0]], [float(v) for v in scores[0]]


class DoubleSwap:
    """Two simultaneous equal-norm coordinate exchanges across a layer band.

    Four vectors per layer: loaded decoy-Q, inserted Q, loaded decoy-belief,
    inserted belief. Pseudoinverse handles correlated J-lens directions.
    """
    def __init__(self, lm, lens, qword, p_yes, _hs, _replace, tap_layers,
                 band=BAND, random_q=False):
        self.lm, self.band = lm, list(range(band[0], band[1] + 1))
        self._hs, self._replace, self.tap_layers = _hs, _replace, tap_layers
        self.V, self.Gi = {}, {}
        for l in self.band:
            J = lens.J[l]
            a = J[lens.index[DECOY_Q]].float()
            b = J[lens.index[RANDOM if random_q else qword]].float()
            c = J[lens.index[DECOY_B]].float()
            # Continuous confidence drawn only from learned GRU marginal,
            # not a Python-computed Bayes posterior.
            d = (p_yes * J[lens.index[GOOD]].float()
                 + (1 - p_yes) * J[lens.index[BAD]].float())
            b = b * (a.norm() / b.norm().clamp_min(1e-12))
            d = d * (c.norm() / d.norm().clamp_min(1e-12))
            V = torch.stack((a, b, c, d))
            G = V @ V.T
            reg = 1e-4 * G.diagonal().mean().clamp_min(1e-8)
            self.V[l] = V
            self.Gi[l] = torch.linalg.pinv(G + reg * torch.eye(4,device=G.device))
        self.want = {}

    def coords(self, layer, z):
        return (z.float() @ self.V[layer].T) @ self.Gi[layer]

    @torch.no_grad()
    def run(self, ids, mode):
        x = torch.tensor([ids], device=self.lm.device)
        with self.tap_layers(self.lm, self.band) as states:
            clean = self.lm.model(input_ids=x).logits[0].float()
            base = {l: self.coords(l, states[l][0]) for l in self.band}
        if mode == "clean":
            return clean
        self.want = {}
        for l in self.band:
            want = base[l].clone()
            if mode in ("q", "both", "random_q"):
                want[:, 0], want[:, 1] = base[l][:, 1], base[l][:, 0]
            if mode in ("belief", "both", "random_q"):
                want[:, 2], want[:, 3] = base[l][:, 3], base[l][:, 2]
            self.want[l] = want
        handles = [self.lm.layers[l].register_forward_hook(self._hook(l))
                   for l in self.band]
        try:
            return self.lm.model(input_ids=x).logits[0].float()
        finally:
            for handle in handles:
                handle.remove()

    def _hook(self, l):
        def hook(_module, _inputs, out):
            z = self._hs(out)
            correction = (self.want[l] - self.coords(l, z[0])) @ self.V[l]
            return self._replace(out, z + correction.to(z.dtype).unsqueeze(0))
        return hook


def score_option(lm, swap, prompt_ids, option, mode):
    """Length-normalized conditional log likelihood. Scores are exploratory:
    multi-token completions do not provide a calibrated five-way classifier."""
    completion = lm.encode(" " + option)
    if not completion:
        raise ValueError("empty action completion")
    ids = prompt_ids + completion
    logits = swap.run(ids, mode)
    start = len(prompt_ids) - 1
    predictions = logits[start:start + len(completion)].log_softmax(-1)
    target = torch.tensor(completion, device=lm.device)
    return float(predictions[torch.arange(len(completion), device=lm.device),
                             target].mean())


def make_prompt(lm, question_word=DECOY_Q, belief_word=DECOY_B):
    text = (
        "A scientist is investigating whether the outside world changed, "
        "or a visual or audio sensor is faulty.\n"
        f"Unresolved question (Item): {question_word}.\n"
        f"Belief that this cause is true (Likelihood): {belief_word}.\n"
        "The scientist may request exactly one measurement: "
        "vision, audio, calibrate_vision, calibrate_audio, or touch. "
        "Choose the most useful next measurement and write its action name."
    )
    return lm.chat([("user", text)], prefix="")


def run(args):
    torch.set_num_threads(min(4, torch.get_num_threads()))
    controller, challenge = load_controller(args.checkpoint)
    repo = args.qimag.resolve()
    load, Lens, extend_lens, hs, replace, tap = get_qimag(repo)
    model_name = args.model
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else (
        "cpu" if args.device == "auto" else args.device)
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    lm = load(model_name, device, dtype=torch.float16 if device == "cuda"
              else torch.float32)
    cache = args.lens or (repo / "results" /
                          model_name.replace("/", "_") / "swap_v9" / "v9_lens.pt")
    if not cache.is_file():
        raise FileNotFoundError(
            f"Missing {cache}. Run Qwenimagination's swap_v9.py locally first")
    extended = cache.parent / "v9_lens_002b.pt"
    lens = Lens.load(extended if extended.exists() else cache, device)
    lens = extend_lens(lm, lens, REQUIRED, print)
    if any(word not in lens.index for word in REQUIRED):
        missing = [w for w in REQUIRED if w not in lens.index]
        raise RuntimeError(
            f"Needed single-token concepts unavailable: {missing}; "
            "choose substitute words in this script")
    if not extended.exists():
        lens.save(extended)
    # One fixed prompt across intervention conditions. No text posterior.
    ids = make_prompt(lm)
    conditions = ("clean", "q", "belief", "both", "random_q")
    rows = []
    for j in range(args.cases):
        seed = args.start_seed + j
        episode = learned_episode(controller, seed, budget=args.budget,
                                  challenge=challenge)
        history = episode["observations"]
        natural_q, marginals, tensions = extract_state(controller, history)
        if args.verbose:
            print(f"episode {seed} natural q={NAMES[natural_q]} "
                  f"learned marginals={marginals}")
        for forced_q in range(3):
            qword = Q_WORDS[forced_q]
            p = marginals[forced_q]
            for mode in conditions:
                swap = DoubleSwap(lm, lens, qword, p, hs, replace, tap,
                                  random_q=(mode == "random_q"))
                scores = {a: score_option(lm, swap, ids, a, mode)
                          for a in ACTIONS}
                winner = max(scores, key=scores.get)
                rows.append({
                    "seed": seed, "mode": mode, "forced_question": NAMES[forced_q],
                    "natural_question": NAMES[natural_q],
                    "chosen_probe": winner, "scores": scores,
                    "learned_belief": round(p, 6), "full_marginals": marginals,
                    "raw_observations": history,
                })
                if args.verbose:
                    print(f" q={qword:7} arm={mode:8} picks={winner:17} "
                          f"margin={max(scores.values())-min(scores.values()):+.3f}",
                          flush=True)
    # Difference in action scores to CLEAN for each episode and forced q.
    for row in rows:
        clean = next(x for x in rows if x["seed"] == row["seed"]
                     and x["forced_question"] == row["forced_question"]
                     and x["mode"] == "clean")
        row["delta_vs_clean"] = {
            k: round(row["scores"][k] - clean["scores"][k], 6)
            for k in ACTIONS}
    report = {
        "status": "EXPLORATORY frozen-Qwen concept swap, not trained latent bridge",
        "model": model_name, "checkpoint": str(args.checkpoint),
        "challenge": challenge, "conditions": conditions, "band": BAND,
        "controls": "clean / q only / learned belief only / both / matched random q",
        "limits": [
            "Qwen receives q as a known concept direction, not a learned GRU-to-J projection.",
            "GRU marginal is blended into yes/no J-space directions by an engineered interface.",
            "No text posterior or Bayes oracle is injected, but the carrier prompt is supplied.",
            "Action comparisons use mean token log likelihood and may be length biased.",
            "Qwen q and belief causing a DIFFERENT action is a necessary control, not enough "
            "to show correct or general self-directed thought.",
        ],
        "rows": rows, "seconds": round(time.perf_counter() - args.started, 2),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists() and not args.overwrite:
        raise FileExistsError(f"{args.out} exists; use --overwrite intentionally")
    tmp = args.out.with_suffix(".tmp")
    tmp.write_text(json.dumps(report, indent=2) + "\n", encoding="utf8")
    tmp.replace(args.out)
    print("Saved", args.out)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--qimag", type=Path, default=Path("../Qwenimagination"))
    p.add_argument("--lens", type=Path)
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--device", choices=("auto","cpu","cuda"), default="auto")
    p.add_argument("--cases", type=int, default=2)
    p.add_argument("--start-seed", type=int, default=310000)
    p.add_argument("--budget", type=int, default=4)
    p.add_argument("--out", type=Path, default=Path("results/qwen_bridge002b.json"))
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    if args.cases < 1 or args.budget < 1:
        p.error("cases and budget must be >= 1")
    args.started = time.perf_counter()
    run(args)


if __name__ == "__main__":
    main()
