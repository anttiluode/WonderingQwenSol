"""Experiment 002B-1: local surprise-pulled, causally question-conditioned GRU.

Training uses exact Bayes teacher LABELS, never posterior as input.
Evaluation uses only the raw (probe, observed-bit) stream.
Argon's central falsifier: can changing the question while h is fixed
alter *which measurement* the controller chooses? See EXPERIMENT_002B_DESIGN.md.

Research status: exploratory code. No Qwen latent intervention here.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import time

import torch
from torch import nn
from torch.nn import functional as F

from learner002 import ACTIONS, STATE_ORDER, WIDTH, encode_observation, make_dataset
from worldloop import World, PROBES, NAMES, metrics, run_episode
from challenge_b import ChangedWorld, independent_prior, transition_after_first_vision, best_probe, query_target

CHANNEL_TARGETS = torch.tensor([
    [1., 1., 0.],  # vision: ambiguous world or vision fault
    [1., 0., 1.],  # audio: ambiguous world or audio fault
    [0., 1., 0.],  # calibrate vision
    [0., 0., 1.],  # calibrate audio
    [1., 0., 0.],  # touch
])
BITS = torch.tensor(STATE_ORDER, dtype=torch.float32)
DEFAULT_TRAIN_SEEDS = (42, 43, 44)


class PulledInquirer(nn.Module):
    """No direct h->action head. Action sees exactly (q, selected slot(h,q))."""

    def __init__(self, hidden=48, slot_size=12):
        super().__init__()
        self.hidden = hidden
        self.slot_size = slot_size
        self.gru = nn.GRUCell(WIDTH, hidden)
        self.belief = nn.Linear(hidden, 8)
        self.predict_sensor = nn.Linear(hidden, len(ACTIONS))
        self.slots = nn.ModuleList([nn.Sequential(nn.Linear(hidden, slot_size),
                                                  nn.Tanh()) for _ in range(3)])
        self.choose_action = nn.Sequential(
            nn.Linear(slot_size + 3, 32), nn.Tanh(), nn.Linear(32, len(ACTIONS)))
        # Shared local salience rule: posterior uncertainty plus decaying
        # *prediction mismatch on the causal channel*. The three questions
        # have separate traces, no arbitrary MLP choosing the label.
        self.raw_surprise_gain = nn.Parameter(torch.tensor(-0.7))
        self.register_buffer("channel_targets", CHANNEL_TARGETS.clone())
        self.register_buffer("bits", BITS.clone())
        self.register_buffer("world_priority", torch.tensor([1.1, 1., 1.]))

    def empty_state(self, batch, device):
        h = torch.zeros(batch, self.hidden, device=device)
        tension = torch.zeros(batch, 3, device=device)
        return h, tension

    def step(self, x, h, tension, forced_question=None, disable_surprise=False):
        """At each step observe x, update learned state and local salience,
        then select a question and condition ONLY that slot into the action."""
        batch = x.shape[0]
        probe_ix = x[:, :len(ACTIONS)].argmax(-1)
        observed = x[:, len(ACTIONS):].argmax(-1).float()
        # This prediction is made from the PRIOR recurrent state, before the
        # new observation arrives, so the error is not contaminated by knowing
        # the outcome being predicted.
        pre = self.predict_sensor(h).gather(1, probe_ix[:, None]).squeeze(1)
        prediction_loss = F.binary_cross_entropy_with_logits(pre, observed,
                                                               reduction="none")
        salience = prediction_loss.unsqueeze(1) * self.channel_targets[probe_ix]
        tension = 0.68 * tension + salience
        h = self.gru(x, h)
        posterior_logits = self.belief(h)
        posterior = posterior_logits.softmax(-1)
        marginals = posterior @ self.bits
        p = marginals.clamp(1e-5, 1 - 1e-5)
        uncertainty = -p * p.log2() - (1 - p) * (1 - p).log2()
        # The tension component *pulls* the question to the channel causing
        # the prediction mismatch. The uncertainty component prevents a loud
        # but already-solved channel from monopolising attention indefinitely.
        gain = 0.0 if disable_surprise else F.softplus(self.raw_surprise_gain)
        question_logits = uncertainty * self.world_priority + gain * tension
        question = (question_logits.argmax(-1) if forced_question is None
                    else torch.as_tensor(forced_question, device=x.device,
                                         dtype=torch.long).expand(batch))
        q_onehot = F.one_hot(question, 3).to(h.dtype)
        slots = torch.stack([fn(h) for fn in self.slots], dim=1)
        relevant = slots.gather(1,
                                question[:, None, None].expand(batch, 1, self.slot_size)
                               ).squeeze(1)
        # Important: DO NOT concatenate h here. Questions are causally
        # load-bearing via both their selected local slot and q_onehot.
        action_logits = self.choose_action(torch.cat((relevant, q_onehot), dim=-1))
        return (h, tension, posterior_logits, question_logits, action_logits,
                pre, question, marginals)

    def forward(self, observations, teacher_questions=None, reset_each=False):
        batch, steps, _ = observations.shape
        h, tension = self.empty_state(batch, observations.device)
        preds = [[] for _ in range(4)]
        for tick in range(steps):
            if reset_each:
                h, tension = self.empty_state(batch, observations.device)
            teacher_q = None if teacher_questions is None else teacher_questions[:, tick]
            h, tension, belief, q, action, pred, _, _ = self.step(
                observations[:, tick], h, tension, teacher_q)
            for store, value in zip(preds, (belief, q, action, pred)):
                store.append(value)
        return [torch.stack(x, dim=1) for x in preds]

    def raw_intervention(self, history, forced_q):
        """No world access: replay observations, then force only the final q."""
        if not history:
            raise ValueError("need at least one observation")
        device = next(self.parameters()).device
        h, tension = self.empty_state(1, device)
        for probe, reading in history[:-1]:
            x = torch.tensor([encode_observation(probe, reading)],
                             device=device)
            h, tension, *_ = self.step(x, h, tension)
        probe, reading = history[-1]
        x = torch.tensor([encode_observation(probe, reading)],
                         device=device)
        return self.step(x, h, tension, forced_question=forced_q)


def make_dataset_b(episodes, seed, teacher_fraction=0.30):
    """Challenge B labels: world changed AFTER initial observation.
    Uses a separate teacher; labels never enter inference."""
    rng = random.Random(seed + 82631)
    xall, posterior, goals, actions = [], [], [], []
    for i in range(episodes):
        world = ChangedWorld.sample(seed + i)
        belief = independent_prior()
        xs, ys, qs, aa = [], [], [], []
        selected = "vision"
        for t in range(5):
            observed = world.observe(selected)
            xs.append(encode_observation(selected, observed))
            belief.observe(selected, observed)
            if t == 0:
                belief = transition_after_first_vision(belief)
            ys.append([belief.p[s] for s in STATE_ORDER])
            goal = query_target(belief)
            optimal = best_probe(belief, goal)
            qs.append(goal)
            aa.append(ACTIONS.index(optimal))
            selected = optimal if rng.random() < teacher_fraction else rng.choice(ACTIONS)
        xall.append(xs)
        posterior.append(ys)
        goals.append(qs)
        actions.append(aa)
    return (torch.tensor(xall, dtype=torch.float32),
            torch.tensor(posterior, dtype=torch.float32),
            torch.tensor(goals, dtype=torch.long),
            torch.tensor(actions, dtype=torch.long))


def train(model, dataset, epochs=35, batch_size=128, lr=0.003, seed=42):
    torch.manual_seed(seed)
    rng = torch.Generator().manual_seed(seed + 99)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=.001)
    x, target_belief, target_question, target_action = dataset
    losses = []
    model.train()
    for epoch in range(epochs):
        indices = torch.randperm(len(x), generator=rng)
        total = 0.
        for ids in indices.split(batch_size):
            batch_x = x[ids]
            # Teacher question is used for training the conditioned action
            # decoder, never for selecting the question in evaluation.
            pb, pq, pa, raw_pred = model(
                batch_x, teacher_questions=target_question[ids])
            kl = F.kl_div(F.log_softmax(pb.reshape(-1, 8), -1),
                          target_belief[ids].reshape(-1, 8),
                          reduction="batchmean")
            qce = F.cross_entropy(pq.reshape(-1, 3), target_question[ids].reshape(-1))
            ace = F.cross_entropy(pa.reshape(-1, 5), target_action[ids].reshape(-1))
            observed = batch_x[:, :, len(ACTIONS):].argmax(-1).float()
            bce = F.binary_cross_entropy_with_logits(raw_pred, observed)
            loss = kl + .15 * qce + .30 * ace + .12 * bce
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.detach()) * len(ids)
        losses.append(round(total / len(x), 5))
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == epochs:
            print(f"seed={seed} epoch={epoch + 1}/{epochs} "
                  f"loss={losses[-1]:.4f} local_gain={float(F.softplus(model.raw_surprise_gain).detach()):.3f}",
                  flush=True)
    model.eval()
    return losses


@torch.no_grad()
def learned_episode(model, seed, budget=4, challenge="a",
                    reset_state=False, force_q=None, disable_surprise=False):
    """Only the simulated environment knows hidden world. Qwen/Bayes NEVER
    execute here; the trained GRU chooses all post-baseline probes."""
    env = World.sample(seed) if challenge == "a" else ChangedWorld.sample(seed)
    device = next(model.parameters()).device
    h, tension = model.empty_state(1, device)
    action = "vision"
    events, questions = [], []
    for t in range(budget + 1):
        if reset_state:
            h, tension = model.empty_state(1, device)
        reading = env.observe(action)
        events.append({"probe": action, "value": reading})
        x = torch.tensor([encode_observation(action, reading)], device=device)
        h, tension, logits, q_logits, a_logits, _, q, _ = model.step(
            x, h, tension, forced_question=force_q,
            disable_surprise=disable_surprise)
        questions.append(int(q[0]))
        action = ACTIONS[int(a_logits[0].argmax())]
    pp = logits.softmax(-1)[0]
    marg = pp @ model.bits
    choice = STATE_ORDER[int(pp.argmax())]
    return {
        "seed": seed,
        "hidden_truth": dict(zip(NAMES, env.state)),
        "posterior": {name: round(float(marg[i]), 6)
                      for i, name in enumerate(NAMES)},
        "map": dict(zip(NAMES, choice)),
        "observations": events, "questions": questions,
        "probe_cost": sum(PROBES[e["probe"]][1] for e in events),
    }


@torch.no_grad()
def intervention_grid(model, histories):
    """Counterfactual q clamps on IDENTICAL h. Measures q action dependency,
    not whether altered choices are automatically useful."""
    flips, diverse = 0, 0
    examples = []
    for history in histories:
        actions = []
        for q in range(3):
            out = model.raw_intervention(history, q)
            logits = out[4][0]
            actions.append({"question": NAMES[q],
                            "action": ACTIONS[int(logits.argmax())],
                            "scores": [round(float(z), 5) for z in logits]})
        is_diverse = len({x["action"] for x in actions}) > 1
        diverse += is_diverse
        if len(examples) < 10:
            examples.append({"observations": history,
                             "forced_question_actions": actions})
    return {"histories": len(histories),
            "fraction_q_changes_top_action": round(diverse / len(histories), 4),
            "examples": examples}


def benchmark(model, seeds, budget=4, challenge="a", with_references=True):
    records = {
        "intact": [learned_episode(model, s, budget, challenge)
                   for s in seeds],
        "state_zeroed": [learned_episode(model, s, budget, challenge,
                                         reset_state=True)
                         for s in seeds],
        "surprise_off": [learned_episode(model, s, budget, challenge,
                                         disable_surprise=True)
                         for s in seeds],
        "fixed_world_q": [learned_episode(model, s, budget, challenge, force_q=0)
                          for s in seeds],
        "fixed_vision_q": [learned_episode(model, s, budget, challenge, force_q=1)
                           for s in seeds],
        "fixed_audio_q": [learned_episode(model, s, budget, challenge, force_q=2)
                          for s in seeds],
    }
    if with_references:
        if challenge == "a":
            from learner002 import calibrate_alternate
            records.update({
                "bayes": [run_episode(s, "active", budget) for s in seeds],
                "random": [run_episode(s, "random", budget) for s in seeds],
                "calibrate_alternate": [calibrate_alternate(s, budget) for s in seeds],
            })
        else:
            from challenge_b import run as run_b
            records.update({
                "bayes": [run_b(s, "active", budget) for s in seeds],
                "random": [run_b(s, "random", budget) for s in seeds],
                "calibrate_alternate": [run_b(s, "calibrate_alternate", budget)
                                        for s in seeds],
            })
    return {
        "metrics": {name: metrics(rows) for name, rows in records.items()},
        "per_episode": {name: [{
            "seed": r["seed"], "truth": r["hidden_truth"], "posterior": r["posterior"],
            "probe_cost": r["probe_cost"], "actions": [e["probe"] for e in r["observations"]],
            "questions": r.get("questions", []),
        } for r in rows] for name, rows in records.items()},
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--challenge", choices=("a", "b"), default="a")
    p.add_argument("--train-episodes", type=int, default=4096)
    p.add_argument("--epochs", type=int, default=35)
    p.add_argument("--train-seed", type=int, default=200000)
    p.add_argument("--test-seed", type=int, default=310000)
    p.add_argument("--test-episodes", type=int, default=200)
    p.add_argument("--budgets", default="4,8")
    p.add_argument("--seeds", default="42,43,44")
    p.add_argument("--hidden", type=int, default=48)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--out", type=Path, default=Path("results/learner002b.json"))
    p.add_argument("--save-checkpoints", action="store_true", help="save versioned state_dict per seed for the Qwen bridge")
    args = p.parse_args()
    if args.quick:
        args.train_episodes, args.epochs, args.test_episodes = 128, 2, 8
        args.seeds = "42"
        args.budgets = "4"
    train_ids = (args.train_seed, args.train_seed + args.train_episodes)
    test_ids = (args.test_seed, args.test_seed + args.test_episodes)
    if max(train_ids[0], test_ids[0]) < min(train_ids[1], test_ids[1]):
        p.error("train and test episode seeds overlap")
    if args.out.exists() and not args.overwrite:
        p.error(f"{args.out} already exists; --overwrite to replace")
    if min(args.train_episodes, args.epochs, args.test_episodes, args.hidden) <= 0:
        p.error("all sizes must be positive")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    t0 = time.perf_counter()
    make = make_dataset if args.challenge == "a" else make_dataset_b
    data = make(args.train_episodes, args.train_seed)
    report = {"status": "SMOKE ONLY" if args.quick else "exploratory 002B GRU, not Qwen",
              "challenge": args.challenge, "teacher_train_only": True,
              "train_seed": args.train_seed, "train_episodes": args.train_episodes,
              "test_seed": args.test_seed, "test_episodes": args.test_episodes,
              "runs": {}}
    for seed in [int(x) for x in args.seeds.split(",")]:
        torch.manual_seed(seed)
        model = PulledInquirer(args.hidden)
        losses = train(model, data, args.epochs, 128, .003, seed)
        tests = {}
        for b in [int(x) for x in args.budgets.split(",")]:
            tests[str(b)] = benchmark(model,
                                      range(args.test_seed, args.test_seed + args.test_episodes),
                                      budget=b, challenge=args.challenge)
        histories = [
            [(e["probe"], e["value"]) for e in learned_episode(
                model, args.test_seed + i, 4, args.challenge)["observations"]]
            for i in range(min(30, args.test_episodes))]
        grid = intervention_grid(model, histories)
        report["runs"][str(seed)] = {
            "losses": losses, "probes": tests,
            "forced_q": grid, "parameter_count": sum(p.numel() for p in model.parameters()),
        }
        print(json.dumps({b: t["metrics"] for b, t in tests.items()}, indent=2))
        print(f"seed {seed} forced-q flips {grid['fraction_q_changes_top_action']}", flush=True)
        if args.save_checkpoints:
            ckpt = args.out.parent / f"002b_{args.challenge}_seed{seed}.pt"
            if ckpt.exists() and not args.overwrite:
                p.error(f"{ckpt} exists; --overwrite intentionally")
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "hidden": args.hidden, "slot_size": model.slot_size,
                "challenge": args.challenge, "train_seed": args.train_seed,
                "train_episodes": args.train_episodes, "training_seed": seed,
                "train_loss": losses[-1],
            }, ckpt)
            print("Checkpoint:", ckpt, flush=True)
    report["elapsed_seconds"] = round(time.perf_counter()-t0, 2)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(args.out.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(report, indent=2) + "\n", encoding="utf8")
    tmp.replace(args.out)
    print("Saved", args.out)


if __name__ == "__main__":
    main()
