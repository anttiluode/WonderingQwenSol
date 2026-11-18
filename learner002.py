"""Experiment 002A: learned recurrent provenance and inquiry without Bayesian inference at test time.

The Bayesian teacher provides training targets only. Evaluation's learned policy
receives (sensor identity, sensor reading) and maintains its own GRU state.
No Qwen model is used in this stage. See EXPERIMENT_002A.md.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import time

import torch
from torch import nn
from torch.nn import functional as F

from worldloop import Belief, Mind, NAMES, PROBES, STATES, World, metrics, run_episode

ACTIONS = tuple(PROBES)
STATE_ORDER = tuple(STATES)
WIDTH = len(ACTIONS) + 2


def encode_observation(action: str, value: int):
    """Seven observable inputs: probe identity and binary answer; no posterior."""
    x = [0.] * WIDTH
    x[ACTIONS.index(action)] = 1.
    x[len(ACTIONS) + value] = 1.
    return x


class LatentInquirer(nn.Module):
    def __init__(self, hidden=48):
        super().__init__()
        self.hidden = hidden
        self.gru = nn.GRU(input_size=WIDTH, hidden_size=hidden, batch_first=True)
        self.belief = nn.Linear(hidden, len(STATE_ORDER))
        self.question = nn.Linear(hidden, len(NAMES))
        self.action = nn.Linear(hidden, len(ACTIONS))

    def forward(self, observations, reset_each=False):
        batch, seq, dim = observations.shape
        if reset_each:
            # A properly trained memoryless comparison, not merely a zeroed
            # recurrent controller. Every timestep has a fresh hidden state.
            flat = observations.reshape(batch * seq, 1, dim)
            features, _ = self.gru(flat)
            features = features.reshape(batch, seq, self.hidden)
        else:
            features, _ = self.gru(observations)
        return self.belief(features), self.question(features), self.action(features)


def make_dataset(episodes: int, seed: int, teacher_fraction=0.30):
    """Build supervised episodes. The Bayes teacher is limited to this function.

    Hidden truth creates sensory data; the teacher computes labels *after*
    seeing readings but is never an input to the learner at inference.
    """
    rng = random.Random(seed + 82631)
    xs, target_belief, target_question, target_action = [], [], [], []
    for i in range(episodes):
        world = World.sample(seed + i)
        mind = Mind()
        x_seq, b_seq, q_seq, a_seq = [], [], [], []
        candidate = 'vision'
        for t in range(5):
            observed = world.observe(candidate)
            x_seq.append(encode_observation(candidate, observed))
            mind.see(candidate, observed)
            b_seq.append([mind.belief.p[s] for s in STATE_ORDER])
            target = mind.next_question()
            ideal_probe = mind.active_probe(target)
            q_seq.append(target)
            a_seq.append(ACTIONS.index(ideal_probe))
            candidate = (ideal_probe if rng.random() < teacher_fraction
                         else rng.choice(ACTIONS))
        xs.append(x_seq)
        target_belief.append(b_seq)
        target_question.append(q_seq)
        target_action.append(a_seq)
    return (torch.tensor(xs, dtype=torch.float32),
            torch.tensor(target_belief, dtype=torch.float32),
            torch.tensor(target_question, dtype=torch.long),
            torch.tensor(target_action, dtype=torch.long))


def fit(model, data, epochs: int, batch_size: int, lr: float, seed: int,
        memoryless=False):
    torch.manual_seed(seed)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.001)
    xs, beliefs, questions, actions = data
    n, steps, _ = xs.shape
    hist = []
    generator = torch.Generator().manual_seed(seed + 7)
    for epoch in range(epochs):
        order = torch.randperm(n, generator=generator)
        running = 0.0
        for ids in order.split(batch_size):
            p, q, a = model(xs[ids], reset_each=memoryless)
            # Distillation of the known Bayes solution, not unsupervised discovery.
            loss_b = F.kl_div(F.log_softmax(p.flatten(0, 1), -1),
                              beliefs[ids].flatten(0, 1), reduction='batchmean')
            loss_q = F.cross_entropy(q.flatten(0, 1), questions[ids].flatten())
            loss_a = F.cross_entropy(a.flatten(0, 1), actions[ids].flatten())
            loss = loss_b + 0.15 * loss_q + 0.30 * loss_a
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += float(loss.item()) * len(ids)
        hist.append(round(running / n, 5))
        if epoch == 0 or epoch + 1 == epochs or (epoch + 1) % 10 == 0:
            print(f"{'memoryless' if memoryless else 'recurrent':11} "
                  f"epoch {epoch+1:2d}/{epochs} loss={hist[-1]:.4f}", flush=True)
    model.eval()
    return hist


@torch.no_grad()
def run_learned(model, seed: int, budget=4, reset_each=False):
    """NO Bayesian updater, teacher, entropy calculator, or truth in decisions.

    The hidden true world is accessed by the separate simulation environment
    solely to obtain noisy observations and evaluation labels.
    """
    world = World.sample(seed)
    h = None
    action = 'vision'
    events = []
    selected_questions = []
    for tick in range(budget + 1):
        reading = world.observe(action)
        events.append({'probe': action, 'value': reading})
        x = torch.tensor([[encode_observation(action, reading)]],
                         dtype=torch.float32)
        step_features, h = model.gru(x, None if reset_each else h)
        features = step_features[0, -1]
        logits = model.belief(features)
        q_logits = model.question(features)
        a_logits = model.action(features)
        selected_questions.append(NAMES[int(q_logits.argmax())])
        action = ACTIONS[int(a_logits.argmax())]
    probabilities = torch.softmax(logits, dim=-1).tolist()
    joint = dict(zip(STATE_ORDER, probabilities))
    marginals = {name: round(sum(p for s, p in joint.items() if s[i]), 6)
                 for i, name in enumerate(NAMES)}
    winning = max(joint, key=joint.get)
    return {
        'seed': seed,
        'policy': 'state_zeroed' if reset_each else 'learned_recurrent',
        'hidden_truth': dict(zip(NAMES, world.state)),
        'posterior': marginals,
        'map': dict(zip(NAMES, winning)),
        'observations': events,
        'chosen_questions': selected_questions,
        'probe_cost': sum(PROBES[e['probe']][1] for e in events),
    }


def calibrate_alternate(seed, budget):
    count = [0]
    def choice(_, __):
        action = 'calibrate_audio' if count[0] % 2 == 0 else 'calibrate_vision'
        count[0] += 1
        return action
    return run_episode(seed, 'qwen', budget, qwen=choice)


def score_policy(records):
    info = metrics(records)
    info['n'] = len(records)
    return info


def evaluate(model, stateless, test_seed: int, test_episodes: int, budget: int):
    records = {key: [] for key in (
        'learned_recurrent', 'state_zeroed', 'memoryless_trained',
        'random', 'calibrate_alternate', 'active_bayes_reference')}
    for seed in range(test_seed, test_seed + test_episodes):
        records['learned_recurrent'].append(run_learned(model, seed, budget))
        records['state_zeroed'].append(run_learned(model, seed, budget,
                                                    reset_each=True))
        records['memoryless_trained'].append(run_learned(stateless, seed,
                                                         budget, reset_each=True))
        records['random'].append(run_episode(seed, 'random', budget))
        records['calibrate_alternate'].append(calibrate_alternate(seed, budget))
        records['active_bayes_reference'].append(run_episode(seed, 'active', budget))
    return {'metrics': {k: score_policy(v) for k, v in records.items()},
            'per_episode': {k: [{key: r[key] for key in (
                'seed', 'hidden_truth', 'posterior', 'map', 'probe_cost')}
                | {'actions': [e['probe'] for e in r['observations']]}
                for r in v] for k, v in records.items()},
            'first_examples': {k: v[:3] for k, v in records.items()}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train-episodes', type=int, default=4096)
    p.add_argument('--epochs', type=int, default=35)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--lr', type=float, default=0.003)
    p.add_argument('--hidden', type=int, default=48)
    p.add_argument('--test-seed', type=int, default=310000)
    p.add_argument('--test-episodes', type=int, default=200)
    p.add_argument('--train-seed', type=int, default=200000)
    p.add_argument('--budgets', default='4,8')
    p.add_argument('--out', type=Path, default=Path('results/learner002.json'))
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--quick', action='store_true',
                   help='Small CPU smoke only; not an experimental result')
    args = p.parse_args()
    if args.quick:
        args.train_episodes, args.epochs, args.test_episodes = 200, 2, 12
        args.budgets = '4'
    if min(args.train_episodes, args.epochs, args.test_episodes,
           args.batch_size, args.hidden) < 1:
        p.error('all counts and hidden width must be positive')
    if args.out.exists() and not args.overwrite:
        p.error(f'{args.out} already exists (use --overwrite intentionally)')
    if (args.test_seed < args.train_seed + args.train_episodes
            and args.test_seed + args.test_episodes > args.train_seed):
        p.error('train and test episode seeds overlap')
    torch.set_num_threads(min(4, torch.get_num_threads()))
    torch.manual_seed(42)
    began = time.perf_counter()
    print('Generating teacher labels for training only...', flush=True)
    data = make_dataset(args.train_episodes, args.train_seed)
    recurrent = LatentInquirer(args.hidden)
    stateless = LatentInquirer(args.hidden)
    rec_loss = fit(recurrent, data, args.epochs, args.batch_size,
                   args.lr, seed=42, memoryless=False)
    mem_loss = fit(stateless, data, args.epochs, args.batch_size,
                   args.lr, seed=43, memoryless=True)
    evaluated = {}
    for b in [int(x) for x in args.budgets.split(',')]:
        if b < 0:
            p.error('budget must be nonnegative')
        evaluated[str(b)] = evaluate(recurrent, stateless,
                                     args.test_seed, args.test_episodes, b)
    report = {
        'status': 'CPU smoke / NOT results' if args.quick
                  else 'Exploratory learned-controller comparison, not Qwen',
        'train_seed': args.train_seed, 'train_episodes': args.train_episodes,
        'test_seed': args.test_seed, 'test_episodes': args.test_episodes,
        'train_observations': 5, 'teacher_fraction': 0.30,
        'epochs': args.epochs, 'hidden_width': args.hidden,
        'architecture': 'GRU, seven observed inputs, eight-state posterior + question + probe heads',
        'teacher': 'Exact Bayesian posterior, query and information-gain probe; TRAIN LABELS ONLY',
        'inference': 'Raw probe identity and reading only; no Bayes computation or oracle policy',
        'trainable_parameters': {'recurrent': sum(p.numel() for p in recurrent.parameters()),
                                 'memoryless': sum(p.numel() for p in stateless.parameters())},
        'training_loss': {'recurrent': rec_loss, 'memoryless': mem_loss},
        'evaluation': evaluated, 'elapsed_seconds': round(time.perf_counter() - began, 1),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(args.out.suffix + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    tmp.replace(args.out)
    print(json.dumps({b: obj['metrics'] for b, obj in evaluated.items()}, indent=2))
    print('Saved', args.out, flush=True)


if __name__ == '__main__':
    main()
