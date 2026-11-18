# Experiment 002B-2 — frozen Qwen early-band bridge

**Status: bridge code and fake-model tests validated; no real Qwen
intervention has been run in this branch.** The current
`bridge_002b.py` is a *small experimental interface*, not evidence of
a self-directed language model.

## The idea

The independent 002B GRU holds a learned recurrent state updated only
from sensor names and binary outcomes. Its local tension/uncertainty
mechanism pulls one of three questions into its limited workspace.
A question-specific GRU slot affects a proposed measurement.

The bridge asks whether Qwen's own mid-network computation can be
made causally sensitive to this (question + belief) pair:

- Read `q` and the relevant **learned** marginal (not the exact Bayes
  posterior) by replaying observed events through a saved 002B model.
- Start from two loaded decoy concepts in one fixed Qwen input:
  `Item: stone`, `Likelihood: quiet`.
- Use the matching J-lens directions to replace the decoy question
  with `world`, `vision` or `audio`, and to replace the decoy
  belief with a length-matched mixture of `yes` and `no` reflecting
  the model's estimated marginal.
- Clamp the exchange across Qwen's early layers **4–12**, following
  the successful Qwenimagination V9 intervention.
- Compare log-likelihood scores for five possible measurement names.
  Multi-token action scores are normalized by length; **these
  scores are not a validated calibrated five-way policy yet**.

## Controls

Quick: `clean`, `q`, `belief`, `both`, `random_q`.

Optional `--full-controls` also adds:

- `printed`: actually put the question and coarse binary belief in
  the text; positive text-information reference.
- `late`: swap across late layers 20–24 instead of early 4–12.
- `flipped_belief`: hold question fixed but invert learned confidence.

The output records **every action's score** and delta versus the
matched clean baseline. A q/b-dependent change in logits is not
enough; a useful result must show improved *correct experimental
choice* over clean, random, printed and simple GRU action-head
baselines, on a separate set of held-out worlds.

## Run on Windows / RTX 3060

This reuses the real Qwenimagination checkout and its **local**
Jacobian lens cache. Clone/pull both repositories into sibling
directories. Start in `D:\\anttiluode\\WonderingQwenSol`.

```powershell
git pull --ff-only
python3.13 -m pip install -r requirements-qwen.txt
python3.13 -m unittest discover -s tests -v
```

Use a saved model checkpoint from the
[002B training run artifact](https://github.com/anttiluode/WonderingQwenSol/actions/runs/44277193144),
or retrain a small controller and explicitly save its weights:

```powershell
python3.13 learner002b.py --challenge b --train-episodes 2048 --epochs 25 --seeds 42 --test-seed 410000 --test-episodes 200 --budgets 4 --save-checkpoints --out results/local_002b_b.json
```

Then (this uses the held-out Challenge B seed range by default):

```powershell
python3.13 bridge_002b.py --checkpoint results/002b_b_seed42.pt --cases 1 --verbose --out results/bridge_b_pilot.json
```

For full controls:

```powershell
python3.13 bridge_002b.py --checkpoint results/002b_b_seed42.pt --cases 2 --full-controls --verbose --out results/bridge_b_full.json
```

Prerequisites: `D:\\anttiluode\\Qwenimagination` and the local cache
`results/Qwen_Qwen2.5-1.5B-Instruct/swap_v9/v9_lens.pt` *within that
checkout* (run Qwenimagination's `swap_v9.py` once if absent).
The cache is not in Git because it is large. The script
may extend it for a few needed concept tokens and save
`v9_lens_002b.pt` for later reuse. It will stop with an explicit
error if a required concept is not a single tokenizer token.

If memory usage is too high, start with `--cases 1` and the five
default controls, not full controls. Qwen is loaded in float16 on CUDA.
There is no reason to retrain the language model for this initial
intervention test.

## Important limits

1. The GRU is teacher-trained on the toy causal graph, not a general
   emergent model of the world.
2. The current question vocabulary and sensor-to-question routes are
   programmer-supplied; local surprise did not independently earn an
   accuracy gain in the first 002B batch.
3. A hand-written vector blend of `yes/no` carries the learned marginal.
   A **learned** GRU-to-Qwen projection has not been implemented or
   validated. No natural-language Bayesian posterior is injected, but
   the semantic adapter still introduces design choices.
4. This is only a scoring harness. It does **not** substitute Qwen's
   chosen action into the actual environment during a closed-loop
   rollout. That is the next experiment once this intervention is
   calibrated against random and printed concepts.
5. Performance claims need real Qwen runs and held-out action-benefit
   metrics. A clean implementation or fake-model test is not evidence
   the real model understands the injected state.
6. Qwenimagination's V11b **failed** its stronger preregistered
   thought-vs-random gate. We should not presume content-specific
   steering without running those controls again.
