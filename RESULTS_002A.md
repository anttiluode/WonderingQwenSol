# Experiment 002A — first trained recurrent-controller results

Date: 2026-11-18. **Exploratory local PyTorch CPU run**, using the code
in `learner002.py`. No Qwen model is involved in this run.

Commands for replication (Python with PyTorch installed):

```powershell
python3.13 -m pip install -r requirements-qwen.txt
python3.13 learner002.py --train-episodes 4096 --epochs 35 --test-seed 310000 --test-episodes 200 --budgets 4,8
```

A 48-unit GRU read only **7 observable inputs per step**: five
one-hot probe identities and two one-hot possible readings. Training
labels came from a Bayesian teacher: the eight-class posterior, next
query variable, and next information-gain probe. At evaluation, the
learned controller's state and action were calculated from its GRU
alone. The environment supplies only actual noisy measurements.
This is supervised **Bayesian-policy distillation**, not open-ended
self-inquiry or a Qwen result.

Training episode seeds 200000..204095; evaluation seeds
310000..310199; same noise model and prior as experiment 001.
All six policies use paired worlds and four selected probes beyond the
initial vision measurement.

## 4 probes — unseen worlds

| Policy | Correct full hidden state | Brier (lower) | Mean probe cost |
|---|---:|---:|---:|
| Learned recurrent controller | **81.0%** | **0.0916** | 5.870 |
| Same trained weights; hidden state reset after every reading | 29.5% | 0.2186 | 6.040 |
| Separately trained memoryless model | 24.0% | 0.2140 | 5.809 |
| Random probes | 65.5% | 0.1380 | 5.958 |
| Alternate calibration sensors | 73.0% | 0.0957 | 6.200 |
| Bayes known-model reference | 80.5% | 0.0929 | 5.870 |

The learned recurrent controller's full-state advantage over its
own zeroed version is **51.5 percentage points** (paired bootstrap
95% interval +43.0 to +60.0 percentage points; 3000 resamples).
Against random, it is +15.5 points (approximately +9.5 to +21.5).
The 0.5-point apparent advantage over exact Bayesian guidance is
too small to interpret and should not be promoted as superiority.

### What the learner actually asks

Across 800 selected probes: `calibrate_audio` 320,
`calibrate_vision` 260, `audio` 208, `vision` 12, `touch` 0.

This does show departure from the all-calibration heuristic in the
8-world Qwen pilot, but the policy was **taught by Bayes action labels**.
It does not establish the ability to generate novel measurements.

## 8 probes — length generalization beyond training

| Policy | Correct full hidden state | Brier (lower) | Mean probe cost |
|---|---:|---:|---:|
| Learned recurrent controller | **87.5%** | 0.0559 | 10.345 |
| Same trained weights, state zeroed | 29.5% | 0.2133 | 11.031 |
| Separately trained memoryless | 30.0% | 0.2191 | 10.475 |
| Random probes | 85.0% | 0.0735 | 10.937 |
| Alternate calibration sensors | 83.0% | **0.0525** | 11.400 |
| Bayes known-model reference | **94.5%** | **0.0302** | 10.207 |

At double the trained horizon, recurrent memory remains essential,
but the exact Bayes policy pulls ahead in accuracy and calibration.
It would be misleading to say the learner has surpassed its teacher.

## Why it matters and why it is not yet a mind

The principal 002A gate is met: **at evaluation the eight-state
Bayesian posterior is not computed for the learned policy**; a recurrent
network carries sufficient history to infer and select actions,
and ablating that history destroys its advantage. This is a meaningful
mechanistic prerequisite for a persistent inner workspace.

But there are firm limits:

1. The causal world, sensor laws, target posterior, question labels
   and optimal action labels were created by Python during training.
   The GRU learned to approximate this supplied solution.
2. Qwen's weights and J-space were **not used**. This is a tiny
   learned controller, not a language model beginning to ruminate.
3. The learned `question` head is an *auxiliary prediction*: the
   chosen question does **not yet causally gate** the action head.
   Calling this independent self-generated inquiry would overclaim.
4. The held-out worlds use **new random seeds, not new causal
   structures**. Re-labeling, changed noise/priors, changing
   sensor mappings, and different world dynamics are untested.
5. The GRU actor learns from a fixed teacher; it does not presently
   evaluate new counterfactuals at inference. Imagination belongs in
   002B.
6. This is a single trained seed and an exploratory comparison.
   Confirm using paired model initialization seeds before publication.

The next experiment should couple the chosen question to the
measurement policy causally, prove it with a question-head ablation,
and then connect a learned persistent state to frozen Qwen's
mid-layer reading/writing **without converting the state into a
complete text posterior**. Preserve the same Bayes, fixed-calibration,
memoryless and state-zeroing baselines.

The local result artifact stores full per-seed model probabilities,
chosen probes and the first three detailed episodes per policy; it is
not a preregistered independent replication.
