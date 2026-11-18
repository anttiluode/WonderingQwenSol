# WonderingQwenSol

**Can an agent maintain an inner model, wonder about its uncertainty, and choose
a useful experiment without being handed the next question?**

Sol's **independent** route toward a thinking Qwen. Claude has the separate
[WonderingQwenClaude](https://github.com/anttiluode/WonderingQwenClaude)
repository; this repo does not edit it.

The long-term goal is a persistent causal workspace in which Qwen can (1) keep
latent hypotheses, (2) simulate counterfactual futures, (3) distinguish
imagining from observing, (4) choose new questions, and (5) revise its own
assumptions when prediction fails. Success would be **functional cognitive
capability**, not evidence of subjective experience.

## Experiment 001 — The broken-eye problem

An underlying world may change, or a visual or auditory channel may be
faulty. The *same visual report* can support two very different causes.

Imagine the world was previously quiet. Now vision reports a disturbance.
There are at least two explanations:

- **World changed:** another independent sense should also detect it.
- **Vision changed:** an independent sense may remain stable.

The program maintains probabilities for all eight combinations of three
latent bits: `world_changed`, `vision_fault`, and `audio_fault`.
It can ask for another visual or auditory report, physically touch the
world, or calibrate either sensor. Every measurement has cost and noise.
**No chosen action gets the hidden truth directly.**

<pre>
sensory observation
    |
    v
persistent hypotheses (world, visual fault, audio fault)
    |                                ^
    +--> imagine each possible test -+
             |                        |
             +--> select inquiry --> observe actual outcome
                    |
               update beliefs
</pre>

The essential protection: **imagined observations cannot update the
posterior**. They live in a separate ledger until the world is actually
queried. That is an engineered source distinction, not learned metacognition.

This is a small transparent Bayesian lab. It is **not a Qwen thought result**,
and it is not an original form of Bayesian experimental design. Its purpose
is to make precise controls for what the eventual learned mind should do.

### Run it now: no packages, no GPU

Python 3.10+:

```powershell
git clone https://github.com/anttiluode/WonderingQwenSol
cd WonderingQwenSol
python3.13 -m unittest discover -s tests -v
python3.13 worldloop.py --episodes 100 --seed 10000
```

The output is `results/experiment001.json`. It contains numerical metrics
and the first three full episodes per policy, including *observed* and
*imagined* events, chosen questions, and forecasts.

Windows setups without `python3.13` can use `python`.

### Compare the right things

All arms run on the **same hidden episodes**, with reproducible sensor
noise and the same measurement budget.

| Arm | What it tests |
|---|---|
| `active` | Explicit posterior + counterfactual information-gain selection |
| `random` | Same possible actions, no purposeful inquiry |
| `repeat` | Repeated vision only: more data without new independent evidence |
| `no_fault_model_same_evidence` | Same observations as `active`, but assumes sensors are *never* wrong |
| `qwen` (optional) | Frozen Qwen chooses the next measurement, given uncertainty and observations |

Key metrics are average Brier error (lower is better), individual hidden-bit
accuracy and joint three-bit MAP accuracy (higher is better), and total
probe cost. Probe costs differ, so cost-adjusted comparisons matter.

**Caveat:** `active` is an engineered Bayesian reference with the correct
likelihoods, not an AI discovery. The `no_fault_model_same_evidence` arm
isolates the effect of allowing sensor-fault explanations. The `repeat`
policy is a weak control; `random` is the fairer budget-matched action
control. The policy chooses a *goal variable* by entropy; it does not
invent arbitrary new concepts yet.

### Optional frozen-Qwen action choice

```powershell
python3.13 -m pip install -r requirements-qwen.txt
python3.13 worldloop.py --policy qwen --episodes 8 --seed 10000 --out results/qwen8.json
```

Uses `Qwen/Qwen2.5-1.5B-Instruct` by default and will download weights if
they are not cached. You can specify a local model directory with `--model`.
For a 12 GB NVIDIA card, this model is normally feasible in float16.
The Qwen policy receives **only** the current posterior, observed history,
candidate measurements and their costs, and the question selected by the
workspace. It never gets the simulation's hidden truth.

It emits one action name. Invalid completions are counted and default to
`vision` rather than silently using the Bayesian policy. Outputs include
the raw generated replies under `qwen_diagnostics`. Crucially, this
interface uses ordinary language tokens—not a silent J-space intervention.
If it works, it shows Qwen can help choose measurements, **not** that we
installed a free-running internal mind.

## Experiment 002A — first learned state gate

Argon's "Python-out-of-the-loop" objection is now testable. A small GRU
takes seven observable bits per step (probe identity and reading) and
learns a persistent latent belief over the eight candidate worlds.
The exact Bayesian posterior and the optimal action are used as
**training labels only**, never as an input at evaluation.

On 200 held-out-seed simulated worlds (4 follow-up probes), the
trained recurrent controller got **81.0%** of full states correct.
Resetting its own hidden state after every reading dropped it to
**29.5%**; a separately trained memoryless model reached **24.0%**.
Random probing reached **65.5%**, a fixed alternating-calibration
policy **73.0%**, and the exact Bayes policy **80.5%**.

At 8 probes, the recurrent learner reached **87.5%**, behind
Bayes at **94.5%**. This is **not a Qwen result** and the controller
was taught by a Bayesian oracle on the same causal world family.
The learner's question head does not yet causally control its
action choice; question-to-action binding and frozen-Qwen J-space
interfaces remain open.

[Code: `learner002.py`](learner002.py) ·
[Protocol](EXPERIMENT_002A.md) · [Results](RESULTS_002A.md)

```powershell
python3.13 -m pip install -r requirements-learner.txt
python3.13 learner002.py --quick
python3.13 learner002.py --train-episodes 4096 --epochs 35 --test-episodes 200 --budgets 4,8
```

## Where this came from

Our [Qwenimagination](https://github.com/anttiluode/Qwenimagination)
experiments showed promising silent concept readouts and limited causal
thought transfer. The goal-directed search in `think.py` could follow
associations toward a question, but its script still determined when to
read, write and check. This Sol route tackles the complementary problem:
**what question is worth asking, and what counts as evidence?**

[WaveFormer](https://github.com/anttiluode/WaveFormer) established that a
small real-valued recurrent pathway can carry bound items beyond a restricted
attention window on one synthetic task. It is a potential memory component,
not proof of an autonomous goal selector. Neither wave oscillations nor a
hand-coded Bayesian controller get credit for novelty by themselves.

The 2026 Anthropic *Verbalizable Representations Form a Global Workspace in
Language Models* paper describes an accessible mid-layer concept workspace.
Its results concern models and interventions that differ from this scaffold.
A frozen Qwen reader/writer must be causally validated, not assumed.

## Next scientific step (not implemented yet)

**002 — Make Qwen operate through a persistent workspace instead of
reading a full text scratchpad.** Replace the hand-coded goal/action
selector with a small learned recurrent controller. Give it only
restricted readings from frozen Qwen and the observation/prediction
residuals, not the hidden world state. Its outputs must decide which
thought to hold, which test to run, and whether to revise the world or
sensor explanation.

A successful 002 must beat same-compute random and fixed-policy controls
on new causal worlds, survive state ablation and a swapped-source test,
and show honest source attribution. A text-history baseline remains
mandatory. If a dictionary, Kalman/Bayesian filter or ordinary prompting
wins, report it.

See [EXPERIMENT_001.md](EXPERIMENT_001.md) for the protocol and
[RESULTS_001.md](RESULTS_001.md) for the verified 200-world CPU results.
A user-uploaded real Qwen **8-world exploratory pilot** has since arrived;
see [PILOT_QWEN8.md](PILOT_QWEN8.md) for the matched controls and
probe-fixation finding. It is not yet evidence of learned self-directed thought.

Research direction: Antti Luode. Sol / ChatGPT, initial engineering.
MIT license. November 18, 2026.
