# Experiment 002A — the Python-out-of-the-loop gate

**Frozen before evaluating trained model results.** This is the narrowest
technical test of Gemini Argon's important objection to Experiment 001.

## Hypothesis

A small recurrent controller can learn to hold competing explanations of
the world's state after a sequence of partial, noisy observations. At
**evaluation time** it receives only measurements, not the exact Bayesian
posterior or any hidden world bit. It chooses the next probe and a goal
variable from its own recurrent state.

This alone is **not** Qwen thinking. The network is trained by imitation
of an engineered Bayes information-gain teacher. A later 002B must make
its state causally usable by Qwen without text serialization.

## Frozen environment

Use the same 8-state World/PROBES/prior as Experiment 001. Train on
200,000+seed episodes; test on 310,000+seed (no overlap). Default
training episodes 4,096, observations per episode 5 (one unrequested
vision observation plus four probes). 70% of training probes chosen
at random, 30% from the Bayes information-gain teacher; the mixture is
not conditioned on hidden truth. Teacher computes the target joint
posterior, preferred question and probe *only while generating labels*.

Network input per observation: one-hot probe identity (five), one-hot
sensor reading (two). A GRU with hidden width 48 and three heads
predicts eight-state posterior, choice of question (three classes),
and next probe (five classes). Supervised losses: posterior KL
(weight 1.0), probe cross entropy (0.30), question cross entropy (0.15).

At **evaluation**, the trained model updates its GRU hidden state with
the new observation. Python does not supply the exact posterior,
its information gain, its labels, or the teacher's next probe.
A later Qwen integration must meet this same no-solved-state contract.

## Primary comparisons (identical hidden test seeds, budget and noise)

- `learned_recurrent`: trained GRU state and policy.
- `state_zeroed`: identical trained weights but reset GRU before
  **each** observation (all history removed).
- `memoryless_trained`: separately trained model with no across-event
  state, same training samples and loss.
- `random`: same measurement budget, independent random choices.
- `calibrate_alternate`: trivial alternating sensor-calibration
  heuristic; catches an easy shortcut seen in the first Qwen8 pilot.
- `active_bayes_reference`: known-model information-gain oracle
  for reference, not a learned baseline.

**Primary scientific metric:** mean Brier across 3 source variables.
Report also full-state joint accuracy, probe cost, and per-seed output.
Require the *same trained recurrent weights* to beat the state-zeroed
arm and memoryless model on held-out seeds; otherwise there is no
evidence recurrence is needed.

## Secondary tests / limitations

- Evaluate at 8 follow-up probes after training on 4. Long-horizon
  improvement must be reported separately, with no retraining.
- Do not claim humanlike self-inquiry, discovery, or phenomenology:
  the controller is supervised by the Bayes teacher.
- If random/heuristic/reference wins, preserve it.
- In 002B, freeze Qwen and attach the learned state through validated
  J-lens read/write interventions. A text-table control is mandatory.
- No explicit KL/surprise gate is required. Expected information gain
  and realized posterior change are distinct quantities; a small
  realized update alone does not make repetition useless.
- In this lab, one initial visual observation has already happened;
  sensor calibration by itself can be informative, so do not force
  an invented rule that a raw vision/audio probe must follow.
