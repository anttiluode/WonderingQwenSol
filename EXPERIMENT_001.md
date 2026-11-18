# Experiment 001 — source attribution and active inquiry

Date frozen: 2026-11-18. These are planned **simulator** controls, not a
claim that Qwen passes them. Any results after modifications require a
new versioned protocol. No "consciousness" inference is licensed.

## The causal problem

Three binary hidden variables:

- `w` — the world changed relative to baseline.
- `v` — vision inverted (fault).
- `a` — audio inverted (fault).

The sensory reports `vision = w XOR v` and `audio = w XOR a`
are each corrupted by independent 10% noise. A `touch` check reports
`w` with 75% accuracy. Sensor calibration independently reports its
fault bit with 85% accuracy.

World priors are fixed: no changes 0.05, exactly one change 0.75
(uniform over 3), exactly two changes 0.18 (uniform over 3), all
three change 0.02. The environment samples the true hidden state,
the model receives no hidden-state information, and each observation
has reproducible noise keyed by seed, sensor and observation count.
All policies start with one unrequested visual observation.

## Frozen evaluation

- Seeds: 10000..10199 (200 paired episodes), unless a run is marked
  exploratory or smoke. Default script uses 100 for faster development.
- Four chosen probes after the first vision reading.
- Physical probe cost: vision/audio 1.0 each; calibration 1.3; touch 1.6.
- Compare active vs random vs repeat, and no-fault inference on the
  active policy's **exact same evidence**.
- For optional Qwen: run the same seeds/budget, cache weights and decode
  greedily, publish invalid action rate and actual generated choices.
  Start with a small exploratory GPU smoke first.

Primary metrics: mean Brier score across all 3 hidden variables and
mean cost. Secondary metrics: marginal accuracies and joint MAP
accuracy. Report all arms and the full cost/accuracy tradeoff; never
call a higher-cost policy unconditionally superior.

### What would count as useful

1. Posterior and imagined paths remain separate under test:
   counterfactual branches cannot silently become evidence.
2. The active-inquiry reference beats repeated vision for joint
   attribution at matched budget. If it does not, check the
   scoring/cost tradeoff rather than editing the hidden episodes.
3. At **the same evidence**, the full source-aware model beats the
   "everything was a world change" model on fault episodes.
4. Frozen Qwen, if tested, should select informative probes better
   than random with no additional information about the hidden state.
   A result without this advantage is a negative.
5. Crucially, meeting 1–4 would demonstrate an interface, **not**
   autonomous neural thinking. That requires learned state,
   learned action choice, unseen-world generalization, and ablation.

### Known limitations / possible falsifiers

- This small world's causal structure and likelihoods are supplied by
  the programmer. Bayesian inference is an engineered upper reference.
- The program itself currently chooses the question by entropy. Qwen
  only chooses an action among supplied probes. That does not establish
  spontaneous goal creation.
- Posterior is stored in external Python memory, not Qwen activations.
- "imagined" and "observed" are explicit bookkeeping labels, not
  neural source attribution.
- Model-generated actions may exploit shallow priors. Test shuffled
  sensor names and changed priors in version 002 to see whether any
  success survives re-labeling.
- The simulator is static within an episode; real continuously
  changing worlds require temporal dynamics and unannounced drift.
- Repeating a noisy independent measurement is useful; do not treat
  the weak repeat-vision control as a proof of intelligence.
- A strong simple Bayesian policy may beat Qwen. If so, the honest
  conclusion is that Qwen did not improve this well-specified task.

## Next experiment contract

002 should put Qwen's readout and a small recurrent controller behind
an interface that sees partial observations and the residual between
actual and imagined observations. It must:

- keep relevant beliefs after long silent gaps and distracted text;
- retain source attribution when one channel corrupts;
- choose an informative new experiment without an operator's goal;
- beat static-context, text-scratchpad, random, and same-budget Bayes
  baselines on held-out families, with recurrent-state ablation.

Use the same train/test split and scoring code for every condition.
Constrain GPU work to the user's RTX 3060 12 GB whenever feasible.
