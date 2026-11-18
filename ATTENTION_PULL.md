# Pulled by surprise — the local attention hypothesis

Antti's observation (18 November 2026): in a biological mind, one is not
free to browse all thoughts independently of one's circuitry. Unexpected
sensory or conceptual events can *capture* attention. The track that
registered an error can pull the next thought into a limited workspace.

We use this as a **functional engineering hypothesis**, not an established
neuroscience explanation of any particular person's conscious experience.

## What 002B presently implements

`learner002b.py` replaces the unconstrained 002A question head with
**local tension plus retained uncertainty**:

1. The recurrent model predicts the result of the next measurement *before
   it sees that result*.
2. It records the surprise of the actual outcome:
   `surprise = -log P(observed value | previous learned state, probe)`.
3. The surprise enters only those local tracks allowed to respond to the
   observed probe (a supplied sparse causal routing mask):
   vision -> world and vision-fault; audio -> world and audio-fault;
   sensor calibration -> that sensor's fault; touch -> world.
4. Each track keeps a decaying tension trace:
   `tension_t = 0.68 * tension_previous + routed_surprise_t`.
5. The question is the track with greatest
   `1.1 * H(world)` or `H(sensor fault)` plus
   `softplus(gain) * tension_t`. Entropy comes from the GRU's *learned*
   joint belief probabilities, not a Python Bayes update.
6. The question pulls a **corresponding local slot** out of the learned
   recurrent state. The action head sees only that slot plus question
   identity—**no direct full-state h -> action bypass**.

A local prediction error makes a track more salient; already-resolved
uncertainty and trace decay allow the attention pull to relax.
This is not a hard rule saying 'zero KL -> abandon it': a small
realized belief shift can still be useful evidence.

## Why this is not yet self-generated cognitive structure

- The three possible questions are supplied by the benchmark.
- The causal routing mask is supplied, not self-discovered.
- The loss comes from a Bayesian teacher at TRAIN time.
- The GRU stores a compact latent code, not objects or linguistic
  propositions with a proven semantics.
- The 'source' distinction is limited to three fixed world/fault bits.
- At test time there is no precomputed Bayesian solution for the learned
  controller; that *does* test learned persistence and action choice.

The full program cannot be described as a free-standing mind.

## Falsifiers and controls

**C1: memory.** Clear the recurrent state; measure damage.

**C2: question.** Clamp each of the three q labels on the same observation
history. A different action-logit is a minimum technical check.
A meaningful behavioral improvement versus a fixed question is stronger.

**C3: surprise.** Explicitly disable the local surprise gain at
evaluation while preserving learned entropy-based uncertainty.
If it does just as well or better, *surprise pull has not earned its place*.
This ablation was added after examining first exploratory results and is
not a pre-registered independent confirmation.

**C4: false alarm.** Unexpected noise shouldn't always capture attention:
test noisy but irrelevant channels, held-out sensor laws and whether the
controller eventually returns to high-value unfinished hypotheses.

**C5: sensor rule changes.** The current Challenge B contains an
unannounced world change after the first observation, but the
environment's sensor mapping is still supplied. Test changed
reliability and re-labeling separately; don't call new random
seeds 'new causal laws'.

**C6: Qwen bridge.** Matched clean, q-only, belief-only, combined,
random-concept and printed-text controls. A model that merely says
'calibrate vision' after the word 'vision' was injected has not
inferred the most informative experiment. Frozen Qwen has not yet
been verified in the 002B harness.

## Beyond the three-track toy

In a richer scene, the tracks might not be named 'world', 'vision'
and 'audio'. They could be learned local predictive modules for
objects, other agents, rules, sensors and actions. Each module would
send a small request to the workspace when its error, uncertainty
and expected relevance warrant processing.

Crucially, attention is **competitive and constrained** by physical
or computational wiring. We are not giving the machine free range
of all imaginable questions; we are attempting to make the world,
its internal predictions, and its connectivity pull inquiry through
specific unfinished problems.
