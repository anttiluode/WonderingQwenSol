# Experiment 002B — make the question causally control the inquiry

Design freeze candidate, 18 November 2026. **Not implemented or tested here.**
This builds on [002A](RESULTS_002A.md), the November 17 repositories,
and Argon's critique. The project must avoid both the Qwenimagination
V4 role-selection shortcut and 002A's epiphenomenal question head.

## The actual object

The cognitive cycle has four distinguishable variables:

- **h**: recurrent learned belief/context state, updated from real
  (sensor, observation) pairs. Not an analytic Python posterior at test.
- **q**: a selected *question about the world or the apparatus*.
- **b(q)**: a question-conditioned view of the beliefs/uncertainty in h.
- **a**: a chosen measurement, plus an explicit stop/continue decision.

No "question" success if q is merely printed while h independently
selects a. No "Qwen thinks" success if Python has already computed the
optimal action or inserted the requested concept.

## 002B-1: prove causal selection in the small CPU model

Begin with 002A's pretrained GRU, not with expensive Qwen inference.

1. GRU receives only raw sensor identity and observed value; updates h.
2. A question proposer produces q from h, as a 3-way categorical choice
   or soft distribution (world changed / vision faulty / audio faulty).
3. Belief is read through **q-conditioned slots** b(q)=Read(h,q).
   Action head sees b(q) and q, **never an independent h→action
   shortcut**. For a discrete q, use an explicit selected slot and
   compare with a learned attention query. Check implementation for
   inadvertent leakage.
4. Compare expected reductions of predictive loss or entropy *after*
   seeing hypothetical future observations, with action cost. A small
   realized KL change is not an automatic switch trigger: repeated
   measurements can correctly build confidence.
5. Train on teacher demonstration as a baseline, then test a
   limited-reward fine-tuning variant; label any teacher inheritance
   honestly.

A forced-q intervention is run on **the same frozen h**. It is not
enough merely to make the output change: the changed action should
prefer a measurement informative about the forced question on a
known state of uncertainty, and improve matched outcomes.

## 002B-2: validate a frozen-Qwen decision bridge

Only after 002B-1 passes do we attach Qwen2.5-1.5B-Instruct.

- Reuse Qwenimagination's **early band swap at layers 4–12**, and
  inspect layer-26-derived concepts as a candidate read path.
- Check that chosen concept directions actually represent q, and
  that the intervention changes a *different downstream question*
  under content-matched controls. The V11b G2 thought-vs-random gate
  failed: DO NOT treat the bridge as already validated for arbitrary
  latent belief vectors.
- Begin with a carrier already bearing a matched concept; implement
  **subtract old / add new**, magnitude matched, consistently clamped
  over the validated band. Record null, same-concept, matched random,
  printed concept, and mid/late-layer control.
- Then test a learned projection of h's **question-relevant belief**
  into a usable Qwen representation. A q-only word token is not
  enough: Qwen must have relevant evidence and uncertainty, or the
  mapping 'audio fault -> calibrate audio' is just a dictionary.
- Score the **five possible probe-action continuations** or a
  constrained output; preserve verbatim model logits and any invalid
  generation. Never let Python choose the action by unbinding q or
  doing a Bayes calculation before the model's decision.
- Freeze Qwen weights initially. If a projector must be trained,
  separate its fitted data from held-out tests; compare with an
  equally-sized scalar/MLP probe selector that has access to the
  same h and q.
- Compare against a text-scratchpad Qwen supplied the same observations
  and costs, not a *solved* posterior when the latent arm does not get it.

The critical intervention is **two-way**: clamp q with fixed h, then
swap/erase h with fixed q. Both should change the relevant choice
in predicted, useful ways. The final Qwen-based action must depend on
both paths, not an uninspected shortcut.

## Test matrix

- **G1 — state is needed:** reset/swap h; held-out performance falls
  relative to intact, with matched sensor readings and budgets.
- **G2 — question is needed:** hold h constant, force q to alternative
  causes; probe rankings shift toward actions informative about that
  question, with controls for arbitrary word perturbation.
- **G3 — evidence matters:** hold q constant and swap h between worlds
  requiring different tests; rankings respond to uncertainty, not
  just 'calibrate' language.
- **G4 — Qwen contributes:** early-band intervention beats its
  no-swap, random/same-concept, late-write and direct scalar control
  at the same cost. If a simple MLP is as good, report that.
- **G5 — unseen rules:** new sensor-name permutations, reliabilities,
  costs and causal combinations tested separately. Check whether
  Qwen receives the *correct new rules* through matched channels and
  whether the controller actually uses them.
- **G6 — source discipline:** imagined observations never update h
  as real evidence without a sensor query. Blind and corrupted-sensor
  periods tested explicitly.

Use joint hidden-world accuracy, proper Brier/log loss, action cost,
question-to-action causal effects, and per-seed failure traces. 
Predeclare seeds and train-test partition before confirmation. Test
several training seeds: the 002A report is currently one-seed exploratory.

## Separate Challenge B: calibration cannot solve world content

Argon correctly notices the strong calibration heuristic (73% on
002A). But **do not modify the original eight-state benchmark simply
to force a 30-point margin.** That would make the task easier for a
favoured mechanism after seeing its competitors.

Instead build and version a distinct challenge environment: provide
a time-0 baseline observation, then let the *outside world change*
at an unobserved time and allow sensor faults to change as well.
Calibration reveals current sensor health but **not the current world
value**. An observation of content (vision/audio/touch) after the
possible transition is now necessary. Prove this in a control that
a calibration-only policy cannot disambiguate the relevant worlds
even with infinite calibrations at fixed time.

State the exact transition and noise laws, include a Bayes upper
reference and fixed heuristic, and choose metrics before training.
Evaluate challenge A (original) and B side by side.

## What November 17 contributes

- WaveComputer: distinct content, prediction error and certainty.
- WaveComputerAstra: addressed routing; preserve scalar baseline.
- ApicalRuleReversalMachine: fast, causal context gate over slow weights.
- WaveCortex: an inferred hidden rule affects imagined consequences.
- TuftSheet: several local contexts can coexist; learning boundaries
  was not solved.
- TuftSheetSol: stored belief versus impaired readout/source reliability.
- SyntheticQualia: functional criteria Q1–Q9; no experience claim.
- Qwenimagination: a candidate frozen-Qwen read/write route, with
  known asymmetric and random-control failures.

**Claim ceiling:** Success would show a causally effective,
question-conditioned learned inquiry controller that uses frozen
Qwen's internal computation. It would not establish a general thinking
mind or phenomenal consciousness.
