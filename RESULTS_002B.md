# Experiment 002B — exploratory three-seed results and ablations

18 November 2026. The first results, then a **post-hoc** surprise ablation,
were run on GitHub Actions. Both are from a small recurrent neural controller
trained to imitate an analytic Bayesian teacher, **not** from frozen Qwen.

- [Initial three-seed run (before surprise-off was added)](https://github.com/anttiluode/WonderingQwenSol/actions/runs/44276480750)
- [Second three-seed run (adds surprise-off and saved weights)](https://github.com/anttiluode/WonderingQwenSol/actions/runs/44277193144)
- The second workflow stores all model-seed histories and GRU checkpoints in
  the `002b-exploratory-receipts` Actions artifact, not in Git.
- [Architecture](learner002b.py) · [separate changed-world task](challenge_b.py) ·
  [attention hypothesis](ATTENTION_PULL.md) · [design protocol](EXPERIMENT_002B_DESIGN.md)

## Setup and honesty

For each challenge, three independently initialized controllers (seeds
42, 43, 44), trained 25 epochs on 2,048 simulated episodes from starting
seed 200000; all evaluated on the **same 200 test worlds**. Thus
600 policy/world outputs are *not* 600 independent test worlds.
Challenge A tests original static hidden state; Challenge B separately
draws a world change after an initial visual reading, so calibration
cannot reveal the new world bit. Challenge B train/test seeds are
200000..202047 and 410000..410199 respectively.

Each model has a GRU state, a learned next-measurement outcome predictor,
and a local decaying surprise trace. A sparse **supplied** map routes
surprise into the three relevant questions (world, vision fault, audio
fault). Question scores combine uncertainty in the *learned* posterior
and local surprise. The selected question gates access to a
question-specific learned slot; the action decoder has no unrestricted
wire from GRU state `h`. Python Bayes computes training targets only,
not state updates or action choices at evaluation.

## Challenge A: same static world (four follow-up probes)

| Policy | Full state correct | Brier (lower) |
|---|---:|---:|
| Surprise-on question-conditioned GRU | **73.5–75.5%** | 0.0881–0.0931 |
| Same weights, surprise contribution disabled | **79.0–79.5%** | 0.0895–0.0941 |
| Same weights, state reset every observation | 17.0–24.5% | 0.2122–0.2175 |
| Force question always 'world changed' | 68.0–71.0% | 0.1059–0.1124 |
| Random probes | 65.5% | 0.1380 |
| Fixed alternate calibration | 73.0% | 0.0957 |
| Exact Bayesian reference | 80.5% | 0.0929 |

On eight probes, surprise-on GRU reaches 86.0–86.5%; surprise-off
88.5–90.5%; Bayes 94.5%. The previous 002A recurrent learner reached
81.0% at four and 87.5% at eight on its original seed. This more
constrained 002B architecture is **not an accuracy improvement** over 002A.

## Challenge B: world changes after initial observation (four probes)

| Policy | Full state correct | Brier (lower) |
|---|---:|---:|
| Surprise-on question-conditioned GRU | **77.0–80.0%** | 0.0686–0.0765 |
| Surprise contribution disabled | **78.5%** | 0.0688–0.0715 |
| Same weights, recurrent state reset each step | 33.5% | 0.2488–0.2500 |
| Force question always 'world changed' | 70.0–74.5% | 0.0914–0.0978 |
| Random probes | 71.5% | 0.1042 |
| Fixed alternate calibration | 44.0% | 0.1144 |
| Exact Bayesian reference | 80.5% | 0.0609 |

At eight probes, surprise-on GRU reaches 88.5–92.5%, surprise-off
88.5–92.0%, and Bayes 96.5%. Calibration-only remains 45.5%.

## The three conclusions

**1. Persistent state matters, with these training labels and tasks.**
Resetting it consistently damages performance. These ablations depend
on a particular trained network and do not imply subjective memory.

**2. Questions have a causal path.** Forcing `q` changes action logits
on the same hidden state (verified by tests). In the first
three-seed run, forced q changed the top action on 93–100% of a
30-history exploratory subset. Fixed-world-q intervention reduced
Challenge B accuracy to 70.0–74.5% from 77.0–80.0%. This is evidence
that the query bottleneck is *used*, though q was supplied during
teacher-supervised action-head training.

**3. Local surprise pull did not earn its place.** Removing the extra
surprise contribution improved Challenge A's full-state accuracy in
all three trained seeds; Challenge B stayed roughly even.
This comparison was added **after examining the first results**:
consider it an exploratory diagnostic, not a pre-registered
confirmation. The likely lesson is not 'surprise is irrelevant',
but 'raw surprisal and fixed routing do not automatically prioritize
valuable inquiry'. A noisy observation can be surprising yet
epistemically unimportant.

## Boundaries of what was learned

- The causal graph, possible questions, observation vocabulary and
  sparse route from sensor to questions were given by the programmer.
- The training teacher knows the sensor noise model and world prior.
- Both A and B were trained *separately* under their own task rules,
  so B does not establish zero-shot generalization to a changed
  causal system. Their test seeds are held out, but their laws are
  not.
- A local question-clamp can force action changes without improving
  the true informativeness of actions.
- The optional [Qwen bridge](bridge_002b.py) is implementation only;
  it is **not yet a trained or confirmed Qwen experiment**.
- All models still have only three fixed hidden causes, not a rich
  scene with objects and action-conditioned counterfactuals.

## Next preregistered question, not yet run

On **new untouched worlds** and new sensor mappings, compare
uncertainty-only attention against a *learned relevance/precision gate*
for local prediction-error messages. A useful gate should distinguish
true unexpected world transitions from high-noise channels and should
release attention after a conflict is resolved. Reusing these same
200 test worlds to choose the gate would be overfitting.

Also, run the Qwen concept-swap bridge with clean/printed/random and
question/belief interventions before drawing any 'thinking mind'
conclusion. The small 002B GRU has not made Qwen's own forward passes
necessary yet.
