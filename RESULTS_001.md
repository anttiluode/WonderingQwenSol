# Experiment 001 — first CPU observations

**Run:** GitHub Actions [WorldLoop CI #44269926682](https://github.com/anttiluode/WonderingQwenSol/actions/runs/44269926682)
on 2026-11-18, commit `999ede4072d4e657ff21552d98ce3c956bf7bc9e`.

**Status: simulation only.** These numbers are from a *programmed exact
Bayesian model of a tiny artificial world*, not a Qwen model and not
self-generated humanlike thought. The 12 unit tests passed; the
200-episode run finished successfully on Python 3.12.

Commands used:

```bash
python -m unittest discover -s tests -v
python -m py_compile worldloop.py qwen_policy.py
python worldloop.py --episodes 200 --seed 10000 --budget 4 --out /tmp/worldloop-smoke.json
```

Fixed seeds 10000..10199; one unsolicited vision measurement followed by
four policy-selected probes. Every comparison used the same sampled
worlds; the no-fault arm used exactly the active arm's evidence.

| Policy | World correct | Visual fault correct | Audio fault correct | Joint correct | Brier (lower) | Mean probe cost |
|---|---:|---:|---:|---:|---:|---:|
| Active Bayesian inquiry | 89.5% | 91.5% | 92.0% | **84.0%** | **0.0690** | 5.848 |
| Random probes | 81.5% | 80.5% | 86.5% | 76.0% | 0.1184 | 5.942 |
| Repeat vision | 65.5% | 57.5% | 76.0% | 54.0% | 0.2037 | 5.000 |
| No fault hypothesis, identical active evidence | 58.0% | 57.5% | 64.0% | 26.0% | 0.3439 | 5.848 |

The active reference has lower Brier error than random and repeat
across the 200 selected episodes. Its joint correctness is 8 percentage
points above random and 30 above repeat. Repeating vision costs less
than active; don't ignore that tradeoff.

**Interpretation:** This is a successful *instrumentation test*. A
correctly specified model that can hold multiple explanations and
choose informative experiments makes better use of evidence. But the
likelihoods, candidate questions, information-gain objective and
posterior updates are all supplied by the programmer. The simulation
does not show Qwen discovering any of them.

**What remains untested:** The `--policy qwen` integration has been
syntax-checked but not run with real Qwen weights. There are no Qwen
results or GPU tests here. A model that just says `vision` every time
should score like the weak repeat control, and that would be a
legitimate negative. Qwen must improve on the random action baseline
with the same observations and budgets before its choices count as
useful. The more ambitious target requires a learned recurrent internal
state, not merely a text prompt containing our externally calculated
posterior.

The 200-seed run was a refinement of the original 24-episode CI smoke;
not a pre-registered independent replication. The number of seeds and
report here are transparent, and we should freeze them before tuning
the future learned controller.
