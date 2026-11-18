# Qwen 1.5B first pilot — 8 worlds

A user-uploaded `results/qwen8.json` appeared on 2026-11-18.
It contains a CUDA run with `Qwen/Qwen2.5-1.5B-Instruct`.
The Qwen integration chose actions by text generation. The external
Bayesian posterior, goal variable and counterfactual simulator remained
programmed, not learned.

**Matched controls:** a GitHub Actions run on the same seeds 10000..10007
and 4-probe budget:
[WorldLoop CI #44270491216](https://github.com/anttiluode/WonderingQwenSol/actions/runs/44270491216).

| Policy | Full explanation correct | Mean Brier (lower) | Probe cost |
|---|---:|---:|---:|
| Frozen Qwen chooses actions | **7/8** | 0.0501 | 6.200 |
| Exact-model active information gain | **7/8** | **0.0410** | 5.900 |
| Random actions | 5/8 | 0.1065 | 6.088 |
| Repeated vision | 4/8 | 0.2253 | 5.000 |
| Model assuming sensors never break, on active evidence | 1/8 | 0.4413 | 5.900 |

All 32 Qwen completions were valid action names (no fallback). But
its choices were heavily restricted:

- `calibrate_audio`: **18**
- `calibrate_vision`: **14**
- `vision`, `audio`, `touch`: **0**

In the first episode (seed 10000), the hidden world has *both*
sensor faults but no world change. Qwen asks `calibrate_audio` four
times, including three times after the scripted goal has switched
to `world_changed`. The posterior becomes very confident about the
audio fault but wrongly discounts the visual fault. This is a concrete
example of **probe fixation**, not an adaptive experiment strategy.

**Interpretation:** Qwen can emit valid probes and gets 7/8 cases in
this tiny first sample. That's worth continuing, but it does NOT
demonstrate that Qwen forms independent goals, tracks uncertainty
internally, or uses silent thought. The script still supplies the
goal variable and a complete posterior; Qwen only chooses one of
five named experiments, using ordinary text.

The result is exploratory, with just 8 episodes and no pre-registered
Qwen performance claim. Because random got 5/8, we cannot call the
difference robust. Qwen's lower Brier accuracy and higher cost than
the exact Bayesian reference on these seeds also caution against
crediting the model with a better policy.

**Next falsifiers:** run a larger paired Qwen series (and preserve all
raw replies), test swapped sensor names, change probe costs and
noise strengths without tuning the prompt, and finally replace the
text posterior with a state updated through a learned recurrent
interface. A useful model should choose *different* measurements when
the causal uncertainty changes, not always ask to calibrate.
