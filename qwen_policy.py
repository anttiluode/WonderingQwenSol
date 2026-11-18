"""Optional frozen Qwen policy for experiment 001.

Qwen chooses a measurement. The posterior update and provenance ledger remain
external and explicit. This is an INTERFACE test, not hidden-state thought.
Install requirements-qwen.txt and run with --policy qwen --episodes 8.
"""
from __future__ import annotations

import re

from worldloop import NAMES, PROBES


class QwenPolicy:
    def __init__(self, model_name="Qwen/Qwen2.5-1.5B-Instruct", device="auto"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but PyTorch cannot use it")
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype
        ).to(self.device).eval()
        self.fallbacks = 0
        self.decisions = []

    def __call__(self, mind, target: int) -> str:
        history = "\n".join(f'- {e["probe"]}: {e["value"]}'
                            for e in mind.events[-8:])
        # The model receives the SAME joint posterior and sensor laws as the
        # explicit information-gain policy, never the hidden sampled truth.
        joint = "; ".join(
            f"W={w},Vfault={v},Afault={a}: {mass:.3f}"
            for (w, v, a), mass in sorted(mind.belief.p.items())
        )
        sensor_laws = (
            "vision reads W XOR Vfault (90% accurate); "
            "audio reads W XOR Afault (90% accurate); "
            "calibrate_vision reads Vfault (85% accurate); "
            "calibrate_audio reads Afault (85% accurate); "
            "touch reads W (75% accurate). "
            "A 1 means yes/changed/fault, 0 means no."
        )
        marginals = ", ".join(f"{k}={v:.3f}"
                              for k, v in mind.belief.signature().items())
        probe_options = ", ".join(f"{p} (cost {c:.1f})"
                                  for p, (_, c) in PROBES.items())
        instructions = (
            "You are choosing the next experiment for a tiny scientist. "
            "A real world may have changed, or vision or audio may have become "
            "unreliable. You do not know the hidden truth. Choose the one "
            "measurement that best reduces uncertainty about the scientist's "
            "current question. Never treat imagined outcomes as observations. "
            "Answer with exactly ONE action name, nothing else."
        )
        prompt = (
            f"Current question: {NAMES[target]}\n"
            f"Posterior marginals: {marginals}\n"
            f"Joint posterior hypotheses: {joint}\n"
            f"Measurement likelihoods: {sensor_laws}\n"
            f"Available measurements: {probe_options}\n"
            f"Observed readings (only real evidence):\n{history}\n"
            "Next measurement:"
        )
        messages = [
            {"role": "system", "content": instructions},
            {"role": "user", "content": prompt},
        ]
        tokenizer = self.tokenizer
        tokens = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.device)
        with self.torch.inference_mode():
            generated = self.model.generate(
                tokens, max_new_tokens=20, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        reply = tokenizer.decode(generated[0, tokens.shape[1]:],
                                 skip_special_tokens=True).strip()
        # Parse only the newly generated continuation, never the prompt.
        match = re.match(
            r"^(calibrate_vision|calibrate_audio|vision|audio|touch)\b",
            reply.lower().strip(" \x60\"'"))
        if match:
            action, valid = match.group(1), True
        else:
            # Explicitly counted invalid completion, not silently changed to
            # an active-policy decision that would inflate Qwen's performance.
            action, valid = "vision", False
            self.fallbacks += 1
        self.decisions.append({
            "question": NAMES[target], "model_reply": reply,
            "chosen_probe": action, "valid": valid,
        })
        return action
