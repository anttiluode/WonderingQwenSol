"""Offline double-swap plumbing checks, no Hugging Face / Qwen weights required."""
import contextlib
import importlib.util
import types
import unittest

HAS_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(HAS_TORCH, "optional torch")
class TestQwenBridge002B(unittest.TestCase):
    def setUp(self):
        import torch
        from bridge_002b import REQUIRED, DoubleSwap, BAND
        self.torch = torch
        self.DoubleSwap = DoubleSwap
        self.BAND = BAND

        class Layer(torch.nn.Module):
            def __init__(self, d):
                super().__init__()
                self.lin = torch.nn.Linear(d, d)
            def forward(self, x):
                return x + .05 * self.lin(x).tanh()

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = torch.nn.Embedding(70, 16)
                self.layers = torch.nn.ModuleList([Layer(16) for _ in range(13)])
                self.head = torch.nn.Linear(16, 70)
            def forward(self, input_ids):
                x = self.emb(input_ids)
                for layer in self.layers:
                    x = layer(x)
                return types.SimpleNamespace(logits=self.head(x))

        @contextlib.contextmanager
        def tap(lm, layers):
            st = {}
            hooks = []
            for i in layers:
                def hook(_m, _ins, out, i=i):
                    st[i] = out
                hooks.append(lm.layers[i].register_forward_hook(hook))
            try:
                yield st
            finally:
                for hook in hooks:
                    hook.remove()

        class LM:
            def __init__(self):
                self.device = "cpu"
                self.model = Model().eval()
                self.layers = self.model.layers
            def encode(self, text):
                return [int(ord(ch) % 60) for ch in text]
        class Lens:
            def __init__(self):
                self.J = torch.randn(13, len(REQUIRED), 16)
                self.index = {word: i for i, word in enumerate(REQUIRED)}

        torch.manual_seed(42)
        self.lm = LM()
        self.lens = Lens()
        self.tap = tap

    def swap(self, forced="vision", belief=0.8, random_q=False):
        from bridge_002b import DoubleSwap
        return DoubleSwap(self.lm, self.lens, forced, belief,
                          lambda out: out[0] if isinstance(out, tuple) else out,
                          lambda old, new: new,
                          self.tap, random_q=random_q)

    def test_question_only_changes_output_with_same_prompt(self):
        torch = self.torch
        with torch.no_grad():
            swap = self.swap()
            a = swap.run([1,2,3,4], "clean")
            b = swap.run([1,2,3,4], "q")
            c = swap.run([1,2,3,4], "belief")
            d = swap.run([1,2,3,4], "both")
        self.assertEqual(tuple(a.shape), (4, 70))
        self.assertFalse(torch.allclose(a, b, atol=1e-5))
        self.assertFalse(torch.allclose(a, c, atol=1e-5))
        self.assertFalse(torch.allclose(b, d, atol=1e-5))

    def test_other_question_changes_q_swap(self):
        torch = self.torch
        with torch.no_grad():
            a = self.swap("vision").run([1,2,3], "q")
            b = self.swap("audio").run([1,2,3], "q")
        self.assertFalse(torch.allclose(a, b, atol=1e-5))

    def test_length_normalized_option_scoring_is_finite(self):
        from bridge_002b import score_option
        score = score_option(self.lm, self.swap(), [1,2,3], "vision", "both")
        self.assertTrue(-100 <= score <= 0)

    def test_extract_state_replays_observations_only(self):
        from bridge_002b import extract_state
        from learner002b import PulledInquirer
        torch = self.torch
        with torch.no_grad():
            model = PulledInquirer(hidden=12)
            q, p, scores = extract_state(model, [
                {"probe":"vision", "value":1},
                {"probe":"calibrate_vision", "value":0},
            ])
        self.assertIn(q, (0,1,2))
        self.assertEqual(len(p), 3)
        self.assertEqual(len(scores), 3)
        self.assertTrue(all(0 <= x <= 1 for x in p))


if __name__ == "__main__":
    unittest.main()
