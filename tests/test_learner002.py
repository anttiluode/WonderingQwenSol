"""Optional PyTorch tests for 002A; baseline CI works without PyTorch."""
import importlib.util
import unittest

HAS_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(HAS_TORCH, "Optional PyTorch tests")
class TestLearner002(unittest.TestCase):
    def setUp(self):
        import torch
        from learner002 import LatentInquirer
        self.torch = torch
        self.model = LatentInquirer(hidden=12).eval()

    def test_only_seven_observable_inputs(self):
        from learner002 import WIDTH, encode_observation, ACTIONS
        self.assertEqual(WIDTH, len(ACTIONS) + 2)
        for action in ACTIONS:
            for value in (0, 1):
                x = encode_observation(action, value)
                self.assertEqual(len(x), 7)
                self.assertEqual(sum(x), 2.0)

    def test_inference_does_not_call_python_belief(self):
        import learner002
        from unittest.mock import patch
        # Patch the module's analytic-memory interfaces. Environment remains
        # free to SAMPLE hidden worlds and produce sensor measurements.
        with patch.object(learner002, "Mind", side_effect=AssertionError("no Mind at test")), \
             patch.object(learner002, "Belief", side_effect=AssertionError("no Bayes at test")):
            out = learner002.run_learned(self.model, seed=310000, budget=3)
        self.assertEqual(len(out["observations"]), 4)
        self.assertEqual(len(out["chosen_questions"]), 4)
        self.assertTrue(all(0 <= p <= 1 for p in out["posterior"].values()))

    def test_state_zeroing_removes_prior_history(self):
        torch = self.torch
        # Two sequences have the same final token but different histories.
        prefix_one = torch.tensor([[1., 0., 0., 0., 0., 1., 0.]])
        prefix_two = torch.tensor([[0., 1., 0., 0., 0., 0., 1.]])
        last = torch.tensor([[0., 0., 1., 0., 0., 1., 0.]])
        a = torch.stack([prefix_one[0], last[0]]).unsqueeze(0)
        b = torch.stack([prefix_two[0], last[0]]).unsqueeze(0)
        with torch.no_grad():
            po, _, _ = self.model(a, reset_each=True)
            qo, _, _ = self.model(b, reset_each=True)
        self.assertTrue(torch.equal(po[0, -1], qo[0, -1]))

    def test_teacher_labels_and_seed_reproducibility(self):
        from learner002 import make_dataset
        x1, p1, q1, a1 = make_dataset(4, 200001)
        x2, p2, q2, a2 = make_dataset(4, 200001)
        self.assertEqual(tuple(x1.shape), (4, 5, 7))
        self.assertEqual(tuple(p1.shape), (4, 5, 8))
        self.assertTrue(self.torch.equal(x1, x2))
        self.assertTrue(self.torch.equal(p1, p2))
        self.assertTrue(self.torch.equal(q1, q2))
        self.assertTrue(self.torch.equal(a1, a2))
        self.assertTrue(self.torch.allclose(p1.sum(-1), self.torch.ones(4, 5)))


if __name__ == "__main__":
    unittest.main()
