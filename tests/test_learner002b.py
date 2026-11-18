"""002B tests: source routing, causal question bottleneck and surprise."""
import importlib.util
import unittest
from unittest.mock import patch

HAS_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(HAS_TORCH, "PyTorch optional for baseline CI")
class TestPulledInquirer(unittest.TestCase):
    def setUp(self):
        import torch
        from learner002b import PulledInquirer
        torch.manual_seed(818)
        self.torch = torch
        self.model = PulledInquirer(hidden=12, slot_size=5).eval()

    def test_no_direct_h_action_connection(self):
        from learner002b import CHANNEL_TARGETS
        self.assertEqual(self.model.choose_action[0].in_features, 5 + 3)
        self.assertNotEqual(self.model.choose_action[0].in_features, 12 + 3)
        self.assertEqual(CHANNEL_TARGETS.shape, (5, 3))

    def test_question_intervention_changes_action_logits_with_h_fixed(self):
        with self.torch.no_grad():
            hist = [("vision", 1), ("calibrate_vision", 0), ("audio", 1)]
            a = self.model.raw_intervention(hist, 0)
            b = self.model.raw_intervention(hist, 1)
            c = self.model.raw_intervention(hist, 2)
        self.assertEqual(int(a[6][0]), 0)
        self.assertEqual(int(b[6][0]), 1)
        self.assertEqual(int(c[6][0]), 2)
        self.assertGreater(float((a[4] - b[4]).abs().max()), 1e-4)
        self.assertGreater(float((b[4] - c[4]).abs().max()), 1e-4)

    def test_unexpected_calibration_locally_pulls_sensor_fault(self):
        from learner002 import encode_observation
        torch = self.torch
        with torch.no_grad():
            for param in self.model.predict_sensor.parameters():
                param.zero_()
            self.model.predict_sensor.bias.fill_(-7.0)
            for param in self.model.belief.parameters():
                param.zero_()
            self.model.raw_surprise_gain.fill_(0.0)
            clean = torch.tensor([encode_observation("calibrate_vision", 0)])
            error = torch.tensor([encode_observation("calibrate_vision", 1)])
            h, t = self.model.empty_state(1, "cpu")
            clean_q = self.model.step(clean, h, t)[3]
            bad_q = self.model.step(error, h, t)[3]
        # Surprise on calibrate_vision must pull *vision_fault*, not audio_fault.
        self.assertGreater(float(bad_q[0, 1] - clean_q[0, 1]), 1.0)
        self.assertAlmostEqual(float(bad_q[0, 2]), float(clean_q[0, 2]), places=6)

    def test_learned_execution_has_no_bayes_teacher_call(self):
        import learner002b
        with patch.object(learner002b, "make_dataset", side_effect=AssertionError), \
             patch.object(learner002b, "make_dataset_b", side_effect=AssertionError), \
             patch.object(learner002b, "best_probe", side_effect=AssertionError):
            with self.torch.no_grad():
                sample = learner002b.learned_episode(self.model, 310000, 2)
        self.assertEqual(len(sample["observations"]), 3)
        self.assertEqual(len(sample["questions"]), 3)

    def test_challenge_b_dataset_shapes(self):
        from learner002b import make_dataset_b
        x, p, q, a = make_dataset_b(8, seed=200000)
        self.assertEqual(tuple(x.shape), (8, 5, 7))
        self.assertEqual(tuple(p.shape), (8, 5, 8))
        self.assertEqual(tuple(q.shape), (8, 5))
        self.assertEqual(tuple(a.shape), (8, 5))
        self.assertTrue(self.torch.allclose(p.sum(-1), self.torch.ones(8, 5)))

    def test_end_to_end_small_train(self):
        from learner002b import make_dataset_b, train, benchmark
        torch = self.torch
        training = make_dataset_b(32, seed=202000)
        model = type(self.model)(hidden=12, slot_size=5)
        losses = train(model, training, epochs=2, batch_size=16)
        self.assertEqual(len(losses), 2)
        rows = benchmark(model, range(410000, 410003), 2, challenge="b")
        self.assertEqual(len(rows["per_episode"]["intact"]), 3)
        self.assertTrue(0 <= rows["metrics"]["intact"]["mean_brier"] <= 1)


if __name__ == "__main__":
    unittest.main()
