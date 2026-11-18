"""Pure-Python regression and falsifier checks for WorldLoop."""
import unittest

from worldloop import (
    Belief, Mind, PROBES, STATES, World, benchmark, metrics, naive_replay,
    prior, run_episode, truth_signal,
)


class TestWorldLoop(unittest.TestCase):
    def test_prior_valid_and_full_support(self):
        p = prior()
        self.assertEqual(set(p), set(STATES))
        self.assertAlmostEqual(sum(p.values()), 1)
        self.assertTrue(all(v > 0 for v in p.values()))

    def test_fault_and_world_are_confounded_in_one_sensor(self):
        self.assertEqual(truth_signal((1, 0, 0), "vision"), 1)
        self.assertEqual(truth_signal((0, 1, 0), "vision"), 1)
        self.assertNotEqual(truth_signal((1, 0, 0), "audio"),
                            truth_signal((0, 1, 0), "audio"))

    def test_imagination_never_becomes_observation(self):
        mind = Mind()
        original = mind.belief.p.copy()
        for p in PROBES:
            f = mind.wonder(p, 0)
            self.assertGreaterEqual(f["p_one"], 0)
            self.assertLessEqual(f["p_one"], 1)
        self.assertEqual(original, mind.belief.p)
        self.assertEqual(len(mind.events), 0)
        self.assertTrue(all(x["source"] == "imagined" for x in mind.imagined))

    def test_observation_updates_belief(self):
        b = Belief()
        prior_world = b.marginal(0)
        b.observe("touch", 1)
        self.assertGreater(b.marginal(0), prior_world)
        self.assertAlmostEqual(sum(b.p.values()), 1)

    def test_counterfactual_information_nonnegative(self):
        b = Belief()
        for p in PROBES:
            for variable in (None, 0, 1, 2):
                gain = b.imagination(p, variable)["information_gain_bits"]
                self.assertGreaterEqual(gain, 0)
                self.assertLessEqual(gain, 3)

    def test_world_noise_is_action_order_invariant(self):
        a = World.sample(12345)
        b = World.sample(12345)
        self.assertEqual(a.observe("audio"), b.observe("audio"))
        a.observe("vision")
        self.assertEqual(a.observe("audio"), b.observe("audio"))

    def test_active_policy_has_no_truth_access(self):
        a = run_episode(101, budget=3)
        self.assertEqual(len(a["steps"]), 3)
        self.assertEqual(len(a["observations"]), 4)
        self.assertTrue(all(x["source"] == "observed"
                            for x in a["observations"]))
        self.assertTrue(all(x["source"] == "imagined"
                            for x in a["counterfactuals"]))
        self.assertEqual({x["self_chosen_question"] for x in a["steps"]}
                         .difference({"world_changed", "vision_fault", "audio_fault"}),
                         set())

    def test_replay_exact_same_evidence(self):
        record = run_episode(666, budget=5)
        replacement = naive_replay(record)
        self.assertEqual(replacement["observations"], record["observations"])
        self.assertEqual(replacement["probe_cost"], record["probe_cost"])
        self.assertEqual(replacement["hidden_truth"], record["hidden_truth"])

    def test_seed_reproducibility(self):
        self.assertEqual(run_episode(44, "active", 4),
                         run_episode(44, "active", 4))
        self.assertEqual(run_episode(44, "random", 4),
                         run_episode(44, "random", 4))

    def test_paired_protocol_and_metrics(self):
        report = benchmark(start=101, episodes=5, budget=3)
        self.assertEqual(report["status"], "simulation, not Qwen cognitive result")
        self.assertEqual(set(report["results"]), {
            "active", "random", "repeat", "no_fault_model_same_evidence"})
        for row in report["results"].values():
            self.assertGreaterEqual(row["metrics"]["mean_brier"], 0)
            self.assertLessEqual(row["metrics"]["mean_brier"], 1)
        self.assertRaises(ValueError, metrics, [])

    def test_invalid_arguments(self):
        with self.assertRaises(ValueError):
            run_episode(1, "qwen", 2)
        with self.assertRaises(ValueError):
            run_episode(1, "unknown", 2)
        with self.assertRaises(ValueError):
            run_episode(1, "active", -1)

    def test_qwen_policy_callback(self):
        chosen = []
        def dummy(mind, goal):
            chosen.append((mind.belief.signature(), goal))
            return "touch"
        record = run_episode(22, "qwen", 2, qwen=dummy)
        self.assertEqual([r["action"] for r in record["steps"]], ["touch", "touch"])
        self.assertEqual(len(chosen), 2)


if __name__ == "__main__":
    unittest.main()
