"""No-dependency tests for the separate changed-world task."""
import unittest

from challenge_b import (ChangedWorld, independent_prior,
                         transition_after_first_vision, run, benchmark)
from worldloop import PROBES


class TestChallengeB(unittest.TestCase):
    def test_first_reading_uses_baseline(self):
        a = ChangedWorld(123, final_w=1, vision_fault=0, audio_fault=0)
        b = ChangedWorld(123, final_w=0, vision_fault=0, audio_fault=0)
        self.assertEqual(a.observe("vision"), b.observe("vision"))

    def test_calibration_cannot_reveal_world_change(self):
        for seed in range(20):
            world = ChangedWorld.sample(seed)
            belief = independent_prior()
            belief.observe("vision", world.observe("vision"))
            belief = transition_after_first_vision(belief)
            self.assertAlmostEqual(belief.marginal(0), 0.5, places=10)
            for i in range(6):
                probe = "calibrate_audio" if i % 2 == 0 else "calibrate_vision"
                belief.observe(probe, world.observe(probe))
                self.assertAlmostEqual(belief.marginal(0), 0.5, places=10)

    def test_post_change_raw_observation_reveals_world_information(self):
        w = ChangedWorld.sample(410005)
        b = independent_prior()
        b.observe("vision", w.observe("vision"))
        b = transition_after_first_vision(b)
        self.assertAlmostEqual(b.marginal(0), 0.5)
        b.observe("touch", w.observe("touch"))
        self.assertGreater(abs(b.marginal(0) - 0.5), 0.1)

    def test_reproducibility(self):
        self.assertEqual(run(410001, "active"), run(410001, "active"))
        self.assertEqual(run(410001, "random"), run(410001, "random"))

    def test_fair_actions(self):
        for policy in ("active", "calibrate_alternate", "random", "repeat"):
            r = run(410000, policy, 4)
            self.assertEqual(len(r["observations"]), 5)
            self.assertEqual(r["observations"][0]["phase"], "baseline")
            self.assertTrue(all(e["probe"] in PROBES for e in r["observations"]))

    def test_calibration_only_world_marginal_exactly_half(self):
        for seed in range(410000, 410010):
            r = run(seed, "calibrate_alternate", 4)
            self.assertEqual(r["posterior"]["world_changed"], 0.5)

    def test_small_benchmark(self):
        r = benchmark(410000, 8, 4)
        self.assertEqual(r["episodes"], 8)
        self.assertEqual(set(r["results"]), {
            "active", "calibrate_alternate", "random", "repeat"})


if __name__ == "__main__":
    unittest.main()
