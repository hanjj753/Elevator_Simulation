import unittest

from validate_results import SETTING_FIELDS, bootstrap_mean_interval, paired_comparison


class PairedAnalysisTests(unittest.TestCase):
    def test_zero_differences_stay_zero(self) -> None:
        result = bootstrap_mean_interval([0.0] * 30, resamples=100)
        self.assertEqual(result["ci95_low_seconds"], 0)
        self.assertEqual(result["ci95_high_seconds"], 0)
        self.assertEqual(result["equal_runs"], 30)

    def test_bootstrap_is_reproducible_and_preserves_difference_sign(self) -> None:
        differences = [-3.0, -2.0, -1.0] * 10
        first = bootstrap_mean_interval(differences, resamples=500, seed=19)
        self.assertEqual(first, bootstrap_mean_interval(differences, resamples=500, seed=19))
        self.assertEqual(first["mean_difference_seconds"], -2.0)
        self.assertLess(first["ci95_high_seconds"], 0)

    def test_duplicate_and_missing_run_pairs_are_rejected(self) -> None:
        row = {"strategy": "priority_eta", "run": "1"}
        with self.assertRaisesRegex(ValueError, "Duplicate run"):
            paired_comparison([row, row], "eta", resamples=100)
        with self.assertRaisesRegex(ValueError, "runs do not match"):
            paired_comparison([row], "eta", resamples=100)

    def test_same_run_number_with_different_seed_is_not_a_pair(self) -> None:
        settings = dict.fromkeys(SETTING_FIELDS, "1")
        candidate = {**settings, "strategy": "priority_eta", "run": "1"}
        reference = {**settings, "strategy": "eta", "run": "1", "seed": "2"}
        with self.assertRaisesRegex(ValueError, "Unpaired settings"):
            paired_comparison([candidate, reference], "eta", resamples=100)


if __name__ == "__main__":
    unittest.main()
