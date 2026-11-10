import unittest

from analyze_passenger_effects import decompose


class PassengerDecompositionTests(unittest.TestCase):
    def test_majority_can_improve_while_mean_worsens(self):
        result = decompose([-1, -1, -1, 10])
        self.assertEqual(result["improved_fraction"], .75)
        self.assertEqual(result["benefit_seconds_per_passenger"], .75)
        self.assertEqual(result["harm_seconds_per_passenger"], 2.5)
        self.assertEqual(result["net_mean_wait_difference_seconds"], 1.75)

    def test_unchanged_and_empty_groups_are_distinguished(self):
        result = decompose([0, 0, 0])
        self.assertEqual(result["equal_fraction"], 1)
        self.assertEqual(result["net_mean_wait_difference_seconds"], 0)
        with self.assertRaises(ValueError):
            decompose([])


if __name__ == "__main__":
    unittest.main()
