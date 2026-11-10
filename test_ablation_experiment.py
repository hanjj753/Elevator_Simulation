import gzip
import json
import tempfile
import unittest
from pathlib import Path

from ablation_experiment import AblationSimulation, run_ablation
from elevator_sim import ElevatorSimulation, PassengerDemand, SimulationConfig


class AblationTests(unittest.TestCase):
    def test_both_redispatch_policies_use_same_clock_even_without_old_calls(self):
        config = SimulationConfig(duration_seconds=90)
        demands = [PassengerDemand(0, 44, 2, 8), PassengerDemand(1, 44, 3, 9)]
        times = []
        for policy in ("redispatch_eta", "age_order_eta"):
            simulation = AblationSimulation(config, policy, demands)
            simulation._add_arrivals(44)
            executed = []
            for now in range(44, 61):
                before = simulation.redispatch_rounds
                simulation._maybe_redispatch_priority_calls(now)
                if simulation.redispatch_rounds > before:
                    executed.append(now)
            times.append(executed)
        self.assertEqual(times, [[45, 50, 55, 60], [45, 50, 55, 60]])

    def test_age_uses_remaining_passengers_not_departed_call_creator(self):
        demands = [PassengerDemand(0, 0, 2, 9), PassengerDemand(1, 100, 2, 9),
                   PassengerDemand(2, 50, 3, 9)]
        simulation = AblationSimulation(SimulationConfig(duration_seconds=120), "age_order_eta", demands)
        for now in (0, 50, 100):
            simulation._add_arrivals(now)
        simulation.calls[(2, 1)].waiting_passenger_ids.remove(0)
        self.assertEqual([call.floor for call in simulation._ordered_calls(105)], [3, 2])

    def test_eta_baseline_is_unchanged(self):
        config = SimulationConfig(duration_seconds=60)
        demands = [PassengerDemand(0, 0, 1, 12), PassengerDemand(1, 15, 6, 1)]
        self.assertEqual(ElevatorSimulation(config, "eta", demands).run(),
                         AblationSimulation(config, "eta", demands).run())

    def test_saved_journeys_and_independent_run_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            report = run_ablation(Path(directory), runs=2, duration=90, seed=101,
                                  order_seeds=(10, 20), resamples=100, patterns=("interfloor",))
            self.assertEqual(report["simulation_count"], 16)
            self.assertEqual(len(report["records"]), 12)
            self.assertTrue(all(row["demand_runs"] == 2 for row in report["summaries"]))
            self.assertTrue(all(row["metrics"]["unserved_passengers"] == 0 for row in report["trials"]))
            with gzip.open(Path(directory) / "passengers.jsonl.gz", "rt", encoding="utf-8") as stream:
                journeys = [json.loads(line) for line in stream]
            self.assertEqual(len(journeys), report["passenger_journey_rows"])
            self.assertTrue(all(row["arrival_time"] <= row["boarding_time"] <= row["alighting_time"] for row in journeys))
            self.assertTrue(all(row["wait_seconds"] == row["boarding_time"] - row["arrival_time"] for row in journeys))
            # Every exact passenger and every selected cohort is paired across policies/orders.
            pairs = {}
            for row in journeys:
                key = (row["traffic_level"], row["run"], row["passenger_id"])
                pairs.setdefault(key, []).append(row)
            for rows in pairs.values():
                self.assertEqual(len(rows), 4)
                self.assertEqual(len({(r["arrival_time"], r["origin"], r["destination"], r["eta_tail5_cohort"]) for r in rows}), 1)
            controls = [row for row in report["records"] if row["strategy"] == "redispatch_eta"]
            self.assertTrue(all(row["order_replicates"] == 2 for row in controls))


if __name__ == "__main__":
    unittest.main()
