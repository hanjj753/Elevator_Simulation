import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

from ablation_experiment import AblationSimulation, POLICIES
from elevator_sim import PassengerDemand, SimulationConfig, generate_demand
from elevator_trace import TracedSimulation
from visualize_elevator import build_replay, write_replay


class TraceTests(unittest.TestCase):
    def test_observer_preserves_metrics_and_every_passenger_for_all_policies(self):
        config = SimulationConfig(duration_seconds=150, arrival_rate_per_second=.5, capacity=2)
        demands = generate_demand(config, 42, "down_peak")
        for policy in POLICIES:
            with self.subTest(policy=policy):
                original = AblationSimulation(config, policy, demands, 6619)
                traced = TracedSimulation(config, policy, demands, 6619, capture_frames=True, capture_events=True)
                self.assertEqual(original.run(), traced.run())
                self.assertEqual({pid: asdict(p) for pid, p in original.passengers.items()},
                                 {pid: asdict(p) for pid, p in traced.passengers.items()})
                for row in traced.response_records():
                    self.assertAlmostEqual(row["before_first_service_seconds"] + row["after_first_service_seconds"], row["wait_seconds"])
                for frame in traced.frames:
                    waiting, onboard, completed, arrived = frame["counts"]
                    self.assertEqual(waiting+onboard+completed, arrived)
                    self.assertEqual(onboard, sum(car[1] for car in frame["cars"]))
                    self.assertTrue(all(1 <= car[0] <= config.floors and car[1] <= config.capacity for car in frame["cars"]))
                self.assertEqual(traced.frames[-1]["counts"], [0, 0, len(demands), len(demands)])
                boards = [pid for event in traced.events if event["type"] == "stop" for pid in event["boarded"]]
                alights = [pid for event in traced.events if event["type"] == "stop" for pid in event["alighted"]]
                self.assertEqual(sorted(boards), list(range(len(demands))))
                self.assertEqual(sorted(alights), sorted(boards))

    def test_first_service_includes_rejection_and_rewaiting(self):
        config = SimulationConfig(floors=3, elevator_count=1, initial_floors=(1,), capacity=1, duration_seconds=1)
        demands = [PassengerDemand(0, 0, 1, 3), PassengerDemand(1, 0, 1, 3)]
        simulation = TracedSimulation(config, "eta", demands, capture_events=True)
        simulation.run()
        rows = {row["passenger_id"]: row for row in simulation.response_records()}
        self.assertEqual(rows[0]["after_first_service_seconds"], 0)
        self.assertEqual(rows[1]["first_service_time"], 0)
        self.assertGreater(rows[1]["after_first_service_seconds"], 0)
        self.assertGreater(rows[1]["failed_boardings"], 0)

    def test_export_is_offline_and_escapes_embedded_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = SimulationConfig(duration_seconds=10, arrival_rate_per_second=0)
            path = build_replay(config, 42, "down_peak", ["redispatch_eta", "age_order_eta"], 6619, Path(tmp)/"replay.html")
            payload = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
            self.assertEqual(len(payload["traces"]), 2)
            self.assertEqual(payload["traces"][0]["frames"][-1]["counts"], [0, 0, 0, 0])
            payload["note"] = '</script><script>alert("test")</script>'
            write_replay(payload, path)
            html = path.read_text(encoding="utf-8")
            self.assertNotIn(payload["note"], html)
            self.assertNotIn("__REPLAY_DATA__", html)
            self.assertNotIn('<script src=', html)
            self.assertEqual(html.count('<script'), 2)


if __name__ == "__main__":
    unittest.main()
