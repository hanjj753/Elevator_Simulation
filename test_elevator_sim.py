import csv
import tempfile
import unittest

from elevator_sim import (
    ElevatorSimulation,
    PassengerDemand,
    SimulationConfig,
    build_look_route,
    generate_demand,
    generate_uniform_demand,
    percentile,
    run_experiment,
    write_csv_results,
)


class ElevatorSimulationTests(unittest.TestCase):
    def test_percentile_interpolates(self) -> None:
        self.assertEqual(percentile([0.0, 10.0], 0.95), 9.5)

    def test_look_route_finishes_current_direction_first(self) -> None:
        self.assertEqual(build_look_route(10, 1, {3, 12, 20}), [12, 20, 3])
        self.assertEqual(build_look_route(10, -1, {3, 12, 20}), [3, 12, 20])

    def test_uniform_demand_is_reproducible_and_valid(self) -> None:
        config = SimulationConfig(duration_seconds=60, arrival_rate_per_second=0.2)
        first = generate_uniform_demand(config, seed=7)
        second = generate_uniform_demand(config, seed=7)
        self.assertEqual(first, second)
        self.assertTrue(first)
        for demand in first:
            self.assertNotEqual(demand.origin, demand.destination)
            self.assertGreaterEqual(demand.origin, 1)
            self.assertLessEqual(demand.destination, config.floors)

    def test_research_traffic_patterns_have_expected_flows(self) -> None:
        config = SimulationConfig(
            duration_seconds=60,
            arrival_rate_per_second=0.5,
            dominant_flow_share=1.0,
        )
        up_peak = generate_demand(config, seed=17, traffic_pattern="up_peak")
        down_peak = generate_demand(config, seed=17, traffic_pattern="down_peak")
        interfloor = generate_demand(config, seed=17, traffic_pattern="interfloor")

        self.assertTrue(all(demand.origin == 1 for demand in up_peak))
        self.assertTrue(all(demand.destination == 1 for demand in down_peak))
        self.assertTrue(
            all(demand.origin >= 2 and demand.destination >= 2 for demand in interfloor)
        )

    def test_arrival_times_are_paired_across_traffic_patterns(self) -> None:
        config = SimulationConfig(duration_seconds=60, arrival_rate_per_second=0.5)
        patterns = ("up_peak", "interfloor", "down_peak")
        streams = [generate_demand(config, seed=23, traffic_pattern=p) for p in patterns]
        reference_times = [demand.arrival_time for demand in streams[0]]
        for stream in streams[1:]:
            self.assertEqual(
                [demand.arrival_time for demand in stream], reference_times
            )

    def test_single_passenger_is_served(self) -> None:
        config = SimulationConfig(duration_seconds=10, arrival_rate_per_second=0.0)
        demand = [PassengerDemand(0, 0.0, 1, 3)]
        metrics = ElevatorSimulation(config, "eta", demand).run()
        self.assertEqual(metrics.passengers, 1)
        self.assertEqual(metrics.unserved_passengers, 0)
        self.assertEqual(metrics.mean_wait_seconds, 0.0)
        self.assertEqual(metrics.boarding_failure_rate, 0.0)

    def test_dwell_time_scales_with_passenger_exchange(self) -> None:
        config = SimulationConfig(
            duration_seconds=10,
            arrival_rate_per_second=0.0,
            door_seconds=4,
            boarding_seconds_per_passenger=0.8,
            alighting_seconds_per_passenger=0.6,
        )
        self.assertEqual(config.dwell_seconds(0, 0), 4)
        self.assertEqual(config.dwell_seconds(2, 1), 7)
        with self.assertRaises(ValueError):
            config.dwell_seconds(-1, 0)

        demands = [
            PassengerDemand(0, 0.0, 1, 3),
            PassengerDemand(1, 0.0, 1, 4),
        ]
        simulation = ElevatorSimulation(config, "eta", demands)
        simulation._add_arrivals(0)
        elevator = simulation.elevators[0]
        simulation._step_elevator(elevator, 0)
        self.assertEqual(len(elevator.onboard_passenger_ids), 2)
        self.assertEqual(elevator.door_remaining, 6)

    def test_travel_time_uses_acceleration_and_distance(self) -> None:
        config = SimulationConfig()
        self.assertEqual(config.travel_time_seconds(0), 0)
        self.assertEqual(config.travel_time_seconds(1), 4)
        self.assertEqual(config.travel_time_seconds(2), 6)
        self.assertEqual(config.travel_time_seconds(10), 17)
        self.assertLess(
            config.travel_time_seconds(10),
            10 * config.travel_time_seconds(1),
        )
        with self.assertRaises(ValueError):
            config.travel_time_seconds(-1)

    def test_all_strategies_drain_a_small_shared_demand(self) -> None:
        config = SimulationConfig(duration_seconds=120, arrival_rate_per_second=0.15)
        for traffic_pattern in ("up_peak", "interfloor", "down_peak"):
            demands = generate_demand(config, seed=11, traffic_pattern=traffic_pattern)
            for strategy in ElevatorSimulation.VALID_STRATEGIES:
                with self.subTest(
                    traffic_pattern=traffic_pattern, strategy=strategy
                ):
                    metrics = ElevatorSimulation(config, strategy, demands).run()
                    self.assertEqual(metrics.unserved_passengers, 0)
                    self.assertEqual(metrics.passengers, len(demands))

    def test_congested_closest_case_has_enough_drain_time(self) -> None:
        config = SimulationConfig(arrival_rate_per_second=0.3)
        demands = generate_demand(config, seed=202410198, traffic_pattern="up_peak")
        metrics = ElevatorSimulation(config, "closest", demands).run()
        self.assertEqual(metrics.passengers, len(demands))
        self.assertEqual(metrics.unserved_passengers, 0)

    def test_demand_levels_and_csv_outputs_are_complete(self) -> None:
        config = SimulationConfig(duration_seconds=60)
        traffic_levels = {"normal": 0.1, "congested": 0.2}
        rows = run_experiment(
            config=config,
            strategies=("eta",),
            runs=2,
            seed=31,
            traffic_patterns=("up_peak", "down_peak"),
            traffic_levels=traffic_levels,
        )
        self.assertEqual(len(rows), 8)

        normal_counts = [
            row.metrics.passengers
            for row in rows
            if row.traffic_level == "normal"
        ]
        congested_counts = [
            row.metrics.passengers
            for row in rows
            if row.traffic_level == "congested"
        ]
        self.assertGreater(sum(congested_counts), sum(normal_counts))

        with tempfile.TemporaryDirectory() as directory:
            runs_path, summary_path = write_csv_results(rows, directory)
            with runs_path.open(encoding="utf-8-sig", newline="") as stream:
                raw_records = list(csv.DictReader(stream))
            with summary_path.open(encoding="utf-8-sig", newline="") as stream:
                summary_records = list(csv.DictReader(stream))

            self.assertEqual(len(raw_records), 8)
            self.assertEqual(len(summary_records), 4)
            self.assertEqual(raw_records[0]["floor_height_m"], "3.5")
            self.assertEqual(raw_records[0]["max_speed_mps"], "2.5")
            self.assertEqual(raw_records[0]["acceleration_mps2"], "1.0")
            self.assertEqual(raw_records[0]["base_door_seconds"], "4")
            self.assertEqual(
                raw_records[0]["boarding_seconds_per_passenger"], "0.8"
            )
            self.assertEqual(
                summary_records[0]["alighting_seconds_per_passenger"], "0.6"
            )
            self.assertEqual(summary_records[0]["floor_height_m"], "3.5")
            self.assertTrue(all(record["runs"] == "2" for record in summary_records))
            self.assertTrue(
                all(record["rank_by_mean_p95"] == "1" for record in summary_records)
            )


if __name__ == "__main__":
    unittest.main()
