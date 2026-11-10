"""One-at-a-time sensitivity analysis for the elevator simulation.

The analysis has two parts that map directly to the final research poster:

1. Compare all dispatch strategies over a five-level demand sweep.
2. Vary one physical assumption at a time for the selected priority-ETA rule.

Every scenario reuses the same base seed so comparable conditions use common
random numbers. Run-level and summary CSV files preserve all assumptions.
"""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Sequence

from elevator_sim import (
    ElevatorSimulation,
    ExperimentRow,
    SimulationConfig,
    TRAFFIC_PATTERNS,
    run_experiment,
)


DEFAULT_PATTERNS = ("up_peak", "interfloor", "down_peak")
ALL_STRATEGIES = ("closest", "eta", "priority_eta")
SELECTED_STRATEGY = "priority_eta"

DEMAND_SWEEP = (
    ("demand_0_08", 0.08),
    ("demand_0_12", 0.12),
    ("demand_0_20", 0.20),
    ("demand_0_30", 0.30),
    ("demand_0_36", 0.36),
)
BASE_TRAFFIC_LEVELS = (
    ("normal", 0.12),
    ("congested", 0.30),
)
PHYSICAL_VARIANTS = (
    ("baseline", "none", {}),
    ("floor_height_low", "floor_height_m", {"floor_height_m": 3.0}),
    ("floor_height_high", "floor_height_m", {"floor_height_m": 4.0}),
    ("max_speed_low", "max_speed_mps", {"max_speed_mps": 2.0}),
    ("max_speed_high", "max_speed_mps", {"max_speed_mps": 3.0}),
    (
        "acceleration_low",
        "acceleration_mps2",
        {"acceleration_mps2": 0.8},
    ),
    (
        "acceleration_high",
        "acceleration_mps2",
        {"acceleration_mps2": 1.2},
    ),
)


@dataclass(frozen=True)
class SensitivityCase:
    analysis_dimension: str
    scenario: str
    changed_parameter: str
    traffic_level: str
    arrival_rate_per_second: float
    config: SimulationConfig
    strategies: tuple[str, ...]


@dataclass(frozen=True)
class SensitivityObservation:
    analysis_dimension: str
    scenario: str
    changed_parameter: str
    result: ExperimentRow


def build_sensitivity_cases(
    base_config: SimulationConfig,
) -> list[SensitivityCase]:
    """Build paired demand and one-at-a-time physical scenarios."""

    cases: list[SensitivityCase] = []
    for scenario, arrival_rate in DEMAND_SWEEP:
        cases.append(
            SensitivityCase(
                analysis_dimension="demand_rate",
                scenario=scenario,
                changed_parameter="arrival_rate_per_second",
                traffic_level=scenario,
                arrival_rate_per_second=arrival_rate,
                config=replace(
                    base_config, arrival_rate_per_second=arrival_rate
                ),
                strategies=ALL_STRATEGIES,
            )
        )

    for scenario, changed_parameter, changes in PHYSICAL_VARIANTS:
        for traffic_level, arrival_rate in BASE_TRAFFIC_LEVELS:
            cases.append(
                SensitivityCase(
                    analysis_dimension="physical_assumption",
                    scenario=scenario,
                    changed_parameter=changed_parameter,
                    traffic_level=traffic_level,
                    arrival_rate_per_second=arrival_rate,
                    config=replace(
                        base_config,
                        arrival_rate_per_second=arrival_rate,
                        **changes,
                    ),
                    strategies=(SELECTED_STRATEGY,),
                )
            )
    return cases


def run_sensitivity_analysis(
    base_config: SimulationConfig,
    runs: int,
    seed: int,
    traffic_patterns: Sequence[str] = DEFAULT_PATTERNS,
) -> list[SensitivityObservation]:
    """Run every case with common random seeds across comparisons."""

    observations: list[SensitivityObservation] = []
    for case in build_sensitivity_cases(base_config):
        rows = run_experiment(
            config=case.config,
            strategies=case.strategies,
            runs=runs,
            seed=seed,
            traffic_patterns=traffic_patterns,
            traffic_levels={
                case.traffic_level: case.arrival_rate_per_second
            },
        )
        observations.extend(
            SensitivityObservation(
                analysis_dimension=case.analysis_dimension,
                scenario=case.scenario,
                changed_parameter=case.changed_parameter,
                result=row,
            )
            for row in rows
        )
    return observations


def _sample_stdev(values: Iterable[float]) -> float:
    observations = list(values)
    return statistics.stdev(observations) if len(observations) > 1 else 0.0


def summarize_sensitivity(
    observations: Sequence[SensitivityObservation],
) -> list[dict[str, object]]:
    """Aggregate runs, rank strategies, and compare with a reference case."""

    grouped: dict[tuple[str, ...], list[SensitivityObservation]] = defaultdict(list)
    for observation in observations:
        row = observation.result
        key = (
            observation.analysis_dimension,
            observation.scenario,
            observation.changed_parameter,
            row.traffic_level,
            row.traffic_pattern,
            row.strategy,
        )
        grouped[key].append(observation)

    records: list[dict[str, object]] = []
    for key, items in grouped.items():
        (
            analysis_dimension,
            scenario,
            changed_parameter,
            traffic_level,
            traffic_pattern,
            strategy,
        ) = key
        rows = [item.result for item in items]
        metrics = [row.metrics for row in rows]
        settings = rows[0]
        records.append(
            {
                "analysis_dimension": analysis_dimension,
                "scenario": scenario,
                "changed_parameter": changed_parameter,
                "traffic_level": traffic_level,
                "arrival_rate_per_second": settings.arrival_rate_per_second,
                "traffic_pattern": traffic_pattern,
                "strategy": strategy,
                "runs": len(items),
                "floor_height_m": settings.floor_height_m,
                "max_speed_mps": settings.max_speed_mps,
                "acceleration_mps2": settings.acceleration_mps2,
                "base_door_seconds": settings.base_door_seconds,
                "boarding_seconds_per_passenger": (
                    settings.boarding_seconds_per_passenger
                ),
                "alighting_seconds_per_passenger": (
                    settings.alighting_seconds_per_passenger
                ),
                "mean_passengers": statistics.fmean(
                    metric.passengers for metric in metrics
                ),
                "mean_wait_seconds": statistics.fmean(
                    metric.mean_wait_seconds for metric in metrics
                ),
                "sd_mean_wait_seconds": _sample_stdev(
                    metric.mean_wait_seconds for metric in metrics
                ),
                "mean_p95_wait_seconds": statistics.fmean(
                    metric.p95_wait_seconds for metric in metrics
                ),
                "sd_p95_wait_seconds": _sample_stdev(
                    metric.p95_wait_seconds for metric in metrics
                ),
                "mean_long_wait_rate": statistics.fmean(
                    metric.long_wait_rate for metric in metrics
                ),
                "mean_boarding_failure_rate": statistics.fmean(
                    metric.boarding_failure_rate for metric in metrics
                ),
                "mean_failed_boarding_events": statistics.fmean(
                    metric.failed_boarding_events for metric in metrics
                ),
                "mean_ride_seconds": statistics.fmean(
                    metric.mean_ride_seconds for metric in metrics
                ),
                "mean_unserved_passengers": statistics.fmean(
                    metric.unserved_passengers for metric in metrics
                ),
            }
        )

    rank_groups: dict[tuple[str, ...], list[dict[str, object]]] = defaultdict(list)
    for record in records:
        group_key = (
            str(record["analysis_dimension"]),
            str(record["scenario"]),
            str(record["traffic_level"]),
            str(record["traffic_pattern"]),
        )
        rank_groups[group_key].append(record)
    for group in rank_groups.values():
        group.sort(
            key=lambda record: (
                float(record["mean_p95_wait_seconds"]),
                float(record["mean_boarding_failure_rate"]),
                float(record["mean_long_wait_rate"]),
                float(record["mean_wait_seconds"]),
            )
        )
        for rank, record in enumerate(group, start=1):
            record["rank_by_mean_p95"] = rank

    index = {
        (
            str(record["analysis_dimension"]),
            str(record["scenario"]),
            str(record["traffic_level"]),
            str(record["traffic_pattern"]),
            str(record["strategy"]),
        ): record
        for record in records
    }
    for record in records:
        dimension = str(record["analysis_dimension"])
        if dimension == "demand_rate":
            reference_scenario = "demand_0_12"
            reference_traffic_level = reference_scenario
        else:
            reference_scenario = "baseline"
            reference_traffic_level = str(record["traffic_level"])
        reference = index[
            (
                dimension,
                reference_scenario,
                reference_traffic_level,
                str(record["traffic_pattern"]),
                str(record["strategy"]),
            )
        ]
        reference_p95 = float(reference["mean_p95_wait_seconds"])
        current_p95 = float(record["mean_p95_wait_seconds"])
        record["reference_scenario"] = reference_scenario
        record["reference_rank"] = int(reference["rank_by_mean_p95"])
        record["rank_change_vs_reference"] = (
            int(record["rank_by_mean_p95"])
            - int(reference["rank_by_mean_p95"])
        )
        record["reference_mean_p95_wait_seconds"] = reference_p95
        record["p95_change_vs_reference_pct"] = (
            100.0 * (current_p95 / reference_p95 - 1.0)
            if reference_p95 > 0
            else 0.0
        )
        reference_failure = float(reference["mean_boarding_failure_rate"])
        record["reference_mean_boarding_failure_rate"] = reference_failure
        record["boarding_failure_change_percentage_points"] = 100.0 * (
            float(record["mean_boarding_failure_rate"])
            - reference_failure
        )

    demand_order = {
        scenario: index for index, (scenario, _) in enumerate(DEMAND_SWEEP)
    }
    physical_order = {
        scenario: index
        for index, (scenario, _, _) in enumerate(PHYSICAL_VARIANTS)
    }
    pattern_order = {
        pattern: index for index, pattern in enumerate(DEFAULT_PATTERNS)
    }
    level_order = {"normal": 0, "congested": 1}

    def sort_key(record: dict[str, object]) -> tuple[object, ...]:
        dimension = str(record["analysis_dimension"])
        scenario = str(record["scenario"])
        if dimension == "demand_rate":
            return (
                0,
                demand_order[scenario],
                pattern_order.get(str(record["traffic_pattern"]), 99),
                int(record["rank_by_mean_p95"]),
            )
        return (
            1,
            physical_order[scenario],
            level_order[str(record["traffic_level"])],
            pattern_order.get(str(record["traffic_pattern"]), 99),
            int(record["rank_by_mean_p95"]),
        )

    records.sort(key=sort_key)
    return records


RAW_FIELDS = [
    "analysis_dimension",
    "scenario",
    "changed_parameter",
    "traffic_level",
    "arrival_rate_per_second",
    "traffic_pattern",
    "strategy",
    "run",
    "seed",
    "duration_seconds",
    "floors",
    "elevator_count",
    "capacity",
    "floor_height_m",
    "max_speed_mps",
    "acceleration_mps2",
    "dominant_flow_share",
    "base_door_seconds",
    "boarding_seconds_per_passenger",
    "alighting_seconds_per_passenger",
    "passengers",
    "mean_wait_seconds",
    "p95_wait_seconds",
    "long_wait_rate",
    "boarding_failure_rate",
    "failed_boarding_events",
    "mean_ride_seconds",
    "unserved_passengers",
]

SUMMARY_FIELDS = [
    "analysis_dimension",
    "scenario",
    "changed_parameter",
    "traffic_level",
    "arrival_rate_per_second",
    "traffic_pattern",
    "rank_by_mean_p95",
    "strategy",
    "runs",
    "reference_scenario",
    "reference_rank",
    "rank_change_vs_reference",
    "floor_height_m",
    "max_speed_mps",
    "acceleration_mps2",
    "base_door_seconds",
    "boarding_seconds_per_passenger",
    "alighting_seconds_per_passenger",
    "mean_passengers",
    "mean_wait_seconds",
    "sd_mean_wait_seconds",
    "mean_p95_wait_seconds",
    "sd_p95_wait_seconds",
    "reference_mean_p95_wait_seconds",
    "p95_change_vs_reference_pct",
    "mean_long_wait_rate",
    "mean_boarding_failure_rate",
    "reference_mean_boarding_failure_rate",
    "boarding_failure_change_percentage_points",
    "mean_failed_boarding_events",
    "mean_ride_seconds",
    "mean_unserved_passengers",
]


def write_sensitivity_csv(
    observations: Sequence[SensitivityObservation],
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """Write machine-readable observations and poster-ready summaries."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    raw_path = destination / "elevator_sensitivity_runs.csv"
    summary_path = destination / "elevator_sensitivity_summary.csv"

    with raw_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=RAW_FIELDS)
        writer.writeheader()
        for observation in observations:
            row = observation.result
            metrics = row.metrics
            writer.writerow(
                {
                    "analysis_dimension": observation.analysis_dimension,
                    "scenario": observation.scenario,
                    "changed_parameter": observation.changed_parameter,
                    "traffic_level": row.traffic_level,
                    "arrival_rate_per_second": row.arrival_rate_per_second,
                    "traffic_pattern": row.traffic_pattern,
                    "strategy": row.strategy,
                    "run": row.run,
                    "seed": row.seed,
                    "duration_seconds": row.duration_seconds,
                    "floors": row.floors,
                    "elevator_count": row.elevator_count,
                    "capacity": row.capacity,
                    "floor_height_m": row.floor_height_m,
                    "max_speed_mps": row.max_speed_mps,
                    "acceleration_mps2": row.acceleration_mps2,
                    "dominant_flow_share": row.dominant_flow_share,
                    "base_door_seconds": row.base_door_seconds,
                    "boarding_seconds_per_passenger": (
                        row.boarding_seconds_per_passenger
                    ),
                    "alighting_seconds_per_passenger": (
                        row.alighting_seconds_per_passenger
                    ),
                    "passengers": metrics.passengers,
                    "mean_wait_seconds": metrics.mean_wait_seconds,
                    "p95_wait_seconds": metrics.p95_wait_seconds,
                    "long_wait_rate": metrics.long_wait_rate,
                    "boarding_failure_rate": metrics.boarding_failure_rate,
                    "failed_boarding_events": metrics.failed_boarding_events,
                    "mean_ride_seconds": metrics.mean_ride_seconds,
                    "unserved_passengers": metrics.unserved_passengers,
                }
            )

    summary = summarize_sensitivity(observations)
    with summary_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summary)
    return raw_path.resolve(), summary_path.resolve()


def print_headline_results(
    summary: Sequence[dict[str, object]],
) -> None:
    demand_winners = [
        record
        for record in summary
        if record["analysis_dimension"] == "demand_rate"
        and record["rank_by_mean_p95"] == 1
    ]
    winner_counts = Counter(
        str(record["strategy"]) for record in demand_winners
    )
    print("\nDemand sweep: p95 winner counts")
    for strategy in ALL_STRATEGIES:
        print(f"  {strategy}: {winner_counts[strategy]} of {len(demand_winners)}")

    physical_changes = [
        record
        for record in summary
        if record["analysis_dimension"] == "physical_assumption"
        and record["scenario"] != "baseline"
    ]
    physical_changes.sort(
        key=lambda record: abs(float(record["p95_change_vs_reference_pct"])),
        reverse=True,
    )
    print("\nLargest physical-assumption effects on priority_eta p95")
    for record in physical_changes[:6]:
        print(
            f"  {record['scenario']}, {record['traffic_level']}, "
            f"{record['traffic_pattern']}: "
            f"{float(record['p95_change_vs_reference_pct']):+.1f}%"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=int, default=1_800)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=202410189)
    parser.add_argument("--output-dir", default="output")
    parser.add_argument(
        "--traffic-patterns",
        nargs="+",
        choices=TRAFFIC_PATTERNS,
        default=list(DEFAULT_PATTERNS),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_config = SimulationConfig(duration_seconds=args.duration)
    observations = run_sensitivity_analysis(
        base_config=base_config,
        runs=args.runs,
        seed=args.seed,
        traffic_patterns=args.traffic_patterns,
    )
    summary = summarize_sensitivity(observations)
    print_headline_results(summary)
    raw_path, summary_path = write_sensitivity_csv(
        observations, args.output_dir
    )
    print(f"\nSaved sensitivity runs: {raw_path}")
    print(f"Saved sensitivity summary: {summary_path}")


if __name__ == "__main__":
    main()
