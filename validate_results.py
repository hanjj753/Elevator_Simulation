"""Audit saved experiments and estimate paired uncertainty without rerunning them.

Uses only the standard library. The sampling unit is one simulation run, not
one passenger. Bootstrap intervals describe Monte Carlo variation conditional
on this model and are exploratory, with no multiple-comparison adjustment.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Sequence

from elevator_sim import ElevatorSimulation, PassengerDemand, SimulationConfig, percentile


RATE_FIELDS = ("long_wait_rate", "boarding_failure_rate")
METRIC_FIELDS = (
    "mean_wait_seconds", "p95_wait_seconds", "long_wait_rate",
    "boarding_failure_rate", "failed_boarding_events", "mean_ride_seconds",
    "unserved_passengers",
)
SETTING_FIELDS = (
    "seed", "duration_seconds", "floors", "elevator_count", "capacity",
    "floor_height_m", "max_speed_mps", "acceleration_mps2",
    "dominant_flow_share", "base_door_seconds",
    "boarding_seconds_per_passenger", "alighting_seconds_per_passenger",
    "arrival_rate_per_second", "passengers",
)


def read_records(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        records = list(csv.DictReader(stream))
    if not records:
        raise ValueError(f"No records in {path.name}")
    for row in records:
        if any(value in (None, "") for value in row.values()):
            raise ValueError(f"Missing value in {path.name}")
        for field in METRIC_FIELDS:
            if field in row and not math.isfinite(float(row[field])):
                raise ValueError(f"Non-finite {field} in {path.name}")
        for field in RATE_FIELDS:
            if field in row and not 0 <= float(row[field]) <= 1:
                raise ValueError(f"Rate outside [0, 1]: {field}")
    return records


def bootstrap_mean_interval(
    differences: Sequence[float], resamples: int = 10_000, seed: int = 202410189,
) -> dict[str, float | int]:
    if len(differences) < 2:
        raise ValueError("At least two paired runs are required")
    if resamples < 100:
        raise ValueError("Use at least 100 bootstrap resamples")
    rng = random.Random(seed)
    n = len(differences)
    draws = [
        statistics.fmean(rng.choices(differences, k=n))
        for _ in range(resamples)
    ]
    return {
        "runs": n,
        "mean_difference_seconds": statistics.fmean(differences),
        "ci95_low_seconds": percentile(draws, 0.025),
        "ci95_high_seconds": percentile(draws, 0.975),
        "candidate_better_runs": sum(value < 0 for value in differences),
        "reference_better_runs": sum(value > 0 for value in differences),
        "equal_runs": sum(value == 0 for value in differences),
    }


def paired_comparison(
    records: Sequence[dict[str, str]], reference: str,
    resamples: int = 10_000, seed: int = 202410189,
) -> dict[str, object]:
    candidate = "priority_eta"
    indexed: dict[str, dict[int, dict[str, str]]] = {
        candidate: {}, reference: {},
    }
    for row in records:
        strategy = row["strategy"]
        if strategy not in indexed:
            continue
        run = int(row["run"])
        if run in indexed[strategy]:
            raise ValueError(f"Duplicate run {run} for {strategy}")
        indexed[strategy][run] = row
    if indexed[candidate].keys() != indexed[reference].keys():
        raise ValueError("Candidate and reference runs do not match")
    differences: list[float] = []
    for run in sorted(indexed[candidate]):
        left, right = indexed[candidate][run], indexed[reference][run]
        if any(left[field] != right[field] for field in SETTING_FIELDS):
            raise ValueError(f"Unpaired settings or demand at run {run}")
        differences.append(float(left["p95_wait_seconds"]) - float(right["p95_wait_seconds"]))
    estimate = bootstrap_mean_interval(differences, resamples, seed)
    candidate_p95 = statistics.fmean(
        float(row["p95_wait_seconds"]) for row in indexed[candidate].values()
    )
    reference_p95 = statistics.fmean(
        float(row["p95_wait_seconds"]) for row in indexed[reference].values()
    )
    return {
        "candidate": candidate,
        "reference": reference,
        "candidate_mean_p95_seconds": candidate_p95,
        "reference_mean_p95_seconds": reference_p95,
        "p95_reduction_pct": 100 * (1 - candidate_p95 / reference_p95)
        if reference_p95 else None,
        **estimate,
        "ci_excludes_zero": estimate["ci95_low_seconds"] > 0
        or estimate["ci95_high_seconds"] < 0,
    }


def audit_summary(
    raw: Sequence[dict[str, str]], summary: Sequence[dict[str, str]],
    group_fields: Sequence[str],
) -> dict[str, int]:
    groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    seen = set()
    for row in raw:
        key = tuple(row[field] for field in group_fields)
        run_key = (*key, row["strategy"], row["run"])
        if run_key in seen:
            raise ValueError(f"Duplicate observation: {run_key}")
        seen.add(run_key)
        groups[(*key, row["strategy"])].append(row)
    summary_seen = set()
    for row in summary:
        key = (*[row[field] for field in group_fields], row["strategy"])
        if key in summary_seen:
            raise ValueError(f"Duplicate summary: {key}")
        summary_seen.add(key)
        items = groups[key]
        if len(items) != int(row["runs"]):
            raise ValueError(f"Run count differs: {key}")
        for field in ("mean_wait_seconds", "p95_wait_seconds", *RATE_FIELDS):
            summary_field = "mean_" + field if field != "mean_wait_seconds" else field
            expected = statistics.fmean(float(item[field]) for item in items)
            if not math.isclose(float(row[summary_field]), expected, abs_tol=1e-10):
                raise ValueError(f"Summary differs: {key}, {field}")
    if summary_seen != groups.keys():
        raise ValueError("Some raw groups are missing from the summary")
    rank_groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in summary:
        rank_groups[tuple(row[field] for field in group_fields)].append(row)
    for key, items in rank_groups.items():
        ordered = sorted(items, key=lambda row: (
            float(row["mean_p95_wait_seconds"]),
            float(row["mean_boarding_failure_rate"]),
            float(row["mean_long_wait_rate"]), float(row["mean_wait_seconds"]),
        ))
        if [int(row["rank_by_mean_p95"]) for row in ordered] != list(range(1, len(items)+1)):
            raise ValueError(f"Rank order differs: {key}")
    return {
        "raw_rows": len(raw), "summary_rows": len(summary),
        "comparison_conditions": len(rank_groups),
        "unserved_passengers": sum(int(row["unserved_passengers"]) for row in raw),
    }


def model_boundary_probe() -> dict[str, object]:
    """Record a diagnostic example, not evidence of real-building accuracy."""
    config = SimulationConfig(
        elevator_count=1, capacity=1, initial_floors=(1,),
        duration_seconds=1, arrival_rate_per_second=0,
    )
    simulation = ElevatorSimulation(config, "eta", [
        PassengerDemand(0, 0, 1, 3), PassengerDemand(1, 0, 2, 3),
    ])
    metrics = simulation.run()
    passenger = simulation.passengers[1]
    return {
        "example": "Full car bypasses floor 2, later returns to collect its passenger",
        "passenger_at_floor_2_wait_seconds": passenger.boarding_time,
        "recorded_failed_boardings": passenger.failed_boardings,
        "all_passengers_completed": all(
            item.alighting_time is not None for item in simulation.passengers.values()
        ),
        "measured_failure_definition": "Rejected at a serviced stop, not capacity-related bypass",
        "metrics": metrics.__dict__,
    }


def build_report(output_dir: Path, resamples: int = 10_000) -> dict[str, object]:
    filenames = (
        "elevator_experiment_runs.csv", "elevator_experiment_summary.csv",
        "elevator_sensitivity_runs.csv", "elevator_sensitivity_summary.csv",
    )
    base, base_summary, sensitivity, sensitivity_summary = [
        read_records(output_dir / name) for name in filenames
    ]
    base_audit = audit_summary(base, base_summary, ("traffic_level", "traffic_pattern"))
    sensitivity_audit = audit_summary(sensitivity, sensitivity_summary, (
        "analysis_dimension", "scenario", "traffic_level", "traffic_pattern",
    ))
    comparisons = []
    sources = [
        ("baseline", base, ("traffic_level", "traffic_pattern")),
        ("demand_sensitivity", [row for row in sensitivity if row["analysis_dimension"] == "demand_rate"],
         ("scenario", "traffic_pattern")),
    ]
    for analysis, rows, fields in sources:
        groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            groups[tuple(row[field] for field in fields)].append(row)
        for group, items in groups.items():
            for reference in ("eta", "closest"):
                label = "|".join((analysis, *group, reference))
                seed = int.from_bytes(hashlib.sha256(label.encode()).digest()[:8], "big")
                comparisons.append({
                    "analysis": analysis, **dict(zip(fields, group)),
                    **paired_comparison(items, reference, resamples, seed),
                })
    winners = Counter(
        row["strategy"] for row in sensitivity_summary
        if row["analysis_dimension"] == "demand_rate" and row["rank_by_mean_p95"] == "1"
    )
    # The physical baseline repeats priority_eta runs already in the demand sweep.
    # Baseline experiments also overlap the 0.12/0.30 demand sweep.
    identity_fields = (
        "strategy", "run", *SETTING_FIELDS, *METRIC_FIELDS,
    )
    def identity(row: dict[str, str]) -> tuple[str, ...]:
        return tuple(row[field] for field in identity_fields)
    unique_base = {identity(row) for row in base}
    unique_sensitivity = {identity(row) for row in sensitivity}
    return {
        "method": {
            "primary_metric": "Mean of per-run passenger waiting-time p95; not pooled p95",
            "difference_sign": "priority_eta minus reference, seconds; negative favors priority_eta",
            "ci": "Paired percentile bootstrap of mean run-level differences, 95%",
            "bootstrap_resamples": resamples,
            "resampling_unit": "One paired simulation run",
            "scope": "Exploratory Monte Carlo uncertainty, unadjusted for multiple comparisons",
            "empirical_building_validation": False,
        },
        "inputs": {name: hashlib.sha256((output_dir / name).read_bytes()).hexdigest() for name in filenames},
        "baseline_audit": base_audit,
        "sensitivity_audit": sensitivity_audit,
        "overlap": {
            "baseline_rows_reproduced_in_sensitivity": len(unique_base & unique_sensitivity),
            "duplicate_simulation_rows_within_sensitivity": len(sensitivity) - len(unique_sensitivity),
            "unique_parameter_seed_strategy_records_across_files": len(unique_base | unique_sensitivity),
            "note": "File row totals must not be added as independent replications",
        },
        "demand_p95_winner_counts": dict(winners),
        "paired_p95_comparisons": comparisons,
        "model_boundary_probe": model_boundary_probe(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--resamples", type=int, default=10_000)
    args = parser.parse_args()
    report = build_report(args.output_dir, args.resamples)
    path = args.output_dir / "research_validation.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "baseline_audit", "sensitivity_audit", "overlap", "demand_p95_winner_counts",
    )}, ensure_ascii=False, indent=2))
    for item in report["paired_p95_comparisons"]:
        if item["analysis"] == "baseline" and item["reference"] == "eta":
            print(item["traffic_level"], item["traffic_pattern"],
                  round(item["mean_difference_seconds"], 3),
                  [round(item["ci95_low_seconds"], 3), round(item["ci95_high_seconds"], 3)])
    print(f"Saved {path.resolve()}")


if __name__ == "__main__":
    main()
