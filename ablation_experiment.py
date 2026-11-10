"""Separate ETA reassignment from call-age ordering in a controlled experiment.

Run with Python 3.10+ and no external packages. Existing experiments are not
overwritten. All policies use the original simulator's movement and service
model, so this is a mechanism comparison conditional on those simplifications.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import random
import statistics
from collections import defaultdict
from dataclasses import asdict, replace
from pathlib import Path
from typing import Sequence, TextIO

from elevator_sim import (
    ElevatorSimulation, HallCall, PassengerDemand, SimulationConfig,
    generate_demand, percentile,
)
from validate_results import bootstrap_mean_interval


POLICIES = ("eta", "redispatch_eta", "age_order_eta")
LEVELS = {"normal": 0.12, "congested": 0.30}
PATTERNS = ("up_peak", "interfloor", "down_peak")
DEFAULT_ORDER_SEEDS = (1103, 2207, 3301, 4409, 5501)
EXTRA_METRICS = (
    "tail5_mean_wait_seconds", "repeated_rejection_rate",
    "eta_tail5_mean_wait_seconds", "eta_rejected_mean_wait_seconds",
    "redispatch_rounds", "assignment_changes",
)


class AblationSimulation(ElevatorSimulation):
    VALID_STRATEGIES = set(POLICIES)

    def __init__(
        self, config: SimulationConfig, strategy: str,
        demands: Sequence[PassengerDemand], order_seed: int = 1103,
        first_redispatch_second: int = 45, redispatch_interval: int = 5,
    ) -> None:
        if first_redispatch_second < 0 or redispatch_interval < 1:
            raise ValueError("Redispatch start must be nonnegative and interval positive")
        super().__init__(config, strategy, demands)
        self.order_seed = order_seed
        self.first_redispatch_second = first_redispatch_second
        self.redispatch_interval = redispatch_interval
        self.redispatch_rounds = 0
        self.assignment_changes = 0

    def _ordered_calls(self, now: int) -> list[HallCall]:
        if self.strategy == "age_order_eta":
            # Ignore the ages of passengers who have already boarded.
            return sorted(self.calls.values(), key=lambda call: (
                min(self.passengers[pid].arrival_time for pid in call.waiting_passenger_ids),
                call.floor, call.direction,
            ))
        # A permutation of the whole floor/direction universe gives each call
        # an age-independent order. Policy RNG never consumes demand RNG.
        keys = [(floor, direction) for floor in range(1, self.config.floors + 1)
                for direction in (-1, 1)]
        random.Random(f"{self.order_seed}:{now}").shuffle(keys)
        order = {key: index for index, key in enumerate(keys)}
        return sorted(self.calls.values(), key=lambda call: order[call.key])

    def _maybe_redispatch_priority_calls(self, now: int) -> None:
        if self.strategy == "eta" or not self.calls:
            return
        if now < self.first_redispatch_second:
            return
        if (now - self.first_redispatch_second) % self.redispatch_interval:
            return
        old_assignments = {key: call.assigned_elevator_id for key, call in self.calls.items()}
        ordered = self._ordered_calls(now)
        for elevator in self.elevators:
            elevator.assigned_calls.clear()
        for call in ordered:
            call.assigned_elevator_id = None
            self._assign_call(call)
            self.assignment_changes += call.assigned_elevator_id != old_assignments[call.key]
        self.redispatch_rounds += 1


def average(values: Sequence[float]) -> float | None:
    return statistics.fmean(values) if values else None


def passenger_measurements(
    simulation: AblationSimulation, eta_tail_ids: set[int], eta_rejected_ids: set[int],
) -> tuple[dict[str, float | int | None], list[dict[str, object]], dict[int, float]]:
    passengers = list(simulation.passengers.values())
    if any(p.boarding_time is None or p.alighting_time is None for p in passengers):
        raise RuntimeError("Incomplete passenger journey")
    waits = {p.passenger_id: p.boarding_time - p.arrival_time for p in passengers}
    ordered_waits = sorted(waits.values(), reverse=True)
    top_count = max(1, math.ceil(0.05 * len(passengers)))
    extras = {
        "tail5_mean_wait_seconds": average(ordered_waits[:top_count]),
        "repeated_rejection_rate": sum(p.failed_boardings >= 2 for p in passengers) / len(passengers)
        if passengers else None,
        "eta_tail5_mean_wait_seconds": average([waits[pid] for pid in eta_tail_ids]),
        "eta_rejected_mean_wait_seconds": average([waits[pid] for pid in eta_rejected_ids]),
        "redispatch_rounds": simulation.redispatch_rounds,
        "assignment_changes": simulation.assignment_changes,
    }
    by_floor: dict[int, list] = defaultdict(list)
    for passenger in passengers:
        by_floor[passenger.origin].append(passenger)
    floors = []
    for floor, items in sorted(by_floor.items()):
        floor_waits = [waits[p.passenger_id] for p in items]
        floors.append({
            "origin_floor": floor, "passengers": len(items),
            "mean_wait_seconds": statistics.fmean(floor_waits),
            "p95_wait_seconds": percentile(floor_waits, 0.95),
            "long_wait_rate": sum(w > simulation.config.long_wait_seconds for w in floor_waits) / len(items),
            "boarding_failure_rate": sum(p.failed_boardings > 0 for p in items) / len(items),
        })
    return extras, floors, waits


def write_passengers(
    stream: TextIO, simulation: AblationSimulation, context: dict[str, object],
    eta_tail_ids: set[int], eta_rejected_ids: set[int],
) -> int:
    for passenger in simulation.passengers.values():
        stream.write(json.dumps({
            **context, **asdict(passenger),
            "wait_seconds": passenger.boarding_time - passenger.arrival_time,
            "eta_tail5_cohort": passenger.passenger_id in eta_tail_ids,
            "eta_rejected_cohort": passenger.passenger_id in eta_rejected_ids,
        }, separators=(",", ":"), allow_nan=False) + "\n")
    return len(simulation.passengers)


def collapse_controls(trials: Sequence[dict[str, object]]) -> dict[str, object]:
    """Average order replicates within a demand seed, never as extra subjects."""
    first = trials[0]
    result = {key: first[key] for key in ("traffic_level", "traffic_pattern", "run", "demand_seed", "passengers")}
    result["strategy"] = "redispatch_eta"
    result["order_replicates"] = len(trials)
    result["metrics"] = {
        key: average([row["metrics"][key] for row in trials if row["metrics"][key] is not None])
        for key in first["metrics"]
    }
    result["order_p95_sd_seconds"] = statistics.stdev(
        row["metrics"]["p95_wait_seconds"] for row in trials
    ) if len(trials) > 1 else 0.0
    floors: dict[int, list] = defaultdict(list)
    for row in trials:
        for floor in row["floors"]:
            floors[floor["origin_floor"]].append(floor)
    result["floors"] = [{
        "origin_floor": floor, "passengers": items[0]["passengers"],
        **{key: statistics.fmean(item[key] for item in items)
           for key in ("mean_wait_seconds", "p95_wait_seconds", "long_wait_rate", "boarding_failure_rate")},
    } for floor, items in sorted(floors.items())]
    return result


def summarize(records: Sequence[dict[str, object]], resamples: int) -> dict[str, object]:
    groups: dict[tuple[str, str], list] = defaultdict(list)
    for record in records:
        groups[(record["traffic_level"], record["traffic_pattern"])].append(record)
    summaries, contrasts, floor_summaries = [], [], []
    comparisons = (
        ("redispatch_effect", "redispatch_eta", "eta"),
        ("age_order_added_effect", "age_order_eta", "redispatch_eta"),
        ("combined_effect", "age_order_eta", "eta"),
    )
    for (level, pattern), items in groups.items():
        by_policy = {policy: {row["run"]: row for row in items if row["strategy"] == policy}
                     for policy in POLICIES}
        run_ids = sorted(by_policy["eta"])
        if any(sorted(group) != run_ids for group in by_policy.values()):
            raise ValueError("Demand runs not paired across policies")
        for policy, runs in by_policy.items():
            means = {}
            counts = {}
            for key in next(iter(runs.values()))["metrics"]:
                values = [row["metrics"][key] for row in runs.values() if row["metrics"][key] is not None]
                means[key] = average(values)
                counts[key] = len(values)
            summaries.append({"traffic_level": level, "traffic_pattern": pattern,
                              "strategy": policy, "demand_runs": len(runs),
                              "metrics": means, "defined_run_counts": counts})
            floor_groups = defaultdict(list)
            for row in runs.values():
                for floor in row["floors"]:
                    floor_groups[floor["origin_floor"]].append(floor)
            for floor, rows in sorted(floor_groups.items()):
                floor_summaries.append({
                    "traffic_level": level, "traffic_pattern": pattern,
                    "strategy": policy, "origin_floor": floor,
                    "observed_demand_runs": len(rows),
                    "passengers_across_demand_runs": sum(row["passengers"] for row in rows),
                    **{key: statistics.fmean(row[key] for row in rows)
                       for key in ("mean_wait_seconds", "p95_wait_seconds", "long_wait_rate", "boarding_failure_rate")},
                })
        for name, candidate, reference in comparisons:
            metrics = {}
            for metric in ("p95_wait_seconds", "mean_wait_seconds", "eta_tail5_mean_wait_seconds"):
                differences = [by_policy[candidate][run]["metrics"][metric]
                               - by_policy[reference][run]["metrics"][metric]
                               for run in run_ids]
                label = f"{level}:{pattern}:{name}:{metric}"
                seed = int.from_bytes(hashlib.sha256(label.encode()).digest()[:8], "big")
                metrics[metric] = bootstrap_mean_interval(differences, resamples, seed)
            for metric in ("long_wait_rate", "boarding_failure_rate", "repeated_rejection_rate"):
                metrics[metric] = {"mean_difference_percentage_points": 100 * statistics.fmean(
                    by_policy[candidate][run]["metrics"][metric]
                    - by_policy[reference][run]["metrics"][metric] for run in run_ids
                )}
            contrasts.append({"traffic_level": level, "traffic_pattern": pattern,
                              "contrast": name, "candidate": candidate, "reference": reference,
                              "metrics": metrics})
    return {"summaries": summaries, "contrasts": contrasts, "floor_summaries": floor_summaries}


def run_ablation(
    output_dir: Path, runs: int = 30, duration: int = 1800,
    seed: int = 202410189, order_seeds: Sequence[int] = DEFAULT_ORDER_SEEDS,
    resamples: int = 10_000, patterns: Sequence[str] = PATTERNS,
) -> dict[str, object]:
    if runs < 2 or not order_seeds or len(set(order_seeds)) != len(order_seeds):
        raise ValueError("Use at least two demand runs and unique nonempty control order seeds")
    if resamples < 100:
        raise ValueError("Use at least 100 bootstrap resamples")
    output_dir.mkdir(parents=True, exist_ok=True)
    base_config = SimulationConfig(duration_seconds=duration)
    protocol = {
        "question": "Does age ordering add benefit beyond redispatch itself?",
        "design": "Three policies; identical fixed redispatch clock for both redispatch policies",
        "first_redispatch_second": 45, "redispatch_interval_seconds": 5,
        "candidate_calls": "All pending hall calls; onboard destinations and in-flight targets retained",
        "age_definition": "Earliest arrival of a passenger still waiting in this hall call",
        "neutral_order": "Age-independent seeded permutation of floor/direction keys at each clock tick",
        "order_seeds": list(order_seeds), "base_demand_seed": seed,
        "demand_runs_per_condition": runs, "traffic_levels": LEVELS,
        "traffic_patterns": list(patterns), "config": asdict(base_config),
        "primary_outcome": "Mean of per-run P95, after averaging neutral order replicates within each demand run",
        "primary_contrast": "age_order_eta minus redispatch_eta",
        "cohort_definition": "Top ceil(5% * N) longest-waiting passenger IDs under ETA, fixed across policies",
        "failure_definition": "Rejection at an actual stop; full-car bypass not counted",
        "bootstrap": {"resamples": resamples, "unit": "Demand-run pair after averaging order replicates",
                      "interval": "95% percentile, exploratory, no multiple-comparison adjustment"},
        "limitations": [
            "Conditional on simplified inherited boarding, movement, and ETA information model",
            "Fixed-clock age_order_eta differs from historical age-triggered priority_eta",
            "Neutral policy averages a finite set of random order streams",
            "ETA-defined long-wait cohort is descriptive, selected using its baseline outcome",
            "Floor summaries average runs with observed arrivals, not equally weighted passengers",
        ],
        "source_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                          for name in ("elevator_sim.py", "ablation_experiment.py", "validate_results.py")},
    }
    # Save the protocol before obtaining outcomes; it is an exploratory design,
    # not a claim of external preregistration.
    (output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    records, trials = [], []
    journey_rows = 0
    raw_path = output_dir / "passengers.jsonl.gz"
    with gzip.open(raw_path, "wt", encoding="utf-8", compresslevel=5) as stream:
        for level, rate in LEVELS.items():
            config = replace(base_config, arrival_rate_per_second=rate)
            for pattern in patterns:
                for run_index in range(runs):
                    demand_seed = seed + run_index
                    demands = generate_demand(config, demand_seed, pattern)
                    if not demands:
                        raise ValueError("Empty demand; increase duration or use another seed")
                    eta = AblationSimulation(config, "eta", demands)
                    eta_metrics = asdict(eta.run())
                    eta_waits = {pid: p.boarding_time - p.arrival_time for pid, p in eta.passengers.items()}
                    count = max(1, math.ceil(0.05 * len(demands)))
                    tail_ids = set(sorted(eta_waits, key=lambda pid: (-eta_waits[pid], pid))[:count])
                    rejected_ids = {pid for pid, p in eta.passengers.items() if p.failed_boardings > 0}
                    context = {"traffic_level": level, "traffic_pattern": pattern,
                               "run": run_index + 1, "demand_seed": demand_seed}
                    trial_specs = [("eta", None), ("age_order_eta", None)] + [
                        ("redispatch_eta", value) for value in order_seeds
                    ]
                    neutral_trials, neutral_waits = [], []
                    age_waits = None
                    for policy, order_seed in trial_specs:
                        simulation = eta if policy == "eta" else AblationSimulation(
                            config, policy, demands,
                            order_seed if order_seed is not None else DEFAULT_ORDER_SEEDS[0],
                        )
                        metrics = eta_metrics.copy() if policy == "eta" else asdict(simulation.run())
                        extras, floors, waits = passenger_measurements(simulation, tail_ids, rejected_ids)
                        metrics.update(extras)
                        trial = {**context, "strategy": policy, "order_seed": order_seed,
                                 "passengers": len(demands), "eta_tail5_cohort_size": len(tail_ids),
                                 "eta_rejected_cohort_size": len(rejected_ids),
                                 "metrics": metrics, "floors": floors}
                        trials.append(trial)
                        journey_rows += write_passengers(stream, simulation,
                                                        {**context, "strategy": policy, "order_seed": order_seed},
                                                        tail_ids, rejected_ids)
                        if policy == "redispatch_eta":
                            neutral_trials.append(trial)
                            neutral_waits.append(waits)
                        else:
                            records.append(trial)
                            if policy == "age_order_eta":
                                age_waits = waits
                    collapsed = collapse_controls(neutral_trials)
                    control_mean_waits = {pid: statistics.fmean(item[pid] for item in neutral_waits) for pid in eta_waits}
                    # Descriptive matched-person effect relative to the five-order
                    # control mean, with ties at floating-point tolerance.
                    differences = {pid: age_waits[pid] - control_mean_waits[pid] for pid in eta_waits}
                    collapsed["age_order_matched_person_effect"] = {
                        "improved_fraction": sum(value < -1e-9 for value in differences.values()) / len(demands),
                        "worsened_fraction": sum(value > 1e-9 for value in differences.values()) / len(demands),
                        "equal_fraction": sum(abs(value) <= 1e-9 for value in differences.values()) / len(demands),
                    }
                    records.append(collapsed)
                print(f"Completed {level}/{pattern}: {runs} demand runs, {runs * (2 + len(order_seeds))} simulations", flush=True)
    summary = summarize(records, resamples)
    report = {
        "protocol": protocol, "simulation_count": len(trials),
        "passenger_journey_rows": journey_rows,
        "passenger_archive_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        "records": records, "trials": trials, **summary,
    }
    (output_dir / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for item in summary["contrasts"]:
        if item["contrast"] == "age_order_added_effect":
            metric = item["metrics"]["p95_wait_seconds"]
            print(item["traffic_level"], item["traffic_pattern"],
                  "age minus neutral P95:", round(metric["mean_difference_seconds"], 2),
                  "CI", round(metric["ci95_low_seconds"], 2), round(metric["ci95_high_seconds"], 2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--duration", type=int, default=1800)
    parser.add_argument("--seed", type=int, default=202410189)
    parser.add_argument("--order-seeds", type=int, nargs="+", default=list(DEFAULT_ORDER_SEEDS))
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--traffic-patterns", nargs="+", choices=PATTERNS, default=list(PATTERNS))
    parser.add_argument("--output-dir", type=Path, default=Path("output/ablation"))
    args = parser.parse_args()
    report = run_ablation(args.output_dir, args.runs, args.duration, args.seed,
                          args.order_seeds, args.resamples, args.traffic_patterns)
    print(f"Saved {report['simulation_count']} simulations to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
