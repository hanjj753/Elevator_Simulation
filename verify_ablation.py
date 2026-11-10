"""Independently reconstruct saved ablation outcomes from passenger journeys."""

import argparse
import gzip
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from elevator_sim import percentile


def verify(output_dir: Path) -> dict:
    report = json.loads((output_dir / "results.json").read_text(encoding="utf-8"))
    protocol = json.loads((output_dir / "protocol.json").read_text(encoding="utf-8"))
    assert protocol == report["protocol"], "Protocol differs"
    archive = output_dir / "passengers.jsonl.gz"
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == report["passenger_archive_sha256"]
    for name, expected in protocol["source_sha256"].items():
        assert hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() == expected, name

    def trial_key(row):
        return tuple(row[field] for field in ("traffic_level", "traffic_pattern", "run", "strategy", "order_seed"))

    trials = {trial_key(row): row for row in report["trials"]}
    assert len(trials) == len(report["trials"]), "Duplicate trial"
    reconstructed = {}
    demand_fingerprints, cohort_fingerprints = {}, {}
    row_count = 0

    def finish(key, passengers):
        trial = trials[key]
        assert len(passengers) == trial["passengers"]
        assert len({row["passenger_id"] for row in passengers}) == len(passengers)
        waits = [row["wait_seconds"] for row in passengers]
        count = len(waits)
        top_count = max(1, math.ceil(0.05 * count))
        computed = {
            "mean_wait_seconds": statistics.fmean(waits),
            "p95_wait_seconds": percentile(waits, .95),
            "long_wait_rate": sum(value > protocol["config"]["long_wait_seconds"] for value in waits) / count,
            "boarding_failure_rate": sum(row["failed_boardings"] > 0 for row in passengers) / count,
            "failed_boarding_events": sum(row["failed_boardings"] for row in passengers),
            "mean_ride_seconds": statistics.fmean(row["alighting_time"] - row["boarding_time"] for row in passengers),
            "tail5_mean_wait_seconds": statistics.fmean(sorted(waits, reverse=True)[:top_count]),
            "repeated_rejection_rate": sum(row["failed_boardings"] >= 2 for row in passengers) / count,
            "eta_tail5_mean_wait_seconds": statistics.fmean(row["wait_seconds"] for row in passengers if row["eta_tail5_cohort"]),
        }
        for field, value in computed.items():
            assert math.isclose(trial["metrics"][field], value, rel_tol=1e-12, abs_tol=1e-9), (key, field)
        assert trial["metrics"]["unserved_passengers"] == 0
        assert sum(row["eta_tail5_cohort"] for row in passengers) == top_count
        demand = [(row["passenger_id"], row["arrival_time"], row["origin"], row["destination"]) for row in passengers]
        cohort = [(row["passenger_id"], row["eta_tail5_cohort"], row["eta_rejected_cohort"]) for row in passengers]
        group = key[:3]
        fingerprint = hashlib.sha256(json.dumps(demand).encode()).hexdigest()
        cohort_fingerprint = hashlib.sha256(json.dumps(cohort).encode()).hexdigest()
        assert demand_fingerprints.setdefault(group, fingerprint) == fingerprint
        assert cohort_fingerprints.setdefault(group, cohort_fingerprint) == cohort_fingerprint
        if key[3] == "eta":
            expected_ids = {row["passenger_id"] for row in sorted(passengers, key=lambda row: (-row["wait_seconds"], row["passenger_id"]))[:top_count]}
            assert {row["passenger_id"] for row in passengers if row["eta_tail5_cohort"]} == expected_ids
        floor_groups = defaultdict(list)
        for row in passengers:
            floor_groups[row["origin"]].append(row)
        for floor in trial["floors"]:
            items = floor_groups[floor["origin_floor"]]
            assert floor["passengers"] == len(items)
            assert math.isclose(floor["mean_wait_seconds"], statistics.fmean(row["wait_seconds"] for row in items), abs_tol=1e-9)
        assert sum(floor["passengers"] for floor in trial["floors"]) == count
        reconstructed[key] = computed

    current, passengers = None, []
    with gzip.open(archive, "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            key = trial_key(row)
            if current is not None and key != current:
                finish(current, passengers)
                passengers = []
            current = key
            assert row["arrival_time"] <= row["boarding_time"] <= row["alighting_time"]
            assert math.isclose(row["wait_seconds"], row["boarding_time"] - row["arrival_time"], abs_tol=1e-12)
            assert row["boarding_attempts"] == row["failed_boardings"] + 1
            passengers.append(row)
            row_count += 1
    if current is not None:
        finish(current, passengers)
    assert reconstructed.keys() == trials.keys()
    assert row_count == report["passenger_journey_rows"]
    for row in report["records"]:
        if row["strategy"] != "redispatch_eta":
            continue
        keys = [key for key in trials if key[:3] == (row["traffic_level"], row["traffic_pattern"], row["run"]) and key[3] == "redispatch_eta"]
        assert len(keys) == len(protocol["order_seeds"])
        for metric in ("p95_wait_seconds", "mean_wait_seconds", "eta_tail5_mean_wait_seconds"):
            assert math.isclose(row["metrics"][metric], statistics.fmean(reconstructed[key][metric] for key in keys), abs_tol=1e-9)
    for summary in report["summaries"]:
        rows = [row for row in report["records"] if all(row[field] == summary[field] for field in ("traffic_level", "traffic_pattern", "strategy"))]
        assert len(rows) == summary["demand_runs"] == protocol["demand_runs_per_condition"]
        for metric in ("p95_wait_seconds", "mean_wait_seconds", "eta_tail5_mean_wait_seconds"):
            assert math.isclose(summary["metrics"][metric], statistics.fmean(row["metrics"][metric] for row in rows), abs_tol=1e-9)
    result = {"simulations": len(trials), "journey_rows": row_count,
              "paired_demand_groups": len(demand_fingerprints),
              "all_journeys_complete": True, "demand_and_cohorts_paired": True,
              "neutral_replicates_averaged_within_demand": True,
              "source_and_archive_hashes_match": True,
              "results_sha256": hashlib.sha256((output_dir / "results.json").read_bytes()).hexdigest()}
    (output_dir / "verification.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("output/ablation"))
    verify(parser.parse_args().output_dir)
