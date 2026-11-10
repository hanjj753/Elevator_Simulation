"""Describe who gains and loses under age ordering, using matched passengers.

Compare each age-order wait with that same passenger's mean wait over all
neutral order replicates. Average within demand runs before statistical
inference; never treat individual journeys as independent experimental units.
"""

import argparse
import gzip
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from validate_results import bootstrap_mean_interval


BANDS = {"lobby": (1, 1), "lower": (2, 9), "middle": (10, 17), "upper": (18, 25)}


def decompose(differences):
    if not differences:
        raise ValueError("Cannot summarize an empty passenger cohort")
    count = len(differences)
    improved = [value for value in differences if value < -1e-9]
    worsened = [value for value in differences if value > 1e-9]
    benefits = sum(-min(value, 0) for value in differences) / count
    harms = sum(max(value, 0) for value in differences) / count
    net = statistics.fmean(differences)
    if not math.isclose(net, harms - benefits, abs_tol=1e-9):
        raise ValueError("Gain/loss decomposition does not reconcile")
    return {
        "passengers": count,
        "improved_fraction": len(improved) / count,
        "worsened_fraction": len(worsened) / count,
        "equal_fraction": (count - len(improved) - len(worsened)) / count,
        "benefit_seconds_per_passenger": benefits,
        "harm_seconds_per_passenger": harms,
        "net_mean_wait_difference_seconds": net,
        "worsened_at_least_30_seconds_fraction": sum(value >= 30 for value in differences) / count,
    }


def analyze(directory: Path, resamples=10_000):
    result_path = directory / "results.json"
    report = json.loads(result_path.read_text(encoding="utf-8"))
    archive = directory / "passengers.jsonl.gz"
    if hashlib.sha256(archive.read_bytes()).hexdigest() != report["passenger_archive_sha256"]:
        raise ValueError("Passenger archive hash mismatch")
    order_seeds = set(report["protocol"]["order_seeds"])
    records = []

    def finish(key, trials):
        if not trials:
            return
        age = trials[("age_order_eta", None)]
        neutral = [trials[("redispatch_eta", order_seed)] for order_seed in sorted(order_seeds)]
        if set(trials) != {("eta", None), ("age_order_eta", None)} | {("redispatch_eta", s) for s in order_seeds}:
            raise ValueError("Unexpected/missing policy replicates")
        if any(set(items) != set(age) for items in trials.values()):
            raise ValueError("Passenger IDs not paired")
        groups = defaultdict(list)
        for pid, passenger in age.items():
            demand = tuple(passenger[field] for field in ("arrival_time", "origin", "destination", "eta_tail5_cohort"))
            if any(tuple(items[pid][field] for field in ("arrival_time", "origin", "destination", "eta_tail5_cohort")) != demand for items in trials.values()):
                raise ValueError("Different passenger demand or cohort across policies")
            delta = passenger["wait_seconds"] - statistics.fmean(items[pid]["wait_seconds"] for items in neutral)
            groups["all"].append(delta)
            for band, (low, high) in BANDS.items():
                if low <= passenger["origin"] <= high:
                    groups[band].append(delta)
            if passenger["eta_tail5_cohort"]:
                groups["eta_tail5"].append(delta)
        for cohort, differences in groups.items():
            records.append({"traffic_level": key[0], "traffic_pattern": key[1], "run": key[2],
                            "demand_seed": key[3], "cohort": cohort, **decompose(differences)})

    current, trials = None, {}
    journeys = 0
    with gzip.open(archive, "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            key = tuple(row[field] for field in ("traffic_level", "traffic_pattern", "run", "demand_seed"))
            if current is not None and key != current:
                finish(current, trials)
                trials = {}
            current = key
            passengers = trials.setdefault((row["strategy"], row["order_seed"]), {})
            if row["passenger_id"] in passengers:
                raise ValueError("Duplicate passenger journey")
            passengers[row["passenger_id"]] = row
            journeys += 1
    finish(current, trials)
    if journeys != report["passenger_journey_rows"]:
        raise ValueError("Passenger journey count mismatch")

    groups = defaultdict(list)
    for row in records:
        groups[(row["traffic_level"], row["traffic_pattern"], row["cohort"])].append(row)
    summaries = []
    for (level, pattern, cohort), rows in groups.items():
        means = {field: statistics.fmean(row[field] for row in rows)
                 for field in rows[0] if field not in ("traffic_level", "traffic_pattern", "run", "demand_seed", "cohort", "passengers")}
        improved, worsened = means["improved_fraction"], means["worsened_fraction"]
        means["mean_shortening_among_improved_seconds"] = means["benefit_seconds_per_passenger"] / improved if improved else None
        means["mean_lengthening_among_worsened_seconds"] = means["harm_seconds_per_passenger"] / worsened if worsened else None
        label = f"{level}:{pattern}:{cohort}"
        seed = int.from_bytes(hashlib.sha256(label.encode()).digest()[:8], "big")
        ci = bootstrap_mean_interval([row["net_mean_wait_difference_seconds"] for row in rows], resamples, seed)
        summaries.append({"traffic_level": level, "traffic_pattern": pattern, "cohort": cohort,
                          "demand_runs": len(rows), "passenger_scenarios": sum(row["passengers"] for row in rows),
                          "metrics": means, "net_difference_ci95_seconds": [ci["ci95_low_seconds"], ci["ci95_high_seconds"]]})
        if cohort == "all":
            contrast = next(item for item in report["contrasts"] if item["traffic_level"] == level and item["traffic_pattern"] == pattern and item["contrast"] == "age_order_added_effect")
            if not math.isclose(means["net_mean_wait_difference_seconds"], contrast["metrics"]["mean_wait_seconds"]["mean_difference_seconds"], abs_tol=1e-9):
                raise ValueError("Passenger decomposition disagrees with primary mean outcome")
    result = {
        "source_results_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
        "archive_verified_journeys": journeys,
        "analysis_status": "Descriptive gain/loss decomposition; not a new routing-cause test",
        "contrast": "Each passenger's age-order wait minus mean of that same passenger's neutral-order waits",
        "weighting": "Equal weight for demand runs; equal passenger weight within each run",
        "conditional_magnitude_definition": "Ratio of average per-passenger contribution to average affected fraction, preserving demand-run weighting",
        "identity": "Net mean wait difference = harm contribution minus benefit contribution",
        "records": records, "summaries": summaries,
    }
    (directory / "passenger_effects.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for row in summaries:
        if row["traffic_level"] == "congested" and row["traffic_pattern"] == "down_peak":
            print(row["cohort"], row["demand_runs"], row["metrics"])
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("output/ablation"))
    analyze(parser.parse_args().output_dir)
