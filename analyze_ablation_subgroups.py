"""Exploratory, post-hoc origin-floor-band analysis of the saved ablation.

The primary ablation was completed first. These equal-sized upper-floor bands
help describe heterogeneity; they are not preregistered confirmatory outcomes.
"""

import argparse
import hashlib
import json
import statistics
from pathlib import Path

from validate_results import bootstrap_mean_interval


def analyze(path: Path, resamples: int = 10_000, replication_plan: Path | None = None) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    plan = None
    if replication_plan is not None:
        plan = json.loads(replication_plan.read_text(encoding="utf-8"))
        for field in ("base_demand_seed", "demand_runs_per_condition", "order_seeds", "traffic_levels", "traffic_patterns"):
            if plan[field] != report["protocol"][field]:
                raise ValueError(f"Replication differs from saved plan: {field}")
        if plan["duration_seconds"] != report["protocol"]["config"]["duration_seconds"]:
            raise ValueError("Replication duration differs from plan")
    if report["protocol"]["config"]["floors"] != 25:
        raise ValueError("This exploratory grouping is defined for the 25-floor study")
    bands = (("lobby", 1, 1), ("lower", 2, 9), ("middle", 10, 17), ("upper", 18, 25))
    groups = {}
    for row in report["records"]:
        if row["strategy"] not in ("redispatch_eta", "age_order_eta"):
            continue
        for band, low, high in bands:
            floors = [floor for floor in row["floors"] if low <= floor["origin_floor"] <= high]
            passengers = sum(floor["passengers"] for floor in floors)
            if not passengers:
                continue
            key = (row["traffic_level"], row["traffic_pattern"], band)
            groups.setdefault(key, {}).setdefault(row["strategy"], {})[row["run"]] = {
                "passengers": passengers,
                "mean_wait_seconds": sum(floor["mean_wait_seconds"] * floor["passengers"] for floor in floors) / passengers,
            }
    effects = []
    for (level, pattern, band), policies in groups.items():
        age, neutral = policies["age_order_eta"], policies["redispatch_eta"]
        assert age.keys() == neutral.keys()
        run_ids = sorted(age)
        assert all(age[run]["passengers"] == neutral[run]["passengers"] for run in run_ids)
        values = [age[run]["mean_wait_seconds"] - neutral[run]["mean_wait_seconds"] for run in run_ids]
        seed = int.from_bytes(hashlib.sha256(f"{level}:{pattern}:{band}".encode()).digest()[:8], "big")
        effects.append({
            "traffic_level": level, "traffic_pattern": pattern, "origin_floor_band": band,
            "passengers_across_demand_runs": sum(age[run]["passengers"] for run in run_ids),
            "age_mean_wait_seconds": statistics.fmean(age[run]["mean_wait_seconds"] for run in run_ids),
            "neutral_mean_wait_seconds": statistics.fmean(neutral[run]["mean_wait_seconds"] for run in run_ids),
            "effect": bootstrap_mean_interval(values, resamples, seed) if len(values) >= 2 else None,
        })
    result = {
        "analysis_status": (
            "Independent-seed follow-up; 18-25 floor band fixed in the saved plan before follow-up simulations; secondary outcome"
            if plan is not None else
            "Exploratory post-hoc; bands chosen after primary results were inspected"
        ),
        "bands": [{"name": name, "lowest_floor": low, "highest_floor": high} for name, low, high in bands],
        "weighting": "Passenger-weighted floor means within a demand run, then equal weight across observed demand runs",
        "contrast": "age_order_eta minus redispatch_eta; negative favors age ordering",
        "ci": "Unadjusted exploratory 95% paired percentile bootstrap",
        "source_results_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "effects": effects,
    }
    if plan is not None:
        result["replication_plan_sha256"] = hashlib.sha256(replication_plan.read_bytes()).hexdigest()
    destination = path.parent / "subgroups.json"
    destination.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for effect in effects:
        if effect["traffic_level"] == "congested" and effect["traffic_pattern"] == "down_peak":
            print(effect["origin_floor_band"], effect["passengers_across_demand_runs"], effect["effect"])
    print(destination.resolve())
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("output/ablation/results.json"))
    parser.add_argument("--replication-plan", type=Path)
    args = parser.parse_args()
    analyze(args.input, replication_plan=args.replication_plan)
