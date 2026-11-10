"""Check a fresh-seed follow-up against its saved plan and discovery results."""

import argparse
import hashlib
import json
from pathlib import Path

from analyze_ablation_subgroups import analyze as analyze_floors
from verify_ablation import verify


def analyze(directory: Path):
    plan_path = directory / "replication_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    report_path = directory / "results.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    discovery_path = Path(__file__).parent / plan["discovery_results"]
    discovery = json.loads(discovery_path.read_text(encoding="utf-8"))
    for field in ("base_demand_seed", "demand_runs_per_condition", "order_seeds", "traffic_levels", "traffic_patterns"):
        if report["protocol"][field] != plan[field]:
            raise ValueError(f"Follow-up does not match plan: {field}")
    if report["simulation_count"] != plan["expected_simulations"]:
        raise ValueError("Follow-up sample size differs from plan")
    old_seeds = {row["demand_seed"] for row in discovery["records"]}
    new_seeds = {row["demand_seed"] for row in report["records"]}
    if new_seeds & old_seeds:
        raise ValueError("Follow-up reuses discovery demand seeds")
    if set(report["protocol"]["order_seeds"]) & set(discovery["protocol"]["order_seeds"]):
        raise ValueError("Follow-up reuses discovery order seeds")
    expected_seeds = set(range(plan["base_demand_seed"], plan["base_demand_seed"] + plan["demand_runs_per_condition"]))
    if new_seeds != expected_seeds:
        raise ValueError("Missing or replaced follow-up demand seeds")
    verification = verify(directory)
    floor_report = analyze_floors(report_path, replication_plan=plan_path)
    old_floor_path = discovery_path.with_name("subgroups.json")
    old_floors = json.loads(old_floor_path.read_text(encoding="utf-8"))
    if old_floors["source_results_sha256"] != hashlib.sha256(discovery_path.read_bytes()).hexdigest():
        raise ValueError("Discovery floor analysis is stale")
    def contrast(result, metric):
        row = next(item for item in result["contrasts"] if item["traffic_level"] == "congested" and item["traffic_pattern"] == "down_peak" and item["contrast"] == "age_order_added_effect")
        return row["metrics"][metric]
    def upper(result):
        row = next(item for item in result["effects"] if item["traffic_level"] == "congested" and item["traffic_pattern"] == "down_peak" and item["origin_floor_band"] == "upper")
        return {**row["effect"], "age_mean_wait_seconds": row["age_mean_wait_seconds"],
                "neutral_mean_wait_seconds": row["neutral_mean_wait_seconds"]}
    primary = contrast(report, "p95_wait_seconds")
    secondary = upper(floor_report)
    result = {
        "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        "discovery_sha256": hashlib.sha256(discovery_path.read_bytes()).hexdigest(),
        "followup_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
        "disjoint_demand_and_order_seeds": True,
        "verification": verification,
        "stage_counts": {"discovery_simulations": discovery["simulation_count"],
                         "followup_simulations": report["simulation_count"],
                         "followup_journey_rows": report["passenger_journey_rows"]},
        "condition": "congested/down_peak",
        "contrast": "age_order_eta minus redispatch_eta",
        "primary_p95_seconds": {"discovery": contrast(discovery, "p95_wait_seconds"), "followup": primary},
        "secondary_upper_band_mean_seconds": {"discovery": upper(old_floors), "followup": secondary},
        "overall_mean_seconds": {"discovery": contrast(discovery, "mean_wait_seconds"), "followup": contrast(report, "mean_wait_seconds")},
        "hypothesis_decisions": {
            "H3a": "supports predicted increase within this model" if primary["ci95_low_seconds"] > 0 else "predicted increase not established",
            "H3b": "supports predicted increase in secondary outcome" if secondary["ci95_low_seconds"] > 0 else "predicted upper-floor increase not established",
        },
        "interpretation": "Fresh-seed robustness in the same simulator; no claim of external preregistration, real-building validity, or identified routing mechanism. Stages are not pooled.",
    }
    (directory / "replication_comparison.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("output/ablation_replication"))
    analyze(parser.parse_args().output_dir)
