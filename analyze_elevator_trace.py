"""Exploratory response-time decomposition of the completed replication.

Replays existing demands with observers, verifies every saved journey and
metric, then averages the five neutral order streams within each demand run.
This does not create a new independent research sample.
"""

import argparse
import gzip
import hashlib
import json
import statistics
from collections import defaultdict
from dataclasses import asdict, replace
from pathlib import Path

from ablation_experiment import passenger_measurements
from elevator_sim import SimulationConfig, generate_demand
from elevator_trace import TracedSimulation
from validate_results import bootstrap_mean_interval


BANDS = {"all": (1, 25), "lower": (2, 9), "middle": (10, 17), "upper": (18, 25)}
FIELDS = ("wait_seconds", "before_first_service_seconds", "after_first_service_seconds",
          "reassignments_while_waiting", "rejected_fraction")


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def aggregate(rows):
    result = {}
    for band, (low, high) in BANDS.items():
        selected = [row for row in rows if low <= row["origin"] <= high]
        if not selected:
            raise ValueError(f"No passengers in planned band {band}")
        result[band] = {field: statistics.fmean(
            (row["failed_boardings"] > 0) if field == "rejected_fraction" else row[field]
            for row in selected) for field in FIELDS}
        result[band]["passengers"] = len(selected)
    return result


def analyze(input_dir, output_dir, resamples=10_000):
    plan_path = output_dir / "analysis_plan.json"
    if not plan_path.exists():
        raise ValueError("Save the exploratory analysis plan before running instrumentation")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    source = input_dir / "results.json"
    report = json.loads(source.read_text(encoding="utf-8"))
    protocol = report["protocol"]
    for name, expected in protocol["source_sha256"].items():
        if hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Frozen simulator changed: {name}")
    archive = input_dir / "passengers.jsonl.gz"
    if hashlib.sha256(archive.read_bytes()).hexdigest() != report["passenger_archive_sha256"]:
        raise ValueError("Saved passenger archive changed")

    def included(row):
        return (row["traffic_level"] == "congested" and row["traffic_pattern"] == "down_peak"
                and row["strategy"] in ("age_order_eta", "redispatch_eta"))

    def key(row):
        return row["demand_seed"], row["strategy"], row["order_seed"]

    trials = [row for row in report["trials"] if included(row)]
    seeds = sorted({row["demand_seed"] for row in trials})
    expected_seeds = list(range(protocol["base_demand_seed"], protocol["base_demand_seed"] + 30))
    if (seeds != expected_seeds or protocol["demand_runs_per_condition"] != 30
            or len(trials) != plan["expected_instrumented_replays"]):
        raise ValueError("This saved plan requires all 30 replication seeds / 180 replays")
    old_rows = defaultdict(dict)
    with gzip.open(archive, "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if included(row):
                old_rows[key(row)][row["passenger_id"]] = row

    # Selection uses the already-saved full set of run contrasts, before any
    # new trace is inspected. One concrete neutral order is not its mean route.
    grouped = defaultdict(dict)
    for row in report["records"]:
        if included({**row, "order_seed": None}):
            grouped[row["demand_seed"]][row["strategy"]] = row["metrics"]["p95_wait_seconds"]
    differences = {seed: item["age_order_eta"] - item["redispatch_eta"] for seed, item in grouped.items()}
    median = statistics.median(differences.values())
    representative = min(seeds, key=lambda seed: (abs(differences[seed] - median), seed))
    first_order = protocol["order_seeds"][0]
    write_json(output_dir / "illustration_selection.json", {
        "criterion": plan["illustration_selection"], "demand_seed": representative,
        "median_saved_p95_difference_seconds": median,
        "selected_saved_p95_difference_seconds": differences[representative],
        "neutral_order_seed": first_order,
    })
    config = SimulationConfig(**{**protocol["config"], "initial_floors": tuple(protocol["config"]["initial_floors"])})
    config = replace(config, arrival_rate_per_second=protocol["traffic_levels"]["congested"])
    summaries, replays = [], []
    journeys_verified = 0
    with gzip.open(output_dir / "responses.jsonl.gz", "wt", encoding="utf-8", compresslevel=5) as stream:
        for index, trial in enumerate(trials):
            saved = old_rows[key(trial)]
            capture = trial["demand_seed"] == representative and trial["order_seed"] in (None, first_order)
            simulation = TracedSimulation(
                config, trial["strategy"], generate_demand(config, trial["demand_seed"], "down_peak"),
                trial["order_seed"] if trial["order_seed"] is not None else first_order,
                first_redispatch_second=protocol["first_redispatch_second"],
                redispatch_interval=protocol["redispatch_interval_seconds"],
                capture_frames=capture, capture_events=capture)
            metrics = simulation.run()
            tail_ids = {pid for pid, row in saved.items() if row["eta_tail5_cohort"]}
            rejected_ids = {pid for pid, row in saved.items() if row["eta_rejected_cohort"]}
            extras, _, _ = passenger_measurements(simulation, tail_ids, rejected_ids)
            if {**asdict(metrics), **extras} != trial["metrics"]:
                raise ValueError(f"Instrumented metrics differ from saved trial: {key(trial)}")
            if simulation.passengers.keys() != saved.keys():
                raise ValueError("Passenger IDs differ")
            for pid, passenger in simulation.passengers.items():
                if any(saved[pid][name] != value for name, value in asdict(passenger).items()):
                    raise ValueError(f"Saved passenger journey changed: {key(trial)}, {pid}")
            rows = simulation.response_records()
            journeys_verified += len(rows)
            context = {name: trial[name] for name in ("demand_seed", "run", "strategy", "order_seed")}
            for row in rows:
                stream.write(json.dumps({**context, **row}, separators=(",", ":"), allow_nan=False) + "\n")
            summaries.append({**context, "bands": aggregate(rows)})
            if capture:
                replays.append(simulation.replay_payload(metrics))
            if (index + 1) % 30 == 0:
                print(f"Verified {index + 1}/{len(trials)} existing runs", flush=True)

    paired = []
    for seed in seeds:
        items = [row for row in summaries if row["demand_seed"] == seed]
        age = [row for row in items if row["strategy"] == "age_order_eta"]
        neutral = [row for row in items if row["strategy"] == "redispatch_eta"]
        if len(age) != 1 or sorted(row["order_seed"] for row in neutral) != sorted(protocol["order_seeds"]):
            raise ValueError("Order replicates missing or duplicated")
        paired.append({"demand_seed": seed, "bands": {band: {
            field: {"age": age[0]["bands"][band][field],
                    "neutral": statistics.fmean(row["bands"][band][field] for row in neutral)}
            for field in FIELDS} for band in BANDS}})
    effects = []
    for band in BANDS:
        outcomes = {}
        for field in FIELDS:
            age = [row["bands"][band][field]["age"] for row in paired]
            neutral = [row["bands"][band][field]["neutral"] for row in paired]
            diff = [a - b for a, b in zip(age, neutral)]
            label = f"trace:{band}:{field}"
            # The total wait is an already-published outcome. Reuse its seed
            # so repeat bootstrap Monte Carlo does not change its old interval.
            if field == "wait_seconds":
                label = ("congested:down_peak:age_order_added_effect:mean_wait_seconds" if band == "all"
                         else f"congested:down_peak:{band}")
            rng_seed = int.from_bytes(hashlib.sha256(label.encode()).digest()[:8], "big")
            interval = bootstrap_mean_interval(diff, resamples, rng_seed)
            # The reused helper names columns 'seconds'; rename for rates/counts.
            outcomes[field] = {"age_mean": statistics.fmean(age), "neutral_mean": statistics.fmean(neutral),
                               "difference": interval["mean_difference_seconds"],
                               "ci95": [interval["ci95_low_seconds"], interval["ci95_high_seconds"]],
                               "unit": "fraction" if field == "rejected_fraction" else
                                       "changes/passenger" if field == "reassignments_while_waiting" else "seconds"}
        if abs(outcomes["wait_seconds"]["difference"] - outcomes["before_first_service_seconds"]["difference"]
               - outcomes["after_first_service_seconds"]["difference"]) > 1e-8:
            raise ValueError("Component differences fail to sum to total difference")
        effects.append({"band": band, "demand_runs": len(seeds), "outcomes": outcomes})
    result = {"stage": plan["stage"], "plan": plan, "source_results_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "bootstrap_resamples": resamples,
              "instrumentation_source_sha256": {
                  name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                  for name in ("elevator_trace.py", "analyze_elevator_trace.py")},
              "verification": {"exact_metric_matches": len(trials), "exact_journey_matches": journeys_verified,
                               "frozen_sources_match": True},
              "inference": "30 paired demand runs; five neutral orders averaged first; unadjusted exploratory 95% bootstrap",
              "effects": effects, "paired_runs": paired, "trial_summaries": summaries}
    write_json(output_dir / "response_decomposition.json", result)
    from visualize_elevator import write_replay
    replay = {"schema_version": 1, "traffic_pattern": "down_peak", "demand_seed": representative,
              "note": "기존 재현 자료의 P95 차이가 중앙값에 가까운 승객 씨앗. B는 순서 씨앗 1개 경로이며 5회 평균 경로가 아님.",
              "traces": sorted(replays, key=lambda trace: trace["strategy"] != "redispatch_eta")}
    write_replay(replay, output_dir / "representative_replay.html")
    write_component_figure(result, output_dir / "response_decomposition.svg")
    for effect in effects:
        print(effect["band"], {name: round(item["difference"], 3) for name, item in effect["outcomes"].items()})
    return result


def write_component_figure(result, path):
    """Dependency-free scientific figure with true zero and explicit units."""
    outcomes = next(row for row in result["effects"] if row["band"] == "upper")["outcomes"]
    width, height = 1040, 470
    before, after = outcomes["before_first_service_seconds"], outcomes["after_first_service_seconds"]
    maximum = max(outcomes["wait_seconds"][f"{policy}_mean"] for policy in ("neutral", "age"))
    axis_max = (int(maximum / 20) + 1) * 20
    scale = 700 / axis_max
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<g font-family="Malgun Gothic, Apple SD Gothic Neo, sans-serif" fill="#172b3a">',
             '<text x="36" y="48" font-size="25" font-weight="700">18–25층 대기시간은 어느 구간에서 늘었는가?</text>',
             '<text x="36" y="78" font-size="16">혼잡 퇴근형 · 기존 재현 자료의 30개 승객 씨앗 · B 순서 5회 평균 후 반복 평균</text>']
    for tick in range(0, axis_max + 1, 40):
        x = 240 + tick * scale
        parts.append(f'<path d="M{x} 115V265" stroke="#e4e9ed"/><text x="{x}" y="290" text-anchor="middle" font-size="14">{tick}</text>')
    for policy, label, y in (("neutral", "B  무작위 호출 순서", 126), ("age", "C  오래된 호출 순서", 206)):
        b, a = before[f"{policy}_mean"], after[f"{policy}_mean"]
        parts.extend([f'<text x="36" y="{y+28}" font-size="18">{label}</text>',
                      f'<rect x="240" y="{y}" width="{b*scale}" height="42" fill="#25658d"/>',
                      f'<rect x="{240+b*scale}" y="{y}" width="{a*scale}" height="42" fill="#ca7a35"/>',
                      f'<text x="{250+(b+a)*scale}" y="{y+28}" font-size="17">{b+a:.2f}초</text>'])
    parts.append('<text x="995" y="290" font-size="14">초</text>')
    for i, (label, color, item) in enumerate((("첫 정차 응답 전", "#25658d", before), ("첫 응답 후 재대기", "#ca7a35", after))):
        y = 338 + i * 42
        parts.append(f'<rect x="36" y="{y-15}" width="16" height="16" fill="{color}"/><text x="63" y="{y}" font-size="18">{label}: C−B {item["difference"]:+.2f}초  (95% 구간 {item["ci95"][0]:+.2f}, {item["ci95"][1]:+.2f})</text>')
    parts.extend(['<text x="36" y="434" font-size="14">첫 응답 = 해당 호출에 배정된 차량의 실제 정차. 만원 무정차 통과는 제외. 탐색적 분해이며 경로 인과 증명은 아님.</text>', '</g></svg>'])
    path.write_text("\n".join(parts), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("output/ablation_replication"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/trace_analysis"))
    parser.add_argument("--resamples", type=int, default=10_000)
    args = parser.parse_args()
    analyze(args.input_dir, args.output_dir, args.resamples)
