"""Run the research simulator and export a standalone offline HTML replay.

Python 3.10+, standard library only. Browser JS replays Python-computed states;
it never makes independent dispatch or boarding decisions.
"""

import argparse
import json
import webbrowser
from dataclasses import asdict
from pathlib import Path

from ablation_experiment import POLICIES
from elevator_sim import SimulationConfig, TRAFFIC_PATTERNS, generate_demand
from elevator_trace import TracedSimulation


def write_replay(payload, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    template = Path(__file__).parent / "visualization" / "elevator_replay_template.html"
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    # The JSON lives in a nonexecuting script tag. Escape HTML delimiters even
    # when metadata is supplied by another local caller.
    safe = raw.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
    output.write_text(template.read_text(encoding="utf-8").replace("__REPLAY_DATA__", safe), encoding="utf-8")
    output.with_suffix(".json").write_text(raw + "\n", encoding="utf-8")
    return output.resolve()


def build_replay(config, seed, pattern, strategies, order_seed, output):
    if not strategies or len(strategies) != len(set(strategies)):
        raise ValueError("Select unique nonempty strategies")
    demands = generate_demand(config, seed, pattern)
    # Save conditions before calculating the illustrative run. This is a
    # demonstration, not additional evidence for the research hypotheses.
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    protocol = {"purpose": "Illustrative replay, not an independent statistical experiment",
                "config": asdict(config), "demand_seed": seed, "traffic_pattern": pattern,
                "strategies": list(strategies), "neutral_order_seed": order_seed}
    output.with_suffix(".protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    traces = []
    for strategy in strategies:
        simulation = TracedSimulation(config, strategy, demands, order_seed,
                                      capture_frames=True, capture_events=True)
        metrics = simulation.run()
        traces.append(simulation.replay_payload(metrics))
        print(f"{strategy}: {metrics.passengers} journeys, P95 {metrics.p95_wait_seconds:.2f}s" if metrics.passengers
              else f"{strategy}: no passengers", flush=True)
    return write_replay({"schema_version": 1, "traffic_pattern": pattern, "demand_seed": seed,
                         "note": "같은 승객 발생 자료를 이용한 단일 실행 예시. 통계 결과는 연구 현황 문서를 참고.",
                         "traces": traces}, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pattern", choices=TRAFFIC_PATTERNS, default="down_peak")
    parser.add_argument("--arrival-rate", type=float, default=0.30)
    parser.add_argument("--duration", type=int, default=300, help="Passenger arrival window; existing passengers then drain")
    parser.add_argument("--seed", type=int, default=202411189)
    parser.add_argument("--order-seed", type=int, default=6619)
    parser.add_argument("--strategies", nargs="+", choices=POLICIES, default=["redispatch_eta", "age_order_eta"])
    parser.add_argument("--floors", type=int, default=25)
    parser.add_argument("--elevators", type=int, default=3)
    parser.add_argument("--capacity", type=int, default=12)
    parser.add_argument("--output", type=Path, default=Path("output/visualization/elevator_replay.html"))
    parser.add_argument("--open", action="store_true", help="Open the saved file in the default browser")
    args = parser.parse_args()
    config = SimulationConfig(floors=args.floors, elevator_count=args.elevators, capacity=args.capacity,
                              initial_floors=(1,) * args.elevators, duration_seconds=args.duration,
                              arrival_rate_per_second=args.arrival_rate)
    path = build_replay(config, args.seed, args.pattern, args.strategies, args.order_seed, args.output)
    print(path)
    if args.open:
        webbrowser.open(path.as_uri())


if __name__ == "__main__":
    main()
