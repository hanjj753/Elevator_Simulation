"""Reproducible elevator group-control simulation for the research proposal.

The model compares three dispatch rules across traffic patterns and demand
levels. It can save run-level and summary CSV files for later analysis.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import statistics
from collections import defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterable, Mapping, Sequence


UP = 1
DOWN = -1
TRAFFIC_PATTERNS = ("uniform", "up_peak", "interfloor", "down_peak")
DEFAULT_TRAFFIC_LEVELS = {
    "normal": 0.12,
    "congested": 0.30,
}
TRAFFIC_PATTERN_LABELS = {
    "uniform": "uniform random",
    "up_peak": "morning/up-peak",
    "interfloor": "interfloor",
    "down_peak": "evening/down-peak",
}
_PATTERN_SEED_OFFSETS = {
    "uniform": 0,
    "up_peak": 1_000_003,
    "interfloor": 2_000_033,
    "down_peak": 3_000_091,
}


@dataclass(frozen=True)
class SimulationConfig:
    floors: int = 25
    elevator_count: int = 3
    capacity: int = 12
    floor_height_m: float = 3.5
    max_speed_mps: float = 2.5
    acceleration_mps2: float = 1.0
    door_seconds: int = 4
    boarding_seconds_per_passenger: float = 0.8
    alighting_seconds_per_passenger: float = 0.6
    duration_seconds: int = 1_800
    arrival_rate_per_second: float = 0.12
    long_wait_seconds: float = 60.0
    priority_wait_seconds: float = 45.0
    priority_redispatch_interval: int = 5
    dominant_flow_share: float = 0.85
    initial_floors: tuple[int, ...] = (1, 1, 1)

    def __post_init__(self) -> None:
        if self.floors < 2:
            raise ValueError("floors must be at least 2")
        if self.elevator_count < 1 or self.capacity < 1:
            raise ValueError("elevator_count and capacity must be positive")
        if len(self.initial_floors) != self.elevator_count:
            raise ValueError("initial_floors must contain one floor per elevator")
        if any(not 1 <= floor <= self.floors for floor in self.initial_floors):
            raise ValueError("every initial floor must be inside the building")
        if (
            self.floor_height_m <= 0
            or self.max_speed_mps <= 0
            or self.acceleration_mps2 <= 0
        ):
            raise ValueError("floor height, speed, and acceleration must be positive")
        if self.door_seconds < 1:
            raise ValueError("door time must be a positive integer")
        if (
            self.boarding_seconds_per_passenger < 0
            or self.alighting_seconds_per_passenger < 0
        ):
            raise ValueError("passenger service times must be nonnegative")
        if self.duration_seconds < 1 or self.arrival_rate_per_second < 0:
            raise ValueError("duration must be positive and arrival rate nonnegative")
        if not 0.0 <= self.dominant_flow_share <= 1.0:
            raise ValueError("dominant_flow_share must be between 0 and 1")

    def dwell_seconds(self, boarding_count: int, alighting_count: int) -> int:
        """Return door time plus sequential passenger exchange time."""

        if boarding_count < 0 or alighting_count < 0:
            raise ValueError("passenger counts must be nonnegative")
        exchange_time = (
            boarding_count * self.boarding_seconds_per_passenger
            + alighting_count * self.alighting_seconds_per_passenger
        )
        return self.door_seconds + math.ceil(exchange_time)

    def travel_time_seconds(self, floors_apart: int) -> int:
        """Return travel time for a direct stop-to-stop movement.

        Short trips use a triangular speed profile because the car cannot
        reach maximum speed. Longer trips use a trapezoidal profile with a
        constant-speed section between acceleration and braking.
        """

        if floors_apart < 0:
            raise ValueError("floors_apart must be nonnegative")
        if floors_apart == 0:
            return 0

        distance_m = floors_apart * self.floor_height_m
        distance_to_accelerate_and_brake = (
            self.max_speed_mps**2 / self.acceleration_mps2
        )
        if distance_m <= distance_to_accelerate_and_brake:
            travel_seconds = 2.0 * math.sqrt(
                distance_m / self.acceleration_mps2
            )
        else:
            travel_seconds = (
                2.0 * self.max_speed_mps / self.acceleration_mps2
                + (distance_m - distance_to_accelerate_and_brake)
                / self.max_speed_mps
            )
        return math.ceil(travel_seconds)


@dataclass(frozen=True)
class PassengerDemand:
    passenger_id: int
    arrival_time: float
    origin: int
    destination: int

    @property
    def direction(self) -> int:
        return UP if self.destination > self.origin else DOWN


@dataclass
class Passenger:
    passenger_id: int
    arrival_time: float
    origin: int
    destination: int
    boarding_time: float | None = None
    alighting_time: float | None = None
    boarding_attempts: int = 0
    failed_boardings: int = 0

    @classmethod
    def from_demand(cls, demand: PassengerDemand) -> "Passenger":
        return cls(
            passenger_id=demand.passenger_id,
            arrival_time=demand.arrival_time,
            origin=demand.origin,
            destination=demand.destination,
        )


CallKey = tuple[int, int]  # (floor, direction)


@dataclass
class HallCall:
    floor: int
    direction: int
    created_time: float
    waiting_passenger_ids: list[int] = field(default_factory=list)
    assigned_elevator_id: int | None = None

    @property
    def key(self) -> CallKey:
        return (self.floor, self.direction)


@dataclass
class Elevator:
    elevator_id: int
    current_floor: int
    capacity: int
    direction: int = 0
    move_remaining: int = 0
    target_floor: int | None = None
    door_remaining: int = 0
    onboard_passenger_ids: list[int] = field(default_factory=list)
    assigned_calls: set[CallKey] = field(default_factory=set)
    floors_travelled: int = 0

    @property
    def available_space(self) -> int:
        return self.capacity - len(self.onboard_passenger_ids)


@dataclass(frozen=True)
class Metrics:
    passengers: int
    mean_wait_seconds: float
    p95_wait_seconds: float
    long_wait_rate: float
    boarding_failure_rate: float
    failed_boarding_events: int
    mean_ride_seconds: float
    unserved_passengers: int

    @classmethod
    def calculate(
        cls, passengers: Iterable[Passenger], long_wait_seconds: float
    ) -> "Metrics":
        passenger_list = list(passengers)
        waits = [
            p.boarding_time - p.arrival_time
            for p in passenger_list
            if p.boarding_time is not None
        ]
        rides = [
            p.alighting_time - p.boarding_time
            for p in passenger_list
            if p.boarding_time is not None and p.alighting_time is not None
        ]
        total = len(passenger_list)
        unserved = total - len(waits)
        if not waits:
            return cls(total, 0.0, 0.0, 0.0, 0.0, 0, 0.0, unserved)

        failed_passengers = sum(p.failed_boardings > 0 for p in passenger_list)
        failed_events = sum(p.failed_boardings for p in passenger_list)
        return cls(
            passengers=total,
            mean_wait_seconds=statistics.fmean(waits),
            p95_wait_seconds=percentile(waits, 0.95),
            long_wait_rate=sum(wait > long_wait_seconds for wait in waits) / len(waits),
            boarding_failure_rate=failed_passengers / total if total else 0.0,
            failed_boarding_events=failed_events,
            mean_ride_seconds=statistics.fmean(rides) if rides else 0.0,
            unserved_passengers=unserved,
        )


def percentile(values: Sequence[float], quantile: float) -> float:
    """Return a linearly interpolated percentile without third-party packages."""

    if not values:
        return 0.0
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be between 0 and 1")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def generate_demand(
    config: SimulationConfig, seed: int, traffic_pattern: str
) -> list[PassengerDemand]:
    """Generate demand while keeping arrival times paired across patterns."""

    if traffic_pattern not in TRAFFIC_PATTERNS:
        raise ValueError(f"unknown traffic pattern: {traffic_pattern}")
    if traffic_pattern == "interfloor" and config.floors < 3:
        raise ValueError("interfloor traffic requires at least 3 floors")
    if config.arrival_rate_per_second == 0:
        return []
    arrival_rng = random.Random(seed)
    od_rng = random.Random(seed + _PATTERN_SEED_OFFSETS[traffic_pattern])
    demands: list[PassengerDemand] = []
    arrival_time = 0.0
    while True:
        arrival_time += arrival_rng.expovariate(config.arrival_rate_per_second)
        if arrival_time >= config.duration_seconds:
            break
        origin, destination = _sample_origin_destination(
            config, od_rng, traffic_pattern
        )
        demands.append(
            PassengerDemand(
                passenger_id=len(demands),
                arrival_time=arrival_time,
                origin=origin,
                destination=destination,
            )
        )
    return demands


def _sample_origin_destination(
    config: SimulationConfig, rng: random.Random, traffic_pattern: str
) -> tuple[int, int]:
    if traffic_pattern == "up_peak":
        if rng.random() < config.dominant_flow_share:
            return 1, rng.randint(2, config.floors)
        return _uniform_origin_destination(rng, config.floors)
    if traffic_pattern == "down_peak":
        if rng.random() < config.dominant_flow_share:
            return rng.randint(2, config.floors), 1
        return _uniform_origin_destination(rng, config.floors)
    if traffic_pattern == "interfloor":
        return _uniform_origin_destination(rng, config.floors, lowest_floor=2)
    return _uniform_origin_destination(rng, config.floors)


def _uniform_origin_destination(
    rng: random.Random, highest_floor: int, lowest_floor: int = 1
) -> tuple[int, int]:
    origin = rng.randint(lowest_floor, highest_floor)
    destination = rng.randint(lowest_floor, highest_floor - 1)
    if destination >= origin:
        destination += 1
    return origin, destination


def generate_uniform_demand(
    config: SimulationConfig, seed: int
) -> list[PassengerDemand]:
    """Backward-compatible helper for the original neutral traffic model."""

    return generate_demand(config, seed, "uniform")


class ElevatorSimulation:
    VALID_STRATEGIES = {"closest", "eta", "priority_eta"}

    def __init__(
        self,
        config: SimulationConfig,
        strategy: str,
        demands: Sequence[PassengerDemand],
    ) -> None:
        if strategy not in self.VALID_STRATEGIES:
            raise ValueError(f"unknown strategy: {strategy}")
        self.config = config
        self.strategy = strategy
        self.passengers = {
            demand.passenger_id: Passenger.from_demand(demand) for demand in demands
        }
        self.arrivals: dict[int, list[PassengerDemand]] = defaultdict(list)
        for demand in demands:
            self.arrivals[math.ceil(demand.arrival_time)].append(demand)
        self.calls: dict[CallKey, HallCall] = {}
        self.elevators = [
            Elevator(index, config.initial_floors[index], config.capacity)
            for index in range(config.elevator_count)
        ]
        self.last_priority_redispatch = -config.priority_redispatch_interval

    def run(self) -> Metrics:
        # The drain period lets everyone who arrived during the observation
        # window finish instead of silently dropping late passengers.
        # A deliberately poor dispatch rule can build a long queue under
        # congested demand. The generous limit is only a runaway safeguard;
        # normal completion still returns as soon as the system is empty.
        max_drain = max(3_600, self.config.duration_seconds * 5)
        deadline = self.config.duration_seconds + max_drain

        for now in range(deadline + 1):
            self._add_arrivals(now)
            self._maybe_redispatch_priority_calls(now)
            for elevator in self.elevators:
                self._step_elevator(elevator, now)

            if not self.arrivals and self._system_is_idle():
                return Metrics.calculate(
                    self.passengers.values(), self.config.long_wait_seconds
                )

        waiting = sum(len(call.waiting_passenger_ids) for call in self.calls.values())
        raise RuntimeError(
            f"simulation did not drain by {deadline}s; {waiting} passengers remain"
        )

    def _add_arrivals(self, now: int) -> None:
        for demand in self.arrivals.pop(now, []):
            key = (demand.origin, demand.direction)
            call = self.calls.get(key)
            if call is None:
                call = HallCall(
                    floor=demand.origin,
                    direction=demand.direction,
                    created_time=demand.arrival_time,
                )
                self.calls[key] = call
            call.waiting_passenger_ids.append(demand.passenger_id)
            if call.assigned_elevator_id is None:
                self._assign_call(call)

    def _assign_call(self, call: HallCall) -> None:
        if self.strategy == "closest":
            elevator = min(self.elevators, key=lambda e: self._closest_score(e, call))
        else:
            elevator = min(self.elevators, key=lambda e: self._eta_score(e, call))
        call.assigned_elevator_id = elevator.elevator_id
        elevator.assigned_calls.add(call.key)

    def _closest_score(self, elevator: Elevator, call: HallCall) -> tuple[int, int, int]:
        workload = len(elevator.assigned_calls) + len(elevator.onboard_passenger_ids)
        return (abs(elevator.current_floor - call.floor), workload, elevator.elevator_id)

    def _eta_score(self, elevator: Elevator, call: HallCall) -> tuple[float, int, int]:
        workload = len(elevator.assigned_calls) + len(elevator.onboard_passenger_ids)
        return (self._estimate_eta(elevator, call.floor), workload, elevator.elevator_id)

    def _estimate_eta(self, elevator: Elevator, call_floor: int) -> float:
        elapsed = float(elevator.door_remaining)
        position = elevator.current_floor
        direction = elevator.direction

        if elevator.move_remaining > 0:
            if elevator.target_floor is None:
                raise RuntimeError("moving elevator has no target floor")
            elapsed += elevator.move_remaining
            position = elevator.target_floor

        stops = {
            self.passengers[passenger_id].destination
            for passenger_id in elevator.onboard_passenger_ids
        }
        stops.update(floor for floor, _ in elevator.assigned_calls)
        stops.add(call_floor)

        for stop in build_look_route(position, direction, stops):
            elapsed += self.config.travel_time_seconds(abs(stop - position))
            if stop == call_floor:
                return elapsed
            elapsed += self._estimated_stop_dwell(elevator, stop)
            position = stop
        return elapsed

    def _estimated_stop_dwell(self, elevator: Elevator, stop: int) -> int:
        alighting_count = sum(
            self.passengers[passenger_id].destination == stop
            for passenger_id in elevator.onboard_passenger_ids
        )
        waiting_count = sum(
            len(self.calls[key].waiting_passenger_ids)
            for key in elevator.assigned_calls
            if key in self.calls and key[0] == stop
        )
        available_after_alighting = (
            elevator.available_space + alighting_count
        )
        boarding_count = min(waiting_count, available_after_alighting)
        return self.config.dwell_seconds(boarding_count, alighting_count)

    def _maybe_redispatch_priority_calls(self, now: int) -> None:
        if self.strategy != "priority_eta" or not self.calls:
            return
        oldest_age = max(now - call.created_time for call in self.calls.values())
        if oldest_age < self.config.priority_wait_seconds:
            return
        if now - self.last_priority_redispatch < self.config.priority_redispatch_interval:
            return

        # Only hall calls are redistributed. Destinations of passengers
        # already inside an elevator never move to another elevator.
        for elevator in self.elevators:
            elevator.assigned_calls.clear()
        ordered_calls = sorted(
            self.calls.values(),
            key=lambda call: (
                0
                if now - call.created_time >= self.config.priority_wait_seconds
                else 1,
                call.created_time,
            ),
        )
        for call in ordered_calls:
            call.assigned_elevator_id = None
            self._assign_call(call)
        self.last_priority_redispatch = now

    def _step_elevator(self, elevator: Elevator, now: int) -> None:
        if elevator.door_remaining > 0:
            elevator.door_remaining -= 1
            if elevator.door_remaining > 0:
                return

        if elevator.move_remaining > 0:
            elevator.move_remaining -= 1
            if elevator.move_remaining > 0:
                return
            if elevator.target_floor is None:
                raise RuntimeError("moving elevator has no target floor")
            floors_apart = abs(elevator.target_floor - elevator.current_floor)
            elevator.current_floor = elevator.target_floor
            elevator.target_floor = None
            elevator.floors_travelled += floors_apart
            if self._should_stop(elevator):
                self._service_floor(elevator, now)
                return

        if self._should_stop(elevator):
            self._service_floor(elevator, now)
            return

        target = self._choose_next_floor(elevator)
        if target is None:
            elevator.direction = 0
            elevator.target_floor = None
            return
        elevator.direction = UP if target > elevator.current_floor else DOWN
        elevator.target_floor = target
        elevator.move_remaining = self.config.travel_time_seconds(
            abs(target - elevator.current_floor)
        )

    def _should_stop(self, elevator: Elevator) -> bool:
        has_dropoff = any(
            self.passengers[passenger_id].destination == elevator.current_floor
            for passenger_id in elevator.onboard_passenger_ids
        )
        has_pickup = elevator.available_space > 0 and any(
            key in self.calls and key[0] == elevator.current_floor
            for key in elevator.assigned_calls
        )
        return has_dropoff or has_pickup

    def _choose_next_floor(self, elevator: Elevator) -> int | None:
        stops = {
            self.passengers[passenger_id].destination
            for passenger_id in elevator.onboard_passenger_ids
        }
        if elevator.available_space > 0:
            stops.update(
                floor
                for floor, direction in elevator.assigned_calls
                if (floor, direction) in self.calls
            )
        stops.discard(elevator.current_floor)
        route = build_look_route(elevator.current_floor, elevator.direction, stops)
        return route[0] if route else None

    def _service_floor(self, elevator: Elevator, now: int) -> None:
        activity = False
        alighting_count = 0
        remaining_onboard: list[int] = []
        for passenger_id in elevator.onboard_passenger_ids:
            passenger = self.passengers[passenger_id]
            if passenger.destination == elevator.current_floor:
                passenger.alighting_time = float(now)
                activity = True
                alighting_count += 1
            else:
                remaining_onboard.append(passenger_id)
        elevator.onboard_passenger_ids = remaining_onboard

        call_keys = [
            key
            for key in elevator.assigned_calls
            if key in self.calls and key[0] == elevator.current_floor
        ]
        waiting_here: list[tuple[float, int, CallKey]] = []
        for key in call_keys:
            call = self.calls[key]
            waiting_here.extend(
                (self.passengers[passenger_id].arrival_time, passenger_id, key)
                for passenger_id in call.waiting_passenger_ids
            )
        waiting_here.sort()

        if waiting_here:
            activity = True
        available = elevator.available_space
        boarding_ids = {passenger_id for _, passenger_id, _ in waiting_here[:available]}
        for _, passenger_id, _ in waiting_here:
            passenger = self.passengers[passenger_id]
            passenger.boarding_attempts += 1
            if passenger_id in boarding_ids:
                passenger.boarding_time = float(now)
                elevator.onboard_passenger_ids.append(passenger_id)
            else:
                passenger.failed_boardings += 1

        for key in call_keys:
            call = self.calls[key]
            call.waiting_passenger_ids = [
                passenger_id
                for passenger_id in call.waiting_passenger_ids
                if passenger_id not in boarding_ids
            ]
            if not call.waiting_passenger_ids:
                self._remove_call(key)

        if activity:
            elevator.door_remaining = self.config.dwell_seconds(
                boarding_count=len(boarding_ids),
                alighting_count=alighting_count,
            )

    def _remove_call(self, key: CallKey) -> None:
        self.calls.pop(key, None)
        for elevator in self.elevators:
            elevator.assigned_calls.discard(key)

    def _system_is_idle(self) -> bool:
        return not self.calls and all(
            not elevator.onboard_passenger_ids and not elevator.assigned_calls
            for elevator in self.elevators
        )


def build_look_route(
    current_floor: int, direction: int, stops: Iterable[int]
) -> list[int]:
    """Order stops with a simple LOOK/collective-control route."""

    remaining = set(stops)
    route: list[int] = []
    position = current_floor
    travel_direction = direction

    while remaining:
        if position in remaining:
            next_floor = position
        elif travel_direction == 0:
            next_floor = min(remaining, key=lambda floor: (abs(floor - position), floor))
            travel_direction = UP if next_floor > position else DOWN
        elif travel_direction == UP:
            above = [floor for floor in remaining if floor > position]
            if not above:
                travel_direction = DOWN
                continue
            next_floor = min(above)
        else:
            below = [floor for floor in remaining if floor < position]
            if not below:
                travel_direction = UP
                continue
            next_floor = max(below)
        route.append(next_floor)
        remaining.remove(next_floor)
        position = next_floor
    return route


@dataclass(frozen=True)
class ExperimentRow:
    traffic_level: str
    arrival_rate_per_second: float
    traffic_pattern: str
    strategy: str
    run: int
    seed: int
    duration_seconds: int
    floors: int
    elevator_count: int
    capacity: int
    floor_height_m: float
    max_speed_mps: float
    acceleration_mps2: float
    dominant_flow_share: float
    base_door_seconds: int
    boarding_seconds_per_passenger: float
    alighting_seconds_per_passenger: float
    metrics: Metrics


def run_experiment(
    config: SimulationConfig,
    strategies: Sequence[str],
    runs: int,
    seed: int,
    traffic_patterns: Sequence[str] = ("uniform",),
    traffic_levels: Mapping[str, float] | None = None,
) -> list[ExperimentRow]:
    if runs < 1:
        raise ValueError("runs must be at least 1")
    level_rates = dict(
        traffic_levels
        if traffic_levels is not None
        else {"custom": config.arrival_rate_per_second}
    )
    if not level_rates:
        raise ValueError("at least one traffic level is required")
    rows: list[ExperimentRow] = []
    for run_index in range(runs):
        run_seed = seed + run_index
        for traffic_level, arrival_rate in level_rates.items():
            level_config = replace(
                config, arrival_rate_per_second=arrival_rate
            )
            for traffic_pattern in traffic_patterns:
                # Arrival times are paired across traffic patterns within each
                # demand level. Only the origin/destination mix changes.
                demands = generate_demand(
                    level_config, run_seed, traffic_pattern
                )
                for strategy in strategies:
                    metrics = ElevatorSimulation(
                        level_config, strategy, demands
                    ).run()
                    rows.append(
                        ExperimentRow(
                            traffic_level=traffic_level,
                            arrival_rate_per_second=arrival_rate,
                            traffic_pattern=traffic_pattern,
                            strategy=strategy,
                            run=run_index + 1,
                            seed=run_seed,
                            duration_seconds=level_config.duration_seconds,
                            floors=level_config.floors,
                            elevator_count=level_config.elevator_count,
                            capacity=level_config.capacity,
                            floor_height_m=level_config.floor_height_m,
                            max_speed_mps=level_config.max_speed_mps,
                            acceleration_mps2=(
                                level_config.acceleration_mps2
                            ),
                            dominant_flow_share=(
                                level_config.dominant_flow_share
                            ),
                            base_door_seconds=level_config.door_seconds,
                            boarding_seconds_per_passenger=(
                                level_config.boarding_seconds_per_passenger
                            ),
                            alighting_seconds_per_passenger=(
                                level_config.alighting_seconds_per_passenger
                            ),
                            metrics=metrics,
                        )
                    )
    return rows


def aggregate_rows(
    rows: Sequence[ExperimentRow], traffic_level: str, traffic_pattern: str
) -> list[tuple[str, Metrics]]:
    grouped: dict[str, list[Metrics]] = defaultdict(list)
    for row in rows:
        if (
            row.traffic_level == traffic_level
            and row.traffic_pattern == traffic_pattern
        ):
            grouped[row.strategy].append(row.metrics)

    result: list[tuple[str, Metrics]] = []
    for strategy, items in grouped.items():
        result.append(
            (
                strategy,
                Metrics(
                    passengers=round(statistics.fmean(m.passengers for m in items)),
                    mean_wait_seconds=statistics.fmean(
                        m.mean_wait_seconds for m in items
                    ),
                    p95_wait_seconds=statistics.fmean(m.p95_wait_seconds for m in items),
                    long_wait_rate=statistics.fmean(m.long_wait_rate for m in items),
                    boarding_failure_rate=statistics.fmean(
                        m.boarding_failure_rate for m in items
                    ),
                    failed_boarding_events=round(
                        statistics.fmean(m.failed_boarding_events for m in items)
                    ),
                    mean_ride_seconds=statistics.fmean(m.mean_ride_seconds for m in items),
                    unserved_passengers=round(
                        statistics.fmean(m.unserved_passengers for m in items)
                    ),
                ),
            )
        )
    # Primary comparison rule: smaller p95 wait is better. The following
    # fields are explicit tie-breakers, so "better" is reproducible.
    result.sort(
        key=lambda item: (
            item[1].p95_wait_seconds,
            item[1].boarding_failure_rate,
            item[1].long_wait_rate,
            item[1].mean_wait_seconds,
        )
    )
    return result


def print_summary(
    rows: Sequence[ExperimentRow],
    runs: int,
    traffic_patterns: Sequence[str] = ("uniform",),
    traffic_levels: Mapping[str, float] | None = None,
) -> None:
    if traffic_levels is None:
        traffic_levels = {
            row.traffic_level: row.arrival_rate_per_second for row in rows
        }
    print(f"\nAverage over {runs} paired run(s)")
    print("Primary ranking metric: p95 waiting time (lower is better)")
    for traffic_level, arrival_rate in traffic_levels.items():
        print(
            f"\nDemand level: {traffic_level} "
            f"({arrival_rate:.3f} passengers/s, {arrival_rate * 60:.1f}/min)"
        )
        for traffic_pattern in traffic_patterns:
            aggregated = aggregate_rows(
                rows, traffic_level, traffic_pattern
            )
            print(
                f"\nTraffic pattern: {traffic_pattern} "
                f"({TRAFFIC_PATTERN_LABELS[traffic_pattern]})\n"
            )
            header = (
                f"{'rank':>4}  {'strategy':<13} {'passengers':>10} "
                f"{'mean(s)':>9} {'p95(s)':>9} {'>60s':>8} "
                f"{'failed':>8} {'ride(s)':>9}"
            )
            print(header)
            print("-" * len(header))
            for rank, (strategy, metrics) in enumerate(
                aggregated, start=1
            ):
                print(
                    f"{rank:>4}  {strategy:<13} {metrics.passengers:>10} "
                    f"{metrics.mean_wait_seconds:>9.2f} "
                    f"{metrics.p95_wait_seconds:>9.2f} "
                    f"{metrics.long_wait_rate:>7.1%} "
                    f"{metrics.boarding_failure_rate:>7.1%} "
                    f"{metrics.mean_ride_seconds:>9.2f}"
                )
            if any(
                metrics.unserved_passengers for _, metrics in aggregated
            ):
                print("\nWarning: at least one passenger was not served.")


def write_csv_results(
    rows: Sequence[ExperimentRow], output_dir: str | Path
) -> tuple[Path, Path]:
    """Save machine-readable run-level data and grouped summary statistics."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    runs_path = destination / "elevator_experiment_runs.csv"
    summary_path = destination / "elevator_experiment_summary.csv"

    raw_fields = [
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
    with runs_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=raw_fields)
        writer.writeheader()
        for row in rows:
            metrics = row.metrics
            writer.writerow(
                {
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

    summary_fields = [
        "traffic_level",
        "arrival_rate_per_second",
        "traffic_pattern",
        "rank_by_mean_p95",
        "strategy",
        "runs",
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
        "mean_long_wait_rate",
        "sd_long_wait_rate",
        "mean_boarding_failure_rate",
        "sd_boarding_failure_rate",
        "mean_failed_boarding_events",
        "mean_ride_seconds",
        "mean_unserved_passengers",
    ]
    level_patterns = list(
        dict.fromkeys((row.traffic_level, row.traffic_pattern) for row in rows)
    )
    with summary_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=summary_fields)
        writer.writeheader()
        for traffic_level, traffic_pattern in level_patterns:
            aggregated = aggregate_rows(rows, traffic_level, traffic_pattern)
            for rank, (strategy, metrics) in enumerate(aggregated, start=1):
                items = [
                    row.metrics
                    for row in rows
                    if row.traffic_level == traffic_level
                    and row.traffic_pattern == traffic_pattern
                    and row.strategy == strategy
                ]
                settings_row = next(
                    row
                    for row in rows
                    if row.traffic_level == traffic_level
                )
                writer.writerow(
                    {
                        "traffic_level": traffic_level,
                        "arrival_rate_per_second": (
                            settings_row.arrival_rate_per_second
                        ),
                        "traffic_pattern": traffic_pattern,
                        "rank_by_mean_p95": rank,
                        "strategy": strategy,
                        "runs": len(items),
                        "floor_height_m": settings_row.floor_height_m,
                        "max_speed_mps": settings_row.max_speed_mps,
                        "acceleration_mps2": (
                            settings_row.acceleration_mps2
                        ),
                        "base_door_seconds": settings_row.base_door_seconds,
                        "boarding_seconds_per_passenger": (
                            settings_row.boarding_seconds_per_passenger
                        ),
                        "alighting_seconds_per_passenger": (
                            settings_row.alighting_seconds_per_passenger
                        ),
                        "mean_passengers": statistics.fmean(
                            item.passengers for item in items
                        ),
                        "mean_wait_seconds": metrics.mean_wait_seconds,
                        "sd_mean_wait_seconds": _sample_stdev(
                            item.mean_wait_seconds for item in items
                        ),
                        "mean_p95_wait_seconds": metrics.p95_wait_seconds,
                        "sd_p95_wait_seconds": _sample_stdev(
                            item.p95_wait_seconds for item in items
                        ),
                        "mean_long_wait_rate": metrics.long_wait_rate,
                        "sd_long_wait_rate": _sample_stdev(
                            item.long_wait_rate for item in items
                        ),
                        "mean_boarding_failure_rate": (
                            metrics.boarding_failure_rate
                        ),
                        "sd_boarding_failure_rate": _sample_stdev(
                            item.boarding_failure_rate for item in items
                        ),
                        "mean_failed_boarding_events": statistics.fmean(
                            item.failed_boarding_events for item in items
                        ),
                        "mean_ride_seconds": metrics.mean_ride_seconds,
                        "mean_unserved_passengers": statistics.fmean(
                            item.unserved_passengers for item in items
                        ),
                    }
                )
    return runs_path.resolve(), summary_path.resolve()


def _sample_stdev(values: Iterable[float]) -> float:
    observations = list(values)
    return statistics.stdev(observations) if len(observations) > 1 else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=int, default=1_800, help="arrival window (s)")
    parser.add_argument(
        "--arrival-rate",
        type=float,
        default=None,
        help="custom single demand rate; overrides --traffic-levels",
    )
    parser.add_argument(
        "--traffic-levels",
        nargs="+",
        choices=tuple(DEFAULT_TRAFFIC_LEVELS),
        default=list(DEFAULT_TRAFFIC_LEVELS),
        help="demand intensity levels to compare",
    )
    parser.add_argument(
        "--normal-rate",
        type=float,
        default=DEFAULT_TRAFFIC_LEVELS["normal"],
        help="normal demand arrivals per second",
    )
    parser.add_argument(
        "--congested-rate",
        type=float,
        default=DEFAULT_TRAFFIC_LEVELS["congested"],
        help="congested demand arrivals per second",
    )
    parser.add_argument("--runs", type=int, default=30, help="paired repetitions")
    parser.add_argument("--seed", type=int, default=202410189)
    parser.add_argument(
        "--output-dir",
        default="output",
        help="directory for run-level and summary CSV files",
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="print results without writing CSV files",
    )
    parser.add_argument(
        "--traffic-patterns",
        nargs="+",
        choices=TRAFFIC_PATTERNS,
        default=["up_peak", "interfloor", "down_peak"],
        help="origin/destination patterns to compare",
    )
    parser.add_argument(
        "--peak-share",
        type=float,
        default=0.85,
        help="dominant lobby flow share for up/down peak patterns",
    )
    parser.add_argument(
        "--floor-height-m",
        type=float,
        default=3.5,
        help="vertical distance between adjacent floors in metres",
    )
    parser.add_argument(
        "--max-speed-mps",
        type=float,
        default=2.5,
        help="elevator maximum speed in metres per second",
    )
    parser.add_argument(
        "--acceleration-mps2",
        type=float,
        default=1.0,
        help="symmetric acceleration and braking rate in metres per second squared",
    )
    parser.add_argument(
        "--base-door-seconds",
        type=int,
        default=4,
        help="fixed door opening/closing time at each stop",
    )
    parser.add_argument(
        "--boarding-seconds-per-passenger",
        type=float,
        default=0.8,
        help="additional dwell time for each boarding passenger",
    )
    parser.add_argument(
        "--alighting-seconds-per-passenger",
        type=float,
        default=0.6,
        help="additional dwell time for each alighting passenger",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=sorted(ElevatorSimulation.VALID_STRATEGIES),
        default=["closest", "eta", "priority_eta"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.arrival_rate is not None:
        traffic_levels = {"custom": args.arrival_rate}
    else:
        available_rates = {
            "normal": args.normal_rate,
            "congested": args.congested_rate,
        }
        traffic_levels = {
            name: available_rates[name] for name in args.traffic_levels
        }
    config = SimulationConfig(
        duration_seconds=args.duration,
        arrival_rate_per_second=next(iter(traffic_levels.values())),
        dominant_flow_share=args.peak_share,
        floor_height_m=args.floor_height_m,
        max_speed_mps=args.max_speed_mps,
        acceleration_mps2=args.acceleration_mps2,
        door_seconds=args.base_door_seconds,
        boarding_seconds_per_passenger=(
            args.boarding_seconds_per_passenger
        ),
        alighting_seconds_per_passenger=(
            args.alighting_seconds_per_passenger
        ),
    )
    rows = run_experiment(
        config,
        args.strategies,
        args.runs,
        args.seed,
        args.traffic_patterns,
        traffic_levels,
    )
    print_summary(rows, args.runs, args.traffic_patterns, traffic_levels)
    if not args.no_csv:
        runs_path, summary_path = write_csv_results(rows, args.output_dir)
        print(f"\nSaved run-level CSV: {runs_path}")
        print(f"Saved summary CSV: {summary_path}")


if __name__ == "__main__":
    main()
