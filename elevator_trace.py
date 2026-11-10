"""Observe the frozen simulator without changing its decisions.

Events, display frames and passenger response times are instrumentation only.
The display interpolates discrete travel legs; boarding/alighting timestamps
and all analysis outcomes retain the original one-second model's semantics.
"""

from dataclasses import asdict
import math

from ablation_experiment import AblationSimulation


class TracedSimulation(AblationSimulation):
    def __init__(self, *args, capture_frames=False, capture_events=False, **kwargs):
        self.capture_frames = capture_frames
        self.capture_events = capture_events
        self.frames = []
        self.events = []
        self.first_service_time = {}
        self.reassignments_while_waiting = {}
        self.now = 0
        self.completed = 0
        self.arrived = 0
        self.active_legs = {}
        self.stop_counts = {}
        super().__init__(*args, **kwargs)

    def _event(self, event):
        if self.capture_events:
            self.events.append({"time": self.now, **event})

    def _add_arrivals(self, now):
        self.now = now
        ids = [demand.passenger_id for demand in self.arrivals.get(now, [])]
        super()._add_arrivals(now)
        self.arrived += len(ids)
        if ids:
            self._event({"type": "arrive", "passengers": ids})

    def _maybe_redispatch_priority_calls(self, now):
        before = {key: call.assigned_elevator_id for key, call in self.calls.items()}
        previous_rounds = self.redispatch_rounds
        super()._maybe_redispatch_priority_calls(now)
        if self.redispatch_rounds == previous_rounds:
            return
        changes = []
        for key, call in self.calls.items():
            if before[key] != call.assigned_elevator_id:
                for pid in call.waiting_passenger_ids:
                    self.reassignments_while_waiting[pid] = self.reassignments_while_waiting.get(pid, 0) + 1
                changes.append({"floor": call.floor, "direction": call.direction,
                                "from_elevator": before[key], "to_elevator": call.assigned_elevator_id,
                                "passengers": list(call.waiting_passenger_ids)})
        self._event({"type": "redispatch", "changes": changes})

    def _service_floor(self, elevator, now):
        before_onboard = set(elevator.onboard_passenger_ids)
        waiting = [pid for key in elevator.assigned_calls if key in self.calls and key[0] == elevator.current_floor
                   for pid in self.calls[key].waiting_passenger_ids]
        for pid in waiting:
            self.first_service_time.setdefault(pid, now)
        super()._service_floor(elevator, now)
        after_onboard = set(elevator.onboard_passenger_ids)
        boarded = sorted(after_onboard - before_onboard)
        alighted = sorted(before_onboard - after_onboard)
        rejected = sorted(set(waiting) - set(boarded))
        self.completed += len(alighted)
        self.stop_counts[elevator.current_floor] = self.stop_counts.get(elevator.current_floor, 0) + 1
        self._event({"type": "stop", "elevator": elevator.elevator_id,
                     "floor": elevator.current_floor, "boarded": boarded,
                     "alighted": alighted, "rejected": rejected,
                     "load_after": len(after_onboard)})

    def _step_elevator(self, elevator, now):
        if elevator.move_remaining == 1:
            self.active_legs.pop(elevator.elevator_id, None)
        super()._step_elevator(elevator, now)
        if elevator.move_remaining > 0 and elevator.elevator_id not in self.active_legs:
            leg = {"from_floor": elevator.current_floor, "target_floor": elevator.target_floor,
                   "start_time": now, "duration": elevator.move_remaining}
            self.active_legs[elevator.elevator_id] = leg
            self._event({"type": "depart", "elevator": elevator.elevator_id,
                         "load": len(elevator.onboard_passenger_ids), **leg})
        if self.capture_frames and elevator.elevator_id == self.config.elevator_count - 1:
            self._capture_frame(now)

    def _display_floor(self, elevator, now):
        leg = self.active_legs.get(elevator.elevator_id)
        if not leg:
            return float(elevator.current_floor)
        span = abs(leg["target_floor"] - leg["from_floor"])
        distance = span * self.config.floor_height_m
        vmax, acceleration = self.config.max_speed_mps, self.config.acceleration_mps2
        peak = min(vmax, math.sqrt(distance * acceleration))
        ramp = peak / acceleration
        cruise = max(0, (distance - peak * peak / acceleration) / peak)
        total = 2 * ramp + cruise
        elapsed = max(0, min(1, (now - leg["start_time"]) / leg["duration"])) * total
        if elapsed < ramp:
            travelled = .5 * acceleration * elapsed * elapsed
        elif elapsed <= ramp + cruise:
            travelled = .5 * acceleration * ramp * ramp + peak * (elapsed - ramp)
        else:
            travelled = distance - .5 * acceleration * (total - elapsed) ** 2
        direction = 1 if leg["target_floor"] > leg["from_floor"] else -1
        return leg["from_floor"] + direction * travelled / self.config.floor_height_m

    def _capture_frame(self, now):
        queues = [[0, 0, 0.0] for _ in range(self.config.floors)]
        waiting = 0
        for call in self.calls.values():
            floor = queues[call.floor - 1]
            floor[0 if call.direction == 1 else 1] += len(call.waiting_passenger_ids)
            if call.waiting_passenger_ids:
                floor[2] = max(floor[2], now - min(self.passengers[pid].arrival_time for pid in call.waiting_passenger_ids))
            waiting += len(call.waiting_passenger_ids)
        cars = [[round(self._display_floor(elevator, now), 4), len(elevator.onboard_passenger_ids),
                 elevator.target_floor, elevator.door_remaining, elevator.direction,
                 list(elevator.onboard_passenger_ids)] for elevator in self.elevators]
        onboard = sum(car[1] for car in cars)
        if waiting + onboard + self.completed != self.arrived:
            raise RuntimeError("Passenger conservation failed in trace frame")
        self.frames.append({"cars": cars, "queues": [[u, d, round(age, 2)] for u, d, age in queues],
                            "counts": [waiting, onboard, self.completed, self.arrived]})

    def response_records(self):
        rows = []
        for passenger in self.passengers.values():
            if passenger.alighting_time is None or passenger.passenger_id not in self.first_service_time:
                raise RuntimeError("Trace includes an incomplete passenger journey")
            first = self.first_service_time[passenger.passenger_id]
            before = first - passenger.arrival_time
            after = passenger.boarding_time - first
            if before < 0 or after < 0:
                raise RuntimeError("Invalid response-time decomposition")
            rows.append({"passenger_id": passenger.passenger_id, "origin": passenger.origin,
                         "first_service_time": first, "before_first_service_seconds": before,
                         "after_first_service_seconds": after,
                         "wait_seconds": passenger.boarding_time - passenger.arrival_time,
                         "reassignments_while_waiting": self.reassignments_while_waiting.get(passenger.passenger_id, 0),
                         "failed_boardings": passenger.failed_boardings})
        return rows

    def replay_payload(self, metrics):
        return {"strategy": self.strategy, "order_seed": self.order_seed if self.strategy == "redispatch_eta" else None,
                "config": asdict(self.config), "metrics": asdict(metrics), "frames": self.frames,
                "events": self.events, "passengers": [asdict(p) for p in self.passengers.values()]}
