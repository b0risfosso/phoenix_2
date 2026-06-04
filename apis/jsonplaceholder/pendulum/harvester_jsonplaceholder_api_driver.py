#!/usr/bin/env python3
"""
JSONPlaceholder + Pendulum Energy Harvester Driver

Place this file in the same folder as:
    pendulum_energy_harvester_simulation.py

Run:
    python harvester_jsonplaceholder_api_driver.py

The API controls each harvester round by converting JSONPlaceholder /todos
into PendulumParams, then the original harvester AI controller still reacts
after rounds. No external packages are required.
"""
from __future__ import annotations

import json
import random
import urllib.error
import urllib.request
from dataclasses import replace
from typing import Any, Dict, List

import pendulum_energy_harvester_simulation as sim

BASE_URL = "https://jsonplaceholder.typicode.com"


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def request_json(endpoint: str, timeout_s: float = 8.0) -> Any:
    request = urllib.request.Request(
        BASE_URL + endpoint,
        headers={"Accept": "application/json", "User-Agent": "harvester-api-driver/1.0"},
        method="GET",
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        return json.loads(response.read().decode("utf-8"))


def get_api_todos(limit: int) -> List[Dict[str, Any]]:
    try:
        return request_json("/todos")[:limit]
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as error:
        print("\nAPI request failed. Using local fallback todo items.")
        print(f"Reason: {error}")
        return [
            {"userId": 1, "id": 1, "title": "fallback light generator baseline", "completed": False},
            {"userId": 2, "id": 2, "title": "fallback high energy wide swing", "completed": True},
            {"userId": 3, "id": 3, "title": "fallback strong load extraction", "completed": False},
            {"userId": 4, "id": 4, "title": "fallback long running low load", "completed": True},
            {"userId": 5, "id": 5, "title": "fallback narrow pickup coil", "completed": False},
            {"userId": 6, "id": 6, "title": "fallback storage fill test", "completed": True},
            {"userId": 7, "id": 7, "title": "fallback fast short harvester", "completed": False},
            {"userId": 8, "id": 8, "title": "fallback efficient balanced harvest", "completed": True},
        ][:limit]


def map_todo_to_params(todo: Dict[str, Any], base: sim.PendulumParams, round_number: int) -> sim.PendulumParams:
    """Convert one JSONPlaceholder todo item into energy-harvester parameters."""
    p = replace(base)
    todo_id = int(todo.get("id", round_number))
    user_id = int(todo.get("userId", 1))
    title = str(todo.get("title", "untitled harvester experiment"))
    completed = bool(todo.get("completed", False))

    title_score = min(1.0, len(title) / 80.0)
    id_cycle = todo_id % 10
    user_cycle = user_id % 5

    if completed:
        # A completed API item becomes an efficient, controlled harvesting setup.
        p.gravity = clamp(9.20 + 0.20 * user_cycle + 0.05 * id_cycle, 1.0, 15.0)
        p.length = clamp(1.00 + 0.85 * title_score + 0.04 * user_cycle, 0.25, 3.0)
        p.mass = clamp(0.90 + 0.10 * user_cycle, 0.1, 10.0)
        p.damping = clamp(0.010 + 0.006 * user_cycle + 0.004 * title_score, 0.0, 0.25)
        p.initial_angle_deg = clamp(42.0 + 35.0 * title_score + 1.0 * id_cycle, -120.0, 120.0)
        p.initial_omega = clamp(-0.25 + 0.06 * id_cycle, -2.5, 2.5)
        p.generator_load = clamp(0.040 + 0.010 * id_cycle + 0.018 * title_score, 0.0, 0.35)
        p.generator_efficiency = clamp(0.68 + 0.03 * user_cycle, 0.05, 0.95)
        p.engagement_angle_deg = clamp(18.0 + 24.0 * title_score, 3.0, 65.0)
        p.cut_in_speed = clamp(0.12 + 0.025 * user_cycle, 0.01, 1.5)
        p.storage_capacity_j = clamp(4.0 + 7.0 * title_score + 0.4 * id_cycle, 0.05, 20.0)
    else:
        # An incomplete API item becomes a more exploratory or lossy harvesting setup.
        p.gravity = clamp(8.20 + 0.30 * user_cycle + 0.12 * id_cycle, 1.0, 15.0)
        p.length = clamp(0.70 + 1.30 * title_score + 0.02 * id_cycle, 0.25, 3.0)
        p.mass = clamp(0.80 + 0.08 * user_cycle + 0.02 * id_cycle, 0.1, 10.0)
        p.damping = clamp(0.018 + 0.010 * user_cycle + 0.006 * (1.0 - title_score), 0.0, 0.25)
        p.initial_angle_deg = clamp(35.0 + 48.0 * title_score + 1.5 * id_cycle, -120.0, 120.0)
        p.initial_omega = clamp(-0.60 + 0.12 * id_cycle, -2.5, 2.5)
        p.generator_load = clamp(0.070 + 0.014 * id_cycle + 0.020 * title_score, 0.0, 0.35)
        p.generator_efficiency = clamp(0.48 + 0.025 * user_cycle + 0.05 * title_score, 0.05, 0.95)
        p.engagement_angle_deg = clamp(10.0 + 28.0 * title_score + 1.0 * user_cycle, 3.0, 65.0)
        p.cut_in_speed = clamp(0.18 + 0.035 * user_cycle + 0.01 * id_cycle, 0.01, 1.5)
        p.storage_capacity_j = clamp(2.0 + 5.5 * title_score + 0.30 * id_cycle, 0.05, 20.0)

    p.storage_initial_j = 0.0
    p.storage_capacitance_f = clamp(0.75 + 0.15 * user_cycle + 0.50 * title_score, 0.05, 100.0)
    return p


def blend_params(api_params: sim.PendulumParams, ai_params: sim.PendulumParams, api_weight: float = 0.70) -> sim.PendulumParams:
    """Blend API-selected parameters with the original AI controller's adjustment."""
    ai_weight = 1.0 - api_weight
    return sim.PendulumParams(
        gravity=api_params.gravity * api_weight + ai_params.gravity * ai_weight,
        length=api_params.length * api_weight + ai_params.length * ai_weight,
        mass=api_params.mass * api_weight + ai_params.mass * ai_weight,
        damping=api_params.damping * api_weight + ai_params.damping * ai_weight,
        initial_angle_deg=api_params.initial_angle_deg * api_weight + ai_params.initial_angle_deg * ai_weight,
        initial_omega=api_params.initial_omega * api_weight + ai_params.initial_omega * ai_weight,
        generator_load=api_params.generator_load * api_weight + ai_params.generator_load * ai_weight,
        generator_efficiency=api_params.generator_efficiency * api_weight + ai_params.generator_efficiency * ai_weight,
        engagement_angle_deg=api_params.engagement_angle_deg * api_weight + ai_params.engagement_angle_deg * ai_weight,
        cut_in_speed=api_params.cut_in_speed * api_weight + ai_params.cut_in_speed * ai_weight,
        storage_capacity_j=api_params.storage_capacity_j * api_weight + ai_params.storage_capacity_j * ai_weight,
        storage_initial_j=0.0,
        storage_capacitance_f=api_params.storage_capacitance_f * api_weight + ai_params.storage_capacitance_f * ai_weight,
    )


def print_api_items(todos: List[Dict[str, Any]]) -> None:
    print("\nJSONPlaceholder todo items controlling this harvester run:")
    print("-" * 104)
    for index, todo in enumerate(todos, start=1):
        completed = bool(todo.get("completed", False))
        mode = "efficient harvest" if completed else "exploratory harvest"
        print(
            f"{index:>2}. id={todo.get('id'):<3} userId={todo.get('userId'):<2} "
            f"completed={str(completed):<5} mode={mode:<20} title={todo.get('title')}"
        )


def print_final_summary(summaries: List[sim.RoundSummary]) -> None:
    print("\n" + "=" * 132)
    print("FINAL JSONPLACEHOLDER HARVESTER COMPARISON")
    print("=" * 132)
    print(f"{'Round':>5} | {'g':>6} | {'L':>6} | {'damp':>7} | {'load':>7} | {'eff':>5} | {'engage':>7} | {'start':>8} | {'harvest':>9} | {'avg_W':>8} | {'peak_W':>8} | {'gen_s':>7} | note")
    print("-" * 132)
    for s in summaries:
        print(
            f"{s.round_number:5d} | {s.params.gravity:6.2f} | {s.params.length:6.2f} | "
            f"{s.params.damping:7.3f} | {s.params.generator_load:7.3f} | "
            f"{s.params.generator_efficiency:5.2f} | {s.params.engagement_angle_deg:7.1f} | "
            f"{s.params.initial_angle_deg:8.2f} | {s.harvested_j:9.4f} | "
            f"{s.average_electrical_power_w:8.5f} | {s.peak_electrical_power_w:8.5f} | "
            f"{s.generator_on_time_s:7.2f} | {s.stability_note}"
        )
    best_harvest = max(summaries, key=lambda s: s.harvested_j)
    best_power = max(summaries, key=lambda s: s.peak_electrical_power_w)
    longest_on = max(summaries, key=lambda s: s.generator_on_time_s)
    print("\nAPI-driven harvester observations:")
    print(f"  Most harvested energy: Round {best_harvest.round_number}, {best_harvest.harvested_j:.4f} J")
    print(f"  Highest peak electrical power: Round {best_power.round_number}, {best_power.peak_electrical_power_w:.5f} W")
    print(f"  Longest generator activity: Round {longest_on.round_number}, {longest_on.generator_on_time_s:.2f} s")


def main() -> None:
    print("JSONPlaceholder-driven Pendulum Energy Harvester Simulation")
    print("The API provides external experiment seeds; the original harvester AI controller still adapts after each round.")

    config = sim.SimConfig(rounds=8, duration=8.0, dt=0.02, print_every_steps=25, sleep_between_prints=0.0, seed=7)
    rng = random.Random(config.seed)
    ai = sim.PendulumAIController(rng)

    todos = get_api_todos(config.rounds)
    print_api_items(todos)

    params = sim.PendulumParams()
    previous_summary = None
    summaries: List[sim.RoundSummary] = []

    for round_number, todo in enumerate(todos, start=1):
        print("\n" + "#" * 120)
        print(f"API CONTROL FOR HARVESTER ROUND {round_number}")
        print("#" * 120)

        api_params = map_todo_to_params(todo, params, round_number)
        print(
            "API-selected harvester setup: "
            f"g={api_params.gravity:.3f}, L={api_params.length:.3f}, damping={api_params.damping:.4f}, "
            f"start_angle={api_params.initial_angle_deg:.2f}, omega0={api_params.initial_omega:.3f}, "
            f"generator_load={api_params.generator_load:.4f}, efficiency={api_params.generator_efficiency:.2f}, "
            f"engagement=±{api_params.engagement_angle_deg:.1f}, cut_in={api_params.cut_in_speed:.2f}, "
            f"storage_capacity={api_params.storage_capacity_j:.2f} J"
        )

        summary = sim.simulate_round(round_number, api_params, config)
        summaries.append(summary)

        if round_number < config.rounds:
            ai_params = ai.choose_next(previous_summary or summary, api_params, round_number + 1)
            params = blend_params(api_params, ai_params, api_weight=0.70)
        previous_summary = summary

    print_final_summary(summaries)


if __name__ == "__main__":
    main()
