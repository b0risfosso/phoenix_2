#!/usr/bin/env python3
"""
JSONPlaceholder + Pendulum AI Rounds Driver

Place this file in the same folder as:
    pendulum_ai_rounds_simulation_returned.py

Run:
    python pendulum_jsonplaceholder_api_driver.py

The API controls each pendulum round by converting JSONPlaceholder /todos
into PendulumParams, then the original AI controller still reacts after rounds.
No external packages are required.
"""
from __future__ import annotations

import json
import random
import urllib.error
import urllib.request
from dataclasses import replace
from typing import Any, Dict, List

import pendulum_ai_rounds_simulation_returned as sim

BASE_URL = "https://jsonplaceholder.typicode.com"


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def request_json(endpoint: str, timeout_s: float = 8.0) -> Any:
    request = urllib.request.Request(
        BASE_URL + endpoint,
        headers={"Accept": "application/json", "User-Agent": "pendulum-api-driver/1.0"},
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
            {"userId": 1, "id": 1, "title": "fallback slow wide swing", "completed": False},
            {"userId": 2, "id": 2, "title": "fallback fast short swing", "completed": True},
            {"userId": 3, "id": 3, "title": "fallback heavy damping test", "completed": False},
            {"userId": 4, "id": 4, "title": "fallback low gravity arc", "completed": True},
            {"userId": 5, "id": 5, "title": "fallback energy boost swing", "completed": False},
            {"userId": 6, "id": 6, "title": "fallback near settle experiment", "completed": True},
            {"userId": 7, "id": 7, "title": "fallback long pendulum period", "completed": False},
            {"userId": 8, "id": 8, "title": "fallback controlled stable oscillation", "completed": True},
        ][:limit]


def map_todo_to_params(todo: Dict[str, Any], base: sim.PendulumParams, round_number: int) -> sim.PendulumParams:
    """Convert one JSONPlaceholder todo item into pendulum parameters."""
    p = replace(base)
    todo_id = int(todo.get("id", round_number))
    user_id = int(todo.get("userId", 1))
    title = str(todo.get("title", "untitled pendulum experiment"))
    completed = bool(todo.get("completed", False))

    title_score = min(1.0, len(title) / 80.0)
    id_cycle = todo_id % 10
    user_cycle = user_id % 5

    if completed:
        # A completed API item becomes a cleaner, controlled swing.
        p.gravity = clamp(8.8 + 0.35 * user_cycle + 0.08 * id_cycle, 1.0, 15.0)
        p.length = clamp(0.90 + 0.85 * title_score + 0.03 * id_cycle, 0.25, 3.0)
        p.mass = clamp(0.80 + 0.12 * user_cycle, 0.1, 10.0)
        p.damping = clamp(0.035 + 0.010 * user_cycle + 0.005 * title_score, 0.0, 0.25)
        p.initial_angle_deg = clamp(18.0 + 28.0 * title_score + 1.5 * id_cycle, -120.0, 120.0)
        p.initial_omega = clamp(-0.25 + 0.05 * id_cycle, -2.5, 2.5)
    else:
        # An incomplete API item becomes a more exploratory swing.
        p.gravity = clamp(5.5 + 0.60 * user_cycle + 0.35 * id_cycle, 1.0, 15.0)
        p.length = clamp(0.45 + 1.65 * title_score + 0.04 * id_cycle, 0.25, 3.0)
        p.mass = clamp(0.70 + 0.10 * user_cycle + 0.03 * id_cycle, 0.1, 10.0)
        p.damping = clamp(0.010 + 0.020 * user_cycle + 0.008 * (1.0 - title_score), 0.0, 0.25)
        p.initial_angle_deg = clamp(28.0 + 55.0 * title_score + 2.0 * id_cycle, -120.0, 120.0)
        p.initial_omega = clamp(-0.90 + 0.20 * id_cycle, -2.5, 2.5)

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
    )


def print_api_items(todos: List[Dict[str, Any]]) -> None:
    print("\nJSONPlaceholder todo items controlling this pendulum run:")
    print("-" * 104)
    for index, todo in enumerate(todos, start=1):
        completed = bool(todo.get("completed", False))
        mode = "controlled swing" if completed else "exploratory swing"
        print(
            f"{index:>2}. id={todo.get('id'):<3} userId={todo.get('userId'):<2} "
            f"completed={str(completed):<5} mode={mode:<18} title={todo.get('title')}"
        )


def print_final_summary(summaries: List[sim.RoundSummary]) -> None:
    print("\n" + "=" * 108)
    print("FINAL JSONPLACEHOLDER PENDULUM COMPARISON")
    print("=" * 108)
    print(f"{'Round':>5} | {'g':>6} | {'L':>6} | {'damp':>7} | {'start':>8} | {'omega0':>8} | {'max_deg':>8} | {'max_v':>8} | {'period':>8} | note")
    print("-" * 108)
    for s in summaries:
        period = "--" if s.estimated_period is None else f"{s.estimated_period:.3f}"
        print(
            f"{s.round_number:5d} | {s.params.gravity:6.2f} | {s.params.length:6.2f} | "
            f"{s.params.damping:7.3f} | {s.params.initial_angle_deg:8.2f} | "
            f"{s.params.initial_omega:8.3f} | {s.max_angle_deg:8.2f} | "
            f"{s.max_speed:8.3f} | {period:>8} | {s.stability_note}"
        )
    fastest = max(summaries, key=lambda s: s.max_speed)
    widest = max(summaries, key=lambda s: s.max_angle_deg)
    settled = min(summaries, key=lambda s: abs(s.final_angle_deg) + abs(s.final_omega) * 10.0)
    print("\nAPI-driven observations:")
    print(f"  Fastest swing: Round {fastest.round_number}, max speed {fastest.max_speed:.3f} m/s")
    print(f"  Widest swing: Round {widest.round_number}, max angle {widest.max_angle_deg:.2f} deg")
    print(f"  Closest to settled: Round {settled.round_number}, final angle {settled.final_angle_deg:.2f} deg")


def main() -> None:
    print("JSONPlaceholder-driven Pendulum AI Rounds Simulation")
    print("The API provides external experiment seeds; the original AI controller still adapts after each round.")

    config = sim.SimConfig(rounds=8, duration=8.0, dt=0.02, print_every_steps=25, sleep_between_prints=0.0, seed=7)
    rng = random.Random(config.seed)
    ai = sim.PendulumAIController(rng)

    todos = get_api_todos(config.rounds)
    print_api_items(todos)

    params = sim.PendulumParams()
    previous_summary = None
    summaries: List[sim.RoundSummary] = []

    for round_number, todo in enumerate(todos, start=1):
        print("\n" + "#" * 104)
        print(f"API CONTROL FOR PENDULUM ROUND {round_number}")
        print("#" * 104)

        api_params = map_todo_to_params(todo, params, round_number)
        print(
            "API-selected pendulum setup: "
            f"g={api_params.gravity:.3f}, L={api_params.length:.3f}, mass={api_params.mass:.3f}, "
            f"damping={api_params.damping:.4f}, start_angle={api_params.initial_angle_deg:.2f}, "
            f"omega0={api_params.initial_omega:.3f}"
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
