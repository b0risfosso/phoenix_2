#!/usr/bin/env python3
"""
JSONPlaceholder + Terminal Water Flow AI Rounds Driver
------------------------------------------------------

This script lets the JSONPlaceholder API interact with:
    terminal_water_flow_ai_rounds.py

Place this file in the same folder as:
    terminal_water_flow_ai_rounds.py

Run:
    python water_jsonplaceholder_api_driver.py

What it does:
- Downloads fake todo items from JSONPlaceholder.
- Converts each todo item into WaterParams.
- Runs the original water-flow run_round() function using API-selected settings.
- Uses the original WaterAIController after each round, then blends API choices
  with AI adjustments.
- Falls back to local fake API items if the network is unavailable.

No external packages are required.
"""

from __future__ import annotations

import json
import random
import urllib.error
import urllib.request
from dataclasses import replace
from typing import Any, Dict, List

import terminal_water_flow_ai_rounds as water_sim


BASE_URL = "https://jsonplaceholder.typicode.com"


# -----------------------------------------------------------------------------
# API ACCESS
# -----------------------------------------------------------------------------

def request_json(endpoint: str, timeout_s: float = 8.0) -> Any:
    """Request JSON from JSONPlaceholder using only the Python standard library."""
    url = BASE_URL + endpoint

    request = urllib.request.Request(
        url=url,
        headers={
            "Accept": "application/json",
            "User-Agent": "water-jsonplaceholder-driver/1.0",
        },
        method="GET",
    )

    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        body = response.read().decode("utf-8")
        return json.loads(body)


def get_api_todos(limit: int) -> List[Dict[str, Any]]:
    """Download todo items from JSONPlaceholder, or use local fallback items."""
    try:
        todos = request_json("/todos")
        return todos[:limit]

    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as error:
        print("\nAPI request failed. Using local fallback todo items.")
        print(f"Reason: {error}")

        return [
            {"userId": 1, "id": 1, "title": "fallback gentle basin fill", "completed": False},
            {"userId": 2, "id": 2, "title": "fallback fast river slope", "completed": True},
            {"userId": 3, "id": 3, "title": "fallback maze flow branching", "completed": False},
            {"userId": 4, "id": 4, "title": "fallback sponge absorption test", "completed": True},
            {"userId": 5, "id": 5, "title": "fallback overflow stress test", "completed": False},
            {"userId": 6, "id": 6, "title": "fallback misty evaporation basin", "completed": True},
            {"userId": 7, "id": 7, "title": "fallback deep pooling experiment", "completed": False},
            {"userId": 8, "id": 8, "title": "fallback stable controlled water run", "completed": True},
        ][:limit]


# -----------------------------------------------------------------------------
# API DATA TO WATER PARAMETERS
# -----------------------------------------------------------------------------

def clamp_int(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def clamp_float(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def choose_obstacle_mode(todo_id: int, completed: bool, title: str) -> str:
    """
    Choose terrain mode from API data.

    completed=True tends to use more controlled modes.
    completed=False tends to use more stressful or exploratory modes.
    """
    title_lower = title.lower()

    if "sponge" in title_lower or "absorb" in title_lower:
        return "sponge"
    if "maze" in title_lower:
        return "maze"
    if "overflow" in title_lower:
        return "overflow"
    if "slope" in title_lower or "river" in title_lower:
        return "slope"

    if completed:
        modes = ["basin", "slope", "sponge", "maze"]
    else:
        modes = ["basin", "maze", "overflow", "slope", "sponge"]

    return modes[todo_id % len(modes)]


def map_todo_to_water_params(
    todo: Dict[str, Any],
    base: water_sim.WaterParams,
    config: water_sim.SimConfig,
    round_number: int,
) -> water_sim.WaterParams:
    """
    Convert one JSONPlaceholder todo item into WaterParams.

    Mapping idea:
    - completed=True means a more controlled water experiment.
    - completed=False means a more chaotic or stress-test water experiment.
    - title length changes source strength, splash, evaporation, and terrain.
    - userId changes absorption/leakiness/source position.
    - id changes gravity flow, lateral flow, and mode selection.
    """

    p = replace(base)

    todo_id = int(todo.get("id", round_number))
    user_id = int(todo.get("userId", 1))
    title = str(todo.get("title", "untitled water experiment"))
    completed = bool(todo.get("completed", False))

    title_len = len(title)
    title_score = min(1.0, title_len / 80.0)
    id_cycle = todo_id % 10
    user_cycle = user_id % 5

    p.obstacle_mode = choose_obstacle_mode(todo_id, completed, title)

    if completed:
        # More controlled flow: lower source, cleaner spread, fewer losses.
        p.source_rate = clamp_int(round(4 + 4 * title_score + user_cycle), 1, 18)
        p.gravity_flow = clamp_int(3 + (todo_id % 4), 1, 12)
        p.lateral_flow = clamp_int(1 + (todo_id % 3), 1, 8)
        p.splash_chance = clamp_float(0.015 + 0.045 * title_score, 0.0, 0.4)
        p.evaporation_rate = clamp_float(0.001 + 0.006 * (1.0 - title_score), 0.0, 0.08)
        p.absorption_rate = clamp_float(0.020 + 0.030 * user_cycle / 4.0, 0.0, 0.3)
        p.basin_leakiness = clamp_float(0.002 + 0.010 * (id_cycle / 10.0), 0.0, 0.1)
    else:
        # Exploratory flow: stronger source, more turbulence, more chance of overflow/loss.
        p.source_rate = clamp_int(round(6 + 8 * title_score + id_cycle * 0.4), 1, 18)
        p.gravity_flow = clamp_int(4 + (todo_id % 6), 1, 12)
        p.lateral_flow = clamp_int(2 + (todo_id % 4), 1, 8)
        p.splash_chance = clamp_float(0.040 + 0.120 * title_score, 0.0, 0.4)
        p.evaporation_rate = clamp_float(0.002 + 0.020 * title_score, 0.0, 0.08)
        p.absorption_rate = clamp_float(0.015 + 0.080 * user_cycle / 4.0, 0.0, 0.3)
        p.basin_leakiness = clamp_float(0.008 + 0.025 * (id_cycle / 10.0), 0.0, 0.1)

    # Source position moves across the grid based on API id/title.
    center = config.width // 2
    offset_seed = ((todo_id * 7 + title_len + user_id * 3) % 17) - 8
    p.source_x = clamp_int(center + offset_seed, 1, config.width - 2)

    # Long titles get slightly more cell capacity so the run can pool more deeply.
    p.max_cell_water = clamp_int(7 + round(4 * title_score), 3, 12)

    return p


def blend_params(
    api_params: water_sim.WaterParams,
    ai_params: water_sim.WaterParams,
    config: water_sim.SimConfig,
    api_weight: float = 0.72,
) -> water_sim.WaterParams:
    """
    Blend numeric API-selected parameters with the original AI controller's choice.

    obstacle_mode stays API-selected so each todo visibly selects the environment.
    """
    ai_weight = 1.0 - api_weight

    p = replace(api_params)
    p.source_rate = clamp_int(round(api_params.source_rate * api_weight + ai_params.source_rate * ai_weight), 1, 18)
    p.source_x = clamp_int(round(api_params.source_x * api_weight + ai_params.source_x * ai_weight), 1, config.width - 2)
    p.gravity_flow = clamp_int(round(api_params.gravity_flow * api_weight + ai_params.gravity_flow * ai_weight), 1, 12)
    p.lateral_flow = clamp_int(round(api_params.lateral_flow * api_weight + ai_params.lateral_flow * ai_weight), 1, 8)
    p.splash_chance = clamp_float(api_params.splash_chance * api_weight + ai_params.splash_chance * ai_weight, 0.0, 0.4)
    p.evaporation_rate = clamp_float(api_params.evaporation_rate * api_weight + ai_params.evaporation_rate * ai_weight, 0.0, 0.08)
    p.absorption_rate = clamp_float(api_params.absorption_rate * api_weight + ai_params.absorption_rate * ai_weight, 0.0, 0.3)
    p.basin_leakiness = clamp_float(api_params.basin_leakiness * api_weight + ai_params.basin_leakiness * ai_weight, 0.0, 0.1)
    p.max_cell_water = clamp_int(round(api_params.max_cell_water * api_weight + ai_params.max_cell_water * ai_weight), 3, 12)
    return p


# -----------------------------------------------------------------------------
# DISPLAY
# -----------------------------------------------------------------------------

def print_api_items(todos: List[Dict[str, Any]]) -> None:
    print("\nJSONPlaceholder todo items controlling this water run:")
    print("-" * 104)

    for index, todo in enumerate(todos, start=1):
        completed = bool(todo.get("completed", False))
        title = str(todo.get("title", ""))
        mode = choose_obstacle_mode(int(todo.get("id", index)), completed, title)
        behavior = "controlled flow" if completed else "exploratory flow"

        print(
            f"{index:>2}. id={todo.get('id'):<3} "
            f"userId={todo.get('userId'):<2} "
            f"completed={str(completed):<5} "
            f"mode={mode:<8} "
            f"behavior={behavior:<18} "
            f"title={title}"
        )


def print_api_final_summary(summaries: List[water_sim.RoundSummary]) -> None:
    print("\n" + "=" * 122)
    print("FINAL JSONPLACEHOLDER WATER FLOW COMPARISON")
    print("=" * 122)
    header = (
        f"{'Round':>5} | {'Mode':>9} | {'Src':>4} | {'X':>3} | {'Grav':>4} | {'Lat':>3} | "
        f"{'Added':>7} | {'Remain':>7} | {'Evap':>5} | {'Abs':>5} | {'Leak':>5} | "
        f"{'Over':>6} | {'MaxD':>4} | {'Move':>8} | Note"
    )
    print(header)
    print("-" * len(header))

    for s in summaries:
        print(
            f"{s.round_number:5d} | "
            f"{s.params.obstacle_mode:>9} | "
            f"{s.params.source_rate:4d} | "
            f"{s.params.source_x:3d} | "
            f"{s.params.gravity_flow:4d} | "
            f"{s.params.lateral_flow:3d} | "
            f"{s.added_water:7d} | "
            f"{s.remaining_water:7d} | "
            f"{s.evaporated:5d} | "
            f"{s.absorbed:5d} | "
            f"{s.leaked:5d} | "
            f"{s.overflowed:6d} | "
            f"{s.max_depth:4d} | "
            f"{s.average_speed_estimate:8.2f} | {s.note}"
        )

    best_storage = max(summaries, key=lambda s: s.remaining_water)
    best_flow = max(summaries, key=lambda s: s.average_speed_estimate)
    most_loss = max(summaries, key=lambda s: s.evaporated + s.absorbed + s.leaked + s.overflowed)

    print("\nAPI-driven water observations:")
    print(f"  Best water storage: Round {best_storage.round_number}, {best_storage.remaining_water} units remaining")
    print(f"  Most active flow: Round {best_flow.round_number}, movement {best_flow.average_speed_estimate:.2f} units/step")
    print(
        f"  Most total loss: Round {most_loss.round_number}, "
        f"{most_loss.evaporated + most_loss.absorbed + most_loss.leaked + most_loss.overflowed} units lost"
    )


# -----------------------------------------------------------------------------
# RUN
# -----------------------------------------------------------------------------

def main() -> None:
    print("JSONPlaceholder-driven Terminal Water Flow AI Rounds Simulation")
    print("The API provides external experiment seeds; the original water AI controller still adapts after each round.")

    config = water_sim.SimConfig(
        rounds=8,
        width=56,
        height=18,
        steps_per_round=80,
        print_every_steps=10,
        sleep_between_frames=0.0,
        seed=11,
    )

    rng = random.Random(config.seed)
    ai = water_sim.WaterAIController(rng, config)

    todos = get_api_todos(config.rounds)
    print_api_items(todos)

    params = water_sim.WaterParams(source_x=config.width // 2)
    previous_summary = None
    summaries: List[water_sim.RoundSummary] = []

    for round_number, todo in enumerate(todos, start=1):
        print("\n" + "#" * 104)
        print(f"API CONTROL FOR WATER ROUND {round_number}")
        print("#" * 104)

        api_params = map_todo_to_water_params(todo, params, config, round_number)

        print(
            "API-selected water setup: "
            f"mode={api_params.obstacle_mode}, "
            f"source_rate={api_params.source_rate}, "
            f"source_x={api_params.source_x}, "
            f"gravity={api_params.gravity_flow}, "
            f"lateral={api_params.lateral_flow}, "
            f"splash={api_params.splash_chance:.3f}, "
            f"evap={api_params.evaporation_rate:.3f}, "
            f"absorb={api_params.absorption_rate:.3f}, "
            f"leak={api_params.basin_leakiness:.3f}, "
            f"max_cell={api_params.max_cell_water}"
        )

        summary = water_sim.run_round(round_number, api_params, config, rng)
        summaries.append(summary)

        if round_number < config.rounds:
            ai_params = ai.choose_next(previous_summary or summary, api_params, round_number + 1)
            params = blend_params(api_params, ai_params, config, api_weight=0.72)

        previous_summary = summary

    print_api_final_summary(summaries)


if __name__ == "__main__":
    main()
