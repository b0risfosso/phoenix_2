#!/usr/bin/env python3
"""
JSONPlaceholder + Simple Flame Simulation Driver
------------------------------------------------

This script lets the JSONPlaceholder API control rounds of the attached
simple_flame_combustion_simulation.py file.

Place this file in the same folder as:
    simple_flame_combustion_simulation.py

Run:
    python flame_jsonplaceholder_api_driver.py

What it does:
- Downloads fake todo items from JSONPlaceholder.
- Converts each todo into a flame experiment.
- Runs the original flame simulation using those API-driven parameters.
- Falls back to local synthetic API items if the network is unavailable.

No external packages are required.
"""

from __future__ import annotations

import json
import random
import urllib.error
import urllib.request
from dataclasses import replace
from typing import Any, Dict, List

import simple_flame_combustion_simulation as flame


BASE_URL = "https://jsonplaceholder.typicode.com"


def request_json(endpoint: str, timeout_s: float = 8.0) -> Any:
    """Request JSON from JSONPlaceholder using only the standard library."""
    url = BASE_URL + endpoint

    request = urllib.request.Request(
        url=url,
        headers={
            "Accept": "application/json",
            "User-Agent": "flame-jsonplaceholder-driver/1.0",
        },
        method="GET",
    )

    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        body = response.read().decode("utf-8")
        return json.loads(body)


def get_api_todos(limit: int) -> List[Dict[str, Any]]:
    """
    Download todo items from JSONPlaceholder.

    JSONPlaceholder returns deterministic fake data. These items are not real
    experiments; they are used here as remote control inputs.
    """
    try:
        todos = request_json("/todos")
        return todos[:limit]

    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as error:
        print("\nAPI request failed. Using local fallback todo items.")
        print(f"Reason: {error}")

        return [
            {"userId": 1, "id": 1, "title": "fallback baseline ignition", "completed": False},
            {"userId": 2, "id": 2, "title": "fallback hotter oxygen assisted burn", "completed": True},
            {"userId": 3, "id": 3, "title": "fallback humid weak flame", "completed": False},
            {"userId": 4, "id": 4, "title": "fallback small chamber flash", "completed": True},
        ][:limit]


def map_todo_to_flame_params(todo: Dict[str, Any], base: flame.FlameParams, round_number: int) -> flame.FlameParams:
    """
    Convert one JSONPlaceholder todo item into FlameParams.

    Mapping idea:
    - completed=True means the controller supplies a cleaner, stronger burn.
    - completed=False means the flame is more uncertain and smoky.
    - title length changes spark energy and chamber behavior.
    - userId changes oxygen/fuel balance and humidity.
    - id changes reaction strength and cooling.
    """

    p = replace(base)

    todo_id = int(todo.get("id", round_number))
    user_id = int(todo.get("userId", 1))
    title = str(todo.get("title", "untitled api experiment"))
    completed = bool(todo.get("completed", False))

    title_len = len(title)
    title_score = min(1.0, title_len / 80.0)
    id_cycle = todo_id % 10
    user_cycle = user_id % 5

    p.fuel_name = f"API todo {todo_id}: {title[:38]}"

    # API-derived starting reservoirs.
    p.initial_fuel = flame.clamp(0.70 + 0.07 * id_cycle + 0.20 * title_score, 0.25, 1.80)
    p.initial_oxygen = flame.clamp(0.75 + 0.12 * user_cycle + (0.35 if completed else -0.10), 0.20, 1.90)

    # API-derived supply behavior.
    p.fuel_supply_rate = flame.clamp(0.06 + 0.012 * (todo_id % 12), 0.02, 0.30)
    p.oxygen_supply_rate = flame.clamp(0.22 + 0.08 * user_cycle + (0.35 if completed else 0.05), 0.05, 1.20)

    # Completed tasks get a stronger, cleaner ignition.
    p.spark_energy_J = flame.clamp(520.0 + 12.0 * title_len + (120.0 if completed else -40.0), 250.0, 1100.0)
    p.reaction_strength = flame.clamp(1.7 + 0.15 * id_cycle + (0.55 if completed else -0.10), 0.5, 5.0)

    # Incomplete tasks are interpreted as messy/uncertain burn conditions.
    p.humidity = flame.clamp(0.12 + 0.07 * user_cycle + (0.00 if completed else 0.22), 0.0, 0.85)
    p.soot_rate = flame.clamp(0.012 + 0.004 * id_cycle + (0.000 if completed else 0.020), 0.004, 0.090)

    # Chamber and cooling are varied by title/id.
    p.chamber_volume = flame.clamp(0.65 + 0.75 * title_score, 0.40, 1.60)
    p.convective_loss_coeff = flame.clamp(0.55 + 0.06 * (todo_id % 8) + (0.20 if not completed else 0.00), 0.20, 1.50)
    p.mixing_loss_coeff = flame.clamp(0.06 + 0.025 * user_cycle + (0.08 if completed else 0.02), 0.00, 0.45)

    return p


def print_api_items(todos: List[Dict[str, Any]]) -> None:
    print("\nJSONPlaceholder todo items controlling this run:")
    print("-" * 90)

    for index, todo in enumerate(todos, start=1):
        print(
            f"{index:>2}. id={todo.get('id'):<3} "
            f"userId={todo.get('userId'):<2} "
            f"completed={str(todo.get('completed')):<5} "
            f"title={todo.get('title')}"
        )


def main() -> None:
    config = flame.SimConfig(
        rounds=6,
        duration_s=4.0,
        dt_s=0.01,
        print_every_s=0.40,
        seed=23,
    )

    random.seed(config.seed)

    todos = get_api_todos(config.rounds)
    print_api_items(todos)

    base_params = flame.FlameParams()
    summaries: List[flame.RoundSummary] = []

    print("\nJSONPlaceholder-driven flame simulation")
    print("Each API todo item is converted into flame fuel, oxygen, spark, cooling, humidity, and soot parameters.")

    for round_number, todo in enumerate(todos, start=1):
        params = map_todo_to_flame_params(todo, base_params, round_number)
        summary = flame.run_round(round_number, params, config)
        summaries.append(summary)

    flame.print_final_summary(summaries)


if __name__ == "__main__":
    main()
