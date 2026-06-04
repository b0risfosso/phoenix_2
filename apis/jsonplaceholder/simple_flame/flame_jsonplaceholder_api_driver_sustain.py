#!/usr/bin/env python3
"""
JSONPlaceholder + Simple Flame Simulation Driver: Sustain-Flame Version
-----------------------------------------------------------------------

This script lets the JSONPlaceholder API control rounds of the attached
simple_flame_combustion_simulation.py file.

This version adds a sustain-flame mode:
- completed=True API todos become stronger sustained-flame experiments.
- every third incomplete todo also receives a moderate sustain boost.
- cooling losses are reduced for sustain rounds.
- heat release, reaction strength, oxygen supply, and burn duration are increased.
- the original simple_flame_combustion_simulation.py file is not overwritten.

Place this file in the same folder as:
    simple_flame_combustion_simulation.py

Run:
    python flame_jsonplaceholder_api_driver_sustain.py

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


# -----------------------------------------------------------------------------
# API access
# -----------------------------------------------------------------------------

def request_json(endpoint: str, timeout_s: float = 8.0) -> Any:
    """Request JSON from JSONPlaceholder using only the standard library."""
    url = BASE_URL + endpoint

    request = urllib.request.Request(
        url=url,
        headers={
            "Accept": "application/json",
            "User-Agent": "flame-jsonplaceholder-driver-sustain/1.0",
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
            {"userId": 5, "id": 5, "title": "fallback sustained blue flame", "completed": True},
            {"userId": 6, "id": 6, "title": "fallback smoky unstable flame", "completed": False},
        ][:limit]


# -----------------------------------------------------------------------------
# Mapping API data to flame experiments
# -----------------------------------------------------------------------------

def should_use_sustain_mode(todo: Dict[str, Any], round_number: int) -> bool:
    """
    Decide whether this API item should create a sustained flame.

    completed=True always creates a sustained-flame experiment.
    Every third round gets a moderate sustain test so incomplete todos can still
    sometimes create a longer flame.
    """
    completed = bool(todo.get("completed", False))
    return completed or round_number % 3 == 0


def map_todo_to_flame_params(todo: Dict[str, Any], base: flame.FlameParams, round_number: int) -> flame.FlameParams:
    """
    Convert one JSONPlaceholder todo item into FlameParams.

    Mapping idea:
    - completed=True means a stronger, cleaner sustained burn.
    - completed=False means a more uncertain and smoky burn.
    - every third incomplete todo receives a moderate sustain boost.
    - title length changes spark energy and chamber behavior.
    - userId changes oxygen/fuel balance and humidity.
    - id changes reaction strength and cooling.
    """

    p = replace(base)

    todo_id = int(todo.get("id", round_number))
    user_id = int(todo.get("userId", 1))
    title = str(todo.get("title", "untitled api experiment"))
    completed = bool(todo.get("completed", False))
    sustain = should_use_sustain_mode(todo, round_number)

    title_len = len(title)
    title_score = min(1.0, title_len / 80.0)
    id_cycle = todo_id % 10
    user_cycle = user_id % 5

    mode_label = "sustain" if sustain else "brief"
    p.fuel_name = f"API {mode_label} todo {todo_id}: {title[:32]}"

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

    if sustain:
        apply_sustain_flame_boost(p, completed=completed, todo_id=todo_id, title_score=title_score)

    return p


def apply_sustain_flame_boost(
    p: flame.FlameParams,
    *,
    completed: bool,
    todo_id: int,
    title_score: float,
) -> None:
    """
    Modify FlameParams in-place to make a longer active flame more likely.

    The original simulation cools very aggressively, so this mode reduces heat
    losses and increases fuel/oxygen/heat generation enough to let some rounds
    stay active instead of immediately dropping below extinction temperature.
    """

    boost = 1.0 if completed else 0.65

    p.initial_fuel = flame.clamp(p.initial_fuel + 0.25 * boost, 0.20, 1.90)
    p.initial_oxygen = flame.clamp(p.initial_oxygen + 0.55 * boost, 0.20, 1.90)

    p.fuel_supply_rate = flame.clamp(p.fuel_supply_rate + 0.10 * boost, 0.02, 0.50)
    p.oxygen_supply_rate = flame.clamp(p.oxygen_supply_rate + 0.45 * boost, 0.05, 1.40)

    p.spark_energy_J = flame.clamp(p.spark_energy_J + 160.0 * boost, 300.0, 1200.0)
    p.reaction_strength = flame.clamp(p.reaction_strength + 1.35 * boost, 0.5, 6.0)

    # The most important sustain changes:
    # more heat per fuel, lower heat capacity penalty, and much weaker losses.
    p.heat_of_combustion_J_per_unit = flame.clamp(
        900.0 + 260.0 * boost + 80.0 * title_score,
        520.0,
        1500.0,
    )
    p.heat_capacity_J_per_K = flame.clamp(0.78 - 0.08 * boost, 0.45, 1.20)

    p.convective_loss_coeff = flame.clamp(p.convective_loss_coeff * (0.32 if completed else 0.45), 0.08, 0.80)
    p.radiative_loss_coeff = flame.clamp(p.radiative_loss_coeff * (0.08 if completed else 0.16), 0.0, 2.6e-8)
    p.mixing_loss_coeff = flame.clamp(p.mixing_loss_coeff * (0.35 if completed else 0.50), 0.0, 0.25)

    p.humidity = flame.clamp(p.humidity * (0.45 if completed else 0.65), 0.0, 0.60)
    p.soot_rate = flame.clamp(p.soot_rate * (0.55 if completed else 0.75), 0.004, 0.070)

    # Slightly easier ignition/extinction thresholds for the sustained toy model.
    p.ignition_temperature_K = 690.0
    p.extinction_temperature_K = 560.0

    # Smaller chamber concentrates fuel/oxygen enough to sustain reaction.
    p.chamber_volume = flame.clamp(p.chamber_volume * (0.72 + 0.02 * (todo_id % 3)), 0.40, 1.20)


def print_api_items(todos: List[Dict[str, Any]]) -> None:
    print("\nJSONPlaceholder todo items controlling this run:")
    print("-" * 104)

    for index, todo in enumerate(todos, start=1):
        sustain = should_use_sustain_mode(todo, index)
        print(
            f"{index:>2}. id={todo.get('id'):<3} "
            f"userId={todo.get('userId'):<2} "
            f"completed={str(todo.get('completed')):<5} "
            f"sustain_mode={str(sustain):<5} "
            f"title={todo.get('title')}"
        )


# -----------------------------------------------------------------------------
# Run
# -----------------------------------------------------------------------------

def main() -> None:
    config = flame.SimConfig(
        rounds=6,
        duration_s=6.0,
        dt_s=0.01,
        print_every_s=0.50,
        seed=23,
    )

    random.seed(config.seed)

    todos = get_api_todos(config.rounds)
    print_api_items(todos)

    base_params = flame.FlameParams()
    summaries: List[flame.RoundSummary] = []

    print("\nJSONPlaceholder-driven flame simulation with sustain-flame mode")
    print("Each API todo item is converted into flame fuel, oxygen, spark, cooling, humidity, soot, and sustain parameters.")
    print("completed=True todos always receive sustain mode; every third todo receives a moderate sustain test.")

    for round_number, todo in enumerate(todos, start=1):
        params = map_todo_to_flame_params(todo, base_params, round_number)
        summary = flame.run_round(round_number, params, config)
        summaries.append(summary)

    flame.print_final_summary(summaries)


if __name__ == "__main__":
    main()
