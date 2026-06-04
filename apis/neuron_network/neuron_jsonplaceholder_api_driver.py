#!/usr/bin/env python3
"""
JSONPlaceholder + Neuron AI Network Driver
------------------------------------------

This script lets the JSONPlaceholder API interact with the provided
neuron_ai_network_rounds_simulation_returned.py script.

Place this file in the same folder as:
    neuron_ai_network_rounds_simulation_returned.py

Run:
    python neuron_jsonplaceholder_api_driver.py

What it does:
- Downloads fake todo items from JSONPlaceholder.
- Converts each todo item into NetworkParams.
- Runs the original neuron network round function using those API-driven settings.
- Still uses the original AI controller after each round, so the API and AI controller both affect the simulation.
- Falls back to local fake API items if the network is unavailable.

No external packages are required.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import replace
from typing import Any, Dict, List

import neuron_ai_network_rounds_simulation_returned as neuron


BASE_URL = "https://jsonplaceholder.typicode.com"


def request_json(endpoint: str, timeout_s: float = 8.0) -> Any:
    """Request JSON from JSONPlaceholder using only Python standard library."""
    url = BASE_URL + endpoint

    request = urllib.request.Request(
        url=url,
        headers={
            "Accept": "application/json",
            "User-Agent": "neuron-jsonplaceholder-driver/1.0",
        },
        method="GET",
    )

    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        body = response.read().decode("utf-8")
        return json.loads(body)


def get_api_todos(limit: int) -> List[Dict[str, Any]]:
    """Download todo items from JSONPlaceholder, or use fallback items."""
    try:
        todos = request_json("/todos")
        return todos[:limit]

    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as error:
        print("\nAPI request failed. Using local fallback todo items.")
        print(f"Reason: {error}")

        return [
            {"userId": 1, "id": 1, "title": "fallback quiet sensory input", "completed": False},
            {"userId": 2, "id": 2, "title": "fallback synchronized pulse train", "completed": True},
            {"userId": 3, "id": 3, "title": "fallback inhibitory stabilization", "completed": False},
            {"userId": 4, "id": 4, "title": "fallback high excitation burst", "completed": True},
            {"userId": 5, "id": 5, "title": "fallback delayed recurrent loop", "completed": False},
            {"userId": 6, "id": 6, "title": "fallback balanced network response", "completed": True},
            {"userId": 7, "id": 7, "title": "fallback adaptation recovery test", "completed": False},
            {"userId": 8, "id": 8, "title": "fallback strong sensory wave", "completed": True},
        ][:limit]


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def map_todo_to_network_params(
    todo: Dict[str, Any],
    base: neuron.NetworkParams,
    round_index: int,
) -> neuron.NetworkParams:
    """
    Convert a JSONPlaceholder todo into neuron NetworkParams.

    Mapping idea:
    - completed=True means a stronger, more organized stimulus.
    - completed=False means weaker/noisier activity with more inhibition.
    - title length changes input waves and pulse timing.
    - userId changes excitation/inhibition balance.
    - id changes delay, threshold, refractory time, and adaptation.
    """

    p = replace(base)

    todo_id = int(todo.get("id", round_index))
    user_id = int(todo.get("userId", 1))
    title = str(todo.get("title", "untitled network input"))
    completed = bool(todo.get("completed", False))

    title_len = len(title)
    title_score = min(1.0, title_len / 80.0)
    id_cycle = todo_id % 10
    user_cycle = user_id % 5

    # External drive: completed items are interpreted as stronger successful signals.
    p.base_current_na = clamp(
        0.92 + 0.045 * id_cycle + (0.18 if completed else -0.02),
        0.30,
        2.50,
    )

    p.pulse_current_na = clamp(
        0.45 + 0.025 * title_len + (0.18 if completed else -0.08),
        0.05,
        2.50,
    )

    # Timing: title length and id control when pulses arrive.
    p.pulse_start_ms = clamp(45.0 + 3.0 * (todo_id % 12), 20.0, 120.0)
    p.pulse_duration_ms = clamp(24.0 + 18.0 * title_score + (10.0 if completed else 0.0), 10.0, 90.0)
    p.pulse_interval_ms = clamp(70.0 + 5.0 * user_cycle + 2.0 * (todo_id % 8), 35.0, 150.0)

    # Excitation/inhibition mapping.
    p.excitatory_weight_na = clamp(
        0.24 + 0.025 * id_cycle + (0.12 if completed else 0.00),
        0.02,
        1.40,
    )

    p.inhibitory_weight_na = clamp(
        0.34 + 0.035 * user_cycle + (0.08 if not completed else -0.03),
        0.02,
        1.60,
    )

    p.exc_to_inh_scale = clamp(1.05 + 0.03 * user_cycle + (0.07 if completed else 0.00), 0.40, 2.50)
    p.inh_to_exc_scale = clamp(1.00 + 0.025 * id_cycle + (0.06 if not completed else -0.02), 0.40, 2.50)

    # Delay and wave behavior.
    p.synaptic_delay_ms = clamp(1.8 + 0.35 * (todo_id % 8), 0.50, 12.0)
    p.delay_spread_ms = clamp(0.45 + 0.08 * user_cycle + 0.30 * title_score, 0.0, 4.0)
    p.synaptic_decay_ms = clamp(6.0 + 5.0 * title_score + (2.0 if completed else 0.0), 2.0, 30.0)

    # Threshold and recovery.
    p.v_threshold_mv = clamp(-55.5 - (0.6 if completed else 0.0) + 0.12 * (todo_id % 6), -62.0, -45.0)
    p.refractory_ms = clamp(3.0 + 0.30 * (todo_id % 5) + (0.4 if not completed else 0.0), 1.0, 12.0)
    p.spike_adaptation_add_na = clamp(0.09 + 0.012 * id_cycle + (0.025 if not completed else 0.00), 0.0, 0.70)
    p.adaptation_decay_ms = clamp(95.0 + 12.0 * user_cycle + 20.0 * title_score, 30.0, 300.0)

    # Sensory wave.
    p.input_wave_strength_na = clamp(0.04 + 0.11 * title_score + (0.03 if completed else 0.00), 0.0, 0.25)
    p.input_wave_period_ms = clamp(45.0 + 4.0 * (todo_id % 11) + 15.0 * title_score, 25.0, 140.0)

    # Small per-neuron bias, so API data changes network asymmetry.
    p.neuron_bias_step_na = clamp(0.025 + 0.006 * user_cycle + 0.012 * title_score, 0.0, 0.12)

    return p


def print_api_items(todos: List[Dict[str, Any]]) -> None:
    print("\nJSONPlaceholder todo items controlling this neuron run:")
    print("-" * 104)

    for index, todo in enumerate(todos, start=1):
        completed = bool(todo.get("completed", False))
        mode = "strong signal" if completed else "unstable signal"
        print(
            f"{index:>2}. id={todo.get('id'):<3} "
            f"userId={todo.get('userId'):<2} "
            f"completed={str(completed):<5} "
            f"mode={mode:<15} "
            f"title={todo.get('title')}"
        )


def combine_api_and_ai_params(
    api_params: neuron.NetworkParams,
    ai_params: neuron.NetworkParams,
    api_weight: float = 0.65,
) -> neuron.NetworkParams:
    """
    Blend API-selected parameters with the built-in AI controller's chosen parameters.

    api_weight=0.65 means the JSONPlaceholder item is the main control signal,
    while the original AI controller still nudges the system toward its target.
    """
    blended = replace(api_params)

    numeric_fields = [
        "v_rest_mv",
        "v_reset_mv",
        "v_threshold_mv",
        "membrane_resistance_mohm",
        "membrane_tau_ms",
        "refractory_ms",
        "base_current_na",
        "pulse_current_na",
        "pulse_start_ms",
        "pulse_duration_ms",
        "pulse_interval_ms",
        "excitatory_weight_na",
        "inhibitory_weight_na",
        "exc_to_inh_scale",
        "inh_to_exc_scale",
        "synaptic_delay_ms",
        "delay_spread_ms",
        "synaptic_decay_ms",
        "adaptation_decay_ms",
        "spike_adaptation_add_na",
        "input_wave_strength_na",
        "input_wave_period_ms",
        "neuron_bias_step_na",
    ]

    ai_weight = 1.0 - api_weight

    for field_name in numeric_fields:
        api_value = getattr(api_params, field_name)
        ai_value = getattr(ai_params, field_name)
        setattr(blended, field_name, api_value * api_weight + ai_value * ai_weight)

    return blended


def main() -> None:
    print("JSONPlaceholder-driven Neuron AI Network Simulation")
    print("The API provides external experiment seeds; the original AI controller still adapts after each round.")

    rounds = neuron.TOTAL_ROUNDS
    todos = get_api_todos(rounds)
    print_api_items(todos)

    params = neuron.NetworkParams()
    results: List[Dict[str, float]] = []

    for round_index, todo in enumerate(todos, start=1):
        print("\n" + "#" * 104)
        print(f"API CONTROL FOR ROUND {round_index}")
        print("#" * 104)

        api_params = map_todo_to_network_params(todo, params, round_index)

        print(
            "API-selected network setup: "
            f"base={api_params.base_current_na:.3f} nA, "
            f"pulse={api_params.pulse_current_na:.3f} nA, "
            f"E_w={api_params.excitatory_weight_na:.3f}, "
            f"I_w={api_params.inhibitory_weight_na:.3f}, "
            f"delay={api_params.synaptic_delay_ms:.2f} ms, "
            f"threshold={api_params.v_threshold_mv:.2f} mV"
        )

        result = neuron.run_round(round_index, api_params)
        results.append(result)

        if round_index < rounds:
            ai_params = neuron.ai_controller(api_params, result)
            params = combine_api_and_ai_params(api_params, ai_params, api_weight=0.65)

    neuron.print_final_summary(results)


if __name__ == "__main__":
    main()
