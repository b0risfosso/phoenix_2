#!/usr/bin/env python3
"""
JSONPlaceholder + Neuron Network Computation/Dream Driver - Accuracy Tuned
---------------------------------------------------------

This script lets the JSONPlaceholder API interact with:
    neuron_network_compute_dream_accuracy_tuned.py

Place this file in the same folder as:
    neuron_network_compute_dream_accuracy_tuned.py

Run:
    python compute_dream_jsonplaceholder_api_driver_accuracy_tuned.py

What it does:
- Downloads fake todo items from JSONPlaceholder.
- Converts each todo item into:
    - a compute or dream RoundPlan
    - neuron excitability parameters
    - pattern drive and output gate strength
    - learning and learned-weight scale
    - dream replay, silence, recombination, and dream cycle timing
- Runs the original run_round() function using API-selected settings.
- Uses the original ai_controller() after each round, then blends API choices
  with AI adjustments.
- Preserves memory traces across rounds.
- Falls back to local fake API items if the network is unavailable.

No external packages are required.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import replace
from typing import Any, Dict, List

import neuron_network_compute_dream_accuracy_tuned as dreamnet


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
            "User-Agent": "compute-dream-jsonplaceholder-driver/1.0",
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
            {"userId": 1, "id": 1, "title": "fallback compute pattern A sensory input", "completed": False},
            {"userId": 2, "id": 2, "title": "fallback compute pattern B sensory input", "completed": True},
            {"userId": 3, "id": 3, "title": "fallback repeat pattern A memory trace", "completed": False},
            {"userId": 4, "id": 4, "title": "fallback repeat pattern B memory trace", "completed": True},
            {"userId": 5, "id": 5, "title": "fallback dream replay stored activity", "completed": False},
            {"userId": 6, "id": 6, "title": "fallback recombined dream trace", "completed": True},
            {"userId": 7, "id": 7, "title": "fallback wake test pattern A", "completed": False},
            {"userId": 8, "id": 8, "title": "fallback wake test pattern B", "completed": True},
        ][:limit]


# -----------------------------------------------------------------------------
# MAPPING API DATA TO ROUND PLANS AND NETWORK PARAMETERS
# -----------------------------------------------------------------------------

def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def choose_api_round_plan(todo: Dict[str, Any], round_number: int) -> dreamnet.RoundPlan:
    """
    Convert API todo data into a compute/dream plan.

    Accuracy-tuned rhythm:
    - rounds 1-4 alternate A/B to build balanced memories
    - rounds 5-6 dream/replay stored traces
    - rounds 7-8 wake-test A/B after replay

    This prevents the API titles from accidentally producing mostly one class.
    """
    title = str(todo.get("title", "")).lower()
    completed = bool(todo.get("completed", False))

    if "dream" in title or "replay" in title or round_number in (5, 6):
        if completed:
            description = "API DREAM: completed todo creates a stronger recombination dream."
        else:
            description = "API DREAM: incomplete todo creates an exploratory replay dream."
        return dreamnet.RoundPlan("dream", "DREAM", description)

    balanced_pattern_sequence = {
        1: "A",
        2: "B",
        3: "A",
        4: "B",
        7: "A",
        8: "B",
    }
    pattern = balanced_pattern_sequence.get(round_number, "A")

    if round_number >= 7:
        description = f"API WAKE TEST: classify Pattern {pattern} after prior replay."
    elif completed:
        description = f"API COMPUTE: completed todo gives a clear Pattern {pattern} classification task."
    else:
        description = f"API COMPUTE: incomplete todo gives a noisier Pattern {pattern} classification task."

    return dreamnet.RoundPlan("compute", pattern, description)


def map_todo_to_network_params(
    todo: Dict[str, Any],
    base: dreamnet.NetworkParams,
    plan: dreamnet.RoundPlan,
    round_number: int,
) -> dreamnet.NetworkParams:
    """
    Convert one JSONPlaceholder todo item into NetworkParams.

    Mapping idea:
    - completed=True means stronger, clearer, more organized signals.
    - completed=False means noisier or more exploratory signals.
    - title length changes pattern drive, dream cycle, and replay/recombination.
    - userId changes E/I balance and timing.
    - id changes thresholds, adaptation, and synaptic delay.
    """

    p = replace(base)

    todo_id = int(todo.get("id", round_number))
    user_id = int(todo.get("userId", 1))
    title = str(todo.get("title", "untitled compute dream item"))
    completed = bool(todo.get("completed", False))

    title_len = len(title)
    title_score = min(1.0, title_len / 80.0)
    id_cycle = todo_id % 10
    user_cycle = user_id % 5

    # Shared excitability and timing.
    p.base_current_na = clamp(
        0.76 + 0.026 * id_cycle + 0.030 * user_cycle + (0.08 if completed else 0.00),
        0.10,
        3.00,
    )
    p.pulse_current_na = clamp(
        0.42 + 0.011 * title_len + (0.10 if completed else 0.00),
        0.00,
        3.00,
    )
    p.pulse_start_ms = clamp(48.0 + 2.5 * (todo_id % 14), 20.0, 140.0)
    p.pulse_duration_ms = clamp(24.0 + 22.0 * title_score + (8.0 if completed else 0.0), 8.0, 110.0)
    p.pulse_interval_ms = clamp(66.0 + 4.0 * user_cycle + 2.0 * (todo_id % 8), 35.0, 160.0)

    # Compute settings.
    if plan.mode == "compute":
        p.pattern_drive_na = clamp(
            0.68 + 0.24 * title_score + 0.020 * id_cycle + (0.12 if completed else 0.06),
            0.05,
            1.50,
        )
        p.output_gate_na = clamp(
            0.28 + 0.12 * title_score + (0.10 if completed else 0.05),
            0.00,
            0.95,
        )
        p.classification_margin_spikes = 1
    else:
        # Dream rounds quiet direct pattern classification drive.
        p.pattern_drive_na = clamp(0.32 + 0.08 * title_score, 0.05, 1.50)
        p.output_gate_na = clamp(0.04 + 0.03 * title_score, 0.00, 0.70)
        p.classification_margin_spikes = 1

    # Synaptic E/I mapping.
    p.excitatory_weight_na = clamp(
        0.22 + 0.014 * id_cycle + 0.018 * title_score + (0.045 if completed else 0.00),
        0.02,
        1.40,
    )
    p.inhibitory_weight_na = clamp(
        0.40 + 0.025 * user_cycle + (0.045 if not completed else 0.00),
        0.02,
        1.60,
    )
    p.exc_to_inh_scale = clamp(1.00 + 0.035 * user_cycle + (0.05 if completed else 0.00), 0.40, 2.50)
    p.inh_to_exc_scale = clamp(1.00 + 0.020 * id_cycle + (0.04 if not completed else -0.02), 0.40, 2.50)

    p.synaptic_delay_ms = clamp(1.8 + 0.32 * (todo_id % 9), 0.50, 12.0)
    p.delay_spread_ms = clamp(0.45 + 0.10 * user_cycle + 0.25 * title_score, 0.0, 4.0)
    p.synaptic_decay_ms = clamp(6.0 + 5.0 * title_score + (1.5 if completed else 0.0), 2.0, 30.0)

    # Learning and dream settings.
    p.learned_weight_scale_na = clamp(
        0.10 + 0.04 * title_score + (0.025 if completed else 0.00),
        0.00,
        0.45,
    )
    p.learning_rate = clamp(
        0.012 + 0.010 * title_score + (0.006 if completed else 0.000),
        0.00,
        0.06,
    )

    if plan.mode == "dream":
        p.dream_external_silence = clamp(0.48 + (0.04 if completed else 0.00), 0.05, 0.90)
        p.dream_replay_boost_na = clamp(0.72 + 0.22 * title_score + (0.10 if completed else 0.04), 0.0, 1.50)
        p.dream_recombination_na = clamp(0.14 + 0.12 * title_score + (0.06 if completed else 0.03), 0.0, 0.80)
        p.dream_cycle_ms = clamp(130.0 + 65.0 * title_score + 5.0 * (todo_id % 6), 80.0, 320.0)
    else:
        p.dream_external_silence = clamp(0.34 + 0.05 * (1.0 - title_score), 0.05, 0.90)
        p.dream_replay_boost_na = clamp(0.25 + 0.08 * title_score, 0.0, 1.50)
        p.dream_recombination_na = clamp(0.06 + 0.05 * title_score, 0.0, 0.80)
        p.dream_cycle_ms = clamp(150.0 + 35.0 * title_score, 80.0, 320.0)

    # Recovery and voltage threshold.
    p.v_threshold_mv = clamp(
        -55.4 - (0.35 if completed else 0.00) + 0.11 * (todo_id % 7),
        -62.0,
        -45.0,
    )
    p.refractory_ms = clamp(3.0 + 0.25 * (todo_id % 5) + (0.25 if not completed else 0.0), 1.0, 12.0)
    p.spike_adaptation_add_na = clamp(
        0.135 + 0.010 * id_cycle + (0.025 if not completed else 0.000),
        0.0,
        0.70,
    )
    p.adaptation_decay_ms = clamp(90.0 + 14.0 * user_cycle + 24.0 * title_score, 30.0, 300.0)

    # Deterministic sensory wave.
    p.input_wave_strength_na = clamp(0.035 + 0.105 * title_score + (0.025 if completed else 0.0), 0.0, 0.25)
    p.input_wave_period_ms = clamp(44.0 + 4.0 * (todo_id % 11) + 18.0 * title_score, 25.0, 140.0)
    p.neuron_bias_step_na = clamp(0.025 + 0.006 * user_cycle + 0.012 * title_score, 0.0, 0.12)

    return p


def blend_network_params(
    api_params: dreamnet.NetworkParams,
    ai_params: dreamnet.NetworkParams,
    api_weight: float = 0.68,
) -> dreamnet.NetworkParams:
    """
    Blend API-selected params with the original AI controller's params.

    The next round is still mainly API-driven, but the original controller's
    behavioral correction is not discarded.
    """
    blended = replace(api_params)
    ai_weight = 1.0 - api_weight

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
        "pattern_drive_na",
        "output_gate_na",
        "excitatory_weight_na",
        "inhibitory_weight_na",
        "exc_to_inh_scale",
        "inh_to_exc_scale",
        "synaptic_delay_ms",
        "delay_spread_ms",
        "synaptic_decay_ms",
        "learned_weight_scale_na",
        "learning_rate",
        "dream_external_silence",
        "dream_replay_boost_na",
        "dream_recombination_na",
        "dream_cycle_ms",
        "adaptation_decay_ms",
        "spike_adaptation_add_na",
        "input_wave_strength_na",
        "input_wave_period_ms",
        "neuron_bias_step_na",
    ]

    for field_name in numeric_fields:
        api_value = getattr(api_params, field_name)
        ai_value = getattr(ai_params, field_name)
        setattr(blended, field_name, api_value * api_weight + ai_value * ai_weight)

    # classification_margin_spikes is discrete.
    blended.classification_margin_spikes = api_params.classification_margin_spikes

    return blended


# -----------------------------------------------------------------------------
# DISPLAY HELPERS
# -----------------------------------------------------------------------------

def print_api_items(todos: List[Dict[str, Any]]) -> None:
    print("\nJSONPlaceholder todo items controlling this compute/dream run:")
    print("-" * 116)

    for index, todo in enumerate(todos, start=1):
        plan = choose_api_round_plan(todo, index)
        completed = bool(todo.get("completed", False))
        title = str(todo.get("title", ""))

        print(
            f"{index:>2}. id={todo.get('id'):<3} "
            f"userId={todo.get('userId'):<2} "
            f"completed={str(completed):<5} "
            f"plan={plan.mode:<7} "
            f"pattern={plan.pattern_label:<5} "
            f"title={title}"
        )


def print_api_final_summary(results: List[Dict[str, float | str | bool]], memory: dreamnet.NetworkMemory) -> None:
    print("\n" + "=" * 140)
    print("FINAL JSONPLACEHOLDER COMPUTATION/DREAM COMPARISON")
    print("=" * 140)
    header = (
        f"{'Round':>5} | {'Mode':>7} | {'Pattern':>7} | {'Total':>6} | {'Hz/N':>8} | "
        f"{'A_out':>6} | {'B_out':>6} | {'Decision':>9} | {'Correct':>7} | "
        f"{'DreamSim':>8} | {'E/I':>7} | {'Events':>7}"
    )
    print(header)
    print("-" * len(header))

    for r in results:
        correct_text = "--" if r["mode"] != "compute" else str(bool(r["correct"]))
        dream_text = "--" if r["mode"] != "dream" else f"{float(r['dream_similarity']):.3f}"
        print(
            f"{int(r['round']):5d} | "
            f"{str(r['mode']):>7} | "
            f"{str(r['pattern']):>7} | "
            f"{int(r['total_spikes']):6d} | "
            f"{float(r['network_rate_hz']):8.2f} | "
            f"{int(r['output_a_spikes']):6d} | "
            f"{int(r['output_b_spikes']):6d} | "
            f"{str(r['decision']):>9} | "
            f"{correct_text:>7} | "
            f"{dream_text:>8} | "
            f"{float(r['balance_ratio']):7.2f} | "
            f"{int(r['delivered_events']):7d}"
        )

    compute_results = [r for r in results if r["mode"] == "compute"]
    dream_results = [r for r in results if r["mode"] == "dream"]

    correct_count = sum(1 for r in compute_results if bool(r["correct"]))
    accuracy = correct_count / max(1, len(compute_results))
    avg_dream_similarity = sum(float(r["dream_similarity"]) for r in dream_results) / max(1, len(dream_results))

    most_active = max(results, key=lambda r: int(r["total_spikes"]))
    clearest_compute = None
    if compute_results:
        clearest_compute = max(
            compute_results,
            key=lambda r: abs(int(r["output_a_spikes"]) - int(r["output_b_spikes"])),
        )

    print("\nAPI-driven observations:")
    print(f"  Computation accuracy: {correct_count}/{len(compute_results)} = {accuracy:.2%}")
    print(f"  Average dream-memory similarity: {avg_dream_similarity:.3f}")
    print(f"  Memory traces stored: {len(memory.traces)}")
    print(f"  Strongest learned links: {dreamnet.learned_weight_summary(memory)}")
    print(f"  Most active round: Round {int(most_active['round'])}, {int(most_active['total_spikes'])} spikes")
    if clearest_compute is not None:
        margin = abs(int(clearest_compute["output_a_spikes"]) - int(clearest_compute["output_b_spikes"]))
        print(f"  Clearest compute output split: Round {int(clearest_compute['round'])}, margin {margin} spikes")


# -----------------------------------------------------------------------------
# RUN
# -----------------------------------------------------------------------------

def main() -> None:
    print("JSONPlaceholder-driven Neuron Network Computation and Dream Simulation - Accuracy Tuned")
    print("The API provides external experiment seeds; the original AI controller still adapts after each round.")

    todos = get_api_todos(dreamnet.TOTAL_ROUNDS)
    print_api_items(todos)

    params = dreamnet.NetworkParams()
    memory = dreamnet.NetworkMemory()
    results: List[Dict[str, float | str | bool]] = []

    for round_number, todo in enumerate(todos, start=1):
        print("\n" + "#" * 116)
        print(f"API CONTROL FOR COMPUTE/DREAM ROUND {round_number}")
        print("#" * 116)

        plan = choose_api_round_plan(todo, round_number)
        api_params = map_todo_to_network_params(todo, params, plan, round_number)

        print(
            "API-selected compute/dream setup: "
            f"mode={plan.mode}, "
            f"pattern={plan.pattern_label}, "
            f"base={api_params.base_current_na:.3f} nA, "
            f"pulse={api_params.pulse_current_na:.3f} nA, "
            f"pattern_drive={api_params.pattern_drive_na:.3f} nA, "
            f"E_w={api_params.excitatory_weight_na:.3f}, "
            f"I_w={api_params.inhibitory_weight_na:.3f}, "
            f"learned_scale={api_params.learned_weight_scale_na:.3f}, "
            f"dream_silence={api_params.dream_external_silence:.2f}, "
            f"dream_replay={api_params.dream_replay_boost_na:.3f}, "
            f"recombination={api_params.dream_recombination_na:.3f}, "
            f"threshold={api_params.v_threshold_mv:.2f} mV"
        )

        result = dreamnet.run_round(round_number, api_params, memory, plan)
        results.append(result)

        if round_number < dreamnet.TOTAL_ROUNDS:
            ai_params = dreamnet.ai_controller(api_params, result)
            params = blend_network_params(api_params, ai_params, api_weight=0.68)

    print_api_final_summary(results, memory)


if __name__ == "__main__":
    main()
