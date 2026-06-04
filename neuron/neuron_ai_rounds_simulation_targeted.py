#!/usr/bin/env python3
"""
Neuron AI Rounds Simulation - Targeted Spike Controller
-------------------------------------------------------

A terminal-only Python simulation of a simplified spiking neuron.

Run:
    python neuron_ai_rounds_simulation_targeted.py

No external packages are required.

The model uses a leaky integrate-and-fire neuron:

    dV/dt = (-(V - V_rest) + R_m * I_total) / tau_m

When membrane voltage crosses threshold, the neuron fires a spike,
then enters a refractory period and resets its voltage.

Corrected behavior:
    The AI controller explicitly targets the requested spike count. If the
    neuron fires too many spikes, the next round becomes less excitable. If it
    fires too few, the next round becomes more excitable. The earlier controller
    treated 7-8 spikes as "near target" and shortened the pulse interval, which
    pushed the neuron away from the 6-spike target.
"""

from dataclasses import dataclass, replace
from math import exp
from typing import List, Dict, Tuple


# ============================================================
# CONFIGURATION
# ============================================================

TOTAL_ROUNDS = 8
ROUND_DURATION_MS = 500.0
DT_MS = 0.25
PRINT_EVERY_MS = 25.0
DISPLAY_WIDTH = 60
TARGET_SPIKES = 6


@dataclass
class NeuronParams:
    # Electrical properties
    v_rest_mv: float = -70.0
    v_reset_mv: float = -75.0
    v_threshold_mv: float = -54.0
    membrane_resistance_mohm: float = 10.0
    membrane_tau_ms: float = 20.0
    refractory_ms: float = 4.0

    # Input current properties
    base_current_na: float = 1.2
    pulse_current_na: float = 0.8
    pulse_start_ms: float = 80.0
    pulse_duration_ms: float = 250.0
    pulse_interval_ms: float = 100.0

    # Ion/recovery terms, simplified
    adaptation_strength_na: float = 0.05
    adaptation_decay_ms: float = 120.0
    spike_adaptation_add_na: float = 0.18

    # Noise-like deterministic oscillation term
    input_wave_strength_na: float = 0.05
    input_wave_period_ms: float = 60.0


@dataclass
class NeuronState:
    t_ms: float = 0.0
    voltage_mv: float = -70.0
    refractory_left_ms: float = 0.0
    adaptation_na: float = 0.0
    spikes: int = 0
    last_spike_ms: float = -999999.0
    max_voltage_mv: float = -70.0
    min_voltage_mv: float = -70.0


def triangular_wave(t_ms: float, period_ms: float) -> float:
    """Deterministic wave in range [-1, 1], used instead of random noise."""
    if period_ms <= 0:
        return 0.0
    phase = (t_ms % period_ms) / period_ms
    if phase < 0.25:
        return phase * 4.0
    if phase < 0.75:
        return 2.0 - phase * 4.0
    return phase * 4.0 - 4.0


def stimulus_current_na(t_ms: float, p: NeuronParams) -> float:
    """Base current plus repeated square pulses plus a small deterministic wave."""
    current = p.base_current_na

    if t_ms >= p.pulse_start_ms and p.pulse_interval_ms > 0:
        phase = (t_ms - p.pulse_start_ms) % p.pulse_interval_ms
        if phase <= p.pulse_duration_ms:
            current += p.pulse_current_na

    current += p.input_wave_strength_na * triangular_wave(t_ms, p.input_wave_period_ms)
    return current


def step_neuron(state: NeuronState, p: NeuronParams, dt_ms: float) -> Tuple[NeuronState, bool, float]:
    """Advance neuron by one time step. Returns updated state, spike flag, input current."""
    s = replace(state)
    input_current = stimulus_current_na(s.t_ms, p)

    # Adaptation acts like a temporary outward current that reduces excitability.
    effective_current = input_current - s.adaptation_na

    if s.refractory_left_ms > 0.0:
        s.voltage_mv = p.v_reset_mv
        s.refractory_left_ms = max(0.0, s.refractory_left_ms - dt_ms)
        spike = False
    else:
        dv_dt = (-(s.voltage_mv - p.v_rest_mv) + p.membrane_resistance_mohm * effective_current) / p.membrane_tau_ms
        s.voltage_mv += dv_dt * dt_ms
        spike = s.voltage_mv >= p.v_threshold_mv
        if spike:
            s.spikes += 1
            s.last_spike_ms = s.t_ms
            s.voltage_mv = p.v_reset_mv
            s.refractory_left_ms = p.refractory_ms
            s.adaptation_na += p.spike_adaptation_add_na

    # Exponential decay for adaptation current.
    if p.adaptation_decay_ms > 0:
        s.adaptation_na *= exp(-dt_ms / p.adaptation_decay_ms)

    s.t_ms += dt_ms
    s.max_voltage_mv = max(s.max_voltage_mv, s.voltage_mv)
    s.min_voltage_mv = min(s.min_voltage_mv, s.voltage_mv)
    return s, spike, input_current


def voltage_bar(voltage_mv: float, v_min: float = -80.0, v_max: float = -45.0) -> str:
    """Small terminal visualization of voltage position."""
    x = int((voltage_mv - v_min) / (v_max - v_min) * (DISPLAY_WIDTH - 1))
    x = max(0, min(DISPLAY_WIDTH - 1, x))
    chars = ["-"] * DISPLAY_WIDTH
    chars[x] = "|"
    return "".join(chars)


def run_round(round_index: int, p: NeuronParams) -> Dict[str, float]:
    print("\n" + "=" * 78)
    print(f"ROUND {round_index}")
    print("=" * 78)
    print("Parameters:")
    print(f"  V_rest={p.v_rest_mv:.1f} mV, V_threshold={p.v_threshold_mv:.1f} mV, V_reset={p.v_reset_mv:.1f} mV")
    print(f"  Rm={p.membrane_resistance_mohm:.2f} Mohm, tau={p.membrane_tau_ms:.2f} ms, refractory={p.refractory_ms:.2f} ms")
    print(f"  base_current={p.base_current_na:.3f} nA, pulse_current={p.pulse_current_na:.3f} nA")
    print(f"  pulse_start={p.pulse_start_ms:.1f} ms, pulse_duration={p.pulse_duration_ms:.1f} ms, pulse_interval={p.pulse_interval_ms:.1f} ms")
    print(f"  adaptation_add={p.spike_adaptation_add_na:.3f} nA, adaptation_decay={p.adaptation_decay_ms:.1f} ms")
    print("-" * 78)

    state = NeuronState(voltage_mv=p.v_rest_mv, max_voltage_mv=p.v_rest_mv, min_voltage_mv=p.v_rest_mv)
    next_print = 0.0
    spike_times: List[float] = []
    voltage_sum = 0.0
    samples = 0

    while state.t_ms < ROUND_DURATION_MS:
        state, spike, current = step_neuron(state, p, DT_MS)
        voltage_sum += state.voltage_mv
        samples += 1
        if spike:
            spike_times.append(state.last_spike_ms)

        if state.t_ms + 1e-9 >= next_print:
            marker = "SPIKE" if spike else "     "
            print(
                f"t={state.t_ms:7.2f} ms | V={state.voltage_mv:7.2f} mV | "
                f"I={current:5.2f} nA | adapt={state.adaptation_na:5.2f} nA | "
                f"spikes={state.spikes:2d} | {marker} | {voltage_bar(state.voltage_mv)}"
            )
            next_print += PRINT_EVERY_MS

    avg_voltage = voltage_sum / max(1, samples)
    duration_s = ROUND_DURATION_MS / 1000.0
    firing_rate_hz = state.spikes / duration_s

    if len(spike_times) >= 2:
        intervals = [b - a for a, b in zip(spike_times[:-1], spike_times[1:])]
        avg_interval_ms = sum(intervals) / len(intervals)
    else:
        avg_interval_ms = 0.0

    result = {
        "round": round_index,
        "spikes": state.spikes,
        "spike_error": TARGET_SPIKES - state.spikes,
        "firing_rate_hz": firing_rate_hz,
        "avg_voltage_mv": avg_voltage,
        "max_voltage_mv": state.max_voltage_mv,
        "min_voltage_mv": state.min_voltage_mv,
        "avg_spike_interval_ms": avg_interval_ms,
        "base_current_na": p.base_current_na,
        "pulse_current_na": p.pulse_current_na,
        "threshold_mv": p.v_threshold_mv,
        "tau_ms": p.membrane_tau_ms,
        "refractory_ms": p.refractory_ms,
        "adaptation_add_na": p.spike_adaptation_add_na,
        "adaptation_decay_ms": p.adaptation_decay_ms,
        "pulse_interval_ms": p.pulse_interval_ms,
        "pulse_duration_ms": p.pulse_duration_ms,
    }

    print("-" * 78)
    print("Round result:")
    print(f"  spikes: {state.spikes}")
    print(f"  target spikes: {TARGET_SPIKES}")
    print(f"  spike error target-observed: {TARGET_SPIKES - state.spikes:+d}")
    print(f"  firing rate: {firing_rate_hz:.2f} Hz")
    print(f"  average voltage: {avg_voltage:.2f} mV")
    print(f"  voltage range: {state.min_voltage_mv:.2f} to {state.max_voltage_mv:.2f} mV")
    if avg_interval_ms:
        print(f"  average spike interval: {avg_interval_ms:.2f} ms")
    else:
        print("  average spike interval: not enough spikes")

    return result


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def ai_controller(previous_params: NeuronParams, result: Dict[str, float], target_spikes: int = TARGET_SPIKES) -> NeuronParams:
    """
    Deterministic target-seeking AI controller.

    Too few spikes  -> make neuron easier to excite.
    Too many spikes -> make neuron harder to excite.
    Exact target    -> preserve parameters with small stabilizing changes.
    """
    p = previous_params
    spikes = int(result["spikes"])
    error = target_spikes - spikes
    abs_error = abs(error)
    new_p = replace(p)

    if error > 0:
        decision = "Neuron fired too little; increase excitability toward the target."
        scale = min(1.0, abs_error / 4.0)

        new_p.base_current_na += 0.10 * scale
        new_p.pulse_current_na += 0.08 * scale
        new_p.v_threshold_mv -= 0.45 * scale
        new_p.refractory_ms -= 0.20 * scale
        new_p.spike_adaptation_add_na -= 0.012 * scale
        new_p.adaptation_decay_ms -= 5.0 * scale
        new_p.pulse_interval_ms -= 6.0 * scale
        new_p.pulse_duration_ms += 4.0 * scale

    elif error < 0:
        decision = "Neuron fired too much; reduce excitability toward the target."
        scale = min(1.0, abs_error / 4.0)

        new_p.base_current_na -= 0.12 * scale
        new_p.pulse_current_na -= 0.10 * scale
        new_p.v_threshold_mv += 0.55 * scale
        new_p.refractory_ms += 0.35 * scale
        new_p.spike_adaptation_add_na += 0.018 * scale
        new_p.adaptation_decay_ms += 8.0 * scale
        new_p.pulse_interval_ms += 8.0 * scale
        new_p.pulse_duration_ms -= 6.0 * scale

    else:
        decision = "Neuron hit the target; preserve parameters with minor stabilization."
        new_p.input_wave_strength_na *= 0.95
        new_p.spike_adaptation_add_na += 0.002

    # Extra interval-based correction. For 6 spikes over 500 ms, a useful
    # interval is roughly 83 ms. If the neuron is overactive and intervals are
    # too short, lengthen recovery and pulse spacing.
    avg_interval = result.get("avg_spike_interval_ms", 0.0)
    ideal_interval = ROUND_DURATION_MS / max(1, target_spikes)

    if spikes > target_spikes and avg_interval and avg_interval < ideal_interval:
        new_p.pulse_interval_ms += 4.0
        new_p.refractory_ms += 0.15
        new_p.spike_adaptation_add_na += 0.006
    elif spikes < target_spikes and (not avg_interval or avg_interval > ideal_interval):
        new_p.pulse_interval_ms -= 3.0
        new_p.refractory_ms -= 0.10

    # Keep parameters in sensible ranges.
    new_p.base_current_na = clamp(new_p.base_current_na, 0.1, 3.0)
    new_p.pulse_current_na = clamp(new_p.pulse_current_na, 0.0, 3.0)
    new_p.v_threshold_mv = clamp(new_p.v_threshold_mv, -62.0, -45.0)
    new_p.membrane_tau_ms = clamp(new_p.membrane_tau_ms, 5.0, 45.0)
    new_p.refractory_ms = clamp(new_p.refractory_ms, 1.0, 12.0)
    new_p.spike_adaptation_add_na = clamp(new_p.spike_adaptation_add_na, 0.0, 0.7)
    new_p.adaptation_decay_ms = clamp(new_p.adaptation_decay_ms, 30.0, 300.0)
    new_p.pulse_duration_ms = clamp(new_p.pulse_duration_ms, 10.0, 250.0)
    new_p.pulse_interval_ms = clamp(new_p.pulse_interval_ms, 45.0, 180.0)
    new_p.input_wave_strength_na = clamp(new_p.input_wave_strength_na, 0.0, 0.25)

    print("\nAI controller decision:")
    print(f"  Target spikes: {target_spikes}")
    print(f"  Observed spikes: {spikes}")
    print(f"  Spike error target-observed: {error:+d}")
    print(f"  {decision}")
    print("  Adjustments:")
    print(f"    base_current -> {new_p.base_current_na:.3f} nA")
    print(f"    pulse_current -> {new_p.pulse_current_na:.3f} nA")
    print(f"    threshold -> {new_p.v_threshold_mv:.2f} mV")
    print(f"    refractory -> {new_p.refractory_ms:.2f} ms")
    print(f"    pulse_interval -> {new_p.pulse_interval_ms:.1f} ms")
    print(f"    pulse_duration -> {new_p.pulse_duration_ms:.1f} ms")
    print(f"    adaptation_add -> {new_p.spike_adaptation_add_na:.3f} nA")
    print("  Next round parameters selected.")

    return new_p


def print_final_summary(results: List[Dict[str, float]]) -> None:
    print("\n" + "=" * 112)
    print("FINAL COMPARISON SUMMARY")
    print("=" * 112)
    header = (
        f"{'Round':>5} | {'Spikes':>6} | {'Err':>4} | {'Hz':>8} | {'Avg V':>9} | "
        f"{'Max V':>9} | {'Base I':>8} | {'Pulse I':>8} | {'Thresh':>8} | "
        f"{'Refrac':>7} | {'PulseInt':>8} | {'PulseDur':>8} | {'Adapt+':>7}"
    )
    print(header)
    print("-" * len(header))

    for r in results:
        print(
            f"{int(r['round']):5d} | {int(r['spikes']):6d} | {int(r['spike_error']):+4d} | "
            f"{r['firing_rate_hz']:8.2f} | {r['avg_voltage_mv']:9.2f} | "
            f"{r['max_voltage_mv']:9.2f} | {r['base_current_na']:8.3f} | "
            f"{r['pulse_current_na']:8.3f} | {r['threshold_mv']:8.2f} | "
            f"{r['refractory_ms']:7.2f} | {r['pulse_interval_ms']:8.1f} | "
            f"{r['pulse_duration_ms']:8.1f} | {r['adaptation_add_na']:7.3f}"
        )

    best = min(results, key=lambda r: (abs(r["spikes"] - TARGET_SPIKES), r["round"]))
    exact = [r for r in results if r["spikes"] == TARGET_SPIKES]

    print("\nBest target match:")
    print(
        f"  Round {int(best['round'])}: {int(best['spikes'])} spikes, "
        f"error {int(best['spike_error']):+d}, {best['firing_rate_hz']:.2f} Hz"
    )
    if exact:
        exact_rounds = ", ".join(str(int(r["round"])) for r in exact)
        print(f"  Exact target achieved in round(s): {exact_rounds}")
    else:
        print("  Exact target was not achieved; closest round is shown above.")


def main() -> None:
    print("Neuron AI Rounds Simulation - Targeted Spike Controller")
    print("Terminal-only simulation. No external packages.")
    print("Model: leaky integrate-and-fire neuron with stimulus pulses, refractory period, and adaptation.")
    print(f"Target spike count: {TARGET_SPIKES}")

    params = NeuronParams()
    results: List[Dict[str, float]] = []

    for round_index in range(1, TOTAL_ROUNDS + 1):
        result = run_round(round_index, params)
        results.append(result)
        if round_index < TOTAL_ROUNDS:
            params = ai_controller(params, result, TARGET_SPIKES)

    print_final_summary(results)


if __name__ == "__main__":
    main()
