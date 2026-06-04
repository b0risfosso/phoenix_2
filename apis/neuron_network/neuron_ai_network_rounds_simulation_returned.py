#!/usr/bin/env python3
"""
Neuron AI Network Rounds Simulation
-----------------------------------
A terminal-only Python simulation of a small spiking neuron network.

Run:
    python neuron_ai_network_rounds_simulation.py

No external packages are required.

The model expands a single leaky integrate-and-fire neuron into a network
of excitatory and inhibitory neurons. Each neuron follows:
    dV/dt = (-(V - V_rest) + R_m * I_total) / tau_m

When a neuron spikes, it sends scheduled synaptic events to other neurons.
Excitatory events increase current. Inhibitory events reduce current.
The AI controller tunes excitation, inhibition, synaptic delay, and firing
balance after each round.
"""

from dataclasses import dataclass, replace
from math import exp
from typing import Dict, List, Tuple


# ============================================================
# CONFIGURATION
# ============================================================

TOTAL_ROUNDS = 8
ROUND_DURATION_MS = 600.0
DT_MS = 0.25
PRINT_EVERY_MS = 25.0

DISPLAY_WIDTH = 34
NETWORK_SIZE = 6
EXCITATORY_COUNT = 4
INHIBITORY_COUNT = NETWORK_SIZE - EXCITATORY_COUNT


@dataclass
class NetworkParams:
    # Electrical properties shared by all neurons
    v_rest_mv: float = -70.0
    v_reset_mv: float = -75.0
    v_threshold_mv: float = -54.0
    membrane_resistance_mohm: float = 10.0
    membrane_tau_ms: float = 20.0
    refractory_ms: float = 4.0

    # External input current properties
    base_current_na: float = 1.05
    pulse_current_na: float = 0.70
    pulse_start_ms: float = 70.0
    pulse_duration_ms: float = 38.0
    pulse_interval_ms: float = 90.0

    # Neuron-to-neuron connection properties
    excitatory_weight_na: float = 0.34
    inhibitory_weight_na: float = 0.42
    exc_to_inh_scale: float = 1.15
    inh_to_exc_scale: float = 1.10
    synaptic_delay_ms: float = 3.0
    delay_spread_ms: float = 1.0
    synaptic_decay_ms: float = 8.0

    # Ion/recovery terms, simplified
    adaptation_decay_ms: float = 130.0
    spike_adaptation_add_na: float = 0.12

    # Deterministic variation terms
    input_wave_strength_na: float = 0.07
    input_wave_period_ms: float = 64.0
    neuron_bias_step_na: float = 0.045


@dataclass
class NeuronState:
    t_ms: float = 0.0
    voltage_mv: float = -70.0
    refractory_left_ms: float = 0.0
    adaptation_na: float = 0.0
    synaptic_current_na: float = 0.0
    spikes: int = 0
    last_spike_ms: float = -999999.0
    max_voltage_mv: float = -70.0
    min_voltage_mv: float = -70.0


@dataclass
class SynapticEvent:
    delivery_ms: float
    target_index: int
    current_na: float
    source_index: int


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


def neuron_type(index: int) -> str:
    return "E" if index < EXCITATORY_COUNT else "I"


def stimulus_current_na(t_ms: float, neuron_index: int, p: NetworkParams) -> float:
    """External current with pulses, waves, and slight per-neuron bias."""
    center = (NETWORK_SIZE - 1) / 2.0
    bias = (neuron_index - center) * p.neuron_bias_step_na
    current = p.base_current_na + bias

    # Stagger pulse phase slightly so the network does not spike as one block.
    local_t = t_ms - neuron_index * 8.0
    if local_t >= p.pulse_start_ms and p.pulse_interval_ms > 0:
        phase = (local_t - p.pulse_start_ms) % p.pulse_interval_ms
        if phase <= p.pulse_duration_ms:
            current += p.pulse_current_na

    wave_period = p.input_wave_period_ms + neuron_index * 5.0
    current += p.input_wave_strength_na * triangular_wave(t_ms + neuron_index * 11.0, wave_period)
    return current


def connection_weight_na(source_index: int, target_index: int, p: NetworkParams) -> float:
    """Return signed synaptic current for source -> target."""
    if source_index == target_index:
        return 0.0

    source_type = neuron_type(source_index)
    target_type = neuron_type(target_index)

    if source_type == "E":
        weight = p.excitatory_weight_na
        if target_type == "I":
            weight *= p.exc_to_inh_scale
    else:
        weight = -p.inhibitory_weight_na
        if target_type == "E":
            weight *= p.inh_to_exc_scale

    # Nearby neurons are slightly more connected than distant neurons.
    distance = abs(source_index - target_index)
    distance_scale = max(0.55, 1.0 - 0.09 * distance)
    return weight * distance_scale


def connection_delay_ms(source_index: int, target_index: int, p: NetworkParams) -> float:
    """Deterministic delay with a small distance-based spread."""
    distance = abs(source_index - target_index)
    return p.synaptic_delay_ms + p.delay_spread_ms * distance


def deliver_synaptic_events(states: List[NeuronState], events: List[SynapticEvent], t_ms: float) -> Tuple[List[NeuronState], List[SynapticEvent], int]:
    """Apply all synaptic events due at the current time."""
    remaining: List[SynapticEvent] = []
    delivered = 0

    for event in events:
        if event.delivery_ms <= t_ms + 1e-9:
            states[event.target_index].synaptic_current_na += event.current_na
            delivered += 1
        else:
            remaining.append(event)

    return states, remaining, delivered


def step_network(
    states: List[NeuronState],
    p: NetworkParams,
    dt_ms: float,
    events: List[SynapticEvent],
) -> Tuple[List[NeuronState], List[SynapticEvent], List[int], List[float], int]:
    """Advance the whole network one time step."""
    t_ms = states[0].t_ms
    states, events, delivered_count = deliver_synaptic_events(states, events, t_ms)

    new_states: List[NeuronState] = []
    spiking_indices: List[int] = []
    external_currents: List[float] = []

    for i, state in enumerate(states):
        s = replace(state)
        external_current = stimulus_current_na(s.t_ms, i, p)
        external_currents.append(external_current)

        effective_current = external_current + s.synaptic_current_na - s.adaptation_na

        if s.refractory_left_ms > 0.0:
            s.voltage_mv = p.v_reset_mv
            s.refractory_left_ms = max(0.0, s.refractory_left_ms - dt_ms)
        else:
            dv_dt = (-(s.voltage_mv - p.v_rest_mv) + p.membrane_resistance_mohm * effective_current) / p.membrane_tau_ms
            s.voltage_mv += dv_dt * dt_ms
            if s.voltage_mv >= p.v_threshold_mv:
                s.spikes += 1
                s.last_spike_ms = s.t_ms
                s.voltage_mv = p.v_reset_mv
                s.refractory_left_ms = p.refractory_ms
                s.adaptation_na += p.spike_adaptation_add_na
                spiking_indices.append(i)

        # Exponential decay for adaptation and synaptic current.
        if p.adaptation_decay_ms > 0:
            s.adaptation_na *= exp(-dt_ms / p.adaptation_decay_ms)
        if p.synaptic_decay_ms > 0:
            s.synaptic_current_na *= exp(-dt_ms / p.synaptic_decay_ms)

        s.t_ms += dt_ms
        s.max_voltage_mv = max(s.max_voltage_mv, s.voltage_mv)
        s.min_voltage_mv = min(s.min_voltage_mv, s.voltage_mv)
        new_states.append(s)

    # Schedule new synaptic events after all neurons have been stepped.
    for source_index in spiking_indices:
        for target_index in range(NETWORK_SIZE):
            weight = connection_weight_na(source_index, target_index, p)
            if weight == 0.0:
                continue
            events.append(
                SynapticEvent(
                    delivery_ms=t_ms + connection_delay_ms(source_index, target_index, p),
                    target_index=target_index,
                    current_na=weight,
                    source_index=source_index,
                )
            )

    return new_states, events, spiking_indices, external_currents, delivered_count


def voltage_bar(voltage_mv: float, v_min: float = -80.0, v_max: float = -45.0) -> str:
    """Small terminal visualization of voltage position."""
    x = int((voltage_mv - v_min) / (v_max - v_min) * (DISPLAY_WIDTH - 1))
    x = max(0, min(DISPLAY_WIDTH - 1, x))
    chars = ["-"] * DISPLAY_WIDTH
    chars[x] = "|"
    return "".join(chars)


def network_snapshot(states: List[NeuronState], spiking_indices: List[int]) -> str:
    """Compact one-line status for all neurons."""
    parts = []
    for i, s in enumerate(states):
        label = f"{neuron_type(i)}{i}"
        spike_mark = "*" if i in spiking_indices else " "
        parts.append(f"{label}{spike_mark}:{s.voltage_mv:6.1f}mV[{voltage_bar(s.voltage_mv)}]")
    return "  ".join(parts)


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def run_round(round_index: int, p: NetworkParams) -> Dict[str, float]:
    print("\n" + "=" * 92)
    print(f"ROUND {round_index}")
    print("=" * 92)
    print("Network:")
    print(f"  neurons={NETWORK_SIZE} ({EXCITATORY_COUNT} excitatory, {INHIBITORY_COUNT} inhibitory)")
    print("Parameters:")
    print(f"  V_rest={p.v_rest_mv:.1f} mV, V_threshold={p.v_threshold_mv:.1f} mV, V_reset={p.v_reset_mv:.1f} mV")
    print(f"  Rm={p.membrane_resistance_mohm:.2f} Mohm, tau={p.membrane_tau_ms:.2f} ms, refractory={p.refractory_ms:.2f} ms")
    print(f"  external base={p.base_current_na:.3f} nA, pulse={p.pulse_current_na:.3f} nA")
    print(f"  synapse E={p.excitatory_weight_na:.3f} nA, I={p.inhibitory_weight_na:.3f} nA, delay={p.synaptic_delay_ms:.2f} ms, spread={p.delay_spread_ms:.2f} ms")
    print(f"  adaptation_add={p.spike_adaptation_add_na:.3f} nA, synaptic_decay={p.synaptic_decay_ms:.1f} ms")
    print("-" * 92)

    states = [
        NeuronState(
            voltage_mv=p.v_rest_mv - 0.4 * i,
            max_voltage_mv=p.v_rest_mv - 0.4 * i,
            min_voltage_mv=p.v_rest_mv - 0.4 * i,
        )
        for i in range(NETWORK_SIZE)
    ]
    events: List[SynapticEvent] = []
    next_print = 0.0
    voltage_sum = [0.0 for _ in range(NETWORK_SIZE)]
    samples = 0
    total_delivered_events = 0
    spike_timeline: List[Tuple[float, int]] = []

    while states[0].t_ms < ROUND_DURATION_MS:
        states, events, spiking_indices, external_currents, delivered = step_network(states, p, DT_MS, events)
        total_delivered_events += delivered
        samples += 1
        for i, s in enumerate(states):
            voltage_sum[i] += s.voltage_mv
        for i in spiking_indices:
            spike_timeline.append((states[i].last_spike_ms, i))

        if states[0].t_ms + 1e-9 >= next_print:
            spike_labels = ",".join(f"{neuron_type(i)}{i}" for i in spiking_indices) if spiking_indices else "none"
            exc_spikes_now = sum(1 for i in spiking_indices if neuron_type(i) == "E")
            inh_spikes_now = len(spiking_indices) - exc_spikes_now
            avg_syn = sum(s.synaptic_current_na for s in states) / NETWORK_SIZE
            print(
                f"t={states[0].t_ms:7.2f} ms | spikes now={spike_labels:10s} | "
                f"E_now={exc_spikes_now} I_now={inh_spikes_now} | pending_events={len(events):3d} | avg_syn={avg_syn:6.3f} nA"
            )
            print("  " + network_snapshot(states, spiking_indices))
            next_print += PRINT_EVERY_MS

    duration_s = ROUND_DURATION_MS / 1000.0
    spike_counts = [s.spikes for s in states]
    exc_spikes = sum(spike_counts[:EXCITATORY_COUNT])
    inh_spikes = sum(spike_counts[EXCITATORY_COUNT:])
    total_spikes = exc_spikes + inh_spikes
    exc_rate_hz = exc_spikes / max(1, EXCITATORY_COUNT) / duration_s
    inh_rate_hz = inh_spikes / max(1, INHIBITORY_COUNT) / duration_s
    network_rate_hz = total_spikes / NETWORK_SIZE / duration_s
    balance_ratio = exc_spikes / max(1, inh_spikes)
    mean_voltage = sum(voltage_sum) / max(1, samples * NETWORK_SIZE)
    max_voltage = max(s.max_voltage_mv for s in states)
    min_voltage = min(s.min_voltage_mv for s in states)

    result = {
        "round": round_index,
        "total_spikes": total_spikes,
        "exc_spikes": exc_spikes,
        "inh_spikes": inh_spikes,
        "network_rate_hz": network_rate_hz,
        "exc_rate_hz": exc_rate_hz,
        "inh_rate_hz": inh_rate_hz,
        "balance_ratio": balance_ratio,
        "mean_voltage_mv": mean_voltage,
        "max_voltage_mv": max_voltage,
        "min_voltage_mv": min_voltage,
        "delivered_events": total_delivered_events,
        "pending_events_end": len(events),
        "base_current_na": p.base_current_na,
        "pulse_current_na": p.pulse_current_na,
        "excitatory_weight_na": p.excitatory_weight_na,
        "inhibitory_weight_na": p.inhibitory_weight_na,
        "synaptic_delay_ms": p.synaptic_delay_ms,
        "delay_spread_ms": p.delay_spread_ms,
        "threshold_mv": p.v_threshold_mv,
    }

    print("-" * 92)
    print("Round result:")
    print(f"  total spikes: {total_spikes}")
    print(f"  excitatory spikes: {exc_spikes}, inhibitory spikes: {inh_spikes}, E/I spike ratio: {balance_ratio:.2f}")
    print(f"  mean network firing rate: {network_rate_hz:.2f} Hz per neuron")
    print(f"  E rate: {exc_rate_hz:.2f} Hz per E neuron, I rate: {inh_rate_hz:.2f} Hz per I neuron")
    print(f"  average voltage: {mean_voltage:.2f} mV")
    print(f"  voltage range: {min_voltage:.2f} to {max_voltage:.2f} mV")
    print(f"  synaptic events delivered: {total_delivered_events}")
    print("  spike counts by neuron:")
    for i, count in enumerate(spike_counts):
        print(f"    {neuron_type(i)}{i}: {count}")

    return result


def ai_controller(previous_params: NetworkParams, result: Dict[str, float]) -> NetworkParams:
    """
    Simple deterministic AI controller.

    Goals:
        1. Keep network activity alive but not saturated.
        2. Keep excitatory and inhibitory firing in a useful balance.
        3. Explore connection delays once the firing pattern is reasonable.
    """
    p = previous_params
    new_p = replace(p)

    total_spikes = result["total_spikes"]
    balance_ratio = result["balance_ratio"]
    target_total_spikes = 28
    target_balance_ratio = 2.0  # four E neurons and two I neurons makes ~2:1 a useful rough target.

    decisions: List[str] = []

    if total_spikes < target_total_spikes - 8:
        decisions.append("Network was underactive; increase drive and strengthen excitatory connections.")
        new_p.base_current_na += 0.10
        new_p.pulse_current_na += 0.08
        new_p.excitatory_weight_na += 0.035
        new_p.v_threshold_mv -= 0.4
    elif total_spikes > target_total_spikes + 10:
        decisions.append("Network was overactive; reduce drive and strengthen inhibitory control.")
        new_p.base_current_na -= 0.08
        new_p.pulse_current_na -= 0.05
        new_p.inhibitory_weight_na += 0.045
        new_p.spike_adaptation_add_na += 0.012
        new_p.v_threshold_mv += 0.35
    else:
        decisions.append("Network activity was near target; tune timing and recovery.")
        new_p.synaptic_delay_ms += 0.35
        new_p.delay_spread_ms += 0.08
        new_p.synaptic_decay_ms += 0.5
        new_p.input_wave_strength_na += 0.006

    if balance_ratio > target_balance_ratio + 0.75:
        decisions.append("Excitation dominated inhibition; strengthen inhibitory output and E-to-I recruitment.")
        new_p.inhibitory_weight_na += 0.035
        new_p.exc_to_inh_scale += 0.035
        new_p.inh_to_exc_scale += 0.025
    elif balance_ratio < target_balance_ratio - 0.75:
        decisions.append("Inhibition dominated or matched excitation too strongly; strengthen excitatory spread.")
        new_p.excitatory_weight_na += 0.03
        new_p.inhibitory_weight_na -= 0.025
        new_p.inh_to_exc_scale -= 0.02
    else:
        decisions.append("E/I balance was acceptable; make a small delay exploration step.")
        if result["network_rate_hz"] > 10.0:
            new_p.synaptic_delay_ms += 0.2
        else:
            new_p.synaptic_delay_ms -= 0.1

    # Keep parameters in sensible ranges.
    new_p.base_current_na = clamp(new_p.base_current_na, 0.1, 3.0)
    new_p.pulse_current_na = clamp(new_p.pulse_current_na, 0.0, 3.0)
    new_p.v_threshold_mv = clamp(new_p.v_threshold_mv, -62.0, -45.0)
    new_p.membrane_tau_ms = clamp(new_p.membrane_tau_ms, 5.0, 45.0)
    new_p.refractory_ms = clamp(new_p.refractory_ms, 1.0, 12.0)
    new_p.excitatory_weight_na = clamp(new_p.excitatory_weight_na, 0.02, 1.4)
    new_p.inhibitory_weight_na = clamp(new_p.inhibitory_weight_na, 0.02, 1.6)
    new_p.exc_to_inh_scale = clamp(new_p.exc_to_inh_scale, 0.4, 2.5)
    new_p.inh_to_exc_scale = clamp(new_p.inh_to_exc_scale, 0.4, 2.5)
    new_p.synaptic_delay_ms = clamp(new_p.synaptic_delay_ms, 0.5, 12.0)
    new_p.delay_spread_ms = clamp(new_p.delay_spread_ms, 0.0, 4.0)
    new_p.synaptic_decay_ms = clamp(new_p.synaptic_decay_ms, 2.0, 30.0)
    new_p.spike_adaptation_add_na = clamp(new_p.spike_adaptation_add_na, 0.0, 0.7)
    new_p.adaptation_decay_ms = clamp(new_p.adaptation_decay_ms, 30.0, 300.0)
    new_p.input_wave_strength_na = clamp(new_p.input_wave_strength_na, 0.0, 0.25)

    print("\nAI controller decision:")
    print(f"  Target total spikes: {target_total_spikes}")
    print(f"  Observed total spikes: {total_spikes}")
    print(f"  Target E/I ratio: {target_balance_ratio:.2f}")
    print(f"  Observed E/I ratio: {balance_ratio:.2f}")
    for decision in decisions:
        print(f"  {decision}")
    print("  Next round network parameters selected.")

    return new_p


def print_final_summary(results: List[Dict[str, float]]) -> None:
    print("\n" + "=" * 112)
    print("FINAL NETWORK COMPARISON SUMMARY")
    print("=" * 112)
    header = (
        f"{'Round':>5} | {'Total':>6} | {'E':>5} | {'I':>5} | {'Hz/N':>8} | "
        f"{'E/I':>7} | {'E_w':>7} | {'I_w':>7} | {'Delay':>7} | {'Events':>8} | {'Thresh':>8}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{int(r['round']):5d} | {int(r['total_spikes']):6d} | {int(r['exc_spikes']):5d} | "
            f"{int(r['inh_spikes']):5d} | {r['network_rate_hz']:8.2f} | {r['balance_ratio']:7.2f} | "
            f"{r['excitatory_weight_na']:7.3f} | {r['inhibitory_weight_na']:7.3f} | "
            f"{r['synaptic_delay_ms']:7.2f} | {int(r['delivered_events']):8d} | {r['threshold_mv']:8.2f}"
        )

    target_total_spikes = 28
    target_balance_ratio = 2.0
    best = min(
        results,
        key=lambda r: abs(r["total_spikes"] - target_total_spikes) + 4.0 * abs(r["balance_ratio"] - target_balance_ratio),
    )
    print("\nBest network target match:")
    print(
        f"  Round {int(best['round'])}: {int(best['total_spikes'])} total spikes, "
        f"E/I ratio {best['balance_ratio']:.2f}, "
        f"mean rate {best['network_rate_hz']:.2f} Hz per neuron"
    )


def main() -> None:
    print("Neuron AI Network Rounds Simulation")
    print("Terminal-only simulation. No external packages.")
    print("Model: small excitatory/inhibitory leaky integrate-and-fire network with delayed synapses.")

    params = NetworkParams()
    results: List[Dict[str, float]] = []

    for round_index in range(1, TOTAL_ROUNDS + 1):
        result = run_round(round_index, params)
        results.append(result)
        if round_index < TOTAL_ROUNDS:
            params = ai_controller(params, result)

    print_final_summary(results)


if __name__ == "__main__":
    main()
