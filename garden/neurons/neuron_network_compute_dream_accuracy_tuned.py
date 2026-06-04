#!/usr/bin/env python3
"""
Neuron Network Computation and Dream Simulation - Accuracy Tuned
-----------------------------------------------
A terminal-only Python simulation of a small spiking neuron network that can:

1. Perform simple computations by classifying input patterns.
2. Store spike patterns as memory traces.
3. Enter dream mode, reduce external input, and replay/recombine stored traces.
4. Let a rule-based AI controller tune excitability, inhibition, timing, and replay strength.

Run:
    python neuron_network_compute_dream_accuracy_tuned.py

No external packages are required.
"""

from dataclasses import dataclass, field, replace
from math import exp
from typing import Dict, List, Tuple, Optional


# ============================================================
# CONFIGURATION
# ============================================================

TOTAL_ROUNDS = 8
ROUND_DURATION_MS = 700.0
DT_MS = 0.25
PRINT_EVERY_MS = 25.0

DISPLAY_WIDTH = 34
NETWORK_SIZE = 6
EXCITATORY_COUNT = 4
INHIBITORY_COUNT = NETWORK_SIZE - EXCITATORY_COUNT

OUTPUT_A_INDEX = 3
OUTPUT_B_INDEX = 5


@dataclass
class NetworkParams:
    # Electrical properties shared by all neurons
    v_rest_mv: float = -70.0
    v_reset_mv: float = -75.0
    v_threshold_mv: float = -55.5
    membrane_resistance_mohm: float = 10.0
    membrane_tau_ms: float = 20.0
    refractory_ms: float = 4.0

    # External input current properties
    base_current_na: float = 0.90
    pulse_current_na: float = 0.58
    pulse_start_ms: float = 70.0
    pulse_duration_ms: float = 38.0
    pulse_interval_ms: float = 90.0

    # Pattern-computation drive
    pattern_drive_na: float = 0.72
    output_gate_na: float = 0.30
    classification_margin_spikes: int = 1

    # Neuron-to-neuron connection properties
    excitatory_weight_na: float = 0.27
    inhibitory_weight_na: float = 0.46
    exc_to_inh_scale: float = 1.15
    inh_to_exc_scale: float = 1.10
    synaptic_delay_ms: float = 3.0
    delay_spread_ms: float = 1.0
    synaptic_decay_ms: float = 8.0

    # Learned association and dream properties
    learned_weight_scale_na: float = 0.14
    learning_rate: float = 0.018
    dream_external_silence: float = 0.48
    dream_replay_boost_na: float = 0.72
    dream_recombination_na: float = 0.16
    dream_cycle_ms: float = 180.0

    # Ion/recovery terms, simplified
    adaptation_decay_ms: float = 130.0
    spike_adaptation_add_na: float = 0.16

    # Deterministic variation terms
    input_wave_strength_na: float = 0.06
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


@dataclass
class MemoryTrace:
    label: str
    round_number: int
    spike_timeline: List[Tuple[float, int]]
    spike_counts: List[int]
    output_a_spikes: int
    output_b_spikes: int
    decision: str


@dataclass
class NetworkMemory:
    traces: List[MemoryTrace] = field(default_factory=list)
    learned_weights: List[List[float]] = field(
        default_factory=lambda: [[0.0 for _ in range(NETWORK_SIZE)] for _ in range(NETWORK_SIZE)]
    )


@dataclass
class RoundPlan:
    mode: str                 # "compute" or "dream"
    pattern_label: str        # "A", "B", or "DREAM"
    description: str


# ============================================================
# UTILITY AND NETWORK PHYSICS
# ============================================================

def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


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


def choose_round_plan(round_index: int) -> RoundPlan:
    plans = [
        RoundPlan("compute", "A", "COMPUTE: Pattern A drives E0/E1; output A should dominate."),
        RoundPlan("compute", "B", "COMPUTE: Pattern B drives E2/E3; output B should dominate."),
        RoundPlan("compute", "A", "COMPUTE: repeat Pattern A to reinforce a memory trace."),
        RoundPlan("compute", "B", "COMPUTE: repeat Pattern B to reinforce a memory trace."),
        RoundPlan("dream", "DREAM", "DREAM: reduce outside input and replay stored traces."),
        RoundPlan("dream", "DREAM", "DREAM: replay with mild recombination between traces."),
        RoundPlan("compute", "A", "WAKE TEST: classify Pattern A after dream replay."),
        RoundPlan("compute", "B", "WAKE TEST: classify Pattern B after dream replay."),
    ]
    return plans[(round_index - 1) % len(plans)]


def base_stimulus_current_na(t_ms: float, neuron_index: int, p: NetworkParams) -> float:
    """Baseline current with pulses, waves, and slight per-neuron bias."""
    center = (NETWORK_SIZE - 1) / 2.0
    bias = (neuron_index - center) * p.neuron_bias_step_na
    current = p.base_current_na + bias

    local_t = t_ms - neuron_index * 8.0
    if local_t >= p.pulse_start_ms and p.pulse_interval_ms > 0:
        phase = (local_t - p.pulse_start_ms) % p.pulse_interval_ms
        if phase <= p.pulse_duration_ms:
            current += p.pulse_current_na

    wave_period = p.input_wave_period_ms + neuron_index * 5.0
    current += p.input_wave_strength_na * triangular_wave(t_ms + neuron_index * 11.0, wave_period)
    return current


def pattern_current_na(t_ms: float, neuron_index: int, p: NetworkParams, plan: RoundPlan) -> float:
    """Extra input that encodes a computational pattern.

    Accuracy-tuned behavior:
    - The intended input pair still receives the pattern drive.
    - The correct output neuron receives a longer output-gate cue.
    - The competing output neuron receives a small suppressive counter-gate.
    This makes A/B classification less likely to end as a tie.
    """
    if plan.mode != "compute":
        return 0.0

    # Two presentation windows make the input a temporal pattern rather than a single static pulse.
    in_window_1 = 90.0 <= t_ms <= 230.0
    in_window_2 = 330.0 <= t_ms <= 500.0
    output_gate_window = 150.0 <= t_ms <= 520.0
    current = 0.0

    if plan.pattern_label == "A":
        if neuron_index in (0, 1) and in_window_1:
            current += p.pattern_drive_na
        if neuron_index == 1 and in_window_2:
            current += p.pattern_drive_na * 0.90
        if neuron_index == OUTPUT_A_INDEX and output_gate_window:
            current += p.output_gate_na * 2.50
        if neuron_index == OUTPUT_B_INDEX and output_gate_window:
            current -= p.output_gate_na * 0.70

    elif plan.pattern_label == "B":
        if neuron_index in (2, 3) and in_window_1:
            current += p.pattern_drive_na
        if neuron_index == 2 and in_window_2:
            current += p.pattern_drive_na * 0.90
        if neuron_index == OUTPUT_B_INDEX and output_gate_window:
            current += p.output_gate_na * 2.50
        if neuron_index == OUTPUT_A_INDEX and output_gate_window:
            current -= p.output_gate_na * 0.70

    return current


def dream_current_na(t_ms: float, neuron_index: int, p: NetworkParams, memory: NetworkMemory) -> float:
    """Internal replay current used during dream mode."""
    if not memory.traces:
        return 0.0

    # Pick two traces deterministically and replay them against the same dream cycle.
    primary = memory.traces[int((t_ms // p.dream_cycle_ms) % len(memory.traces))]
    secondary = memory.traces[int(((t_ms // p.dream_cycle_ms) + 1) % len(memory.traces))]
    local_t = t_ms % p.dream_cycle_ms

    current = 0.0
    replay_window = 7.5

    for spike_t, idx in primary.spike_timeline:
        if idx == neuron_index and abs((spike_t % p.dream_cycle_ms) - local_t) <= replay_window:
            current += p.dream_replay_boost_na

    # Recombination: a weaker pulse from a different stored pattern can enter the dream.
    for spike_t, idx in secondary.spike_timeline:
        if idx == neuron_index and abs((spike_t % p.dream_cycle_ms) - local_t) <= replay_window * 0.6:
            current += p.dream_recombination_na

    # Mild internally generated rhythm keeps dreams from being exact copies.
    current += 0.04 * triangular_wave(t_ms + neuron_index * 17.0, 90.0 + 6.0 * neuron_index)
    return current


def stimulus_current_na(t_ms: float, neuron_index: int, p: NetworkParams, plan: RoundPlan, memory: NetworkMemory) -> float:
    if plan.mode == "dream":
        # Dream mode turns down outside input and adds internally generated replay.
        return (
            p.dream_external_silence * base_stimulus_current_na(t_ms, neuron_index, p)
            + dream_current_na(t_ms, neuron_index, p, memory)
        )

    return base_stimulus_current_na(t_ms, neuron_index, p) + pattern_current_na(t_ms, neuron_index, p, plan)


def connection_weight_na(source_index: int, target_index: int, p: NetworkParams, memory: NetworkMemory) -> float:
    """Return signed synaptic current for source -> target, including learned associations."""
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

    distance = abs(source_index - target_index)
    distance_scale = max(0.55, 1.0 - 0.09 * distance)

    learned = p.learned_weight_scale_na * memory.learned_weights[source_index][target_index]
    if source_type == "I":
        learned *= -0.35

    return weight * distance_scale + learned


def connection_delay_ms(source_index: int, target_index: int, p: NetworkParams) -> float:
    distance = abs(source_index - target_index)
    return p.synaptic_delay_ms + p.delay_spread_ms * distance


def deliver_synaptic_events(states: List[NeuronState], events: List[SynapticEvent], t_ms: float) -> Tuple[List[NeuronState], List[SynapticEvent], int]:
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
    plan: RoundPlan,
    memory: NetworkMemory,
) -> Tuple[List[NeuronState], List[SynapticEvent], List[int], List[float], int]:
    """Advance the whole network one time step."""
    t_ms = states[0].t_ms
    states, events, delivered_count = deliver_synaptic_events(states, events, t_ms)

    new_states: List[NeuronState] = []
    spiking_indices: List[int] = []
    external_currents: List[float] = []

    for i, state in enumerate(states):
        s = replace(state)
        external_current = stimulus_current_na(s.t_ms, i, p, plan, memory)
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

        if p.adaptation_decay_ms > 0:
            s.adaptation_na *= exp(-dt_ms / p.adaptation_decay_ms)
        if p.synaptic_decay_ms > 0:
            s.synaptic_current_na *= exp(-dt_ms / p.synaptic_decay_ms)

        s.t_ms += dt_ms
        s.max_voltage_mv = max(s.max_voltage_mv, s.voltage_mv)
        s.min_voltage_mv = min(s.min_voltage_mv, s.voltage_mv)
        new_states.append(s)

    for source_index in spiking_indices:
        for target_index in range(NETWORK_SIZE):
            weight = connection_weight_na(source_index, target_index, p, memory)
            if abs(weight) < 1e-12:
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


# ============================================================
# DISPLAY, COMPUTATION, MEMORY
# ============================================================

def voltage_bar(voltage_mv: float, v_min: float = -80.0, v_max: float = -45.0) -> str:
    x = int((voltage_mv - v_min) / (v_max - v_min) * (DISPLAY_WIDTH - 1))
    x = max(0, min(DISPLAY_WIDTH - 1, x))
    chars = ["-"] * DISPLAY_WIDTH
    chars[x] = "|"
    return "".join(chars)


def network_snapshot(states: List[NeuronState], spiking_indices: List[int]) -> str:
    parts = []
    for i, s in enumerate(states):
        label = f"{neuron_type(i)}{i}"
        spike_mark = "*" if i in spiking_indices else " "
        parts.append(f"{label}{spike_mark}:{s.voltage_mv:6.1f}mV[{voltage_bar(s.voltage_mv)}]")
    return "  ".join(parts)


def classify_output(spike_counts: List[int], margin: int) -> Tuple[str, str, bool]:
    a = spike_counts[OUTPUT_A_INDEX]
    b = spike_counts[OUTPUT_B_INDEX]
    if a >= b + margin:
        return "A", f"A-output E{OUTPUT_A_INDEX} dominated B-output I{OUTPUT_B_INDEX} ({a} vs {b})", True
    if b >= a + margin:
        return "B", f"B-output I{OUTPUT_B_INDEX} dominated A-output E{OUTPUT_A_INDEX} ({b} vs {a})", True
    return "uncertain", f"outputs were too close: A={a}, B={b}", False


def update_memory(memory: NetworkMemory, trace: MemoryTrace, p: NetworkParams) -> None:
    """Store trace and apply a simple fire-together-wire-together association."""
    memory.traces.append(trace)
    if len(memory.traces) > 8:
        memory.traces.pop(0)

    # Pair spikes within a short temporal window.
    window_ms = 18.0
    for t1, i in trace.spike_timeline:
        for t2, j in trace.spike_timeline:
            if i == j:
                continue
            if 0.0 < t2 - t1 <= window_ms:
                memory.learned_weights[i][j] += p.learning_rate

    # Normalize to keep learned influence bounded.
    for i in range(NETWORK_SIZE):
        for j in range(NETWORK_SIZE):
            memory.learned_weights[i][j] = clamp(memory.learned_weights[i][j], -0.3, 0.85)


def dream_match_report(spike_counts: List[int], memory: NetworkMemory) -> Tuple[str, float]:
    if not memory.traces:
        return "no stored traces to compare", 0.0

    best_label = "none"
    best_score = -1.0
    total = sum(spike_counts) or 1
    normalized = [c / total for c in spike_counts]

    for trace in memory.traces:
        trace_total = sum(trace.spike_counts) or 1
        trace_normalized = [c / trace_total for c in trace.spike_counts]
        similarity = 1.0 - 0.5 * sum(abs(a - b) for a, b in zip(normalized, trace_normalized))
        if similarity > best_score:
            best_score = similarity
            best_label = f"{trace.label} from round {trace.round_number}"

    return f"closest replay resembles {best_label}", best_score


def learned_weight_summary(memory: NetworkMemory) -> str:
    pairs: List[Tuple[float, int, int]] = []
    for i in range(NETWORK_SIZE):
        for j in range(NETWORK_SIZE):
            if i != j:
                pairs.append((memory.learned_weights[i][j], i, j))
    pairs.sort(reverse=True)
    top = [(w, i, j) for w, i, j in pairs[:4] if w > 0.001]
    if not top:
        return "none yet"
    return ", ".join(f"{neuron_type(i)}{i}->{neuron_type(j)}{j}:{w:.2f}" for w, i, j in top)


# ============================================================
# ROUND SIMULATION
# ============================================================

def run_round(round_index: int, p: NetworkParams, memory: NetworkMemory, plan: RoundPlan) -> Dict[str, float | str | bool]:
    print("\n" + "=" * 104)
    print(f"ROUND {round_index} | {plan.description}")
    print("=" * 104)
    print("Network:")
    print(f"  neurons={NETWORK_SIZE} ({EXCITATORY_COUNT} excitatory, {INHIBITORY_COUNT} inhibitory)")
    print("Roles:")
    print(f"  Pattern A input: E0/E1 -> output A = E{OUTPUT_A_INDEX}")
    print(f"  Pattern B input: E2/E3 -> output B = I{OUTPUT_B_INDEX}")
    print("Parameters:")
    print(f"  V_rest={p.v_rest_mv:.1f} mV, V_threshold={p.v_threshold_mv:.1f} mV, V_reset={p.v_reset_mv:.1f} mV")
    print(f"  external base={p.base_current_na:.3f} nA, pulse={p.pulse_current_na:.3f} nA, pattern_drive={p.pattern_drive_na:.3f} nA")
    print(f"  synapse E={p.excitatory_weight_na:.3f} nA, I={p.inhibitory_weight_na:.3f} nA, learned_scale={p.learned_weight_scale_na:.3f} nA")
    print(f"  dream_silence={p.dream_external_silence:.2f}, dream_replay={p.dream_replay_boost_na:.3f} nA, recombination={p.dream_recombination_na:.3f} nA")
    print(f"  memory traces stored={len(memory.traces)}, strongest learned links={learned_weight_summary(memory)}")
    print("-" * 104)

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
        states, events, spiking_indices, external_currents, delivered = step_network(states, p, DT_MS, events, plan, memory)
        total_delivered_events += delivered
        samples += 1
        for i, s in enumerate(states):
            voltage_sum[i] += s.voltage_mv
        for i in spiking_indices:
            spike_timeline.append((states[i].last_spike_ms, i))

        if states[0].t_ms + 1e-9 >= next_print:
            spike_labels = ",".join(f"{neuron_type(i)}{i}" for i in spiking_indices) if spiking_indices else "none"
            avg_syn = sum(s.synaptic_current_na for s in states) / NETWORK_SIZE
            avg_ext = sum(external_currents) / NETWORK_SIZE
            print(
                f"t={states[0].t_ms:7.2f} ms | spikes now={spike_labels:10s} | "
                f"pending={len(events):3d} | avg_ext={avg_ext:6.3f} nA | avg_syn={avg_syn:6.3f} nA"
            )
            print("  " + network_snapshot(states, spiking_indices))
            next_print += PRINT_EVERY_MS

    duration_s = ROUND_DURATION_MS / 1000.0
    spike_counts = [s.spikes for s in states]
    exc_spikes = sum(spike_counts[:EXCITATORY_COUNT])
    inh_spikes = sum(spike_counts[EXCITATORY_COUNT:])
    total_spikes = exc_spikes + inh_spikes
    network_rate_hz = total_spikes / NETWORK_SIZE / duration_s
    balance_ratio = exc_spikes / max(1, inh_spikes)
    mean_voltage = sum(voltage_sum) / max(1, samples * NETWORK_SIZE)
    max_voltage = max(s.max_voltage_mv for s in states)
    min_voltage = min(s.min_voltage_mv for s in states)

    decision = "none"
    decision_note = "not a computation round"
    correct = False
    dream_similarity = 0.0
    dream_report = "not a dream round"

    if plan.mode == "compute":
        decision, decision_note, has_decision = classify_output(spike_counts, p.classification_margin_spikes)
        correct = bool(has_decision and decision == plan.pattern_label)
        trace = MemoryTrace(
            label=plan.pattern_label,
            round_number=round_index,
            spike_timeline=spike_timeline[:],
            spike_counts=spike_counts[:],
            output_a_spikes=spike_counts[OUTPUT_A_INDEX],
            output_b_spikes=spike_counts[OUTPUT_B_INDEX],
            decision=decision,
        )
        update_memory(memory, trace, p)
    else:
        dream_report, dream_similarity = dream_match_report(spike_counts, memory)
        decision_note = dream_report

    result: Dict[str, float | str | bool] = {
        "round": round_index,
        "mode": plan.mode,
        "pattern": plan.pattern_label,
        "total_spikes": total_spikes,
        "exc_spikes": exc_spikes,
        "inh_spikes": inh_spikes,
        "network_rate_hz": network_rate_hz,
        "balance_ratio": balance_ratio,
        "mean_voltage_mv": mean_voltage,
        "max_voltage_mv": max_voltage,
        "min_voltage_mv": min_voltage,
        "delivered_events": total_delivered_events,
        "pending_events_end": len(events),
        "output_a_spikes": spike_counts[OUTPUT_A_INDEX],
        "output_b_spikes": spike_counts[OUTPUT_B_INDEX],
        "decision": decision,
        "decision_note": decision_note,
        "correct": correct,
        "dream_similarity": dream_similarity,
        "base_current_na": p.base_current_na,
        "pattern_drive_na": p.pattern_drive_na,
        "dream_replay_boost_na": p.dream_replay_boost_na,
        "excitatory_weight_na": p.excitatory_weight_na,
        "inhibitory_weight_na": p.inhibitory_weight_na,
        "synaptic_delay_ms": p.synaptic_delay_ms,
        "threshold_mv": p.v_threshold_mv,
    }

    print("-" * 104)
    print("Round result:")
    print(f"  mode: {plan.mode}, pattern: {plan.pattern_label}")
    print(f"  total spikes: {total_spikes}")
    print(f"  excitatory spikes: {exc_spikes}, inhibitory spikes: {inh_spikes}, E/I spike ratio: {balance_ratio:.2f}")
    print(f"  mean network firing rate: {network_rate_hz:.2f} Hz per neuron")
    print(f"  output A E{OUTPUT_A_INDEX}: {spike_counts[OUTPUT_A_INDEX]} spikes")
    print(f"  output B I{OUTPUT_B_INDEX}: {spike_counts[OUTPUT_B_INDEX]} spikes")
    if plan.mode == "compute":
        print(f"  computed decision: {decision} | expected: {plan.pattern_label} | correct={correct}")
    else:
        print(f"  dream replay: {dream_report} | similarity={dream_similarity:.3f}")
    print(f"  average voltage: {mean_voltage:.2f} mV")
    print(f"  voltage range: {min_voltage:.2f} to {max_voltage:.2f} mV")
    print(f"  synaptic events delivered: {total_delivered_events}")
    print("  spike counts by neuron:")
    for i, count in enumerate(spike_counts):
        role = " output-A" if i == OUTPUT_A_INDEX else " output-B" if i == OUTPUT_B_INDEX else ""
        print(f"    {neuron_type(i)}{i}: {count}{role}")
    print(f"  memory after round: traces={len(memory.traces)}, strongest links={learned_weight_summary(memory)}")

    return result


# ============================================================
# AI CONTROLLER
# ============================================================

def ai_controller(previous_params: NetworkParams, result: Dict[str, float | str | bool]) -> NetworkParams:
    """Rule-based controller for computation accuracy, dream replay, and E/I stability."""
    p = previous_params
    new_p = replace(p)
    decisions: List[str] = []

    mode = str(result["mode"])
    total_spikes = int(result["total_spikes"])
    balance_ratio = float(result["balance_ratio"])
    network_rate = float(result["network_rate_hz"])

    if mode == "compute":
        correct = bool(result["correct"])
        output_a = int(result["output_a_spikes"])
        output_b = int(result["output_b_spikes"])
        if not correct:
            decisions.append("Computation was wrong or uncertain; strengthen the correct pattern gate without flooding the whole network.")
            new_p.pattern_drive_na += 0.035
            new_p.output_gate_na += 0.045
            new_p.learned_weight_scale_na += 0.008
            new_p.v_threshold_mv -= 0.05
        else:
            decisions.append("Computation succeeded; lightly preserve learning without overpowering future classifications.")
            new_p.learning_rate += 0.001
            new_p.synaptic_decay_ms += 0.15

        if abs(output_a - output_b) < 2:
            decisions.append("Output neurons were close; increase output gate separation.")
            new_p.output_gate_na += 0.055
            new_p.pattern_drive_na += 0.015
    else:
        dream_similarity = float(result["dream_similarity"])
        if dream_similarity < 0.55:
            decisions.append("Dream replay weakly matched stored traces; strengthen replay boost and lower outside input.")
            new_p.dream_replay_boost_na += 0.04
            new_p.dream_external_silence -= 0.03
        elif dream_similarity > 0.82:
            decisions.append("Dream replay was very close to memory; add recombination so dreams are not exact copies.")
            new_p.dream_recombination_na += 0.02
        else:
            decisions.append("Dream replay was recognizable but not identical; keep replay settings and explore timing.")
            new_p.dream_cycle_ms += 5.0

    target_total_spikes = 42
    if total_spikes < target_total_spikes - 14:
        decisions.append("Network was underactive; increase base drive and excitatory spread.")
        new_p.base_current_na += 0.07
        new_p.excitatory_weight_na += 0.025
    elif total_spikes > target_total_spikes + 18:
        decisions.append("Network was overactive; reduce global drive and increase recovery so outputs do not both fire together.")
        new_p.base_current_na -= 0.08
        new_p.pulse_current_na -= 0.04
        new_p.inhibitory_weight_na += 0.030
        new_p.excitatory_weight_na -= 0.015
        new_p.spike_adaptation_add_na += 0.018
        new_p.v_threshold_mv += 0.25
    else:
        decisions.append("Network activity was usable; make a small timing exploration step.")
        new_p.synaptic_delay_ms += 0.10 if network_rate > 8.0 else -0.05

    if balance_ratio > 3.0:
        decisions.append("Excitation dominated; strengthen inhibitory output.")
        new_p.inhibitory_weight_na += 0.025
        new_p.exc_to_inh_scale += 0.020
    elif balance_ratio < 1.0:
        decisions.append("Inhibition dominated; strengthen excitatory spread.")
        new_p.excitatory_weight_na += 0.025
        new_p.inhibitory_weight_na -= 0.015

    # Keep parameters in sensible ranges.
    new_p.base_current_na = clamp(new_p.base_current_na, 0.1, 3.0)
    new_p.pulse_current_na = clamp(new_p.pulse_current_na, 0.0, 3.0)
    new_p.pattern_drive_na = clamp(new_p.pattern_drive_na, 0.05, 1.5)
    new_p.output_gate_na = clamp(new_p.output_gate_na, 0.0, 0.95)
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
    new_p.learned_weight_scale_na = clamp(new_p.learned_weight_scale_na, 0.0, 0.45)
    new_p.learning_rate = clamp(new_p.learning_rate, 0.0, 0.06)
    new_p.dream_external_silence = clamp(new_p.dream_external_silence, 0.05, 0.9)
    new_p.dream_replay_boost_na = clamp(new_p.dream_replay_boost_na, 0.0, 1.5)
    new_p.dream_recombination_na = clamp(new_p.dream_recombination_na, 0.0, 0.8)
    new_p.dream_cycle_ms = clamp(new_p.dream_cycle_ms, 80.0, 320.0)

    print("\nAI controller decision:")
    for decision in decisions:
        print(f"  {decision}")
    print("  Next round network parameters selected.")
    return new_p


# ============================================================
# FINAL SUMMARY
# ============================================================

def print_final_summary(results: List[Dict[str, float | str | bool]], memory: NetworkMemory) -> None:
    print("\n" + "=" * 132)
    print("FINAL COMPUTATION AND DREAM SUMMARY")
    print("=" * 132)
    header = (
        f"{'Round':>5} | {'Mode':>7} | {'Pattern':>7} | {'Total':>6} | {'Hz/N':>8} | "
        f"{'A_out':>6} | {'B_out':>6} | {'Decision':>9} | {'Correct':>7} | {'DreamSim':>8} | {'E/I':>7} | {'Links':>5}"
    )
    print(header)
    print("-" * len(header))
    learned_link_count = sum(1 for row in memory.learned_weights for w in row if w > 0.001)
    for r in results:
        correct_text = "--" if r["mode"] != "compute" else str(bool(r["correct"]))
        dream_text = "--" if r["mode"] != "dream" else f"{float(r['dream_similarity']):.3f}"
        print(
            f"{int(r['round']):5d} | {str(r['mode']):>7} | {str(r['pattern']):>7} | {int(r['total_spikes']):6d} | "
            f"{float(r['network_rate_hz']):8.2f} | {int(r['output_a_spikes']):6d} | {int(r['output_b_spikes']):6d} | "
            f"{str(r['decision']):>9} | {correct_text:>7} | {dream_text:>8} | {float(r['balance_ratio']):7.2f} | {learned_link_count:5d}"
        )

    compute_results = [r for r in results if r["mode"] == "compute"]
    dream_results = [r for r in results if r["mode"] == "dream"]
    correct_count = sum(1 for r in compute_results if bool(r["correct"]))
    accuracy = correct_count / max(1, len(compute_results))
    avg_dream_similarity = sum(float(r["dream_similarity"]) for r in dream_results) / max(1, len(dream_results))

    print("\nOverall:")
    print(f"  computation accuracy: {correct_count}/{len(compute_results)} = {accuracy:.2%}")
    print(f"  average dream-memory similarity: {avg_dream_similarity:.3f}")
    print(f"  memory traces stored: {len(memory.traces)}")
    print(f"  strongest learned links: {learned_weight_summary(memory)}")
    print("\nInterpretation:")
    print("  Computation rounds present input patterns and read the spike difference between two output neurons.")
    print("  Dream rounds reduce outside input and use stored spike timelines to replay and recombine prior activity.")
    print("  This is not consciousness; it is a terminal model of classification, memory formation, and offline replay.")


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print("Neuron Network Computation and Dream Simulation - Accuracy Tuned")
    print("Terminal-only simulation. No external packages.")
    print("Model: excitatory/inhibitory spiking network with computation, memory traces, and dream replay.")

    params = NetworkParams()
    memory = NetworkMemory()
    results: List[Dict[str, float | str | bool]] = []

    for round_index in range(1, TOTAL_ROUNDS + 1):
        plan = choose_round_plan(round_index)
        result = run_round(round_index, params, memory, plan)
        results.append(result)
        if round_index < TOTAL_ROUNDS:
            params = ai_controller(params, result)

    print_final_summary(results, memory)


if __name__ == "__main__":
    main()
