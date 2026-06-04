#!/usr/bin/env python3
"""
Terminal Neuron Network Growth, Pruning, and Self-Exploration Simulation v3 - Homeostatic Control
------------------------------------------------------------------------

Run:
    python terminal_neuron_network_self_exploration_homeostatic_v3.py

No external packages are required.

This v3 version adds homeostatic control so richer spike activity does not saturate into runaway firing.

This is a terminal-only toy model of a small spiking neuron network that can:

    - grow new neurons
    - grow new synapses
    - strengthen useful synapses
    - weaken unused or noisy synapses
    - prune weak synapses
    - prune silent neurons
    - explore itself through internal stimulation
    - form temporary activity motifs
    - compare each round to previous network structure

It is not a biological brain model and not a clinical neuroscience model.
It is a compact simulation for studying network dynamics in the terminal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import exp, sin, cos, pi
from random import Random
from typing import Dict, List, Tuple, Optional


# ============================================================
# CONFIGURATION
# ============================================================

TOTAL_ROUNDS = 10
ROUND_DURATION_MS = 700.0
DT_MS = 1.0
PRINT_EVERY_MS = 50.0
SEED = 1127

INITIAL_EXCITATORY = 4
INITIAL_INHIBITORY = 2
MAX_NEURONS = 16

# Membrane defaults.
V_REST = -70.0
V_RESET = -75.0
V_THRESHOLD = -57.0
MEMBRANE_TAU_MS = 18.0
REFRACTORY_MS = 3.0

# Structural dynamics.
WEAK_SYNAPSE_PRUNE_THRESHOLD = 0.018
STRONG_SYNAPSE_THRESHOLD = 0.30
MAX_SYNAPSES = 80

# Learning dynamics.
HEBBIAN_WINDOW_MS = 28.0
HEBBIAN_GAIN = 0.025
WEIGHT_DECAY_PER_ROUND = 0.988
SYNAPSE_MIN_WEIGHT = 0.0
SYNAPSE_MAX_WEIGHT = 0.90

# Homeostatic control targets.
HOMEOSTATIC_TARGET_LOW_HZ = 8.0
HOMEOSTATIC_TARGET_HIGH_HZ = 20.0
HOMEOSTATIC_TARGET_CENTER_HZ = 14.0
HOMEOSTATIC_MAX_THRESHOLD_SHIFT_MV = 4.0
HOMEOSTATIC_MAX_INHIBITION_GAIN = 1.85
HOMEOSTATIC_MIN_INHIBITION_GAIN = 0.75
HOMEOSTATIC_STRONG_SYNAPSE_FRACTION_LIMIT = 0.72


# ============================================================
# DATA MODEL
# ============================================================

@dataclass
class Neuron:
    nid: int
    kind: str  # "E" or "I"
    role: str
    voltage_mv: float = V_REST
    refractory_left_ms: float = 0.0
    adaptation_na: float = 0.0
    spike_count: int = 0
    last_spike_ms: float = -999999.0
    age_rounds: int = 0
    activity_trace: float = 0.0
    novelty_trace: float = 0.0
    homeostatic_bias_na: float = 0.0
    dynamic_threshold_mv: float = V_THRESHOLD
    alive: bool = True

    @property
    def name(self) -> str:
        return f"{self.kind}{self.nid}"


@dataclass
class Synapse:
    src: int
    dst: int
    weight_na: float
    delay_ms: float
    kind: str  # "E" or "I", based on src neuron
    age_rounds: int = 0
    use_count: int = 0
    last_used_ms: float = -999999.0
    alive: bool = True

    @property
    def sign(self) -> float:
        return 1.0 if self.kind == "E" else -1.0


@dataclass
class PendingEvent:
    deliver_ms: float
    src: int
    dst: int
    current_na: float


@dataclass
class HomeostaticState:
    inhibition_gain: float = 1.0
    threshold_shift_mv: float = 0.0
    synaptic_scaling_factor: float = 1.0
    excitability_bias_na: float = 0.0
    state_label: str = "balanced"


@dataclass
class Network:
    neurons: Dict[int, Neuron] = field(default_factory=dict)
    synapses: List[Synapse] = field(default_factory=list)
    next_neuron_id: int = 0
    memory_motifs: List[Tuple[str, List[int]]] = field(default_factory=list)
    homeostasis: HomeostaticState = field(default_factory=HomeostaticState)
    round_index: int = 0

    def living_neurons(self) -> List[Neuron]:
        return [n for n in self.neurons.values() if n.alive]

    def living_synapses(self) -> List[Synapse]:
        return [s for s in self.synapses if s.alive and self.neurons[s.src].alive and self.neurons[s.dst].alive]

    def excitatory_count(self) -> int:
        return sum(1 for n in self.living_neurons() if n.kind == "E")

    def inhibitory_count(self) -> int:
        return sum(1 for n in self.living_neurons() if n.kind == "I")


@dataclass
class RoundParams:
    mode: str
    base_drive_na: float
    exploration_drive_na: float
    noise_drive_na: float
    growth_pressure: float
    prune_pressure: float
    curiosity: float
    target_activity_low: float
    target_activity_high: float



@dataclass
class RoundSummary:
    round_index: int
    mode: str
    neurons_before: int
    neurons_after: int
    synapses_before: int
    synapses_after: int
    total_spikes: int
    excitatory_spikes: int
    inhibitory_spikes: int
    mean_rate_hz: float
    avg_voltage_mv: float
    active_neurons: int
    silent_neurons: int
    strongest_synapse: str
    new_neurons: int
    new_synapses: int
    pruned_neurons: int
    pruned_synapses: int
    motifs_found: int
    homeostatic_state: str
    global_inhibition_gain: float
    threshold_shift_mv: float
    synaptic_scaling_factor: float
    controller_note: str


# ============================================================
# UTILITY
# ============================================================

def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def voltage_bar(v: float, width: int = 28) -> str:
    lo = -78.0
    hi = -52.0
    idx = int((v - lo) / (hi - lo) * (width - 1))
    idx = max(0, min(width - 1, idx))
    chars = ["-"] * width
    chars[idx] = "|"
    return "".join(chars)


def format_synapse(net: Network, syn: Optional[Synapse]) -> str:
    if syn is None:
        return "none"
    src = net.neurons[syn.src].name
    dst = net.neurons[syn.dst].name
    sign = "+" if syn.kind == "E" else "-"
    return f"{src}->{dst}:{sign}{syn.weight_na:.3f}"


def connection_exists(net: Network, src: int, dst: int) -> bool:
    return any(s.alive and s.src == src and s.dst == dst for s in net.synapses)


def average_voltage(net: Network) -> float:
    living = net.living_neurons()
    if not living:
        return V_REST
    return sum(n.voltage_mv for n in living) / len(living)


def structural_signature(net: Network) -> str:
    living = net.living_neurons()
    synapses = net.living_synapses()
    if not living:
        return "empty"
    strong = sum(1 for s in synapses if s.weight_na >= STRONG_SYNAPSE_THRESHOLD)
    return (
        f"N={len(living)} E={net.excitatory_count()} I={net.inhibitory_count()} "
        f"S={len(synapses)} strong={strong}"
    )


# ============================================================
# NETWORK INITIALIZATION
# ============================================================

def add_neuron(net: Network, kind: str, role: str) -> Neuron:
    nid = net.next_neuron_id
    net.next_neuron_id += 1
    neuron = Neuron(nid=nid, kind=kind, role=role)
    net.neurons[nid] = neuron
    return neuron


def add_synapse(net: Network, src: int, dst: int, weight_na: float, delay_ms: float) -> Optional[Synapse]:
    if src == dst:
        return None
    if src not in net.neurons or dst not in net.neurons:
        return None
    if not net.neurons[src].alive or not net.neurons[dst].alive:
        return None
    if connection_exists(net, src, dst):
        return None
    if len(net.living_synapses()) >= MAX_SYNAPSES:
        return None

    src_kind = net.neurons[src].kind
    syn = Synapse(
        src=src,
        dst=dst,
        weight_na=clamp(weight_na, SYNAPSE_MIN_WEIGHT, SYNAPSE_MAX_WEIGHT),
        delay_ms=delay_ms,
        kind=src_kind,
    )
    net.synapses.append(syn)
    return syn


def make_initial_network(rng: Random) -> Network:
    net = Network()

    for i in range(INITIAL_EXCITATORY):
        role = "sensor" if i < 2 else "processor"
        add_neuron(net, "E", role)

    for i in range(INITIAL_INHIBITORY):
        add_neuron(net, "I", "regulator")

    ids = [n.nid for n in net.living_neurons()]

    # Seed a sparse recurrent network.
    for src in ids:
        for dst in ids:
            if src == dst:
                continue
            p = 0.34 if net.neurons[src].kind == "E" else 0.28
            if rng.random() < p:
                if net.neurons[src].kind == "E":
                    w = rng.uniform(0.20, 0.36)
                else:
                    w = rng.uniform(0.20, 0.36)
                add_synapse(net, src, dst, w, rng.uniform(2.0, 8.0))

    return net


# ============================================================
# DYNAMICS
# ============================================================

def round_mode(round_index: int) -> str:
    modes = [
        "seed exploration",
        "growth search",
        "self-test pulse",
        "prune quiet paths",
        "motif rehearsal",
        "novelty search",
        "stability test",
        "branch growth",
        "deep self-exploration",
        "consolidation",
    ]
    return modes[(round_index - 1) % len(modes)]


def update_homeostatic_state(net: Network, previous: Optional[RoundSummary]) -> None:
    """
    Adjust global excitability based on the previous round.

    If the network is overactive:
        - raise effective spike threshold
        - increase inhibition gain
        - scale down excitatory synapses
        - add a small negative current bias

    If the network is underactive:
        - lower effective threshold
        - reduce inhibition gain
        - scale excitatory synapses up slightly
        - add a small positive current bias

    This does not replace growth/pruning. It acts as a stabilizing layer on top.
    """
    h = net.homeostasis

    if previous is None:
        h.state_label = "initial"
        h.inhibition_gain = 1.0
        h.threshold_shift_mv = 0.0
        h.synaptic_scaling_factor = 1.0
        h.excitability_bias_na = 0.0
        return

    rate = previous.mean_rate_hz
    strong_fraction = 0.0
    synapses = net.living_synapses()
    if synapses:
        strong_fraction = sum(1 for s in synapses if s.weight_na >= STRONG_SYNAPSE_THRESHOLD) / len(synapses)

    if rate > HOMEOSTATIC_TARGET_HIGH_HZ or strong_fraction > HOMEOSTATIC_STRONG_SYNAPSE_FRACTION_LIMIT:
        overload = max(
            (rate - HOMEOSTATIC_TARGET_CENTER_HZ) / max(1.0, HOMEOSTATIC_TARGET_CENTER_HZ),
            (strong_fraction - HOMEOSTATIC_STRONG_SYNAPSE_FRACTION_LIMIT) / 0.28,
        )
        overload = clamp(overload, 0.0, 1.0)

        h.state_label = "dampening overactivity"
        h.threshold_shift_mv = clamp(
            h.threshold_shift_mv + 0.45 + 0.90 * overload,
            0.0,
            HOMEOSTATIC_MAX_THRESHOLD_SHIFT_MV,
        )
        h.inhibition_gain = clamp(
            h.inhibition_gain + 0.08 + 0.22 * overload,
            HOMEOSTATIC_MIN_INHIBITION_GAIN,
            HOMEOSTATIC_MAX_INHIBITION_GAIN,
        )
        h.excitability_bias_na = clamp(
            h.excitability_bias_na - 0.035 - 0.060 * overload,
            -0.28,
            0.22,
        )
        h.synaptic_scaling_factor = 0.985 - 0.055 * overload

    elif rate < HOMEOSTATIC_TARGET_LOW_HZ:
        under = clamp((HOMEOSTATIC_TARGET_LOW_HZ - rate) / HOMEOSTATIC_TARGET_LOW_HZ, 0.0, 1.0)

        h.state_label = "lifting underactivity"
        h.threshold_shift_mv = clamp(
            h.threshold_shift_mv - 0.35 - 0.45 * under,
            -1.50,
            HOMEOSTATIC_MAX_THRESHOLD_SHIFT_MV,
        )
        h.inhibition_gain = clamp(
            h.inhibition_gain - 0.06 - 0.12 * under,
            HOMEOSTATIC_MIN_INHIBITION_GAIN,
            HOMEOSTATIC_MAX_INHIBITION_GAIN,
        )
        h.excitability_bias_na = clamp(
            h.excitability_bias_na + 0.030 + 0.050 * under,
            -0.28,
            0.22,
        )
        h.synaptic_scaling_factor = 1.010 + 0.025 * under

    else:
        h.state_label = "balanced"
        # Relax slowly toward neutral values.
        h.threshold_shift_mv *= 0.90
        h.inhibition_gain += (1.0 - h.inhibition_gain) * 0.12
        h.excitability_bias_na *= 0.85
        h.synaptic_scaling_factor = 1.0


def apply_homeostatic_synaptic_scaling(net: Network) -> None:
    """
    Apply global synaptic scaling after the homeostatic state is updated.

    Excitatory synapses are scaled more strongly than inhibitory synapses. During
    overactivity, this prevents the network from ending with nearly every synapse
    at maximum strength. During underactivity, it provides mild rescue.
    """
    factor = net.homeostasis.synaptic_scaling_factor
    if abs(factor - 1.0) < 1e-6:
        return

    for syn in net.living_synapses():
        if syn.kind == "E":
            syn.weight_na *= factor
        else:
            # Inhibitory synapses are preserved or slightly boosted when damping.
            if factor < 1.0:
                syn.weight_na *= 1.0 + (1.0 - factor) * 0.35
            else:
                syn.weight_na *= 1.0 + (factor - 1.0) * 0.10
        syn.weight_na = clamp(syn.weight_na, SYNAPSE_MIN_WEIGHT, SYNAPSE_MAX_WEIGHT)



def make_round_params(round_index: int, previous: Optional[RoundSummary]) -> RoundParams:
    mode = round_mode(round_index)

    # Defaults.
    params = RoundParams(
        mode=mode,
        base_drive_na=1.05,
        exploration_drive_na=0.62,
        noise_drive_na=0.10,
        growth_pressure=0.50,
        prune_pressure=0.18,
        curiosity=0.50,
        target_activity_low=8.0,
        target_activity_high=22.0,
    )

    if mode == "seed exploration":
        params.base_drive_na = 1.12
        params.exploration_drive_na = 0.72
        params.growth_pressure = 0.45
        params.prune_pressure = 0.10
        params.curiosity = 0.65

    elif mode == "growth search":
        params.base_drive_na = 1.18
        params.exploration_drive_na = 0.82
        params.growth_pressure = 0.80
        params.prune_pressure = 0.10
        params.curiosity = 0.75

    elif mode == "self-test pulse":
        params.base_drive_na = 1.10
        params.exploration_drive_na = 0.94
        params.growth_pressure = 0.35
        params.prune_pressure = 0.20
        params.curiosity = 0.55

    elif mode == "prune quiet paths":
        params.base_drive_na = 1.20
        params.exploration_drive_na = 0.76
        params.growth_pressure = 0.20
        params.prune_pressure = 0.48
        params.curiosity = 0.35

    elif mode == "motif rehearsal":
        params.base_drive_na = 1.08
        params.exploration_drive_na = 0.72
        params.growth_pressure = 0.35
        params.prune_pressure = 0.20
        params.curiosity = 0.45

    elif mode == "novelty search":
        params.base_drive_na = 1.22
        params.exploration_drive_na = 0.94
        params.noise_drive_na = 0.22
        params.growth_pressure = 0.70
        params.prune_pressure = 0.18
        params.curiosity = 0.90

    elif mode == "stability test":
        params.base_drive_na = 1.20
        params.exploration_drive_na = 0.46
        params.growth_pressure = 0.20
        params.prune_pressure = 0.34
        params.curiosity = 0.25

    elif mode == "branch growth":
        params.base_drive_na = 1.20
        params.exploration_drive_na = 0.76
        params.growth_pressure = 0.90
        params.prune_pressure = 0.16
        params.curiosity = 0.60

    elif mode == "deep self-exploration":
        params.base_drive_na = 1.28
        params.exploration_drive_na = 1.00
        params.noise_drive_na = 0.22
        params.growth_pressure = 0.78
        params.prune_pressure = 0.28
        params.curiosity = 0.95

    elif mode == "consolidation":
        params.base_drive_na = 1.02
        params.exploration_drive_na = 0.55
        params.growth_pressure = 0.18
        params.prune_pressure = 0.42
        params.curiosity = 0.25

    # Feedback from previous round.
    if previous is not None:
        if previous.mean_rate_hz < params.target_activity_low:
            params.base_drive_na += 0.14
            params.exploration_drive_na += 0.14
            params.growth_pressure += 0.16
            params.prune_pressure *= 0.82
        elif previous.mean_rate_hz > params.target_activity_high:
            params.base_drive_na -= 0.08
            params.prune_pressure += 0.18

        if previous.silent_neurons > 0:
            params.curiosity += 0.16
            params.growth_pressure += 0.08
            params.prune_pressure *= 0.90

        if previous.synapses_after > 55:
            params.prune_pressure += 0.18
            params.growth_pressure -= 0.08

    params.base_drive_na = clamp(params.base_drive_na, 0.45, 1.70)
    params.exploration_drive_na = clamp(params.exploration_drive_na, 0.10, 1.35)
    params.noise_drive_na = clamp(params.noise_drive_na, 0.00, 0.38)
    params.growth_pressure = clamp(params.growth_pressure, 0.00, 1.00)
    params.prune_pressure = clamp(params.prune_pressure, 0.00, 1.00)
    params.curiosity = clamp(params.curiosity, 0.00, 1.00)

    return params


def external_drive(neuron: Neuron, t_ms: float, params: RoundParams, rng: Random, net: Network) -> float:
    # Base drive keeps network near threshold.
    drive = params.base_drive_na

    # Exploratory pulses sweep across neuron ids.
    phase = (t_ms / 90.0 + neuron.nid * 0.27) % 1.0
    if phase < 0.30:
        drive += params.exploration_drive_na * (1.0 - phase / 0.30)

    # Role-specific behavior.
    if neuron.role == "sensor":
        drive += 0.08 * sin(2 * pi * t_ms / 180.0 + neuron.nid)
    elif neuron.role == "newborn":
        drive += 0.26 * params.curiosity
    elif neuron.role == "motif":
        drive += 0.22 + 0.16 * sin(2 * pi * t_ms / 110.0)

    # Dream/self-exploration modes replay internal traces rather than external sensory input.
    if "rehearsal" in params.mode or "self-exploration" in params.mode:
        if net.memory_motifs:
            motif_ids = net.memory_motifs[-1][1]
            if neuron.nid in motif_ids:
                drive += 0.42 + 0.38 * params.curiosity

    # Deterministic noise-like term.
    drive += params.noise_drive_na * sin(0.031 * t_ms + neuron.nid * 1.7)

    # Homeostatic bias reduces or lifts excitability globally.
    drive += net.homeostasis.excitability_bias_na

    return max(0.0, drive)


def deliver_pending_events(pending: List[PendingEvent], t_ms: float, syn_current: Dict[int, float]) -> int:
    delivered = 0
    keep: List[PendingEvent] = []

    for event in pending:
        if event.deliver_ms <= t_ms:
            syn_current[event.dst] = syn_current.get(event.dst, 0.0) + event.current_na
            delivered += 1
        else:
            keep.append(event)

    pending[:] = keep
    return delivered


def update_neuron(neuron: Neuron, current_na: float, dt_ms: float, net: Network) -> bool:
    if neuron.refractory_left_ms > 0.0:
        neuron.voltage_mv = V_RESET
        neuron.refractory_left_ms = max(0.0, neuron.refractory_left_ms - dt_ms)
        neuron.adaptation_na *= exp(-dt_ms / 105.0)
        return False

    effective_current = current_na - neuron.adaptation_na

    # Input current is translated into voltage pressure with a fixed resistance-like multiplier.
    dv_dt = (-(neuron.voltage_mv - V_REST) + 13.2 * effective_current) / MEMBRANE_TAU_MS
    neuron.voltage_mv += dv_dt * dt_ms

    neuron.dynamic_threshold_mv = V_THRESHOLD + net.homeostasis.threshold_shift_mv
    spiked = neuron.voltage_mv >= neuron.dynamic_threshold_mv
    if spiked:
        neuron.voltage_mv = V_RESET
        neuron.refractory_left_ms = REFRACTORY_MS
        neuron.spike_count += 1
        neuron.adaptation_na += 0.085 if neuron.kind == "E" else 0.060
        neuron.activity_trace += 1.0
        neuron.novelty_trace += 0.35
    else:
        neuron.activity_trace *= exp(-dt_ms / 260.0)
        neuron.novelty_trace *= exp(-dt_ms / 400.0)

    neuron.adaptation_na *= exp(-dt_ms / 105.0)
    return spiked


def schedule_synaptic_events(net: Network, src_id: int, t_ms: float, pending: List[PendingEvent]) -> int:
    count = 0
    for syn in net.living_synapses():
        if syn.src == src_id:
            if syn.kind == "I":
                current = syn.sign * syn.weight_na * net.homeostasis.inhibition_gain
            else:
                current = syn.sign * syn.weight_na
            pending.append(
                PendingEvent(
                    deliver_ms=t_ms + syn.delay_ms,
                    src=syn.src,
                    dst=syn.dst,
                    current_na=current,
                )
            )
            syn.use_count += 1
            syn.last_used_ms = t_ms
            count += 1
    return count


def hebbian_update(net: Network, spiking_ids: List[int], t_ms: float) -> None:
    if not spiking_ids:
        return

    spiking_set = set(spiking_ids)

    for syn in net.living_synapses():
        src = net.neurons[syn.src]
        dst = net.neurons[syn.dst]

        src_recent = abs(t_ms - src.last_spike_ms) <= HEBBIAN_WINDOW_MS
        dst_recent = abs(t_ms - dst.last_spike_ms) <= HEBBIAN_WINDOW_MS

        if net.homeostasis.state_label == "dampening overactivity":
            learning_gate = 0.42
        elif net.homeostasis.state_label == "lifting underactivity":
            learning_gate = 1.18
        else:
            learning_gate = 0.82

        if syn.src in spiking_set and dst_recent:
            syn.weight_na += HEBBIAN_GAIN * learning_gate * (1.0 if syn.kind == "E" else 0.55)
        elif syn.dst in spiking_set and src_recent:
            syn.weight_na += HEBBIAN_GAIN * learning_gate * 0.7
        else:
            syn.weight_na *= 0.9993

        syn.weight_na = clamp(syn.weight_na, SYNAPSE_MIN_WEIGHT, SYNAPSE_MAX_WEIGHT)


def detect_motif(net: Network, round_index: int) -> Optional[Tuple[str, List[int]]]:
    active = sorted(
        [n for n in net.living_neurons() if n.spike_count >= 2],
        key=lambda n: (-n.spike_count, n.nid),
    )
    if len(active) < 3:
        return None

    motif_ids = [n.nid for n in active[:4]]
    motif_name = f"R{round_index}-motif-" + "-".join(net.neurons[i].name for i in motif_ids)
    return motif_name, motif_ids


# ============================================================
# STRUCTURAL PLASTICITY
# ============================================================

def grow_network(net: Network, params: RoundParams, rng: Random) -> Tuple[int, int]:
    new_neurons = 0
    new_synapses = 0
    living = net.living_neurons()

    if not living:
        return 0, 0

    active = [n for n in living if n.spike_count >= 3]
    silent = [n for n in living if n.spike_count == 0]

    # Add synapses between active neurons and underconnected / silent neurons.
    attempts = int(3 + params.growth_pressure * 9)
    for _ in range(attempts):
        living = net.living_neurons()
        if len(net.living_synapses()) >= MAX_SYNAPSES:
            break

        if active and silent and rng.random() < 0.45:
            src = rng.choice(active).nid
            dst = rng.choice(silent).nid
        else:
            src = rng.choice(living).nid
            dst = rng.choice(living).nid

        if src == dst:
            continue
        if connection_exists(net, src, dst):
            continue

        src_kind = net.neurons[src].kind
        if src_kind == "E":
            weight = rng.uniform(0.11, 0.26) + 0.10 * params.curiosity
        else:
            weight = rng.uniform(0.13, 0.30)

        syn = add_synapse(net, src, dst, weight, rng.uniform(2.0, 9.0))
        if syn is not None:
            new_synapses += 1

    # Add a newborn neuron if growth pressure and novelty are high enough.
    if len(living) < MAX_NEURONS:
        should_grow = rng.random() < (0.18 + 0.38 * params.growth_pressure)
        if should_grow:
            # Prefer excitatory growth, but add inhibitory cells if E/I ratio is too high.
            e_spikes = sum(n.spike_count for n in living if n.kind == "E")
            i_spikes = sum(n.spike_count for n in living if n.kind == "I")
            if e_spikes > max(1, i_spikes) * 2.5:
                kind = "I"
                role = "regulator"
            else:
                kind = "E"
                role = "newborn"

            newborn = add_neuron(net, kind, role)
            new_neurons += 1

            # Connect newborn to/from active neurons.
            partners = active[:]
            if not partners:
                partners = rng.sample(living, min(3, len(living)))

            for partner in partners[:4]:
                if rng.random() < 0.75:
                    syn = add_synapse(
                        net,
                        partner.nid,
                        newborn.nid,
                        rng.uniform(0.10, 0.24),
                        rng.uniform(2.0, 8.0),
                    )
                    if syn is not None:
                        new_synapses += 1
                if rng.random() < 0.70:
                    syn = add_synapse(
                        net,
                        newborn.nid,
                        partner.nid,
                        rng.uniform(0.10, 0.22),
                        rng.uniform(2.0, 8.0),
                    )
                    if syn is not None:
                        new_synapses += 1

    return new_neurons, new_synapses


def prune_network(net: Network, params: RoundParams, rng: Random) -> Tuple[int, int]:
    pruned_synapses = 0
    pruned_neurons = 0

    # Decay all synapses once per round.
    for syn in net.living_synapses():
        syn.weight_na *= WEIGHT_DECAY_PER_ROUND
        syn.age_rounds += 1

    # Prune weak, old, or unused synapses. Higher prune pressure raises removal probability.
    for syn in net.living_synapses():
        weak = syn.weight_na < WEAK_SYNAPSE_PRUNE_THRESHOLD
        unused = syn.use_count == 0 and syn.age_rounds > 1
        too_many = len(net.living_synapses()) > int(MAX_SYNAPSES * 0.80)
        removal_probability = 0.0

        network_spikes = sum(n.spike_count for n in net.living_neurons())
        # v2: if the round was fully silent, avoid aggressive pruning. The earlier
        # version repeatedly pruned before enough activity could develop.
        if weak:
            removal_probability += 0.35 if network_spikes == 0 else 0.55
        if unused:
            removal_probability += (0.05 + 0.18 * params.prune_pressure) if network_spikes == 0 else (0.20 + 0.50 * params.prune_pressure)
        if too_many:
            removal_probability += 0.12 if network_spikes == 0 else 0.20
        if net.homeostasis.state_label == "dampening overactivity" and syn.kind == "E" and syn.weight_na > 0.78:
            # prune a small fraction of redundant saturated excitatory links
            removal_probability += 0.08 + 0.16 * params.prune_pressure
        if net.homeostasis.state_label == "dampening overactivity" and syn.kind == "I":
            # preserve inhibition during damping
            removal_probability *= 0.55

        if rng.random() < removal_probability:
            syn.alive = False
            pruned_synapses += 1

    # Prune silent newborn neurons only if they are old enough and not structurally needed.
    for neuron in net.living_neurons():
        if neuron.age_rounds < 3:
            continue
        if neuron.role != "newborn":
            continue
        if neuron.spike_count == 0 and rng.random() < 0.30 + 0.50 * params.prune_pressure:
            neuron.alive = False
            pruned_neurons += 1
            for syn in net.living_synapses():
                if syn.src == neuron.nid or syn.dst == neuron.nid:
                    syn.alive = False
                    pruned_synapses += 1

    return pruned_neurons, pruned_synapses


def reset_round_state(net: Network) -> None:
    for neuron in net.living_neurons():
        neuron.voltage_mv = V_REST - (neuron.nid % 6) * 0.35
        neuron.refractory_left_ms = 0.0
        neuron.adaptation_na = 0.0
        neuron.spike_count = 0
        neuron.last_spike_ms = -999999.0
        neuron.activity_trace *= 0.35
        neuron.novelty_trace *= 0.45

    for syn in net.living_synapses():
        syn.use_count = 0
        syn.last_used_ms = -999999.0


def age_network(net: Network) -> None:
    for neuron in net.living_neurons():
        neuron.age_rounds += 1


# ============================================================
# REPORTING
# ============================================================

def print_network_snapshot(net: Network, t_ms: float, spiking: List[int], pending_count: int, avg_ext: float, avg_syn: float) -> None:
    names = ",".join(net.neurons[i].name for i in spiking) if spiking else "none"
    print(
        f"t={t_ms:7.2f} ms | spikes now={names:<18} | pending={pending_count:3d} | "
        f"avg_ext={avg_ext:6.3f} nA | avg_syn={avg_syn:7.3f} nA"
    )

    living = sorted(net.living_neurons(), key=lambda n: n.nid)
    chunks = []
    for n in living:
        star = "*" if n.nid in spiking else " "
        chunks.append(f"{n.name}{star}: {n.voltage_mv:6.1f}mV[{voltage_bar(n.voltage_mv)}]")
    print("  " + "  ".join(chunks))


def summarize_round(
    net: Network,
    round_index: int,
    params: RoundParams,
    neurons_before: int,
    synapses_before: int,
    voltage_sum: float,
    voltage_samples: int,
    new_neurons: int,
    new_synapses: int,
    pruned_neurons: int,
    pruned_synapses: int,
    motif_added: bool,
    controller_note: str,
) -> RoundSummary:
    living = net.living_neurons()
    total_spikes = sum(n.spike_count for n in living)
    e_spikes = sum(n.spike_count for n in living if n.kind == "E")
    i_spikes = sum(n.spike_count for n in living if n.kind == "I")
    active = sum(1 for n in living if n.spike_count > 0)
    silent = sum(1 for n in living if n.spike_count == 0)
    duration_s = ROUND_DURATION_MS / 1000.0
    mean_rate = total_spikes / max(1, len(living)) / duration_s
    avg_v = voltage_sum / max(1, voltage_samples)
    strongest = max(net.living_synapses(), key=lambda s: s.weight_na, default=None)

    return RoundSummary(
        round_index=round_index,
        mode=params.mode,
        neurons_before=neurons_before,
        neurons_after=len(living),
        synapses_before=synapses_before,
        synapses_after=len(net.living_synapses()),
        total_spikes=total_spikes,
        excitatory_spikes=e_spikes,
        inhibitory_spikes=i_spikes,
        mean_rate_hz=mean_rate,
        avg_voltage_mv=avg_v,
        active_neurons=active,
        silent_neurons=silent,
        strongest_synapse=format_synapse(net, strongest),
        new_neurons=new_neurons,
        new_synapses=new_synapses,
        pruned_neurons=pruned_neurons,
        pruned_synapses=pruned_synapses,
        motifs_found=1 if motif_added else 0,
        homeostatic_state=net.homeostasis.state_label,
        global_inhibition_gain=net.homeostasis.inhibition_gain,
        threshold_shift_mv=net.homeostasis.threshold_shift_mv,
        synaptic_scaling_factor=net.homeostasis.synaptic_scaling_factor,
        controller_note=controller_note,
    )


def print_round_summary(summary: RoundSummary, net: Network) -> None:
    print("-" * 128)
    print("Round result:")
    print(f"  mode: {summary.mode}")
    print(f"  structure before: neurons={summary.neurons_before}, synapses={summary.synapses_before}")
    print(f"  structure after:  neurons={summary.neurons_after}, synapses={summary.synapses_after}")
    print(f"  total spikes: {summary.total_spikes}")
    print(f"  excitatory spikes: {summary.excitatory_spikes}")
    print(f"  inhibitory spikes: {summary.inhibitory_spikes}")
    print(f"  mean firing rate: {summary.mean_rate_hz:.2f} Hz per neuron")
    print(f"  active neurons: {summary.active_neurons}, silent neurons: {summary.silent_neurons}")
    print(f"  average voltage: {summary.avg_voltage_mv:.2f} mV")
    print(f"  strongest synapse: {summary.strongest_synapse}")
    print(f"  growth: +{summary.new_neurons} neurons, +{summary.new_synapses} synapses")
    print(f"  pruning: -{summary.pruned_neurons} neurons, -{summary.pruned_synapses} synapses")
    print(f"  motifs found this round: {summary.motifs_found}")
    print(
        f"  homeostasis: {summary.homeostatic_state}, threshold_shift={summary.threshold_shift_mv:+.2f} mV, "
        f"inhibition_gain={summary.global_inhibition_gain:.2f}, synaptic_scale={summary.synaptic_scaling_factor:.3f}"
    )
    print("  spike counts by neuron:")
    for n in sorted(net.living_neurons(), key=lambda x: x.nid):
        print(f"    {n.name:<4} kind={n.kind} role={n.role:<9} spikes={n.spike_count:3d} age={n.age_rounds:2d}")


def print_final_summary(summaries: List[RoundSummary], net: Network) -> None:
    print("\n" + "=" * 152)
    print("FINAL NEURON NETWORK GROWTH / PRUNING / SELF-EXPLORATION SUMMARY")
    print("=" * 152)

    header = (
        f"{'Round':>5} | {'Mode':>22} | {'N before':>8} | {'N after':>7} | "
        f"{'S before':>8} | {'S after':>7} | {'Spikes':>6} | {'Hz/N':>7} | "
        f"{'Active':>6} | {'Silent':>6} | {'Homeo':>21} | {'Grow N/S':>9} | {'Prune N/S':>10} | Strongest"
    )
    print(header)
    print("-" * len(header))

    for s in summaries:
        print(
            f"{s.round_index:5d} | {s.mode:>22} | {s.neurons_before:8d} | {s.neurons_after:7d} | "
            f"{s.synapses_before:8d} | {s.synapses_after:7d} | {s.total_spikes:6d} | "
            f"{s.mean_rate_hz:7.2f} | {s.active_neurons:6d} | {s.silent_neurons:6d} | "
            f"{s.homeostatic_state[:21]:>21} | +{s.new_neurons}/{s.new_synapses:<5d} | "
            f"-{s.pruned_neurons}/{s.pruned_synapses:<6d} | {s.strongest_synapse}"
        )

    total_new_neurons = sum(s.new_neurons for s in summaries)
    total_new_synapses = sum(s.new_synapses for s in summaries)
    total_pruned_neurons = sum(s.pruned_neurons for s in summaries)
    total_pruned_synapses = sum(s.pruned_synapses for s in summaries)
    total_motifs = sum(s.motifs_found for s in summaries)

    print("\nOverall:")
    print(f"  final structure: {structural_signature(net)}")
    print(f"  total neurons grown: {total_new_neurons}")
    print(f"  total synapses grown: {total_new_synapses}")
    print(f"  total neurons pruned: {total_pruned_neurons}")
    print(f"  total synapses pruned: {total_pruned_synapses}")
    print(f"  memory motifs stored: {len(net.memory_motifs)}")
    print(f"  motifs detected during run: {total_motifs}")

    strongest = sorted(net.living_synapses(), key=lambda s: -s.weight_na)[:8]
    print("\nStrongest final synapses:")
    for syn in strongest:
        print(f"  {format_synapse(net, syn)} delay={syn.delay_ms:.1f}ms uses={syn.use_count}")

    print("\nInterpretation:")
    print("  Growth adds neurons and synapses when the network has curiosity pressure or underused regions.")
    print("  Pruning removes weak, unused, or silent structures so the network does not grow without limit.")
    print("  Self-exploration drives internal activity, detects recurring motifs, and reinforces active pathways.")
    print("  Homeostatic control raises threshold, increases inhibition, scales synapses, and prunes redundant excitation when activity is too high.")
    print("  The result is not consciousness; it is a terminal model of adaptive structural plasticity.")


# ============================================================
# ROUND EXECUTION
# ============================================================

def run_round(net: Network, params: RoundParams, rng: Random) -> RoundSummary:
    round_index = net.round_index
    reset_round_state(net)

    neurons_before = len(net.living_neurons())
    synapses_before = len(net.living_synapses())

    print("\n" + "=" * 128)
    print(f"ROUND {round_index} | {params.mode.upper()}")
    print("=" * 128)
    print(f"Structure: {structural_signature(net)}")
    print(
        f"Controller params: base={params.base_drive_na:.3f} nA, explore={params.exploration_drive_na:.3f} nA, "
        f"noise={params.noise_drive_na:.3f} nA, growth={params.growth_pressure:.2f}, "
        f"prune={params.prune_pressure:.2f}, curiosity={params.curiosity:.2f}"
    )
    print(
        f"Homeostasis: state={net.homeostasis.state_label}, threshold_shift={net.homeostasis.threshold_shift_mv:+.2f} mV, "
        f"inhibition_gain={net.homeostasis.inhibition_gain:.2f}, bias={net.homeostasis.excitability_bias_na:+.3f} nA, "
        f"syn_scale={net.homeostasis.synaptic_scaling_factor:.3f}"
    )
    print("-" * 128)

    t_ms = 0.0
    next_print = 0.0
    pending: List[PendingEvent] = []
    voltage_sum = 0.0
    voltage_samples = 0
    delivered_total = 0

    while t_ms <= ROUND_DURATION_MS:
        syn_current: Dict[int, float] = {n.nid: 0.0 for n in net.living_neurons()}
        delivered = deliver_pending_events(pending, t_ms, syn_current)
        delivered_total += delivered

        ext_currents: Dict[int, float] = {}
        spiking_ids: List[int] = []

        for neuron in sorted(net.living_neurons(), key=lambda n: n.nid):
            ext = external_drive(neuron, t_ms, params, rng, net)
            syn = syn_current.get(neuron.nid, 0.0)
            total_current = ext + syn
            ext_currents[neuron.nid] = ext

            spiked = update_neuron(neuron, total_current, DT_MS, net)
            if spiked:
                neuron.last_spike_ms = t_ms
                spiking_ids.append(neuron.nid)

        for nid in spiking_ids:
            schedule_synaptic_events(net, nid, t_ms, pending)

        hebbian_update(net, spiking_ids, t_ms)

        for neuron in net.living_neurons():
            voltage_sum += neuron.voltage_mv
            voltage_samples += 1

        if t_ms + 1e-9 >= next_print:
            avg_ext = sum(ext_currents.values()) / max(1, len(ext_currents))
            avg_syn = sum(syn_current.values()) / max(1, len(syn_current))
            print_network_snapshot(net, t_ms + DT_MS, spiking_ids, len(pending), avg_ext, avg_syn)
            next_print += PRINT_EVERY_MS

        t_ms += DT_MS

    # Detect motif before growth/pruning so current round activity can be stored.
    motif = detect_motif(net, round_index)
    motif_added = False
    if motif is not None:
        net.memory_motifs.append(motif)
        # Keep most recent motifs only.
        net.memory_motifs = net.memory_motifs[-8:]
        motif_added = True

        # Mark motif neurons.
        for nid in motif[1]:
            if nid in net.neurons and net.neurons[nid].alive:
                if net.neurons[nid].role not in ("sensor", "regulator"):
                    net.neurons[nid].role = "motif"

    new_neurons, new_synapses = grow_network(net, params, rng)
    pruned_neurons, pruned_synapses = prune_network(net, params, rng)

    living_after = net.living_neurons()
    total_spikes = sum(n.spike_count for n in living_after)
    mean_rate = total_spikes / max(1, len(living_after)) / (ROUND_DURATION_MS / 1000.0)

    if mean_rate < params.target_activity_low:
        controller_note = "underactive: future rounds will increase drive and growth pressure"
    elif mean_rate > params.target_activity_high:
        controller_note = "overactive: future rounds will increase pruning and reduce drive"
    elif motif_added:
        controller_note = "useful motif found: preserve and explore around this activity pattern"
    else:
        controller_note = "balanced activity: continue exploration with light structural changes"

    summary = summarize_round(
        net=net,
        round_index=round_index,
        params=params,
        neurons_before=neurons_before,
        synapses_before=synapses_before,
        voltage_sum=voltage_sum,
        voltage_samples=voltage_samples,
        new_neurons=new_neurons,
        new_synapses=new_synapses,
        pruned_neurons=pruned_neurons,
        pruned_synapses=pruned_synapses,
        motif_added=motif_added,
        controller_note=controller_note,
    )

    print_round_summary(summary, net)
    print("\nAI controller decision:")
    print(f"  {controller_note}")
    if motif_added:
        print(f"  Stored motif: {motif[0]}")
    print("  Next round parameters will be selected from activity, silence, growth, and pruning pressure.")

    age_network(net)
    return summary


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    rng = Random(SEED)
    net = make_initial_network(rng)
    summaries: List[RoundSummary] = []
    previous: Optional[RoundSummary] = None

    print("Terminal Neuron Network Growth, Pruning, and Self-Exploration Simulation v3 - Homeostatic Control")
    print("No graphics. No external packages.")
    print("Model: adaptive spiking network with structural plasticity.")
    print("Goal: let the network grow, prune, and explore its own activity motifs.")
    print("v3 tuning: increased excitability plus homeostatic threshold, inhibition, synaptic scaling, and redundant-link pruning.")
    print(f"Initial structure: {structural_signature(net)}")

    for round_index in range(1, TOTAL_ROUNDS + 1):
        net.round_index = round_index
        update_homeostatic_state(net, previous)
        apply_homeostatic_synaptic_scaling(net)
        params = make_round_params(round_index, previous)
        # If homeostasis is damping overactivity, slightly reduce growth and increase pruning.
        if net.homeostasis.state_label == "dampening overactivity":
            params.base_drive_na *= 0.90
            params.exploration_drive_na *= 0.86
            params.growth_pressure *= 0.72
            params.prune_pressure = clamp(params.prune_pressure + 0.20, 0.0, 1.0)
        elif net.homeostasis.state_label == "lifting underactivity":
            params.base_drive_na *= 1.05
            params.exploration_drive_na *= 1.08
            params.growth_pressure = clamp(params.growth_pressure + 0.10, 0.0, 1.0)
            params.prune_pressure *= 0.75
        summary = run_round(net, params, rng)
        summaries.append(summary)
        previous = summary

    print_final_summary(summaries, net)


if __name__ == "__main__":
    main()
