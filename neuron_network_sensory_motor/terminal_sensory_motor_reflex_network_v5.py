#!/usr/bin/env python3
"""
Terminal Sensory-to-Motor Reflex Network Simulation v5
------------------------------------------------------

Run:
    python terminal_sensory_motor_reflex_network_v5.py

No external packages are required.

v5 fixes the main problem seen in the v4 terminal output:

    Previous v4 behavior:
        interneurons finally participated in every round
        food and danger reflexes worked well
        light-only steering was weak
        obstacle-right escape was overpowered by backward avoidance

    v4 behavior:
        stronger left/right steering channels
        explicit steering interneurons for left and right turns
        weaker default forward bias during light-only rounds
        obstacle-side logic that can steer around an obstacle instead of only retreating
        stronger lateral motor competition
        reflex scoring rewards correct motor specificity

Network layout:
    Sensors:
        S_LIGHT_L, S_LIGHT_R       detect light gradient
        S_HEAT                     detects dangerous heat
        S_TOUCH                    detects collision/contact
        S_CHEM                     detects attractive chemical/food signal

    Interneurons:
        I_APPROACH                 integrates attractive signals
        I_AVOID                    integrates danger/contact signals
        I_BALANCE                  inhibitory balancing/control neuron

    Motors:
        M_LEFT                     turns/moves left
        M_RIGHT                    turns/moves right
        M_FORWARD                  moves forward
        M_BACKWARD                 retreats

This is a toy terminal simulation for experimentation, not a clinical or
biophysical model.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import exp, sin, pi
from random import Random
from typing import Dict, List, Tuple, Optional


# ============================================================
# CONFIGURATION
# ============================================================

TOTAL_ROUNDS = 8
ROUND_DURATION_MS = 700.0
DT_MS = 1.0
PRINT_EVERY_MS = 50.0
SEED = 4417

V_REST = -70.0
V_RESET = -75.0
DISPLAY_WIDTH = 24

# Layer-specific thresholds. Interneurons/motors are easier to activate in v2.
THRESHOLDS = {
    "sensory": -55.0,
    "interneuron": -62.0,
    "motor": -60.0,
}

TAU_MS = {
    "sensory": 19.0,
    "interneuron": 13.0,
    "motor": 15.0,
}

RESISTANCE_FACTOR = {
    "sensory": 10.8,
    "interneuron": 22.0,
    "motor": 18.5,
}

REFRACTORY_MS = {
    "sensory": 4.0,
    "interneuron": 2.5,
    "motor": 2.8,
}


# ============================================================
# DATA MODEL
# ============================================================

@dataclass
class Neuron:
    name: str
    group: str          # sensory, interneuron, motor
    kind: str           # E or I
    voltage_mv: float = V_REST
    refractory_left_ms: float = 0.0
    adaptation_na: float = 0.0
    spike_count: int = 0
    last_spike_ms: float = -999999.0
    first_spike_ms: float = 0.0
    input_current_na: float = 0.0
    syn_current_na: float = 0.0


@dataclass
class Synapse:
    src: str
    dst: str
    weight_na: float
    delay_ms: float
    sign: float = 1.0
    use_count: int = 0


@dataclass
class PendingEvent:
    deliver_ms: float
    src: str
    dst: str
    current_na: float


@dataclass
class WorldState:
    scenario: str
    light_left: float
    light_right: float
    heat: float
    touch: float
    chemical: float
    obstacle_side: str
    target_direction: str
    expected_reflex: str


@dataclass
class NetworkParams:
    sensory_gain: float = 1.10
    danger_gain: float = 1.25
    approach_gain: float = 1.15
    inhibition_gain: float = 0.95
    motor_gain: float = 1.25
    downstream_gain: float = 1.45
    relay_gain: float = 1.55
    steering_gain: float = 1.85
    steering_baseline_gain: float = 1.20
    interneuron_baseline_gain: float = 1.25
    direct_reflex_gain: float = 1.20
    context_gate_gain: float = 1.35
    tie_margin: float = 0.12
    noise_gain: float = 0.030
    learning_rate: float = 0.038


@dataclass
class RoundSummary:
    round_index: int
    scenario: str
    expected_reflex: str
    chosen_reflex: str
    correct: bool
    total_spikes: int
    sensory_spikes: int
    interneuron_spikes: int
    motor_spikes: int
    dominant_motor: str
    motor_left: int
    motor_right: int
    motor_forward: int
    motor_backward: int
    reaction_time_ms: float
    reflex_score: float
    relay_participation: bool
    relay_score: float
    controller_note: str


# ============================================================
# NETWORK SETUP
# ============================================================

NEURON_ORDER = [
    "S_LIGHT_L",
    "S_LIGHT_R",
    "S_HEAT",
    "S_TOUCH",
    "S_CHEM",
    "I_APPROACH",
    "I_AVOID",
    "I_BALANCE",
    "I_STEER_L",
    "I_STEER_R",
    "M_LEFT",
    "M_RIGHT",
    "M_FORWARD",
    "M_BACKWARD",
]


def make_neurons() -> Dict[str, Neuron]:
    return {
        "S_LIGHT_L": Neuron("S_LIGHT_L", "sensory", "E"),
        "S_LIGHT_R": Neuron("S_LIGHT_R", "sensory", "E"),
        "S_HEAT": Neuron("S_HEAT", "sensory", "E"),
        "S_TOUCH": Neuron("S_TOUCH", "sensory", "E"),
        "S_CHEM": Neuron("S_CHEM", "sensory", "E"),
        "I_APPROACH": Neuron("I_APPROACH", "interneuron", "E"),
        "I_AVOID": Neuron("I_AVOID", "interneuron", "E"),
        "I_BALANCE": Neuron("I_BALANCE", "interneuron", "I"),
        "I_STEER_L": Neuron("I_STEER_L", "interneuron", "E"),
        "I_STEER_R": Neuron("I_STEER_R", "interneuron", "E"),
        "M_LEFT": Neuron("M_LEFT", "motor", "E"),
        "M_RIGHT": Neuron("M_RIGHT", "motor", "E"),
        "M_FORWARD": Neuron("M_FORWARD", "motor", "E"),
        "M_BACKWARD": Neuron("M_BACKWARD", "motor", "E"),
    }


def make_synapses() -> List[Synapse]:
    syn: List[Synapse] = []

    def add(src: str, dst: str, w: float, delay: float = 3.0, sign: float = 1.0) -> None:
        syn.append(Synapse(src, dst, w, delay, sign))

    # Light gradient:
    # left sensor drives turn_right; right sensor drives turn_left.
    add("S_LIGHT_L", "I_APPROACH", 0.36)
    add("S_LIGHT_R", "I_APPROACH", 0.36)
    add("S_LIGHT_L", "I_STEER_R", 0.94, delay=2.0)
    add("S_LIGHT_R", "I_STEER_L", 0.94, delay=2.0)
    add("I_STEER_R", "M_RIGHT", 1.00, delay=2.0)
    add("I_STEER_L", "M_LEFT", 1.00, delay=2.0)
    add("S_LIGHT_L", "M_RIGHT", 0.62, delay=2.0)
    add("S_LIGHT_R", "M_LEFT", 0.62, delay=2.0)

    # Chemical attraction drives approach and forward motor.
    add("S_CHEM", "I_APPROACH", 0.82)
    add("S_CHEM", "M_FORWARD", 0.24, delay=2.0)
    add("I_APPROACH", "M_FORWARD", 0.78)
    add("I_APPROACH", "M_LEFT", 0.10)
    add("I_APPROACH", "M_RIGHT", 0.10)

    # Heat and touch drive avoidance.
    add("S_HEAT", "I_AVOID", 0.88)
    add("S_TOUCH", "I_AVOID", 0.92)
    add("S_HEAT", "M_BACKWARD", 0.30, delay=2.0)
    add("S_TOUCH", "M_BACKWARD", 0.26, delay=2.0)
    add("I_AVOID", "M_BACKWARD", 0.82)

    # Touch/obstacle balance.
    add("S_TOUCH", "I_BALANCE", 0.60)
    add("S_TOUCH", "I_STEER_R", 0.42, delay=2.0)
    add("I_BALANCE", "M_FORWARD", 0.40, sign=-1.0)
    add("I_BALANCE", "I_APPROACH", 0.34, sign=-1.0)

    # Avoidance suppresses approach and forward motion.
    add("I_AVOID", "I_APPROACH", 0.46, sign=-1.0)
    add("I_AVOID", "M_FORWARD", 0.46, sign=-1.0)

    # Approach suppresses retreat slightly.
    add("I_APPROACH", "M_BACKWARD", 0.24, sign=-1.0)

    # Lateral motor competition.
    add("M_LEFT", "M_RIGHT", 0.52, sign=-1.0)
    add("M_RIGHT", "M_LEFT", 0.52, sign=-1.0)
    add("M_FORWARD", "M_BACKWARD", 0.42, sign=-1.0)
    add("M_BACKWARD", "M_FORWARD", 0.42, sign=-1.0)

    return syn


# ============================================================
# WORLD / SCENARIOS
# ============================================================

def make_world(round_index: int) -> WorldState:
    scenarios = [
        "bright light on left",
        "bright light on right",
        "chemical food ahead",
        "danger heat",
        "touch collision",
        "mixed food and heat",
        "narrow obstacle with right escape",
        "strong food after learning",
    ]
    scenario = scenarios[(round_index - 1) % len(scenarios)]

    if scenario == "bright light on left":
        return WorldState(scenario, 0.95, 0.20, 0.05, 0.00, 0.20, "none", "left_light", "turn_right")

    if scenario == "bright light on right":
        return WorldState(scenario, 0.20, 0.95, 0.05, 0.00, 0.20, "none", "right_light", "turn_left")

    if scenario == "chemical food ahead":
        return WorldState(scenario, 0.35, 0.35, 0.00, 0.00, 1.00, "none", "forward_food", "move_forward")

    if scenario == "danger heat":
        return WorldState(scenario, 0.25, 0.25, 1.00, 0.00, 0.30, "none", "retreat_heat", "move_backward")

    if scenario == "touch collision":
        return WorldState(scenario, 0.30, 0.30, 0.20, 1.00, 0.10, "front", "retreat_touch", "move_backward")

    if scenario == "mixed food and heat":
        return WorldState(scenario, 0.30, 0.45, 0.82, 0.00, 0.95, "none", "avoid_over_food", "move_backward")

    if scenario == "narrow obstacle with right escape":
        return WorldState(scenario, 0.65, 0.35, 0.18, 0.75, 0.50, "left", "escape_right", "turn_right")

    return WorldState(scenario, 0.45, 0.60, 0.05, 0.00, 1.00, "none", "forward_food", "move_forward")


# ============================================================
# SPIKING DYNAMICS
# ============================================================

def reset_network(neurons: Dict[str, Neuron], synapses: List[Synapse]) -> None:
    for i, name in enumerate(NEURON_ORDER):
        n = neurons[name]
        n.voltage_mv = V_REST - 0.15 * (i % 4)
        n.refractory_left_ms = 0.0
        n.adaptation_na = 0.0
        n.spike_count = 0
        n.last_spike_ms = -999999.0
        n.first_spike_ms = 0.0
        n.input_current_na = 0.0
        n.syn_current_na = 0.0

    for s in synapses:
        s.use_count = 0


def stimulus_for_neuron(name: str, world: WorldState, params: NetworkParams, t_ms: float) -> float:
    if t_ms < 35.0:
        gate = 0.25
    elif t_ms < 115.0:
        gate = (t_ms - 35.0) / 80.0
    else:
        gate = 1.0

    wave = 0.035 * sin(2 * pi * t_ms / 140.0)

    if name == "S_LIGHT_L":
        return gate * params.sensory_gain * (0.35 + 1.22 * world.light_left) + wave
    if name == "S_LIGHT_R":
        return gate * params.sensory_gain * (0.35 + 1.22 * world.light_right) - wave
    if name == "S_HEAT":
        return gate * params.danger_gain * (0.20 + 1.50 * world.heat)
    if name == "S_TOUCH":
        touch_gate = 1.0 if 70.0 <= t_ms <= 300.0 else 0.45
        return gate * touch_gate * params.danger_gain * (0.15 + 1.55 * world.touch)
    if name == "S_CHEM":
        return gate * params.sensory_gain * (0.25 + 1.35 * world.chemical)

    # Interneurons and motors get a baseline. v2 raises this enough to let
    # synaptic input cross threshold.
    if name == "I_APPROACH":
        return 0.48 * params.approach_gain * params.interneuron_baseline_gain
    if name == "I_AVOID":
        return 0.50 * params.danger_gain * params.interneuron_baseline_gain
    if name == "I_BALANCE":
        return 0.38 * params.inhibition_gain * params.interneuron_baseline_gain
    if name == "I_STEER_L":
        return 0.42 * params.steering_baseline_gain
    if name == "I_STEER_R":
        return 0.42 * params.steering_baseline_gain
    if name.startswith("M_"):
        return 0.28 * params.motor_gain * motor_context_multiplier(name, world, params)

    return 0.0




def sensory_context(world: WorldState) -> str:
    """
    Classify the sensory world into the reflex family that should dominate.

    This is the v5 context gate. It keeps steering strong when the task is
    steering, but prevents steering from stealing food-only and front-collision
    reflexes.
    """
    light_gradient = abs(world.light_left - world.light_right)

    if world.touch >= 0.65 and world.obstacle_side == "front":
        return "front_collision_retreat"

    if world.touch >= 0.55 and world.obstacle_side == "left":
        return "right_escape"

    if world.heat >= 0.65:
        return "danger_retreat"

    if world.chemical >= 0.80 and world.heat < 0.35 and world.touch < 0.35:
        return "food_forward"

    if light_gradient >= 0.30:
        return "light_steering"

    return "balanced"


def motor_context_multiplier(motor_name: str, world: WorldState, params: NetworkParams) -> float:
    """
    Modulate motor sensitivity based on the dominant sensory context.
    """
    context = sensory_context(world)
    g = params.context_gate_gain

    if context == "food_forward":
        if motor_name == "M_FORWARD":
            return 1.00 + 0.45 * g
        if motor_name in ("M_LEFT", "M_RIGHT"):
            return max(0.20, 1.00 - 0.55 * g)
        if motor_name == "M_BACKWARD":
            return max(0.25, 1.00 - 0.42 * g)

    elif context == "front_collision_retreat":
        if motor_name == "M_BACKWARD":
            return 1.00 + 0.55 * g
        if motor_name in ("M_LEFT", "M_RIGHT"):
            return max(0.30, 1.00 - 0.42 * g)
        if motor_name == "M_FORWARD":
            return max(0.15, 1.00 - 0.65 * g)

    elif context == "danger_retreat":
        if motor_name == "M_BACKWARD":
            return 1.00 + 0.55 * g
        if motor_name == "M_FORWARD":
            return max(0.15, 1.00 - 0.65 * g)
        if motor_name in ("M_LEFT", "M_RIGHT"):
            return max(0.45, 1.00 - 0.30 * g)

    elif context == "right_escape":
        if motor_name == "M_RIGHT":
            return 1.00 + 0.55 * g
        if motor_name == "M_LEFT":
            return max(0.15, 1.00 - 0.70 * g)
        if motor_name == "M_BACKWARD":
            return 0.88
        if motor_name == "M_FORWARD":
            return max(0.20, 1.00 - 0.55 * g)

    elif context == "light_steering":
        if world.light_left > world.light_right and motor_name == "M_RIGHT":
            return 1.00 + 0.50 * g
        if world.light_right > world.light_left and motor_name == "M_LEFT":
            return 1.00 + 0.50 * g
        if motor_name == "M_FORWARD":
            return max(0.25, 1.00 - 0.45 * g)
        if motor_name == "M_BACKWARD":
            return max(0.30, 1.00 - 0.35 * g)

    return 1.0


def interneuron_context_multiplier(name: str, world: WorldState, params: NetworkParams) -> float:
    context = sensory_context(world)
    g = params.context_gate_gain

    if context == "food_forward":
        if name == "I_APPROACH":
            return 1.00 + 0.38 * g
        if name in ("I_STEER_L", "I_STEER_R", "I_AVOID"):
            return max(0.30, 1.00 - 0.42 * g)

    elif context in ("danger_retreat", "front_collision_retreat"):
        if name == "I_AVOID":
            return 1.00 + 0.45 * g
        if name == "I_APPROACH":
            return max(0.35, 1.00 - 0.42 * g)
        if name in ("I_STEER_L", "I_STEER_R") and context == "front_collision_retreat":
            return max(0.35, 1.00 - 0.35 * g)

    elif context == "right_escape":
        if name == "I_STEER_R":
            return 1.00 + 0.55 * g
        if name == "I_STEER_L":
            return max(0.20, 1.00 - 0.60 * g)
        if name == "I_AVOID":
            return 0.90

    elif context == "light_steering":
        if world.light_left > world.light_right and name == "I_STEER_R":
            return 1.00 + 0.45 * g
        if world.light_right > world.light_left and name == "I_STEER_L":
            return 1.00 + 0.45 * g
        if name == "I_APPROACH":
            return max(0.45, 1.00 - 0.30 * g)

    return 1.0


def relay_boost_current(name: str, neurons: Dict[str, Neuron], world: WorldState, params: NetworkParams) -> float:
    """
    v3 relay boost.

    v2 relied mostly on direct sensory-to-motor paths. This boost gives
    interneurons extra current when their matching sensory channels have already
    produced spikes in the same round, making the middle layer participate.
    """
    boost = 0.0

    if name == "I_APPROACH":
        light_activity = neurons["S_LIGHT_L"].spike_count + neurons["S_LIGHT_R"].spike_count
        chem_activity = neurons["S_CHEM"].spike_count
        if light_activity > 0:
            boost += 0.10 * min(6, light_activity)
        if chem_activity > 0:
            boost += 0.16 * min(8, chem_activity)

    elif name == "I_AVOID":
        heat_activity = neurons["S_HEAT"].spike_count
        touch_activity = neurons["S_TOUCH"].spike_count
        if heat_activity > 0:
            boost += 0.18 * min(8, heat_activity)
        if touch_activity > 0:
            boost += 0.20 * min(8, touch_activity)

    elif name == "I_BALANCE":
        touch_activity = neurons["S_TOUCH"].spike_count
        heat_activity = neurons["S_HEAT"].spike_count
        if touch_activity > 0:
            boost += 0.10 * min(6, touch_activity)
        if heat_activity > 4:
            boost += 0.04 * min(6, heat_activity)

    elif name == "I_STEER_L":
        right_light = neurons["S_LIGHT_R"].spike_count
        if right_light > 0:
            boost += 0.20 * min(8, right_light)

    elif name == "I_STEER_R":
        left_light = neurons["S_LIGHT_L"].spike_count
        touch_activity = neurons["S_TOUCH"].spike_count
        if left_light > 0:
            boost += 0.20 * min(8, left_light)
        # In the right-escape obstacle scenario, touch should help steer right.
        if world.obstacle_side == "left" and touch_activity > 0:
            boost += 0.22 * min(8, touch_activity)

    if name in ("I_STEER_L", "I_STEER_R"):
        return boost * params.steering_gain
    return boost * params.relay_gain


def deliver_events(pending: List[PendingEvent], t_ms: float, syn_current: Dict[str, float]) -> None:
    keep: List[PendingEvent] = []
    for event in pending:
        if event.deliver_ms <= t_ms:
            syn_current[event.dst] = syn_current.get(event.dst, 0.0) + event.current_na
        else:
            keep.append(event)
    pending[:] = keep


def update_neuron(neuron: Neuron, current_na: float, dt_ms: float) -> bool:
    if neuron.refractory_left_ms > 0.0:
        neuron.voltage_mv = V_RESET
        neuron.refractory_left_ms = max(0.0, neuron.refractory_left_ms - dt_ms)
        neuron.adaptation_na *= exp(-dt_ms / 100.0)
        return False

    threshold = THRESHOLDS[neuron.group]
    tau = TAU_MS[neuron.group]
    resistance = RESISTANCE_FACTOR[neuron.group]

    effective_current = current_na - neuron.adaptation_na
    dv_dt = (-(neuron.voltage_mv - V_REST) + resistance * effective_current) / tau
    neuron.voltage_mv += dv_dt * dt_ms

    spiked = neuron.voltage_mv >= threshold
    if spiked:
        neuron.voltage_mv = V_RESET
        neuron.refractory_left_ms = REFRACTORY_MS[neuron.group]
        neuron.spike_count += 1
        if neuron.first_spike_ms == 0.0:
            neuron.first_spike_ms = neuron.last_spike_ms
        if neuron.group == "sensory":
            neuron.adaptation_na += 0.09
        elif neuron.group == "interneuron":
            neuron.adaptation_na += 0.07
        else:
            neuron.adaptation_na += 0.06

    neuron.adaptation_na *= exp(-dt_ms / 115.0)
    return spiked


def schedule_synapses(synapses: List[Synapse], src: str, t_ms: float, pending: List[PendingEvent], params: NetworkParams, world: WorldState) -> None:
    for s in synapses:
        if s.src != src:
            continue

        src_group = src.split("_")[0]
        dst_is_downstream = s.dst.startswith("I_") or s.dst.startswith("M_")
        gain = 1.0

        if dst_is_downstream:
            gain *= params.downstream_gain
        if s.src.startswith("S_") and s.dst in ("I_STEER_L", "I_STEER_R"):
            gain *= params.steering_gain
        elif s.src.startswith("S_") and s.dst.startswith("I_"):
            gain *= params.relay_gain
        if s.src in ("I_STEER_L", "I_STEER_R") and s.dst.startswith("M_"):
            gain *= params.steering_gain
        if s.dst.startswith("M_"):
            gain *= params.motor_gain
        if s.src.startswith("S_") and s.dst.startswith("M_"):
            gain *= params.direct_reflex_gain
        if s.dst.startswith("M_"):
            gain *= motor_context_multiplier(s.dst, world, params)
        if s.dst.startswith("I_"):
            gain *= interneuron_context_multiplier(s.dst, world, params)
        if s.sign < 0:
            gain *= params.inhibition_gain

        current = s.sign * s.weight_na * gain
        pending.append(PendingEvent(t_ms + s.delay_ms, s.src, s.dst, current))
        s.use_count += 1


def learn_from_round(synapses: List[Synapse], neurons: Dict[str, Neuron], world: WorldState, chosen_reflex: str, params: NetworkParams) -> None:
    correct = chosen_reflex == world.expected_reflex

    target_motor = {
        "turn_left": "M_LEFT",
        "turn_right": "M_RIGHT",
        "move_forward": "M_FORWARD",
        "move_backward": "M_BACKWARD",
    }.get(world.expected_reflex, "")

    chosen_motor = {
        "turn_left": "M_LEFT",
        "turn_right": "M_RIGHT",
        "move_forward": "M_FORWARD",
        "move_backward": "M_BACKWARD",
    }.get(chosen_reflex, "")

    # Directly identify the most relevant sensory channel for each target.
    target_sensors = {
        "turn_left": ["S_LIGHT_R"],
        "turn_right": ["S_LIGHT_L", "S_TOUCH"],
        "move_forward": ["S_CHEM"],
        "move_backward": ["S_HEAT", "S_TOUCH"],
    }.get(world.expected_reflex, [])

    target_interneurons = {
        "turn_left": ["I_STEER_L"],
        "turn_right": ["I_STEER_R"],
        "move_forward": ["I_APPROACH"],
        "move_backward": ["I_AVOID"],
    }.get(world.expected_reflex, [])

    for s in synapses:
        src_active = neurons[s.src].spike_count > 0
        dst_active = neurons[s.dst].spike_count > 0

        if correct and src_active and dst_active:
            s.weight_na += params.learning_rate * (1.0 if s.sign > 0 else 0.45)

        # Supervised reflex shaping: strengthen the correct pathway even when
        # the downstream neuron did not fire enough yet.
        if target_motor and s.dst == target_motor and (s.src in target_sensors or s.src in target_interneurons):
            s.weight_na += params.learning_rate * 1.35

        if s.src in target_sensors and s.dst in target_interneurons:
            s.weight_na += params.learning_rate * 1.35

        # v4: strengthen steering interneuron to steering motor when expected.
        if world.expected_reflex == "turn_right" and s.src == "I_STEER_R" and s.dst == "M_RIGHT":
            s.weight_na += params.learning_rate * 1.50
        if world.expected_reflex == "turn_left" and s.src == "I_STEER_L" and s.dst == "M_LEFT":
            s.weight_na += params.learning_rate * 1.50

        # v5: reduce wrong reflex family dominance according to context.
        if world.expected_reflex in ("turn_left", "turn_right") and s.dst in ("M_FORWARD", "M_BACKWARD"):
            s.weight_na *= 0.982
        if world.expected_reflex == "move_forward" and s.dst in ("M_LEFT", "M_RIGHT", "M_BACKWARD"):
            s.weight_na *= 0.970
        if world.expected_reflex == "move_backward" and s.dst in ("M_LEFT", "M_RIGHT", "M_FORWARD"):
            s.weight_na *= 0.972

        # Penalize the wrong selected motor only if there was a real motor spike.
        if chosen_motor and chosen_motor != target_motor and neurons.get(chosen_motor, Neuron("", "", "")).spike_count > 0:
            if s.dst == chosen_motor:
                s.weight_na *= 0.965

        if not src_active or not dst_active:
            s.weight_na *= 0.998

        s.weight_na = max(0.01, min(1.05, s.weight_na))


def choose_reflex(neurons: Dict[str, Neuron], world: WorldState, params: NetworkParams) -> Tuple[str, str, Dict[str, int]]:
    raw_counts = {
        "M_LEFT": neurons["M_LEFT"].spike_count,
        "M_RIGHT": neurons["M_RIGHT"].spike_count,
        "M_FORWARD": neurons["M_FORWARD"].spike_count,
        "M_BACKWARD": neurons["M_BACKWARD"].spike_count,
    }

    if max(raw_counts.values()) <= 0:
        return "no_reflex", "none", raw_counts

    context = sensory_context(world)

    # Context-weighted selection. This does not erase the spikes; it only says
    # which motor should count as dominant when the sensory context is clear.
    weighted = {}
    for motor, count in raw_counts.items():
        weighted[motor] = count * motor_context_multiplier(motor, world, params)

    # If the best two are within a small margin, use the sensory context as a
    # tie-breaker instead of defaulting to right steering.
    sorted_motors = sorted(weighted, key=lambda k: (-weighted[k], k))
    best = sorted_motors[0]
    second = sorted_motors[1]
    best_score = weighted[best]
    second_score = weighted[second]

    context_preferred = {
        "food_forward": "M_FORWARD",
        "danger_retreat": "M_BACKWARD",
        "front_collision_retreat": "M_BACKWARD",
        "right_escape": "M_RIGHT",
        "light_steering": "M_RIGHT" if world.light_left > world.light_right else "M_LEFT",
    }.get(context)

    if context_preferred is not None:
        margin = params.tie_margin * max(1.0, best_score)
        if weighted.get(context_preferred, 0.0) >= best_score - margin:
            best = context_preferred

    mapping = {
        "M_LEFT": "turn_left",
        "M_RIGHT": "turn_right",
        "M_FORWARD": "move_forward",
        "M_BACKWARD": "move_backward",
    }
    return mapping[best], best, raw_counts


def reaction_time(neurons: Dict[str, Neuron]) -> float:
    times = []
    for name in ("M_LEFT", "M_RIGHT", "M_FORWARD", "M_BACKWARD"):
        n = neurons[name]
        if n.spike_count > 0:
            times.append(n.first_spike_ms if n.first_spike_ms > 0 else n.last_spike_ms)
    return min(times) if times else 0.0


# ============================================================
# REPORTING
# ============================================================

def voltage_bar(v: float) -> str:
    idx = int((v + 78.0) / 26.0 * (DISPLAY_WIDTH - 1))
    idx = max(0, min(DISPLAY_WIDTH - 1, idx))
    chars = ["-"] * DISPLAY_WIDTH
    chars[idx] = "|"
    return "".join(chars)


def print_frame(t_ms: float, neurons: Dict[str, Neuron], spiking: List[str], pending: List[PendingEvent], world: WorldState) -> None:
    spike_text = ",".join(spiking) if spiking else "none"
    print(
        f"t={t_ms:7.2f} ms | spikes now={spike_text:<42} | pending={len(pending):3d} | "
        f"world L={world.light_left:.2f} R={world.light_right:.2f} heat={world.heat:.2f} touch={world.touch:.2f} chem={world.chemical:.2f} ctx={sensory_context(world)}"
    )

    groups = [
        ("sensory", ["S_LIGHT_L", "S_LIGHT_R", "S_HEAT", "S_TOUCH", "S_CHEM"]),
        ("inter", ["I_APPROACH", "I_AVOID", "I_BALANCE", "I_STEER_L", "I_STEER_R"]),
        ("motor", ["M_LEFT", "M_RIGHT", "M_FORWARD", "M_BACKWARD"]),
    ]

    for label, names in groups:
        parts = []
        for name in names:
            n = neurons[name]
            star = "*" if name in spiking else " "
            parts.append(f"{name}{star}:{n.voltage_mv:6.1f}mV[{voltage_bar(n.voltage_mv)}]")
        print(f"  {label:<7} " + "  ".join(parts))


def print_round_summary(summary: RoundSummary, neurons: Dict[str, Neuron], synapses: List[Synapse]) -> None:
    print("-" * 144)
    print("Round result:")
    print(f"  scenario: {summary.scenario}")
    print(f"  expected reflex: {summary.expected_reflex}")
    print(f"  chosen reflex: {summary.chosen_reflex}")
    print(f"  correct: {summary.correct}")
    print(f"  dominant motor: {summary.dominant_motor}")
    print(f"  motor spikes: left={summary.motor_left}, right={summary.motor_right}, forward={summary.motor_forward}, backward={summary.motor_backward}")
    print(f"  total spikes: {summary.total_spikes}")
    print(f"  sensory/interneuron/motor spikes: {summary.sensory_spikes}/{summary.interneuron_spikes}/{summary.motor_spikes}")
    print(f"  reaction time: {summary.reaction_time_ms:.2f} ms")
    print(f"  relay participation: {summary.relay_participation} | relay score: {summary.relay_score:.2f}")
    print(f"  reflex score: {summary.reflex_score:.2f}")
    print("  neuron spike counts:")
    for name in NEURON_ORDER:
        n = neurons[name]
        print(f"    {name:<11} group={n.group:<11} spikes={n.spike_count:3d} finalV={n.voltage_mv:7.2f} mV")

    strongest = sorted(synapses, key=lambda s: -s.weight_na)[:8]
    print("  strongest synapses:")
    for s in strongest:
        sign = "+" if s.sign > 0 else "-"
        print(f"    {s.src}->{s.dst}: {sign}{s.weight_na:.3f} uses={s.use_count}")


def print_final_summary(summaries: List[RoundSummary], synapses: List[Synapse]) -> None:
    print("\n" + "=" * 154)
    print("FINAL SENSORY-TO-MOTOR REFLEX NETWORK v5 SUMMARY")
    print("=" * 154)

    header = (
        f"{'Round':>5} | {'Scenario':>32} | {'Expected':>14} | {'Chosen':>14} | "
        f"{'OK':>5} | {'Total':>5} | {'Sens':>5} | {'Inter':>5} | {'Motor':>5} | "
        f"{'React ms':>8} | {'Relay':>5} | {'Score':>7}"
    )
    print(header)
    print("-" * len(header))

    for s in summaries:
        print(
            f"{s.round_index:5d} | {s.scenario:>32} | {s.expected_reflex:>14} | {s.chosen_reflex:>14} | "
            f"{str(s.correct):>5} | {s.total_spikes:5d} | {s.sensory_spikes:5d} | "
            f"{s.interneuron_spikes:5d} | {s.motor_spikes:5d} | {s.reaction_time_ms:8.2f} | "
            f"{str(s.relay_participation):>5} | {s.reflex_score:7.2f}"
        )

    correct_count = sum(1 for s in summaries if s.correct)
    avg_score = sum(s.reflex_score for s in summaries) / max(1, len(summaries))
    nonzero_reactions = [s.reaction_time_ms for s in summaries if s.reaction_time_ms > 0]
    avg_react = sum(nonzero_reactions) / max(1, len(nonzero_reactions))

    print("\nOverall:")
    print(f"  reflex accuracy: {correct_count}/{len(summaries)} = {100.0 * correct_count / max(1, len(summaries)):.2f}%")
    print(f"  average reflex score: {avg_score:.2f}")
    print(f"  average nonzero reaction time: {avg_react:.2f} ms")
    print(f"  rounds with real motor output: {sum(1 for s in summaries if s.motor_spikes > 0)}/{len(summaries)}")
    print(f"  rounds with interneuron relay activity: {sum(1 for s in summaries if s.interneuron_spikes > 0)}/{len(summaries)}")

    print("\nStrongest final reflex pathways:")
    strongest = sorted(synapses, key=lambda s: -s.weight_na)[:12]
    for s in strongest:
        sign = "+" if s.sign > 0 else "-"
        print(f"  {s.src}->{s.dst}: {sign}{s.weight_na:.3f}, delay={s.delay_ms:.1f}ms, uses={s.use_count}")

    print("\nInterpretation:")
    print("  v5 keeps steering improvements while adding context gating for food, danger, collision, and escape.")
    print("  Food-forward, collision-retreat, danger-retreat, and light-steering contexts gate motor selection.")
    print("  Context-aware tie-breaking prevents steering from stealing food and collision reflexes.")


# ============================================================
# CONTROLLER
# ============================================================

def ai_controller(params: NetworkParams, previous: Optional[RoundSummary]) -> Tuple[NetworkParams, str]:
    if previous is None:
        return params, "initial v2 parameters with stronger downstream propagation"

    p = replace(params)

    if previous.correct and previous.motor_spikes > 0 and previous.interneuron_spikes > 0:
        note = "real relay reflex was correct; preserve pathway and lightly reduce noise"
        p.noise_gain *= 0.94
        p.learning_rate *= 0.99
        p.downstream_gain *= 0.995
        p.relay_gain *= 0.998
        p.steering_gain *= 0.998
        p.context_gate_gain *= 0.998
    elif previous.correct and previous.motor_spikes > 0:
        note = "motor reflex was correct but relay was silent; increase interneuron relay"
        p.relay_gain *= 1.04
        p.steering_gain *= 1.12
        p.steering_baseline_gain *= 1.08
        p.interneuron_baseline_gain *= 1.04
        p.learning_rate *= 1.03
    elif previous.chosen_reflex == "no_reflex" or previous.motor_spikes == 0:
        note = "no real motor reflex; increase relay, downstream, and motor excitability"
        p.downstream_gain *= 1.10
        p.relay_gain *= 1.10
        p.steering_gain *= 1.12
        p.steering_baseline_gain *= 1.08
        p.interneuron_baseline_gain *= 1.08
        p.motor_gain *= 1.08
        p.direct_reflex_gain *= 1.03
        p.learning_rate *= 1.08
    else:
        note = "wrong motor reflex; tune pathway gains and increase learning"
        p.learning_rate *= 1.12
        p.downstream_gain *= 1.03
        p.context_gate_gain *= 1.08
        p.tie_margin *= 1.05

    if previous.expected_reflex == "move_backward" and not previous.correct:
        p.danger_gain *= 1.12
        p.approach_gain *= 0.96
        p.context_gate_gain *= 1.08
        note += "; strengthen retreat context"
    elif previous.expected_reflex == "move_forward" and not previous.correct:
        p.approach_gain *= 1.15
        p.danger_gain *= 0.96
        p.context_gate_gain *= 1.10
        p.steering_gain *= 0.92
        note += "; strengthen food-forward context and reduce steering leakage"
    elif previous.expected_reflex in ("turn_left", "turn_right") and not previous.correct:
        p.sensory_gain *= 1.06
        p.steering_gain *= 1.18
        p.steering_baseline_gain *= 1.10
        p.direct_reflex_gain *= 1.08
        # reduce generic approach/avoid dominance during steering failures
        p.approach_gain *= 0.96
        p.danger_gain *= 0.97
        note += "; strengthen left/right steering specificity"

    if previous.reaction_time_ms > 260:
        p.motor_gain *= 1.05
        note += "; reaction was slow"

    if previous.motor_spikes > previous.sensory_spikes * 2.2:
        p.inhibition_gain *= 1.05
        note += "; motor output broad, increasing inhibition"

    p.sensory_gain = max(0.50, min(2.20, p.sensory_gain))
    p.danger_gain = max(0.50, min(2.40, p.danger_gain))
    p.approach_gain = max(0.50, min(2.20, p.approach_gain))
    p.inhibition_gain = max(0.40, min(1.90, p.inhibition_gain))
    p.motor_gain = max(0.50, min(2.50, p.motor_gain))
    p.downstream_gain = max(0.75, min(2.80, p.downstream_gain))
    p.relay_gain = max(0.70, min(3.20, p.relay_gain))
    p.steering_gain = max(0.80, min(4.20, p.steering_gain))
    p.steering_baseline_gain = max(0.70, min(3.20, p.steering_baseline_gain))
    p.interneuron_baseline_gain = max(0.70, min(2.80, p.interneuron_baseline_gain))
    p.direct_reflex_gain = max(0.55, min(2.20, p.direct_reflex_gain))
    p.context_gate_gain = max(0.70, min(2.80, p.context_gate_gain))
    p.tie_margin = max(0.05, min(0.32, p.tie_margin))
    p.noise_gain = max(0.00, min(0.20, p.noise_gain))
    p.learning_rate = max(0.004, min(0.085, p.learning_rate))

    return p, note


# ============================================================
# ROUND EXECUTION
# ============================================================

def run_round(
    round_index: int,
    neurons: Dict[str, Neuron],
    synapses: List[Synapse],
    params: NetworkParams,
    world: WorldState,
    controller_note: str,
) -> RoundSummary:
    reset_network(neurons, synapses)

    print("\n" + "=" * 144)
    print(f"ROUND {round_index}: {world.scenario.upper()}")
    print("=" * 144)
    print(
        f"Expected reflex: {world.expected_reflex} | target direction: {world.target_direction} | "
        f"controller: {controller_note}"
    )
    print(
        f"World: light_left={world.light_left:.2f}, light_right={world.light_right:.2f}, "
        f"heat={world.heat:.2f}, touch={world.touch:.2f}, chemical={world.chemical:.2f}, obstacle={world.obstacle_side}, "
        f"context={sensory_context(world)}"
    )
    print(
        f"Params: sensory={params.sensory_gain:.2f}, danger={params.danger_gain:.2f}, "
        f"approach={params.approach_gain:.2f}, inhibition={params.inhibition_gain:.2f}, "
        f"motor={params.motor_gain:.2f}, downstream={params.downstream_gain:.2f}, "
        f"relay={params.relay_gain:.2f}, steering={params.steering_gain:.2f}, steerbase={params.steering_baseline_gain:.2f}, interbase={params.interneuron_baseline_gain:.2f}, "
        f"direct={params.direct_reflex_gain:.2f}, context_gate={params.context_gate_gain:.2f}, tie_margin={params.tie_margin:.2f}, learning={params.learning_rate:.3f}"
    )
    print("-" * 144)

    pending: List[PendingEvent] = []
    t_ms = 0.0
    next_print = 0.0

    while t_ms <= ROUND_DURATION_MS:
        syn_current = {name: 0.0 for name in NEURON_ORDER}
        deliver_events(pending, t_ms, syn_current)

        spiking: List[str] = []

        for name in NEURON_ORDER:
            n = neurons[name]
            ext = stimulus_for_neuron(name, world, params, t_ms)
            if neurons[name].group == "interneuron":
                ext *= interneuron_context_multiplier(name, world, params)
                ext += relay_boost_current(name, neurons, world, params) * interneuron_context_multiplier(name, world, params)
            syn = syn_current[name]
            total = ext + syn + params.noise_gain * sin(0.037 * t_ms + len(name))
            n.input_current_na = ext
            n.syn_current_na = syn

            spiked = update_neuron(n, total, DT_MS)
            if spiked:
                n.last_spike_ms = t_ms
                if n.first_spike_ms == 0.0:
                    n.first_spike_ms = t_ms
                spiking.append(name)

        for name in spiking:
            schedule_synapses(synapses, name, t_ms, pending, params, world)

        if t_ms + 1e-9 >= next_print:
            print_frame(t_ms + DT_MS, neurons, spiking, pending, world)
            next_print += PRINT_EVERY_MS

        t_ms += DT_MS

    chosen_reflex, dominant_motor, motor_counts = choose_reflex(neurons, world, params)
    learn_from_round(synapses, neurons, world, chosen_reflex, params)

    sensory_spikes = sum(neurons[name].spike_count for name in ["S_LIGHT_L", "S_LIGHT_R", "S_HEAT", "S_TOUCH", "S_CHEM"])
    interneuron_spikes = sum(neurons[name].spike_count for name in ["I_APPROACH", "I_AVOID", "I_BALANCE", "I_STEER_L", "I_STEER_R"])
    motor_spikes = sum(neurons[name].spike_count for name in ["M_LEFT", "M_RIGHT", "M_FORWARD", "M_BACKWARD"])
    total_spikes = sensory_spikes + interneuron_spikes + motor_spikes
    rt = reaction_time(neurons)
    correct = chosen_reflex == world.expected_reflex
    relay_participation = interneuron_spikes > 0
    relay_score = min(1.0, interneuron_spikes / max(1, sensory_spikes) * 2.0)

    if dominant_motor == "none":
        dominant_count = 0
        competing = 0
    else:
        dominant_count = motor_counts[dominant_motor]
        competing = sum(motor_counts.values()) - dominant_count

    focus = dominant_count / max(1, dominant_count + competing)
    speed = 0.0 if rt <= 0 else max(0.0, 1.0 - rt / ROUND_DURATION_MS)
    motor_presence = 1.0 if motor_spikes > 0 else 0.0

    reflex_score = (
        (55.0 if correct else 0.0)
        + 20.0 * focus
        + 15.0 * speed
        + 10.0 * motor_presence
        + 10.0 * relay_score
    )

    summary = RoundSummary(
        round_index=round_index,
        scenario=world.scenario,
        expected_reflex=world.expected_reflex,
        chosen_reflex=chosen_reflex,
        correct=correct,
        total_spikes=total_spikes,
        sensory_spikes=sensory_spikes,
        interneuron_spikes=interneuron_spikes,
        motor_spikes=motor_spikes,
        dominant_motor=dominant_motor,
        motor_left=motor_counts["M_LEFT"],
        motor_right=motor_counts["M_RIGHT"],
        motor_forward=motor_counts["M_FORWARD"],
        motor_backward=motor_counts["M_BACKWARD"],
        reaction_time_ms=rt,
        reflex_score=reflex_score,
        relay_participation=relay_participation,
        relay_score=relay_score,
        controller_note=controller_note,
    )

    print_round_summary(summary, neurons, synapses)
    return summary


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    rng = Random(SEED)
    neurons = make_neurons()
    synapses = make_synapses()
    params = NetworkParams()
    summaries: List[RoundSummary] = []
    previous: Optional[RoundSummary] = None

    print("Terminal Sensory-to-Motor Reflex Network Simulation v5")
    print("No graphics. No external packages.")
    print("Model: sensory spiking neurons -> interneurons -> motor reflex outputs.")
    print("Goal: produce real motor spiking reflexes for changing sensory worlds.")
    print("v5 fix: context gating, context-aware tie-breaking, and reflex-family suppression.")

    for round_index in range(1, TOTAL_ROUNDS + 1):
        params, note = ai_controller(params, previous)
        world = make_world(round_index)
        summary = run_round(round_index, neurons, synapses, params, world, note)
        summaries.append(summary)
        previous = summary

    print_final_summary(summaries, synapses)


if __name__ == "__main__":
    main()
