#!/usr/bin/env python3
"""
Terminal-Only Single Bird 3D Octant Wander Simulation - Coverage Tuned

Standard-library only. No graphics.

This version fixes the prior problem where the bird remained trapped in O1.
The bird now has:
- stronger steering
- target advancement only after reaching target or timeout
- explicit axis-crossing bonuses
- longer rounds
- octant coverage controller
- behavior-specific target paths that force all signs of x, y, z

Run:
    python terminal_single_bird_3d_octant_wander_coverage.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import sin, cos, sqrt, atan2, degrees, pi
from typing import Dict, List, Tuple
import random


# ============================================================
# Vector math
# ============================================================

@dataclass
class Vec3:
    x: float
    y: float
    z: float

    def __add__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> "Vec3":
        return Vec3(self.x * scalar, self.y * scalar, self.z * scalar)

    def __truediv__(self, scalar: float) -> "Vec3":
        if abs(scalar) < 1e-12:
            return Vec3(0.0, 0.0, 0.0)
        return Vec3(self.x / scalar, self.y / scalar, self.z / scalar)

    def mag(self) -> float:
        return sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    def horizontal_mag(self) -> float:
        return sqrt(self.x * self.x + self.y * self.y)

    def unit(self) -> "Vec3":
        m = self.mag()
        if m < 1e-9:
            return Vec3(1.0, 0.0, 0.0)
        return self / m

    def dot(self, other: "Vec3") -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def fmt_vec(v: Vec3) -> str:
    return f"({v.x:+8.2f}, {v.y:+8.2f}, {v.z:+8.2f})"


def sign_label(value: float) -> str:
    return "+" if value >= 0 else "-"


def octant_of(pos: Vec3) -> str:
    sx = "+" if pos.x >= 0 else "-"
    sy = "+" if pos.y >= 0 else "-"
    sz = "+" if pos.z >= 0 else "-"
    mapping = {
        ("+", "+", "+"): "O1(+,+,+)",
        ("-", "+", "+"): "O2(-,+,+)",
        ("-", "-", "+"): "O3(-,-,+)",
        ("+", "-", "+"): "O4(+,-,+)",
        ("+", "+", "-"): "O5(+,+,-)",
        ("-", "+", "-"): "O6(-,+,-)",
        ("-", "-", "-"): "O7(-,-,-)",
        ("+", "-", "-"): "O8(+,-,-)",
    }
    return mapping[(sx, sy, sz)]


def octant_short(pos: Vec3) -> str:
    return f"{sign_label(pos.x)}{sign_label(pos.y)}{sign_label(pos.z)}"


def distance(a: Vec3, b: Vec3) -> float:
    return (a - b).mag()


# ============================================================
# Data structures
# ============================================================

@dataclass
class BirdState:
    pos: Vec3
    vel: Vec3
    acc: Vec3
    energy: float
    mode: str
    active_target_index: int = 0
    target_age: int = 0
    bank_angle: float = 0.0
    pitch_angle: float = 0.0
    heading_angle: float = 0.0
    distance_traveled: float = 0.0
    path_curvature_sum: float = 0.0
    last_direction: Vec3 = field(default_factory=lambda: Vec3(1.0, 0.0, 0.0))
    octant_visits: Dict[str, int] = field(default_factory=dict)
    octant_entry_step: Dict[str, int] = field(default_factory=dict)
    octant_transitions: int = 0
    target_hits: int = 0
    forced_target_advances: int = 0
    loop_count: int = 0
    near_origin_passes: int = 0
    axis_crossings_x: int = 0
    axis_crossings_y: int = 0
    axis_crossings_z: int = 0
    max_radius: float = 0.0
    min_radius: float = 0.0


@dataclass
class RoundParams:
    name: str
    description: str
    steps: int
    dt: float
    max_speed: float
    control_strength: float
    wander_strength: float
    chaos_strength: float
    turn_strength: float
    radial_limit: float
    energy_saving: float
    hit_radius: float
    target_timeout: int
    wind: Vec3
    target_sequence: List[Vec3]


@dataclass
class RoundSummary:
    round_index: int
    name: str
    final_pos: Vec3
    final_speed: float
    final_energy: float
    distance_traveled: float
    octants_visited: int
    octant_transitions: int
    target_hits: int
    forced_target_advances: int
    loop_count: int
    near_origin_passes: int
    max_radius: float
    path_curvature: float
    axis_crossings_total: int
    final_mode: str


# ============================================================
# Configuration
# ============================================================

RANDOM_SEED = 31
random.seed(RANDOM_SEED)

START_ENERGY = 96.0
PRINT_STEPS = {1, 20, 40, 60, 80, 100, 120, 150, 180, 210, 240}

O1 = Vec3(+92, +92, +92)
O2 = Vec3(-92, +92, +92)
O3 = Vec3(-92, -92, +92)
O4 = Vec3(+92, -92, +92)
O5 = Vec3(+92, +92, -92)
O6 = Vec3(-92, +92, -92)
O7 = Vec3(-92, -92, -92)
O8 = Vec3(+92, -92, -92)
ORIGIN = Vec3(0, 0, 0)

ALL_OCTANTS = [O1, O2, O3, O4, O5, O6, O7, O8]

ROUNDS: List[RoundParams] = [
    RoundParams(
        name="direct_octant_chain",
        description="directly visit O1 through O8, advancing target only on hit or timeout",
        steps=240,
        dt=0.10,
        max_speed=24.0,
        control_strength=0.240,
        wander_strength=0.40,
        chaos_strength=0.04,
        turn_strength=0.45,
        radial_limit=170.0,
        energy_saving=0.80,
        hit_radius=22.0,
        target_timeout=34,
        wind=Vec3(0.15, -0.05, 0.05),
        target_sequence=ALL_OCTANTS,
    ),
    RoundParams(
        name="axis_cross_survey",
        description="force repeated x/y/z sign changes with intermediate origin passes",
        steps=240,
        dt=0.10,
        max_speed=23.0,
        control_strength=0.260,
        wander_strength=0.55,
        chaos_strength=0.05,
        turn_strength=0.65,
        radial_limit=170.0,
        energy_saving=0.85,
        hit_radius=24.0,
        target_timeout=28,
        wind=Vec3(0.05, 0.10, -0.05),
        target_sequence=[O1, ORIGIN, O7, ORIGIN, O2, ORIGIN, O8, ORIGIN, O3, ORIGIN, O5, ORIGIN, O4, ORIGIN, O6],
    ),
    RoundParams(
        name="large_looping_orbit",
        description="large circular orbit that sweeps through all horizontal quadrants while z oscillates sign",
        steps=240,
        dt=0.10,
        max_speed=22.0,
        control_strength=0.210,
        wander_strength=1.15,
        chaos_strength=0.05,
        turn_strength=1.35,
        radial_limit=175.0,
        energy_saving=1.00,
        hit_radius=28.0,
        target_timeout=24,
        wind=Vec3(0.05, 0.15, 0.00),
        target_sequence=ALL_OCTANTS,
    ),
    RoundParams(
        name="diagonal_opposites",
        description="cut across opposite octants with strong diagonal transfers",
        steps=240,
        dt=0.10,
        max_speed=25.0,
        control_strength=0.270,
        wander_strength=0.60,
        chaos_strength=0.07,
        turn_strength=0.70,
        radial_limit=180.0,
        energy_saving=0.75,
        hit_radius=24.0,
        target_timeout=30,
        wind=Vec3(0.10, -0.15, 0.10),
        target_sequence=[O1, O7, O2, O8, O3, O5, O4, O6, O1],
    ),
    RoundParams(
        name="spiral_octant_expansion",
        description="spiral expands outward while the target octant changes every arc",
        steps=240,
        dt=0.10,
        max_speed=23.0,
        control_strength=0.220,
        wander_strength=1.30,
        chaos_strength=0.06,
        turn_strength=1.55,
        radial_limit=185.0,
        energy_saving=0.95,
        hit_radius=28.0,
        target_timeout=26,
        wind=Vec3(-0.10, 0.10, 0.05),
        target_sequence=ALL_OCTANTS + [O1, O7],
    ),
    RoundParams(
        name="chaotic_octant_hunt",
        description="unpredictable octant jumps with strong correction toward the current target",
        steps=240,
        dt=0.10,
        max_speed=25.0,
        control_strength=0.280,
        wander_strength=1.05,
        chaos_strength=0.20,
        turn_strength=1.10,
        radial_limit=180.0,
        energy_saving=0.70,
        hit_radius=24.0,
        target_timeout=24,
        wind=Vec3(0.10, -0.20, 0.00),
        target_sequence=[O6, O4, O3, O5, O2, O8, O7, O1, O4, O6],
    ),
    RoundParams(
        name="figure_eight_all_axes",
        description="3D figure-eight designed to cross x, y, and z axes many times",
        steps=240,
        dt=0.10,
        max_speed=22.5,
        control_strength=0.225,
        wander_strength=1.20,
        chaos_strength=0.04,
        turn_strength=1.60,
        radial_limit=170.0,
        energy_saving=1.05,
        hit_radius=28.0,
        target_timeout=26,
        wind=Vec3(0.00, 0.10, -0.05),
        target_sequence=ALL_OCTANTS,
    ),
    RoundParams(
        name="full_coverage_mixed",
        description="mixed direct moves, loops, origin dives, and diagonal cuts to cover the full coordinate system",
        steps=240,
        dt=0.10,
        max_speed=24.0,
        control_strength=0.255,
        wander_strength=1.00,
        chaos_strength=0.10,
        turn_strength=1.20,
        radial_limit=185.0,
        energy_saving=0.90,
        hit_radius=25.0,
        target_timeout=25,
        wind=Vec3(0.05, -0.05, 0.05),
        target_sequence=[O1, O3, ORIGIN, O6, O4, ORIGIN, O7, O5, O2, O8, ORIGIN],
    ),
]


# ============================================================
# Initialization and target logic
# ============================================================

def make_initial_state() -> BirdState:
    state = BirdState(
        pos=Vec3(0.0, 0.0, 0.0),
        vel=Vec3(10.0, 7.5, 5.0),
        acc=Vec3(0.0, 0.0, 0.0),
        energy=START_ENERGY,
        mode="coordinate_wander",
    )
    octant = octant_of(state.pos)
    state.octant_visits[octant] = 1
    state.octant_entry_step[octant] = 0
    return state


def base_target(params: RoundParams, state: BirdState) -> Vec3:
    if not params.target_sequence:
        return ORIGIN
    return params.target_sequence[state.active_target_index % len(params.target_sequence)]


def procedural_target(params: RoundParams, state: BirdState, step: int, target: Vec3) -> Vec3:
    t = step * params.dt

    if params.name == "large_looping_orbit":
        r = 112.0
        return Vec3(
            r * cos(t * 0.62),
            r * sin(t * 0.62),
            95.0 * sin(t * 0.31),
        )

    if params.name == "spiral_octant_expansion":
        r = min(150.0, 22.0 + step * 0.65)
        return Vec3(
            r * cos(t * 0.78),
            r * sin(t * 0.78),
            r * 0.78 * sin(t * 0.43),
        )

    if params.name == "figure_eight_all_axes":
        a = 105.0
        return Vec3(
            a * sin(t * 0.62),
            a * sin(t * 0.62) * cos(t * 0.62),
            95.0 * sin(t * 1.24),
        )

    # Direct-ish rounds still weave, but target sign dominates.
    return Vec3(
        target.x + 10.0 * sin(t * params.turn_strength),
        target.y + 10.0 * cos(t * params.turn_strength * 1.2),
        target.z + 10.0 * sin(t * params.turn_strength * 1.4 + 0.5),
    )


def update_target_progress(state: BirdState, params: RoundParams, step: int, target: Vec3) -> str:
    state.target_age += 1
    d = distance(state.pos, target)

    if d <= params.hit_radius:
        state.target_hits += 1
        state.active_target_index = (state.active_target_index + 1) % len(params.target_sequence)
        state.target_age = 0
        return "target_hit_advance"

    if state.target_age >= params.target_timeout:
        state.forced_target_advances += 1
        state.active_target_index = (state.active_target_index + 1) % len(params.target_sequence)
        state.target_age = 0
        return "timeout_advance"

    return "tracking"


# ============================================================
# Flight control
# ============================================================

def wind_at_step(params: RoundParams, step: int) -> Tuple[Vec3, str]:
    t = step * params.dt
    event = "steady"
    wind = Vec3(
        params.wind.x + 0.18 * sin(t * 1.0),
        params.wind.y + 0.18 * cos(t * 0.8),
        params.wind.z + 0.14 * sin(t * 0.6),
    )

    if step % 53 == 0:
        event = "x_axis_gust"
        wind.x += 1.2 * sin(t)
    elif step % 67 == 0:
        event = "y_axis_gust"
        wind.y += 1.4 * cos(t)
    elif step % 79 == 0:
        event = "z_axis_gust"
        wind.z += 1.3 * sin(t * 0.7)

    return wind, event


def lateral_wander(state: BirdState, params: RoundParams, step: int) -> Vec3:
    t = step * params.dt
    f = state.vel.unit()
    side_a = Vec3(-f.y, f.x, 0.0).unit()
    side_b = Vec3(-f.z, 0.0, f.x).unit()

    signal_a = sin(t * params.turn_strength * 1.7) + 0.45 * cos(t * 0.9)
    signal_b = cos(t * params.turn_strength * 1.3 + 0.8) + 0.35 * sin(t * 1.4)

    energy_scale = 0.40 + 0.60 * (state.energy / 100.0)
    return (side_a * signal_a + side_b * signal_b) * params.wander_strength * energy_scale


def boundary_force(state: BirdState, params: RoundParams) -> Vec3:
    r = state.pos.mag()
    if r <= params.radial_limit:
        return Vec3(0.0, 0.0, 0.0)
    return state.pos.unit() * (-(r - params.radial_limit) * 0.22)


def axis_crossing_bonus(state: BirdState, target: Vec3) -> Vec3:
    """Extra force when the target is in a different sign region.

    This deliberately pulls the bird across x=0, y=0, and z=0 instead of allowing
    it to get trapped in its current octant.
    """
    bonus = Vec3(0.0, 0.0, 0.0)

    if state.pos.x * target.x < 0:
        bonus.x += -0.75 if state.pos.x > 0 else 0.75
    if state.pos.y * target.y < 0:
        bonus.y += -0.75 if state.pos.y > 0 else 0.75
    if state.pos.z * target.z < 0:
        bonus.z += -0.75 if state.pos.z > 0 else 0.75

    # If target is origin, pull straight inward.
    if target.mag() < 1.0 and state.pos.mag() > 2.0:
        bonus = bonus + state.pos.unit() * -1.0

    return bonus


def coverage_bonus(state: BirdState, params: RoundParams) -> Vec3:
    """Softly push toward unvisited octants after the bird has explored too little."""
    if len(state.octant_visits) >= 6:
        return Vec3(0.0, 0.0, 0.0)

    unvisited = [t for t in ALL_OCTANTS if octant_of(t) not in state.octant_visits]
    if not unvisited:
        return Vec3(0.0, 0.0, 0.0)

    # Pick the unvisited octant farthest in sign-space from current position.
    target = max(unvisited, key=lambda p: distance(state.pos, p))
    return (target - state.pos).unit() * 0.55


def control_force(state: BirdState, params: RoundParams, step: int, target: Vec3) -> Vec3:
    to_target = target - state.pos
    d = to_target.mag()
    desired_dir = to_target.unit()

    approach = clamp(d / 55.0, 0.45, 1.0)
    energy_scale = 0.40 + 0.60 * (state.energy / 100.0)
    desired_speed = params.max_speed * approach * energy_scale
    desired_velocity = desired_dir * desired_speed

    steering = (desired_velocity - state.vel) * params.control_strength
    wander = lateral_wander(state, params, step)
    axis_bonus = axis_crossing_bonus(state, target)
    coverage = coverage_bonus(state, params)

    chaos = Vec3(
        sin(step * 0.37) * params.chaos_strength,
        cos(step * 0.29) * params.chaos_strength,
        sin(step * 0.23 + 1.2) * params.chaos_strength,
    )

    return steering + wander + axis_bonus + coverage + chaos + boundary_force(state, params)


def classify_mode(params: RoundParams, state: BirdState, target: Vec3, progress_event: str) -> str:
    if progress_event == "target_hit_advance":
        return "target_pass"
    if progress_event == "timeout_advance":
        return "retarget"

    if state.energy < 22.0:
        return "energy_saving_wander"
    if target.mag() < 1.0:
        return "origin_dive"
    if params.name == "large_looping_orbit":
        return "large_looping_orbit"
    if params.name == "figure_eight_all_axes":
        return "figure_eight_axis_cross"
    if params.name == "spiral_octant_expansion":
        return "spiral_axis_sweep"
    if params.name == "chaotic_octant_hunt":
        return "chaotic_octant_hunt"
    if params.name == "diagonal_opposites":
        return "diagonal_opposite_cut"
    if params.name == "axis_cross_survey":
        return "axis_crossing_survey"
    if params.name == "full_coverage_mixed":
        return "mixed_coverage_wander"
    return "direct_octant_move"


def update_angles(state: BirdState) -> None:
    horiz = max(state.vel.horizontal_mag(), 1e-9)
    state.heading_angle = degrees(atan2(state.vel.y, state.vel.x))
    state.pitch_angle = degrees(atan2(state.vel.z, horiz))
    state.bank_angle = clamp(-state.vel.y * 1.6 + state.acc.y * 3.2, -70.0, 70.0)


def update_energy(state: BirdState, params: RoundParams) -> None:
    speed = state.vel.mag()
    turn_cost = abs(state.bank_angle) * 0.001
    accel_cost = state.acc.mag() * 0.0035
    base_cost = 0.020 + speed * 0.0025
    cost = base_cost + turn_cost + accel_cost

    if state.mode in ("large_looping_orbit", "figure_eight_axis_cross", "spiral_axis_sweep", "mixed_coverage_wander"):
        cost *= 0.72
    if state.mode in ("target_pass", "origin_dive", "energy_saving_wander"):
        cost *= 0.55

    # Inward travel and smooth arcing are efficient.
    if state.pos.dot(state.vel) < 0:
        cost -= 0.020 * params.energy_saving
    if state.path_curvature_sum > 0 and "orbit" in state.mode:
        cost -= 0.010 * params.energy_saving

    state.energy = clamp(state.energy - cost, 0.0, 100.0)


def step_simulation(state: BirdState, params: RoundParams, step: int) -> Tuple[Dict[str, float], Vec3, Vec3, str, str]:
    base = base_target(params, state)
    target = procedural_target(params, state, step, base)
    wind, wind_event = wind_at_step(params, step)

    progress_event = update_target_progress(state, params, step, target)

    # Target may change immediately after progress update.
    base = base_target(params, state)
    target = procedural_target(params, state, step, base)

    ctrl = control_force(state, params, step, target)
    rel = state.vel - wind
    drag = rel * (-0.018 * rel.mag())
    thrust = state.vel.unit() * (0.55 + 0.45 * state.energy / 100.0)

    total = ctrl + drag + thrust
    state.acc = total
    state.vel = state.vel + state.acc * params.dt

    max_speed = params.max_speed * (0.58 + 0.42 * state.energy / 100.0)
    if state.vel.mag() > max_speed:
        state.vel = state.vel.unit() * max_speed

    old_pos = state.pos
    old_oct = octant_of(old_pos)
    old_dir = state.last_direction

    state.pos = state.pos + state.vel * params.dt
    state.distance_traveled += distance(old_pos, state.pos)

    new_oct = octant_of(state.pos)
    state.octant_visits[new_oct] = state.octant_visits.get(new_oct, 0) + 1
    if new_oct not in state.octant_entry_step:
        state.octant_entry_step[new_oct] = step
    if new_oct != old_oct:
        state.octant_transitions += 1

    if old_pos.x * state.pos.x < 0:
        state.axis_crossings_x += 1
    if old_pos.y * state.pos.y < 0:
        state.axis_crossings_y += 1
    if old_pos.z * state.pos.z < 0:
        state.axis_crossings_z += 1

    new_dir = state.vel.unit()
    turn_amount = 1.0 - clamp(old_dir.dot(new_dir), -1.0, 1.0)
    state.path_curvature_sum += turn_amount
    state.last_direction = new_dir

    if state.pos.mag() < 10.0 and old_pos.mag() >= 10.0:
        state.near_origin_passes += 1

    if state.path_curvature_sum > (state.loop_count + 1) * 0.85:
        state.loop_count += 1

    state.max_radius = max(state.max_radius, state.pos.mag())
    state.min_radius = min(state.min_radius, state.pos.mag())

    state.mode = classify_mode(params, state, target, progress_event)
    update_angles(state)
    update_energy(state, params)

    metrics = {
        "control": ctrl.mag(),
        "drag": drag.mag(),
        "thrust": thrust.mag(),
        "total": total.mag(),
        "radius": state.pos.mag(),
        "distance_to_target": distance(state.pos, target),
        "curvature": state.path_curvature_sum,
        "turn_step": turn_amount,
        "speed": state.vel.mag(),
        "axis_cross_total": state.axis_crossings_x + state.axis_crossings_y + state.axis_crossings_z,
    }

    event = progress_event if progress_event != "tracking" else wind_event
    return metrics, wind, target, event, progress_event


# ============================================================
# Printing and summaries
# ============================================================

def print_header() -> None:
    print("TERMINAL SINGLE-BIRD 3D OCTANT WANDER SIMULATION - COVERAGE TUNED")
    print("Standard-library only. No graphics.")
    print("World model: centered 3D coordinate system, not an xy grid.")
    print("Coverage improvements:")
    print("  - stronger steering")
    print("  - target advances only on hit or timeout")
    print("  - axis-crossing bonus pulls bird through x=0, y=0, z=0")
    print("  - coverage bonus pulls toward unvisited octants")
    print("  - longer 240-step rounds")
    print("State update:")
    print("  a = steering_to_active_target + lateral_wander + axis_bonus + coverage_bonus + chaos + boundary + drag + thrust")
    print("  v(t+dt) = energy_limited(v + a dt)")
    print("  x(t+dt) = x + v(t+dt) dt")
    print("Coordinate octants:")
    print("  O1 +++ | O2 -++ | O3 --+ | O4 +-+ | O5 ++- | O6 -+- | O7 --- | O8 +--")
    print("=" * 124)


def print_round_start(round_idx: int, params: RoundParams) -> None:
    print("=" * 124)
    print(f"ROUND {round_idx:02d} START | WANDER BEHAVIOR: {params.name}")
    print(f"Description: {params.description}")
    print("PARAMETERS")
    print(f"steps={params.steps}, dt={params.dt:.2f}, max_speed={params.max_speed:.2f}")
    print(f"control={params.control_strength:.3f}, wander={params.wander_strength:.3f}, chaos={params.chaos_strength:.3f}, turn={params.turn_strength:.3f}")
    print(f"hit_radius={params.hit_radius:.1f}, target_timeout={params.target_timeout}, radial_limit={params.radial_limit:.1f}")
    print(f"energy_saving={params.energy_saving:.2f}, wind={fmt_vec(params.wind)}")
    print("Target sequence:")
    for i, t in enumerate(params.target_sequence, 1):
        print(f"  T{i:02d}: {fmt_vec(t)} -> {octant_of(t)}")
    print("Behavior intent: force real 3D coordinate exploration and octant transitions.")


def print_step(round_idx: int, params: RoundParams, step: int, state: BirdState, metrics: Dict[str, float], wind: Vec3, target: Vec3, event: str) -> None:
    t = step * params.dt
    print("-" * 124)
    print(f"ROUND {round_idx:02d} | {params.name} | STEP {step:03d} | t={t:6.2f}s | event={event}")
    print(f"Mode:                {state.mode}")
    print(f"Current octant:      {octant_of(state.pos)}      short={octant_short(state.pos)}")
    print(f"Position:            {fmt_vec(state.pos)}")
    print(f"Velocity:            {fmt_vec(state.vel)}      speed={state.vel.mag():6.2f}")
    print(f"Acceleration:        {fmt_vec(state.acc)}")
    print(f"Active target index: {state.active_target_index + 1:4d} / {len(params.target_sequence):4d}      target_age={state.target_age:4d}")
    print(f"Target:              {fmt_vec(target)}      target_octant={octant_of(target)}")
    print(f"Distance to target:  {metrics['distance_to_target']:8.2f} m")
    print(f"Wind vector:         {fmt_vec(wind)}")
    print(f"Radius from origin:  {metrics['radius']:8.2f} m       radial_limit={params.radial_limit:7.2f} m")
    print(f"Heading:             {state.heading_angle:8.2f} deg     pitch={state.pitch_angle:8.2f} deg     bank={state.bank_angle:8.2f} deg")
    print(f"Energy:              {state.energy:8.2f}%       distance traveled={state.distance_traveled:8.2f} m")
    print(f"Octants visited:     {len(state.octant_visits):8d}       transitions={state.octant_transitions:8d}")
    print(f"Target hits:         {state.target_hits:8d}       timeout advances={state.forced_target_advances:8d}")
    print(f"Loops:               {state.loop_count:8d}       origin passes={state.near_origin_passes:8d}")
    print(f"Axis crossings:      x={state.axis_crossings_x:3d}, y={state.axis_crossings_y:3d}, z={state.axis_crossings_z:3d}, total={metrics['axis_cross_total']:3.0f}")
    print(
        "Force magnitudes:    "
        f"control={metrics['control']:.3f}, drag={metrics['drag']:.3f}, thrust={metrics['thrust']:.3f}, "
        f"total={metrics['total']:.3f}, turn_step={metrics['turn_step']:.4f}, curvature={metrics['curvature']:.3f}"
    )


def interpret_round(summary: RoundSummary) -> str:
    if summary.octants_visited >= 8:
        coverage = "full octant coverage achieved"
    elif summary.octants_visited >= 5:
        coverage = "broad coordinate coverage achieved"
    elif summary.octants_visited >= 3:
        coverage = "partial coordinate coverage achieved"
    else:
        coverage = "limited coordinate coverage"

    if "loop" in summary.name or "figure" in summary.name:
        style = "curved looping motion dominated"
    elif "direct" in summary.name:
        style = "direct target-to-target motion dominated"
    elif "diagonal" in summary.name:
        style = "diagonal coordinate crossing dominated"
    elif "chaotic" in summary.name:
        style = "sharp unpredictable target hunting dominated"
    elif "origin" in summary.name:
        style = "origin-diving and rebound motion dominated"
    else:
        style = "mixed coordinate wandering dominated"

    return f"{coverage}; {style}."


def summarize_round(round_idx: int, params: RoundParams, state: BirdState) -> RoundSummary:
    axis_total = state.axis_crossings_x + state.axis_crossings_y + state.axis_crossings_z
    summary = RoundSummary(
        round_index=round_idx,
        name=params.name,
        final_pos=state.pos,
        final_speed=state.vel.mag(),
        final_energy=state.energy,
        distance_traveled=state.distance_traveled,
        octants_visited=len(state.octant_visits),
        octant_transitions=state.octant_transitions,
        target_hits=state.target_hits,
        forced_target_advances=state.forced_target_advances,
        loop_count=state.loop_count,
        near_origin_passes=state.near_origin_passes,
        max_radius=state.max_radius,
        path_curvature=state.path_curvature_sum,
        axis_crossings_total=axis_total,
        final_mode=state.mode,
    )

    print("\n" + "#" * 124)
    print(f"ROUND {round_idx:02d} SUMMARY | Wander behavior: {params.name}")
    print(f"Final position:        {fmt_vec(summary.final_pos)}")
    print(f"Final octant:          {octant_of(summary.final_pos)}")
    print(f"Final mode:            {summary.final_mode}")
    print(f"Final speed:           {summary.final_speed:8.2f} m/s")
    print(f"Final energy:          {summary.final_energy:8.2f}%")
    print(f"Distance traveled:     {summary.distance_traveled:8.2f} m")
    print(f"Octants visited:       {summary.octants_visited:8d} / 8")
    print(f"Octant transitions:    {summary.octant_transitions:8d}")
    print(f"Target hits:           {summary.target_hits:8d}")
    print(f"Timeout advances:      {summary.forced_target_advances:8d}")
    print(f"Loop count proxy:      {summary.loop_count:8d}")
    print(f"Near-origin passes:    {summary.near_origin_passes:8d}")
    print(f"Axis crossings total:  {summary.axis_crossings_total:8d}")
    print(f"Max radius:            {summary.max_radius:8.2f} m")
    print(f"Path curvature:        {summary.path_curvature:8.3f}")
    print(f"Visited octants:       {', '.join(sorted(state.octant_visits.keys()))}")
    print("Interpretation:        " + interpret_round(summary))
    print("#" * 124 + "\n")

    return summary


def final_summary(summaries: List[RoundSummary]) -> None:
    print("=" * 124)
    print("FINAL COMPARISON SUMMARY")
    print("Round | behavior                | energy | distance | octants | transitions | targets | timeouts | loops | axes | max radius | curvature | final octant")
    for s in summaries:
        print(
            f"{s.round_index:5d} | {s.name:<23} | {s.final_energy:6.2f} | {s.distance_traveled:8.2f} | "
            f"{s.octants_visited:7d} | {s.octant_transitions:11d} | {s.target_hits:7d} | "
            f"{s.forced_target_advances:8d} | {s.loop_count:5d} | {s.axis_crossings_total:4d} | "
            f"{s.max_radius:10.2f} | {s.path_curvature:9.3f} | {octant_of(s.final_pos)}"
        )

    best_coverage = max(summaries, key=lambda s: (s.octants_visited, s.octant_transitions, s.axis_crossings_total))
    most_axes = max(summaries, key=lambda s: s.axis_crossings_total)
    most_loops = max(summaries, key=lambda s: s.loop_count)
    most_targets = max(summaries, key=lambda s: s.target_hits)
    most_distance = max(summaries, key=lambda s: s.distance_traveled)
    most_curved = max(summaries, key=lambda s: s.path_curvature)
    best_energy = max(summaries, key=lambda s: s.final_energy)

    print("\nHighlights:")
    print(f"- Best coordinate coverage:   Round {best_coverage.round_index} ({best_coverage.name})")
    print(f"- Most axis crossings:        Round {most_axes.round_index} ({most_axes.name})")
    print(f"- Most looping behavior:      Round {most_loops.round_index} ({most_loops.name})")
    print(f"- Most target hits:           Round {most_targets.round_index} ({most_targets.name})")
    print(f"- Longest coordinate travel:  Round {most_distance.round_index} ({most_distance.name})")
    print(f"- Highest path curvature:     Round {most_curved.round_index} ({most_curved.name})")
    print(f"- Best final energy:          Round {best_energy.round_index} ({best_energy.name})")
    print("=" * 124)


# ============================================================
# Main
# ============================================================

def run_round(round_idx: int, params: RoundParams) -> RoundSummary:
    print_round_start(round_idx, params)
    state = make_initial_state()

    for step in range(1, params.steps + 1):
        metrics, wind, target, event, progress_event = step_simulation(state, params, step)
        if step in PRINT_STEPS or step == params.steps or progress_event != "tracking":
            print_step(round_idx, params, step, state, metrics, wind, target, event)

    return summarize_round(round_idx, params, state)


def main() -> None:
    print_header()
    summaries: List[RoundSummary] = []
    for i, params in enumerate(ROUNDS, start=1):
        summaries.append(run_round(i, params))
    final_summary(summaries)


if __name__ == "__main__":
    main()
