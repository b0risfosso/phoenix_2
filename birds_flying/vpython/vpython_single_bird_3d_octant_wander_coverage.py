"""
VPython Single Bird 3D Octant Wander Simulation - Coverage Tuned

Based on the terminal output from:
terminal_single_bird_3d_octant_wander_coverage.py

The scene is a centered 3D coordinate system, not an xy grid.
A single bird explores all eight octants using different behaviors per round:
1. direct_octant_chain
2. axis_cross_survey
3. large_looping_orbit
4. diagonal_opposites
5. spiral_octant_expansion
6. chaotic_octant_hunt
7. figure_eight_all_axes
8. full_coverage_mixed

Run:
    python vpython_single_bird_3d_octant_wander_coverage.py

Controls:
    Space   pause/play
    N       next round
    R       restart current round
    Q       quit
    + / -   speed up / slow down
    C       cycle camera mode
    F       toggle camera follow
    Z / X   zoom in/out while tracking
    T       toggle trail
    M       toggle target marker
    O       toggle octant labels
    H       toggle help
"""

from vpython import *
from dataclasses import dataclass, field
from math import sin, cos, sqrt, atan2, degrees, pi
from typing import Dict, List, Tuple
import random
import time

# ============================================================
# Vector helpers
# ============================================================

def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def mag_safe(v):
    m = mag(v)
    return m if m > 1e-9 else 1e-9


def norm_safe(v):
    m = mag(v)
    if m < 1e-9:
        return vector(1, 0, 0)
    return v / m


def fmt_vec(v):
    return f"({v.x:+7.1f}, {v.y:+7.1f}, {v.z:+7.1f})"


def octant_name(pos):
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


def octant_short(pos):
    return ("+" if pos.x >= 0 else "-") + ("+" if pos.y >= 0 else "-") + ("+" if pos.z >= 0 else "-")


# ============================================================
# Scene
# ============================================================

scene = canvas(
    title="Single Bird 3D Octant Wander - Coverage Tuned",
    width=1280,
    height=760,
    background=vector(0.88, 0.94, 1.0),
    center=vector(0, 0, 0),
)
scene.range = 170
scene.forward = vector(-1.0, -0.9, -0.55)

RANDOM_SEED = 31
random.seed(RANDOM_SEED)

START_ENERGY = 96.0
DT = 0.10
STEPS_PER_ROUND = 240
PRINT_EVERY = 20
TRAIL_LIMIT = 260
CAMERA_SMOOTHING = 0.070

# Coordinate target points
O1 = vector(+92, +92, +92)
O2 = vector(-92, +92, +92)
O3 = vector(-92, -92, +92)
O4 = vector(+92, -92, +92)
O5 = vector(+92, +92, -92)
O6 = vector(-92, +92, -92)
O7 = vector(-92, -92, -92)
O8 = vector(+92, -92, -92)
ORIGIN = vector(0, 0, 0)
ALL_OCTANTS = [O1, O2, O3, O4, O5, O6, O7, O8]

# ============================================================
# Data
# ============================================================

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
    wind: vector
    target_sequence: list


@dataclass
class BirdState:
    pos: vector
    vel: vector
    acc: vector
    energy: float
    mode: str
    active_target_index: int = 0
    target_age: int = 0
    bank_angle: float = 0.0
    pitch_angle: float = 0.0
    heading_angle: float = 0.0
    distance_traveled: float = 0.0
    path_curvature_sum: float = 0.0
    last_direction: vector = field(default_factory=lambda: vector(1, 0, 0))
    octant_visits: Dict[str, int] = field(default_factory=dict)
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


ROUNDS = [
    RoundParams(
        "direct_octant_chain",
        "directly visit O1 through O8, advancing target only on hit or timeout",
        240, 0.10, 24.0, 0.240, 0.40, 0.04, 0.45, 170.0, 0.80, 22.0, 34,
        vector(0.15, -0.05, 0.05), ALL_OCTANTS,
    ),
    RoundParams(
        "axis_cross_survey",
        "force repeated x/y/z sign changes with intermediate origin passes",
        240, 0.10, 23.0, 0.260, 0.55, 0.05, 0.65, 170.0, 0.85, 24.0, 28,
        vector(0.05, 0.10, -0.05),
        [O1, ORIGIN, O7, ORIGIN, O2, ORIGIN, O8, ORIGIN, O3, ORIGIN, O5, ORIGIN, O4, ORIGIN, O6],
    ),
    RoundParams(
        "large_looping_orbit",
        "large circular orbit that sweeps through all horizontal quadrants while z oscillates sign",
        240, 0.10, 22.0, 0.210, 1.15, 0.05, 1.35, 175.0, 1.00, 28.0, 24,
        vector(0.05, 0.15, 0.00), ALL_OCTANTS,
    ),
    RoundParams(
        "diagonal_opposites",
        "cut across opposite octants with strong diagonal transfers",
        240, 0.10, 25.0, 0.270, 0.60, 0.07, 0.70, 180.0, 0.75, 24.0, 30,
        vector(0.10, -0.15, 0.10),
        [O1, O7, O2, O8, O3, O5, O4, O6, O1],
    ),
    RoundParams(
        "spiral_octant_expansion",
        "spiral expands outward while the target octant changes every arc",
        240, 0.10, 23.0, 0.220, 1.30, 0.06, 1.55, 185.0, 0.95, 28.0, 26,
        vector(-0.10, 0.10, 0.05), ALL_OCTANTS + [O1, O7],
    ),
    RoundParams(
        "chaotic_octant_hunt",
        "unpredictable octant jumps with strong correction toward the current target",
        240, 0.10, 25.0, 0.280, 1.05, 0.20, 1.10, 180.0, 0.70, 24.0, 24,
        vector(0.10, -0.20, 0.00),
        [O6, O4, O3, O5, O2, O8, O7, O1, O4, O6],
    ),
    RoundParams(
        "figure_eight_all_axes",
        "3D figure-eight designed to cross x, y, and z axes many times",
        240, 0.10, 22.5, 0.225, 1.20, 0.04, 1.60, 170.0, 1.05, 28.0, 26,
        vector(0.00, 0.10, -0.05), ALL_OCTANTS,
    ),
    RoundParams(
        "full_coverage_mixed",
        "mixed direct moves, loops, origin dives, and diagonal cuts to cover the full coordinate system",
        240, 0.10, 24.0, 0.255, 1.00, 0.10, 1.20, 185.0, 0.90, 25.0, 25,
        vector(0.05, -0.05, 0.05),
        [O1, O3, ORIGIN, O6, O4, ORIGIN, O7, O5, O2, O8, ORIGIN],
    ),
]

# ============================================================
# Visual construction
# ============================================================

def make_axes():
    # Main coordinate axes
    axis_len = 165
    curve(pos=[vector(-axis_len, 0, 0), vector(axis_len, 0, 0)], color=vector(0.85, 0.15, 0.15), radius=0.18)
    curve(pos=[vector(0, -axis_len, 0), vector(0, axis_len, 0)], color=vector(0.15, 0.55, 0.15), radius=0.18)
    curve(pos=[vector(0, 0, -axis_len), vector(0, 0, axis_len)], color=vector(0.15, 0.25, 0.85), radius=0.18)

    cone(pos=vector(axis_len, 0, 0), axis=vector(8, 0, 0), radius=2.3, color=vector(0.85, 0.15, 0.15))
    cone(pos=vector(0, axis_len, 0), axis=vector(0, 8, 0), radius=2.3, color=vector(0.15, 0.55, 0.15))
    cone(pos=vector(0, 0, axis_len), axis=vector(0, 0, 8), radius=2.3, color=vector(0.15, 0.25, 0.85))

    label(pos=vector(axis_len + 16, 0, 0), text="+X", height=16, box=False, opacity=0, color=vector(0.55, 0.05, 0.05))
    label(pos=vector(-axis_len - 16, 0, 0), text="-X", height=16, box=False, opacity=0, color=vector(0.55, 0.05, 0.05))
    label(pos=vector(0, axis_len + 16, 0), text="+Y", height=16, box=False, opacity=0, color=vector(0.05, 0.35, 0.05))
    label(pos=vector(0, -axis_len - 16, 0), text="-Y", height=16, box=False, opacity=0, color=vector(0.05, 0.35, 0.05))
    label(pos=vector(0, 0, axis_len + 16), text="+Z", height=16, box=False, opacity=0, color=vector(0.05, 0.10, 0.55))
    label(pos=vector(0, 0, -axis_len - 16), text="-Z", height=16, box=False, opacity=0, color=vector(0.05, 0.10, 0.55))

    # Thin reference planes through origin, translucent
    box(pos=vector(0, 0, 0), size=vector(0.20, axis_len * 2, axis_len * 2), color=vector(0.75, 0.15, 0.15), opacity=0.045)
    box(pos=vector(0, 0, 0), size=vector(axis_len * 2, 0.20, axis_len * 2), color=vector(0.15, 0.65, 0.15), opacity=0.045)
    box(pos=vector(0, 0, 0), size=vector(axis_len * 2, axis_len * 2, 0.20), color=vector(0.15, 0.25, 0.85), opacity=0.045)

    # Coordinate cube edges
    L = 120
    corners = [vector(x, y, z) for x in (-L, L) for y in (-L, L) for z in (-L, L)]
    for c in corners:
        for axis_vec in (vector(2 * L, 0, 0), vector(0, 2 * L, 0), vector(0, 0, 2 * L)):
            start = c
            end = c + axis_vec
            if abs(end.x) <= L and abs(end.y) <= L and abs(end.z) <= L:
                curve(pos=[start, end], color=vector(0.62, 0.68, 0.72), radius=0.035)

    # Ring at origin for reference. Use ring, not torus.
    ring(pos=vector(0, 0, 0), axis=vector(0, 0, 1), radius=8, thickness=0.20, color=vector(0.15, 0.15, 0.15), opacity=0.55)
    ring(pos=vector(0, 0, 0), axis=vector(0, 1, 0), radius=8, thickness=0.20, color=vector(0.15, 0.15, 0.15), opacity=0.45)


def make_octant_labels():
    positions = [
        (O1, "O1\n+ + +"), (O2, "O2\n- + +"), (O3, "O3\n- - +"), (O4, "O4\n+ - +"),
        (O5, "O5\n+ + -"), (O6, "O6\n- + -"), (O7, "O7\n- - -"), (O8, "O8\n+ - -"),
    ]
    labels = []
    for pos, txt in positions:
        labels.append(label(pos=pos * 1.28, text=txt, height=13, box=False, opacity=0, color=vector(0.08, 0.09, 0.12)))
    return labels


def make_bird(pos):
    # Bird-shaped primitive; no sphere body.
    body_color = vector(0.10, 0.32, 0.95)
    body = cone(pos=pos - vector(1.2, 0, 0), axis=vector(2.8, 0, 0), radius=0.80, color=body_color)
    tail = cone(pos=pos - vector(1.3, 0, 0), axis=vector(-1.1, 0, 0), radius=0.62, color=vector(0.06, 0.18, 0.62))
    left_wing = box(pos=pos + vector(0.0, -0.85, 0), size=vector(2.7, 0.18, 0.12), color=vector(0.08, 0.25, 0.82))
    right_wing = box(pos=pos + vector(0.0, 0.85, 0), size=vector(2.7, 0.18, 0.12), color=vector(0.08, 0.25, 0.82))
    beak = cone(pos=pos + vector(1.55, 0, 0), axis=vector(0.65, 0, 0), radius=0.16, color=vector(0.95, 0.62, 0.08))
    return compound([body, tail, left_wing, right_wing, beak], pos=pos)


def make_target_marker():
    # A three-ring target marker, no sphere.
    r1 = ring(pos=vector(0,0,0), axis=vector(0,0,1), radius=5.0, thickness=0.25, color=vector(0.95, 0.35, 0.05), opacity=0.85)
    r2 = ring(pos=vector(0,0,0), axis=vector(0,1,0), radius=5.0, thickness=0.25, color=vector(0.95, 0.35, 0.05), opacity=0.65)
    r3 = ring(pos=vector(0,0,0), axis=vector(1,0,0), radius=5.0, thickness=0.25, color=vector(0.95, 0.35, 0.05), opacity=0.65)
    return [r1, r2, r3]


def set_marker_pos(marker, pos):
    for m in marker:
        m.pos = pos


def set_marker_visible(marker, visible):
    for m in marker:
        m.visible = visible


def energy_color(e):
    if e > 80:
        return vector(0.12, 0.58, 0.16)
    if e > 60:
        return vector(0.62, 0.55, 0.08)
    if e > 35:
        return vector(0.90, 0.45, 0.05)
    return vector(0.78, 0.08, 0.06)


def mode_color(mode):
    if "chaotic" in mode:
        return vector(0.85, 0.12, 0.08)
    if "figure" in mode:
        return vector(0.65, 0.15, 0.82)
    if "spiral" in mode or "loop" in mode:
        return vector(0.08, 0.55, 0.85)
    if "diagonal" in mode:
        return vector(0.95, 0.55, 0.08)
    if "origin" in mode:
        return vector(0.15, 0.15, 0.15)
    if "target" in mode:
        return vector(0.10, 0.65, 0.20)
    if "retarget" in mode:
        return vector(0.95, 0.35, 0.05)
    return vector(0.10, 0.32, 0.95)


# ============================================================
# Simulation logic
# ============================================================

def make_initial_state():
    state = BirdState(
        pos=vector(0, 0, 0),
        vel=vector(10.0, 7.5, 5.0),
        acc=vector(0, 0, 0),
        energy=START_ENERGY,
        mode="coordinate_wander",
    )
    state.octant_visits[octant_name(state.pos)] = 1
    return state


def base_target(params, state):
    if not params.target_sequence:
        return ORIGIN
    return params.target_sequence[state.active_target_index % len(params.target_sequence)]


def procedural_target(params, state, step, target):
    t = step * params.dt

    if params.name == "large_looping_orbit":
        r = 112.0
        return vector(r * cos(t * 0.62), r * sin(t * 0.62), 95.0 * sin(t * 0.31))

    if params.name == "spiral_octant_expansion":
        r = min(150.0, 22.0 + step * 0.65)
        return vector(r * cos(t * 0.78), r * sin(t * 0.78), r * 0.78 * sin(t * 0.43))

    if params.name == "figure_eight_all_axes":
        a = 105.0
        return vector(a * sin(t * 0.62), a * sin(t * 0.62) * cos(t * 0.62), 95.0 * sin(t * 1.24))

    return vector(
        target.x + 10.0 * sin(t * params.turn_strength),
        target.y + 10.0 * cos(t * params.turn_strength * 1.2),
        target.z + 10.0 * sin(t * params.turn_strength * 1.4 + 0.5),
    )


def update_target_progress(state, params, step, target):
    state.target_age += 1
    d = mag(state.pos - target)

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


def wind_at_step(params, step):
    t = step * params.dt
    event = "steady"
    wind = vector(
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


def lateral_wander(state, params, step):
    t = step * params.dt
    f = norm_safe(state.vel)
    side_a = norm_safe(vector(-f.y, f.x, 0))
    side_b = norm_safe(vector(-f.z, 0, f.x))
    signal_a = sin(t * params.turn_strength * 1.7) + 0.45 * cos(t * 0.9)
    signal_b = cos(t * params.turn_strength * 1.3 + 0.8) + 0.35 * sin(t * 1.4)
    energy_scale = 0.40 + 0.60 * (state.energy / 100.0)
    return (side_a * signal_a + side_b * signal_b) * params.wander_strength * energy_scale


def boundary_force(state, params):
    r = mag(state.pos)
    if r <= params.radial_limit:
        return vector(0, 0, 0)
    return norm_safe(state.pos) * (-(r - params.radial_limit) * 0.22)


def axis_crossing_bonus(state, target):
    bonus = vector(0, 0, 0)
    if state.pos.x * target.x < 0:
        bonus.x += -0.75 if state.pos.x > 0 else 0.75
    if state.pos.y * target.y < 0:
        bonus.y += -0.75 if state.pos.y > 0 else 0.75
    if state.pos.z * target.z < 0:
        bonus.z += -0.75 if state.pos.z > 0 else 0.75
    if mag(target) < 1.0 and mag(state.pos) > 2.0:
        bonus = bonus + norm_safe(state.pos) * -1.0
    return bonus


def coverage_bonus(state, params):
    if len(state.octant_visits) >= 6:
        return vector(0, 0, 0)
    unvisited = [t for t in ALL_OCTANTS if octant_name(t) not in state.octant_visits]
    if not unvisited:
        return vector(0, 0, 0)
    target = max(unvisited, key=lambda p: mag(state.pos - p))
    return norm_safe(target - state.pos) * 0.55


def control_force(state, params, step, target):
    to_target = target - state.pos
    d = mag(to_target)
    desired_dir = norm_safe(to_target)

    approach = clamp(d / 55.0, 0.45, 1.0)
    energy_scale = 0.40 + 0.60 * (state.energy / 100.0)
    desired_speed = params.max_speed * approach * energy_scale
    desired_velocity = desired_dir * desired_speed

    steering = (desired_velocity - state.vel) * params.control_strength
    wander = lateral_wander(state, params, step)
    axis_bonus = axis_crossing_bonus(state, target)
    coverage = coverage_bonus(state, params)

    chaos = vector(
        sin(step * 0.37) * params.chaos_strength,
        cos(step * 0.29) * params.chaos_strength,
        sin(step * 0.23 + 1.2) * params.chaos_strength,
    )

    return steering + wander + axis_bonus + coverage + chaos + boundary_force(state, params)


def classify_mode(params, state, target, progress_event):
    if progress_event == "target_hit_advance":
        return "target_pass"
    if progress_event == "timeout_advance":
        return "retarget"
    if state.energy < 22.0:
        return "energy_saving_wander"
    if mag(target) < 1.0:
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


def update_angles(state):
    horiz = max((state.vel.x * state.vel.x + state.vel.y * state.vel.y) ** 0.5, 1e-9)
    state.heading_angle = degrees(atan2(state.vel.y, state.vel.x))
    state.pitch_angle = degrees(atan2(state.vel.z, horiz))
    state.bank_angle = clamp(-state.vel.y * 1.6 + state.acc.y * 3.2, -70.0, 70.0)


def update_energy(state, params):
    speed = mag(state.vel)
    turn_cost = abs(state.bank_angle) * 0.001
    accel_cost = mag(state.acc) * 0.0035
    base_cost = 0.020 + speed * 0.0025
    cost = base_cost + turn_cost + accel_cost

    if state.mode in ("large_looping_orbit", "figure_eight_axis_cross", "spiral_axis_sweep", "mixed_coverage_wander"):
        cost *= 0.72
    if state.mode in ("target_pass", "origin_dive", "energy_saving_wander"):
        cost *= 0.55
    if dot(state.pos, state.vel) < 0:
        cost -= 0.020 * params.energy_saving
    if state.path_curvature_sum > 0 and "orbit" in state.mode:
        cost -= 0.010 * params.energy_saving

    state.energy = clamp(state.energy - cost, 0, 100)


def step_simulation(state, params, step):
    base = base_target(params, state)
    target = procedural_target(params, state, step, base)
    wind, wind_event = wind_at_step(params, step)

    progress_event = update_target_progress(state, params, step, target)

    base = base_target(params, state)
    target = procedural_target(params, state, step, base)

    ctrl = control_force(state, params, step, target)
    rel = state.vel - wind
    drag = rel * (-0.018 * mag(rel))
    thrust = norm_safe(state.vel) * (0.55 + 0.45 * state.energy / 100.0)

    total = ctrl + drag + thrust
    state.acc = total
    state.vel = state.vel + state.acc * params.dt

    max_speed = params.max_speed * (0.58 + 0.42 * state.energy / 100.0)
    if mag(state.vel) > max_speed:
        state.vel = norm_safe(state.vel) * max_speed

    old_pos = vector(state.pos.x, state.pos.y, state.pos.z)
    old_oct = octant_name(old_pos)
    old_dir = state.last_direction

    state.pos = state.pos + state.vel * params.dt
    state.distance_traveled += mag(state.pos - old_pos)

    new_oct = octant_name(state.pos)
    state.octant_visits[new_oct] = state.octant_visits.get(new_oct, 0) + 1
    if new_oct != old_oct:
        state.octant_transitions += 1

    if old_pos.x * state.pos.x < 0:
        state.axis_crossings_x += 1
    if old_pos.y * state.pos.y < 0:
        state.axis_crossings_y += 1
    if old_pos.z * state.pos.z < 0:
        state.axis_crossings_z += 1

    new_dir = norm_safe(state.vel)
    turn_amount = 1.0 - clamp(dot(old_dir, new_dir), -1, 1)
    state.path_curvature_sum += turn_amount
    state.last_direction = new_dir

    if mag(state.pos) < 10.0 and mag(old_pos) >= 10.0:
        state.near_origin_passes += 1

    if state.path_curvature_sum > (state.loop_count + 1) * 0.85:
        state.loop_count += 1

    state.max_radius = max(state.max_radius, mag(state.pos))
    state.min_radius = min(state.min_radius, mag(state.pos))

    state.mode = classify_mode(params, state, target, progress_event)
    update_angles(state)
    update_energy(state, params)

    metrics = {
        "control": mag(ctrl),
        "drag": mag(drag),
        "thrust": mag(thrust),
        "total": mag(total),
        "radius": mag(state.pos),
        "distance_to_target": mag(state.pos - target),
        "curvature": state.path_curvature_sum,
        "turn_step": turn_amount,
        "speed": mag(state.vel),
        "axis_cross_total": state.axis_crossings_x + state.axis_crossings_y + state.axis_crossings_z,
    }
    event = progress_event if progress_event != "tracking" else wind_event
    return metrics, wind, target, event, progress_event


# ============================================================
# UI / controls
# ============================================================

controls = {
    "paused": False,
    "quit": False,
    "next_round": False,
    "restart_round": False,
    "speed": 1.0,
    "camera_mode": 0,
    "camera_follow": True,
    "trail": True,
    "target": True,
    "octants": True,
    "help": True,
}

camera_modes = ["follow", "wide", "top", "x_axis", "y_axis", "fixed"]

hud = label(
    pos=vector(-140, -150, 150),
    text="",
    height=12,
    box=True,
    border=8,
    color=vector(0.04, 0.05, 0.07),
    background=vector(0.96, 0.98, 1.0),
    opacity=0.72,
)

help_label = label(
    pos=vector(0, -155, -145),
    text="",
    height=11,
    box=False,
    opacity=0,
    color=vector(0.08, 0.09, 0.12),
)

round_label = label(
    pos=vector(0, 0, 178),
    text="",
    height=16,
    box=False,
    opacity=0,
    color=vector(0.05, 0.05, 0.08),
)


def on_key(evt):
    k = evt.key
    if k == " ":
        controls["paused"] = not controls["paused"]
    elif k in ("n", "N"):
        controls["next_round"] = True
    elif k in ("r", "R"):
        controls["restart_round"] = True
    elif k in ("q", "Q"):
        controls["quit"] = True
    elif k in ("+", "="):
        controls["speed"] = min(5.0, controls["speed"] * 1.25)
    elif k in ("-", "_"):
        controls["speed"] = max(0.20, controls["speed"] / 1.25)
    elif k in ("c", "C"):
        controls["camera_mode"] = (controls["camera_mode"] + 1) % len(camera_modes)
        controls["camera_follow"] = True
    elif k in ("f", "F"):
        controls["camera_follow"] = not controls["camera_follow"]
    elif k in ("z", "Z"):
        scene.range = max(18, scene.range * 0.82)
    elif k in ("x", "X"):
        scene.range = min(420, scene.range * 1.22)
    elif k in ("t", "T"):
        controls["trail"] = not controls["trail"]
    elif k in ("m", "M"):
        controls["target"] = not controls["target"]
    elif k in ("o", "O"):
        controls["octants"] = not controls["octants"]
    elif k in ("h", "H"):
        controls["help"] = not controls["help"]

scene.bind("keydown", on_key)


def update_help():
    if not controls["help"]:
        help_label.text = ""
        return
    help_label.text = "Space pause | N next | R restart | +/- speed | C camera | F follow | Z/X zoom | T trail | M target | O octants | H help | Q quit"


def update_camera(state):
    if not controls["camera_follow"]:
        return

    mode = camera_modes[controls["camera_mode"]]
    if mode == "follow":
        target = state.pos * 0.65
        scene.forward = vector(-1.0, -0.75, -0.45)
    elif mode == "wide":
        target = vector(0, 0, 0)
        scene.forward = vector(-1.0, -1.0, -0.65)
    elif mode == "top":
        target = vector(0, 0, 0)
        scene.forward = vector(0, 0, -1)
    elif mode == "x_axis":
        target = vector(0, 0, 0)
        scene.forward = vector(-1, 0, 0)
    elif mode == "y_axis":
        target = vector(0, 0, 0)
        scene.forward = vector(0, -1, 0)
    else:
        return
    scene.center = scene.center * (1.0 - CAMERA_SMOOTHING) + target * CAMERA_SMOOTHING


def hud_text(round_idx, params, step, state, metrics, event, target):
    visited = ", ".join([name.split("(")[0] for name in sorted(state.octant_visits.keys())])
    return (
        f"ROUND {round_idx + 1}/8 | {params.name} | step {step:03d}/{params.steps} | event={event}\n"
        f"{params.description}\n"
        f"Mode: {state.mode} | Octant: {octant_name(state.pos)} short={octant_short(state.pos)}\n"
        f"Position {fmt_vec(state.pos)} | Velocity {fmt_vec(state.vel)} | Speed {mag(state.vel):5.2f}\n"
        f"Target {fmt_vec(target)} | target_octant={octant_name(target)} | distance={metrics.get('distance_to_target', 0):6.1f}\n"
        f"Target index {state.active_target_index + 1}/{len(params.target_sequence)} | age={state.target_age} | hits={state.target_hits} | timeouts={state.forced_target_advances}\n"
        f"Energy {state.energy:5.1f}% | radius={metrics.get('radius', mag(state.pos)):6.1f} | distance traveled={state.distance_traveled:6.1f}\n"
        f"Heading={state.heading_angle:6.1f}° pitch={state.pitch_angle:6.1f}° bank={state.bank_angle:6.1f}°\n"
        f"Octants visited {len(state.octant_visits)}/8 | transitions={state.octant_transitions} | axes x/y/z={state.axis_crossings_x}/{state.axis_crossings_y}/{state.axis_crossings_z}\n"
        f"Loops={state.loop_count} | origin passes={state.near_origin_passes} | curvature={state.path_curvature_sum:.3f}\n"
        f"Visited: {visited}\n"
        f"Camera={camera_modes[controls['camera_mode']]} follow={controls['camera_follow']} speed={controls['speed']:.2f}"
    )


def print_terminal(round_idx, params, step, state, metrics, event, target):
    print("-" * 124)
    print(f"ROUND {round_idx + 1:02d} | {params.name} | STEP {step:03d} | event={event}")
    print(f"Mode:           {state.mode}")
    print(f"Octant:         {octant_name(state.pos)} short={octant_short(state.pos)}")
    print(f"Position:       {fmt_vec(state.pos)}")
    print(f"Velocity:       {fmt_vec(state.vel)} speed={mag(state.vel):.2f}")
    print(f"Target:         {fmt_vec(target)} target_octant={octant_name(target)} dist={metrics['distance_to_target']:.2f}")
    print(f"Energy:         {state.energy:.2f}%")
    print(f"Target hits:    {state.target_hits} timeout advances={state.forced_target_advances}")
    print(f"Octants:        visited={len(state.octant_visits)}/8 transitions={state.octant_transitions}")
    print(f"Axis crossings: x={state.axis_crossings_x}, y={state.axis_crossings_y}, z={state.axis_crossings_z}, total={metrics['axis_cross_total']}")
    print(f"Loops:          {state.loop_count} origin_passes={state.near_origin_passes} curvature={state.path_curvature_sum:.3f}")
    print(f"Forces:         control={metrics['control']:.3f}, drag={metrics['drag']:.3f}, thrust={metrics['thrust']:.3f}, total={metrics['total']:.3f}")


def round_summary(round_idx, params, state):
    axis_total = state.axis_crossings_x + state.axis_crossings_y + state.axis_crossings_z
    print("\n" + "#" * 124)
    print(f"ROUND {round_idx + 1:02d} SUMMARY | {params.name}")
    print(f"Final position:       {fmt_vec(state.pos)}")
    print(f"Final octant:         {octant_name(state.pos)}")
    print(f"Final mode:           {state.mode}")
    print(f"Final speed:          {mag(state.vel):.2f}")
    print(f"Final energy:         {state.energy:.2f}%")
    print(f"Distance traveled:    {state.distance_traveled:.2f}")
    print(f"Octants visited:      {len(state.octant_visits)}/8")
    print(f"Octant transitions:   {state.octant_transitions}")
    print(f"Target hits:          {state.target_hits}")
    print(f"Timeout advances:     {state.forced_target_advances}")
    print(f"Loop count proxy:     {state.loop_count}")
    print(f"Near-origin passes:   {state.near_origin_passes}")
    print(f"Axis crossings total: {axis_total}")
    print(f"Max radius:           {state.max_radius:.2f}")
    print(f"Path curvature:       {state.path_curvature_sum:.3f}")
    print(f"Visited octants:      {', '.join(sorted(state.octant_visits.keys()))}")
    print("#" * 124 + "\n")
    return {
        "round": round_idx + 1,
        "behavior": params.name,
        "energy": state.energy,
        "distance": state.distance_traveled,
        "octants": len(state.octant_visits),
        "transitions": state.octant_transitions,
        "targets": state.target_hits,
        "timeouts": state.forced_target_advances,
        "loops": state.loop_count,
        "axes": axis_total,
        "max_radius": state.max_radius,
        "curvature": state.path_curvature_sum,
        "final_octant": octant_name(state.pos),
    }


def final_summary(summaries):
    print("=" * 124)
    print("FINAL COMPARISON SUMMARY")
    print("Round | behavior                | energy | distance | octants | transitions | targets | timeouts | loops | axes | max radius | curvature | final octant")
    for s in summaries:
        print(
            f"{s['round']:5d} | {s['behavior']:<23} | {s['energy']:6.2f} | {s['distance']:8.2f} | "
            f"{s['octants']:7d} | {s['transitions']:11d} | {s['targets']:7d} | "
            f"{s['timeouts']:8d} | {s['loops']:5d} | {s['axes']:4d} | "
            f"{s['max_radius']:10.2f} | {s['curvature']:9.3f} | {s['final_octant']}"
        )
    if summaries:
        best = max(summaries, key=lambda s: (s["octants"], s["transitions"], s["axes"]))
        curved = max(summaries, key=lambda s: s["curvature"])
        print("\nHighlights:")
        print(f"- Best coordinate coverage: Round {best['round']} ({best['behavior']})")
        print(f"- Highest curvature:        Round {curved['round']} ({curved['behavior']})")
    print("=" * 124)


# ============================================================
# Run loop
# ============================================================

make_axes()
octant_labels = make_octant_labels()
bird_visual = make_bird(vector(0, 0, 0))
trail = curve(color=vector(0.10, 0.30, 0.95), radius=0.30)
target_marker = make_target_marker()
energy_bar = box(pos=vector(-150, 150, -150), size=vector(30, 3, 3), color=energy_color(START_ENERGY))
bird_label = label(pos=vector(0, 0, 8), text="", height=11, box=False, opacity=0, color=vector(0.03, 0.04, 0.06))


def update_visuals(state, target, params):
    bird_visual.pos = state.pos
    bird_visual.axis = norm_safe(state.vel) * 4.0
    bird_visual.color = mode_color(state.mode)

    bird_label.pos = state.pos + vector(0, 0, 9)
    bird_label.text = f"bird\n{state.mode}\n{octant_name(state.pos).split('(')[0]} E {state.energy:.1f}%"

    energy_bar.pos = state.pos + vector(0, 0, -9)
    energy_bar.size = vector(2 + 22 * state.energy / 100.0, 2.0, 2.0)
    energy_bar.color = energy_color(state.energy)

    set_marker_pos(target_marker, target)
    set_marker_visible(target_marker, controls["target"])

    for lbl in octant_labels:
        lbl.visible = controls["octants"]

    if controls["trail"]:
        trail.append(pos=state.pos)
        if hasattr(trail, "npoints") and trail.npoints > TRAIL_LIMIT:
            try:
                trail.pop(0)
            except Exception:
                trail.clear()
    else:
        trail.clear()


def reset_visuals():
    trail.clear()


def run_round(round_idx):
    params = ROUNDS[round_idx]
    state = make_initial_state()
    reset_visuals()
    round_label.text = f"Round {round_idx + 1}: {params.name}\n{params.description}"

    print("=" * 124)
    print(f"ROUND {round_idx + 1:02d} START | WANDER BEHAVIOR: {params.name}")
    print(params.description)
    print(f"steps={params.steps}, max_speed={params.max_speed:.2f}, control={params.control_strength:.3f}, wander={params.wander_strength:.3f}")

    step = 1
    last_metrics = {"distance_to_target": 0, "radius": 0}
    last_target = base_target(params, state)
    last_event = "steady"

    while step <= params.steps:
        rate(70 * controls["speed"])
        update_help()

        if controls["quit"]:
            return None, "quit"

        if controls["next_round"]:
            controls["next_round"] = False
            break

        if controls["restart_round"]:
            controls["restart_round"] = False
            return None, "restart"

        if controls["paused"]:
            hud.text = hud_text(round_idx, params, step, state, last_metrics, "paused", last_target)
            update_camera(state)
            continue

        metrics, wind, target, event, progress_event = step_simulation(state, params, step)
        last_metrics = metrics
        last_target = target
        last_event = event

        update_visuals(state, target, params)
        hud.text = hud_text(round_idx, params, step, state, metrics, event, target)

        if step == 1 or step % PRINT_EVERY == 0 or progress_event != "tracking" or step == params.steps:
            print_terminal(round_idx, params, step, state, metrics, event, target)

        update_camera(state)
        step += 1

    summary = round_summary(round_idx, params, state)
    time.sleep(0.15)
    return summary, "done"


def main():
    print("VPYTHON SINGLE-BIRD 3D OCTANT WANDER SIMULATION - COVERAGE TUNED")
    print("Centered 3D coordinate system; one bird-shaped agent; no sphere bird.")
    print("Controls: Space pause/play | N next round | R restart | +/- speed | C camera | F follow | Z/X zoom | T trail | M target | O octants | H help | Q quit")

    summaries = []
    idx = 0
    while idx < len(ROUNDS):
        summary, status = run_round(idx)
        if status == "quit":
            break
        if status == "restart":
            continue
        if summary:
            summaries.append(summary)
        idx += 1

    final_summary(summaries)
    hud.text = "Simulation complete.\nClose the VPython browser tab/window to exit."
    round_label.text = "Simulation complete"


if __name__ == "__main__":
    main()
