#!/usr/bin/env python3
"""
VPython Sensory-to-Motor Reflex Network v5 Visualization
--------------------------------------------------------

Visual simulation based on the terminal output from:
terminal_sensory_motor_reflex_network_v5.py

Run:
    python vpython_sensory_motor_reflex_network_v5.py

Requires:
    pip install vpython

Controls:
    Space  - pause/resume
    N      - next round
    B      - previous round
    R      - restart current round
    A      - toggle auto-cycle
    S      - toggle spike particles
    L      - toggle synapse links
    C      - toggle context gate display
    T      - toggle sensory world panel

This visualization maps the terminal-output behavior into VPython:

    - 5 sensory neurons:
        S_LIGHT_L, S_LIGHT_R, S_HEAT, S_TOUCH, S_CHEM

    - 5 interneurons:
        I_APPROACH, I_AVOID, I_BALANCE, I_STEER_L, I_STEER_R

    - 4 motor neurons:
        M_LEFT, M_RIGHT, M_FORWARD, M_BACKWARD

    - 8 context-gated reflex rounds:
        light_steering
        food_forward
        danger_retreat
        front_collision_retreat
        right_escape

    - all rounds produce real motor output
    - all rounds use interneuron relay activity
    - final accuracy: 8/8 = 100%
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sin, cos, pi
from random import Random

from vpython import (
    arrow,
    box,
    canvas,
    color,
    curve,
    label,
    local_light,
    rate,
    ring,
    sphere,
    vector,
    wtext,
)


# -----------------------------------------------------------------------------
# Data reconstructed from the terminal output
# -----------------------------------------------------------------------------

@dataclass
class RoundData:
    number: int
    scenario: str
    context: str
    expected: str
    chosen: str
    correct: bool
    total_spikes: int
    sensory_spikes: int
    inter_spikes: int
    motor_spikes: int
    reaction_ms: float
    relay: bool
    score: float
    world: dict
    spike_counts: dict
    strongest: list[str]
    note: str


ROUNDS = [
    RoundData(
        1,
        "bright light on left",
        "light_steering",
        "turn_right",
        "turn_right",
        True,
        245,
        11,
        184,
        50,
        577.00,
        True,
        96.04,
        {"L": 0.95, "R": 0.20, "heat": 0.05, "touch": 0.00, "chem": 0.20, "obstacle": "none"},
        {
            "S_LIGHT_L": 11, "S_LIGHT_R": 0, "S_HEAT": 0, "S_TOUCH": 0, "S_CHEM": 0,
            "I_APPROACH": 28, "I_AVOID": 29, "I_BALANCE": 10, "I_STEER_L": 14, "I_STEER_R": 103,
            "M_LEFT": 4, "M_RIGHT": 46, "M_FORWARD": 0, "M_BACKWARD": 0,
        },
        ["S_LIGHT_L->I_STEER_R", "I_STEER_R->M_RIGHT"],
        "left light is gated into right steering",
    ),
    RoundData(
        2,
        "bright light on right",
        "light_steering",
        "turn_left",
        "turn_left",
        True,
        253,
        11,
        186,
        56,
        500.00,
        True,
        98.21,
        {"L": 0.20, "R": 0.95, "heat": 0.05, "touch": 0.00, "chem": 0.20, "obstacle": "none"},
        {
            "S_LIGHT_L": 0, "S_LIGHT_R": 11, "S_HEAT": 0, "S_TOUCH": 0, "S_CHEM": 0,
            "I_APPROACH": 28, "I_AVOID": 29, "I_BALANCE": 12, "I_STEER_L": 103, "I_STEER_R": 14,
            "M_LEFT": 53, "M_RIGHT": 3, "M_FORWARD": 0, "M_BACKWARD": 0,
        },
        ["S_LIGHT_R->I_STEER_L", "I_STEER_L->M_LEFT"],
        "right light is gated into left steering",
    ),
    RoundData(
        3,
        "chemical food ahead",
        "food_forward",
        "move_forward",
        "move_forward",
        True,
        149,
        13,
        109,
        27,
        697.00,
        True,
        95.06,
        {"L": 0.35, "R": 0.35, "heat": 0.00, "touch": 0.00, "chem": 1.00, "obstacle": "none"},
        {
            "S_LIGHT_L": 0, "S_LIGHT_R": 0, "S_HEAT": 0, "S_TOUCH": 0, "S_CHEM": 13,
            "I_APPROACH": 94, "I_AVOID": 1, "I_BALANCE": 14, "I_STEER_L": 0, "I_STEER_R": 0,
            "M_LEFT": 0, "M_RIGHT": 0, "M_FORWARD": 27, "M_BACKWARD": 0,
        },
        ["S_CHEM->I_APPROACH", "I_APPROACH->M_FORWARD"],
        "food context suppresses steering and boosts forward motion",
    ),
    RoundData(
        4,
        "danger heat",
        "danger_retreat",
        "move_backward",
        "move_backward",
        True,
        213,
        18,
        161,
        34,
        685.00,
        True,
        95.32,
        {"L": 0.25, "R": 0.25, "heat": 1.00, "touch": 0.00, "chem": 0.30, "obstacle": "none"},
        {
            "S_LIGHT_L": 0, "S_LIGHT_R": 0, "S_HEAT": 18, "S_TOUCH": 0, "S_CHEM": 0,
            "I_APPROACH": 0, "I_AVOID": 105, "I_BALANCE": 28, "I_STEER_L": 14, "I_STEER_R": 14,
            "M_LEFT": 0, "M_RIGHT": 0, "M_FORWARD": 0, "M_BACKWARD": 34,
        },
        ["S_HEAT->I_AVOID", "I_AVOID->M_BACKWARD"],
        "danger context suppresses forward and selects retreat",
    ),
    RoundData(
        5,
        "touch collision",
        "front_collision_retreat",
        "move_backward",
        "move_backward",
        True,
        195,
        7,
        151,
        37,
        688.00,
        True,
        95.26,
        {"L": 0.30, "R": 0.30, "heat": 0.20, "touch": 1.00, "chem": 0.10, "obstacle": "front"},
        {
            "S_LIGHT_L": 0, "S_LIGHT_R": 0, "S_HEAT": 0, "S_TOUCH": 7, "S_CHEM": 0,
            "I_APPROACH": 0, "I_AVOID": 104, "I_BALANCE": 47, "I_STEER_L": 0, "I_STEER_R": 0,
            "M_LEFT": 0, "M_RIGHT": 0, "M_FORWARD": 0, "M_BACKWARD": 37,
        },
        ["S_TOUCH->I_AVOID", "I_AVOID->M_BACKWARD"],
        "front collision context turns touch into backward retreat",
    ),
    RoundData(
        6,
        "mixed food and heat",
        "danger_retreat",
        "move_backward",
        "move_backward",
        True,
        246,
        24,
        184,
        38,
        683.00,
        True,
        95.36,
        {"L": 0.30, "R": 0.45, "heat": 0.82, "touch": 0.00, "chem": 0.95, "obstacle": "none"},
        {
            "S_LIGHT_L": 0, "S_LIGHT_R": 0, "S_HEAT": 13, "S_TOUCH": 0, "S_CHEM": 11,
            "I_APPROACH": 28, "I_AVOID": 101, "I_BALANCE": 27, "I_STEER_L": 14, "I_STEER_R": 14,
            "M_LEFT": 0, "M_RIGHT": 0, "M_FORWARD": 0, "M_BACKWARD": 38,
        },
        ["S_HEAT->I_AVOID", "I_AVOID->M_BACKWARD", "S_CHEM->I_APPROACH"],
        "danger overrides food when heat and chemical signals coexist",
    ),
    RoundData(
        7,
        "narrow obstacle with right escape",
        "right_escape",
        "turn_right",
        "turn_right",
        True,
        275,
        4,
        208,
        63,
        694.00,
        True,
        91.32,
        {"L": 0.65, "R": 0.35, "heat": 0.18, "touch": 0.75, "chem": 0.50, "obstacle": "left"},
        {
            "S_LIGHT_L": 0, "S_LIGHT_R": 0, "S_HEAT": 0, "S_TOUCH": 4, "S_CHEM": 0,
            "I_APPROACH": 20, "I_AVOID": 57, "I_BALANCE": 38, "I_STEER_L": 0, "I_STEER_R": 93,
            "M_LEFT": 0, "M_RIGHT": 51, "M_FORWARD": 0, "M_BACKWARD": 12,
        },
        ["S_LIGHT_L->I_STEER_R", "I_STEER_R->M_RIGHT", "S_TOUCH->I_AVOID"],
        "left-side obstacle is gated into right escape, not full retreat",
    ),
    RoundData(
        8,
        "strong food after learning",
        "food_forward",
        "move_forward",
        "move_forward",
        True,
        157,
        13,
        110,
        34,
        684.00,
        True,
        95.34,
        {"L": 0.45, "R": 0.60, "heat": 0.05, "touch": 0.00, "chem": 1.00, "obstacle": "none"},
        {
            "S_LIGHT_L": 0, "S_LIGHT_R": 0, "S_HEAT": 0, "S_TOUCH": 0, "S_CHEM": 13,
            "I_APPROACH": 93, "I_AVOID": 0, "I_BALANCE": 17, "I_STEER_L": 0, "I_STEER_R": 0,
            "M_LEFT": 0, "M_RIGHT": 0, "M_FORWARD": 34, "M_BACKWARD": 0,
        },
        ["S_CHEM->I_APPROACH", "I_APPROACH->M_FORWARD"],
        "learned food-forward pathway stays clean after gating",
    ),
]


SENSORY = ["S_LIGHT_L", "S_LIGHT_R", "S_HEAT", "S_TOUCH", "S_CHEM"]
INTER = ["I_APPROACH", "I_AVOID", "I_BALANCE", "I_STEER_L", "I_STEER_R"]
MOTOR = ["M_LEFT", "M_RIGHT", "M_FORWARD", "M_BACKWARD"]
ALL_NEURONS = SENSORY + INTER + MOTOR

NEURON_POS = {
    # sensory layer
    "S_LIGHT_L": vector(-5.3, 2.25, 0.0),
    "S_LIGHT_R": vector(-5.3, 1.35, 0.0),
    "S_HEAT": vector(-5.3, 0.45, 0.0),
    "S_TOUCH": vector(-5.3, -0.45, 0.0),
    "S_CHEM": vector(-5.3, -1.35, 0.0),
    # relay layer
    "I_APPROACH": vector(-1.35, 1.75, 0.0),
    "I_AVOID": vector(-1.35, 0.85, 0.0),
    "I_BALANCE": vector(-1.35, -0.05, 0.0),
    "I_STEER_L": vector(-1.35, -0.95, 0.0),
    "I_STEER_R": vector(-1.35, -1.85, 0.0),
    # motor layer
    "M_LEFT": vector(3.25, 1.35, 0.0),
    "M_RIGHT": vector(3.25, 0.45, 0.0),
    "M_FORWARD": vector(3.25, -0.45, 0.0),
    "M_BACKWARD": vector(3.25, -1.35, 0.0),
}

LINKS = [
    ("S_LIGHT_L", "I_STEER_R"),
    ("I_STEER_R", "M_RIGHT"),
    ("S_LIGHT_R", "I_STEER_L"),
    ("I_STEER_L", "M_LEFT"),
    ("S_CHEM", "I_APPROACH"),
    ("I_APPROACH", "M_FORWARD"),
    ("S_HEAT", "I_AVOID"),
    ("S_TOUCH", "I_AVOID"),
    ("I_AVOID", "M_BACKWARD"),
    ("S_TOUCH", "I_BALANCE"),
    ("I_BALANCE", "M_FORWARD"),
    ("I_BALANCE", "I_APPROACH"),
    ("I_AVOID", "I_APPROACH"),
    ("I_AVOID", "M_FORWARD"),
    ("I_APPROACH", "M_BACKWARD"),
    ("S_LIGHT_L", "M_RIGHT"),
    ("S_LIGHT_R", "M_LEFT"),
    ("S_TOUCH", "I_STEER_R"),
]

CONTEXT_COLORS = {
    "light_steering": vector(0.25, 0.55, 1.0),
    "food_forward": vector(0.20, 0.75, 0.25),
    "danger_retreat": vector(1.0, 0.28, 0.10),
    "front_collision_retreat": vector(1.0, 0.50, 0.10),
    "right_escape": vector(0.65, 0.35, 1.0),
    "balanced": vector(0.7, 0.7, 0.7),
}


# -----------------------------------------------------------------------------
# Scene
# -----------------------------------------------------------------------------

scene = canvas(
    title="Sensory-to-Motor Reflex Network v5",
    width=1320,
    height=780,
    background=vector(0.86, 0.90, 0.96),
    center=vector(-0.8, 0.35, 0),
)
scene.camera.pos = vector(-0.8, 4.8, 10.8)
scene.camera.axis = vector(0, -3.9, -10.8)

status = wtext(text="")
wtext(text="\nControls: Space pause/resume | N next | B previous | R restart | A auto-cycle | S spikes | L links | C context | T world\n\n")

rng = Random(9155)


# -----------------------------------------------------------------------------
# Utility
# -----------------------------------------------------------------------------

def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def mix(a, b, u):
    u = clamp(u, 0.0, 1.0)
    return a * (1.0 - u) + b * u


def neuron_base_color(name: str):
    if name.startswith("S_"):
        return vector(0.20, 0.52, 1.0)
    if name == "I_AVOID" or name == "I_BALANCE":
        return vector(0.95, 0.25, 0.75)
    if name.startswith("I_"):
        return vector(0.55, 0.35, 1.0)
    if name == "M_FORWARD":
        return vector(0.20, 0.75, 0.25)
    if name == "M_BACKWARD":
        return vector(1.0, 0.30, 0.12)
    return vector(1.0, 0.78, 0.15)


def neuron_active_color(name: str):
    if name.startswith("S_"):
        return vector(1.0, 0.85, 0.15)
    if name.startswith("I_"):
        return vector(0.95, 0.45, 1.0)
    return vector(1.0, 0.90, 0.18)


def round_spike_times(rd: RoundData, name: str):
    count = rd.spike_counts.get(name, 0)
    if count <= 0:
        return []
    # Spread spikes through a 700 ms window but bias selected motors later
    # to match terminal reaction times around 500-700 ms.
    if name.startswith("M_"):
        start = max(50.0, rd.reaction_ms - 120.0)
        end = 700.0
    elif name.startswith("I_"):
        start = 40.0 + (ALL_NEURONS.index(name) % 4) * 3.0
        end = 700.0
    else:
        start = 65.0
        end = 690.0

    if count == 1:
        return [(start + end) / 2.0]
    return [start + i * (end - start) / max(1, count - 1) for i in range(count)]


def spikes_so_far(rd: RoundData, name: str, t_ms: float):
    return sum(1 for st in round_spike_times(rd, name) if st <= t_ms)


def just_spiked(rd: RoundData, name: str, t_ms: float):
    return any(0.0 <= t_ms - st <= 5.5 for st in round_spike_times(rd, name))


def link_weight(rd: RoundData, src: str, dst: str):
    base = 0.10
    token = f"{src}->{dst}"
    if token in rd.strongest:
        base = 1.0
    elif src in rd.spike_counts and dst in rd.spike_counts:
        src_max = max(1, rd.spike_counts[src])
        dst_max = max(1, rd.spike_counts[dst])
        base = 0.08 + 0.60 * clamp((src_max + dst_max) / 160.0, 0.0, 1.0)
    if src.startswith("I_AVOID") and rd.context.startswith("danger"):
        base *= 1.2
    return clamp(base, 0.0, 1.0)


def selected_motor_name(reflex: str):
    return {
        "turn_left": "M_LEFT",
        "turn_right": "M_RIGHT",
        "move_forward": "M_FORWARD",
        "move_backward": "M_BACKWARD",
    }.get(reflex, "")


# -----------------------------------------------------------------------------
# Static objects
# -----------------------------------------------------------------------------

floor = box(pos=vector(-0.8, -2.25, 0), size=vector(12.8, 0.08, 6.0), color=vector(0.72, 0.76, 0.78))

# Layer labels
label(pos=vector(-5.3, 2.85, 0), text="SENSORY", height=13, box=False, opacity=0, color=color.black)
label(pos=vector(-1.35, 2.85, 0), text="INTERNEURON RELAY", height=13, box=False, opacity=0, color=color.black)
label(pos=vector(3.25, 2.85, 0), text="MOTOR OUTPUT", height=13, box=False, opacity=0, color=color.black)

neurons = {}
rings = {}
neuron_labels = {}
for name in ALL_NEURONS:
    pos = NEURON_POS[name]
    s = sphere(pos=pos, radius=0.30, color=neuron_base_color(name), opacity=0.92)
    r = ring(pos=pos, axis=vector(0, 1, 0), radius=0.44, thickness=0.018, color=s.color, opacity=0.28)
    lab = label(pos=pos + vector(0, -0.48, 0), text=name, height=9, box=False, opacity=0, color=color.black)
    neurons[name] = s
    rings[name] = r
    neuron_labels[name] = lab

link_curves = {}
for src, dst in LINKS:
    start = NEURON_POS[src]
    end = NEURON_POS[dst]
    mid = (start + end) * 0.5 + vector(0, 0.18, 0.12)
    c = curve(pos=[start, mid, end], radius=0.010, color=vector(0.35, 0.35, 0.35), opacity=0.16)
    link_curves[(src, dst)] = c

# Sensory world bars
world_bars = {}
world_x0 = -6.35
for i, key in enumerate(["L", "R", "heat", "touch", "chem"]):
    b = box(pos=vector(world_x0 + i * 0.5, -2.02, -2.35), size=vector(0.30, 0.10, 0.25), color=vector(0.45, 0.55, 0.70))
    world_bars[key] = b
    label(pos=vector(world_x0 + i * 0.5, -2.24, -2.35), text=key, height=8, box=False, opacity=0, color=color.black)

# Context gate panel
context_box = box(pos=vector(5.1, 1.9, -2.25), size=vector(1.45, 0.85, 0.16), color=vector(0.7, 0.7, 0.7), opacity=0.42)
context_arrow = arrow(pos=vector(5.1, 1.25, -2.25), axis=vector(0, 0.65, 0), shaftwidth=0.055, color=vector(1, 1, 1))

# Motor result arrows
motor_arrows = {
    "M_LEFT": arrow(pos=NEURON_POS["M_LEFT"] + vector(0.50, 0, 0), axis=vector(-0.75, 0, 0), shaftwidth=0.06, color=vector(1.0, 0.85, 0.10), visible=True),
    "M_RIGHT": arrow(pos=NEURON_POS["M_RIGHT"] + vector(0.25, 0, 0), axis=vector(0.75, 0, 0), shaftwidth=0.06, color=vector(1.0, 0.85, 0.10), visible=True),
    "M_FORWARD": arrow(pos=NEURON_POS["M_FORWARD"] + vector(0.0, 0.15, 0), axis=vector(0, 0.75, 0), shaftwidth=0.06, color=vector(0.20, 0.85, 0.25), visible=True),
    "M_BACKWARD": arrow(pos=NEURON_POS["M_BACKWARD"] + vector(0.0, -0.15, 0), axis=vector(0, -0.75, 0), shaftwidth=0.06, color=vector(1.0, 0.30, 0.12), visible=True),
}

# Round bars
round_bars = []
for i, rd in enumerate(ROUNDS):
    x = -5.75 + i * 1.25
    b = box(pos=vector(x, -2.02, 2.55), size=vector(0.34, 0.15, 0.30), color=vector(0.30, 0.60, 0.90))
    round_bars.append(b)
    label(pos=vector(x, -2.27, 2.55), text=str(rd.number), height=9, box=False, opacity=0, color=color.black)

# Particles
spike_particles = []
for _ in range(210):
    p = sphere(pos=vector(0, -9, 0), radius=0.045, color=color.yellow, opacity=0, emissive=True)
    p.life = 0.0
    p.v = vector(0, 0, 0)
    spike_particles.append(p)

pulse_particles = []
for _ in range(120):
    p = sphere(pos=vector(0, -9, 0), radius=0.040, color=color.yellow, opacity=0, emissive=True)
    p.life = 0.0
    p.u = 0.0
    p.start = vector(0, 0, 0)
    p.end = vector(0, 0, 0)
    p.speed = 0.0
    pulse_particles.append(p)

context_particles = []
for _ in range(50):
    p = sphere(pos=vector(0, -9, 0), radius=0.035, color=vector(0.8, 0.8, 1.0), opacity=0, emissive=True)
    p.life = 0.0
    p.v = vector(0, 0, 0)
    context_particles.append(p)

title_label = label(pos=vector(-0.8, 3.45, 0), text="", height=18, box=False, opacity=0, color=color.black)
state_label = label(pos=vector(-0.8, 3.10, 0), text="", height=12, box=False, opacity=0, color=color.black)
left_label = label(pos=vector(-6.55, 2.55, 0), text="", height=10, box=False, opacity=0, color=color.black, align="left")
right_label = label(pos=vector(4.35, 2.55, 0), text="", height=10, box=False, opacity=0, color=color.black, align="left")

network_light = local_light(pos=vector(-0.8, 3.0, 1.4), color=vector(0.35, 0.45, 0.8))


# -----------------------------------------------------------------------------
# Runtime state
# -----------------------------------------------------------------------------

current_round = 0
round_time_ms = 0.0
ROUND_MS = 700.0
dt_ms = 2.0
paused = False
auto_cycle = True
show_spikes = True
show_links = True
show_context = True
show_world = True
last_counts = {name: 0 for name in ALL_NEURONS}


def reset_round():
    global round_time_ms, last_counts
    round_time_ms = 0.0
    last_counts = {name: 0 for name in ALL_NEURONS}
    for p in spike_particles + pulse_particles + context_particles:
        p.pos = vector(0, -9, 0)
        p.opacity = 0
        p.life = 0


def set_round(idx):
    global current_round
    current_round = idx % len(ROUNDS)
    reset_round()


def on_key(evt):
    global paused, auto_cycle, show_spikes, show_links, show_context, show_world
    k = evt.key.lower()
    if k == " ":
        paused = not paused
    elif k == "n":
        set_round(current_round + 1)
    elif k == "b":
        set_round(current_round - 1)
    elif k == "r":
        reset_round()
    elif k == "a":
        auto_cycle = not auto_cycle
    elif k == "s":
        show_spikes = not show_spikes
    elif k == "l":
        show_links = not show_links
        for c in link_curves.values():
            c.visible = show_links
    elif k == "c":
        show_context = not show_context
    elif k == "t":
        show_world = not show_world


scene.bind("keydown", on_key)


# -----------------------------------------------------------------------------
# Particle helpers
# -----------------------------------------------------------------------------

def spawn_spike(name: str):
    if not show_spikes:
        return
    pos = NEURON_POS[name]
    col = neuron_active_color(name)
    for _ in range(5):
        for p in spike_particles:
            if p.life <= 0:
                a = rng.random() * 2 * pi
                p.pos = pos
                p.v = vector(0.025 * cos(a), rng.uniform(0.004, 0.035), 0.025 * sin(a))
                p.radius = rng.uniform(0.022, 0.050)
                p.color = col
                p.opacity = 0.88
                p.life = rng.uniform(0.35, 0.85)
                break


def spawn_pulse(src: str, dst: str, rd: RoundData):
    if not show_spikes:
        return
    if src not in NEURON_POS or dst not in NEURON_POS:
        return
    for p in pulse_particles:
        if p.life <= 0:
            p.start = NEURON_POS[src]
            p.end = NEURON_POS[dst]
            p.pos = p.start
            p.u = 0
            p.speed = rng.uniform(0.025, 0.055)
            p.color = CONTEXT_COLORS.get(rd.context, vector(1.0, 0.85, 0.12))
            p.opacity = 0.82
            p.life = 1.0
            return


def spawn_context_particle(rd: RoundData):
    if not show_context:
        return
    col = CONTEXT_COLORS.get(rd.context, vector(0.8, 0.8, 1.0))
    for p in context_particles:
        if p.life <= 0:
            p.pos = context_box.pos + vector(rng.uniform(-0.55, 0.55), rng.uniform(-0.35, 0.35), rng.uniform(-0.03, 0.03))
            p.v = vector(rng.uniform(-0.006, 0.006), rng.uniform(0.003, 0.014), rng.uniform(-0.003, 0.003))
            p.color = col
            p.opacity = rng.uniform(0.18, 0.45)
            p.radius = rng.uniform(0.025, 0.055)
            p.life = rng.uniform(1.2, 2.8)
            return


def update_particles():
    for p in spike_particles:
        if p.life > 0:
            p.pos += p.v
            p.v *= 0.985
            p.life -= 0.025
            p.opacity *= 0.955
            if p.life <= 0:
                p.opacity = 0
                p.pos = vector(0, -9, 0)

    for p in pulse_particles:
        if p.life > 0:
            p.u += p.speed
            u = clamp(p.u, 0, 1)
            lift = vector(0, 0.18 * sin(pi * u), 0.10 * sin(pi * u))
            p.pos = p.start * (1 - u) + p.end * u + lift
            p.opacity = 0.82 * (1 - 0.45 * u)
            if p.u >= 1:
                p.life = 0
                p.opacity = 0
                p.pos = vector(0, -9, 0)

    for p in context_particles:
        if p.life > 0:
            p.pos += p.v
            p.life -= 0.018
            p.opacity *= 0.992
            if p.life <= 0:
                p.opacity = 0
                p.pos = vector(0, -9, 0)


# -----------------------------------------------------------------------------
# Main visual update
# -----------------------------------------------------------------------------

def update_scene():
    global last_counts

    rd = ROUNDS[current_round]
    t = round_time_ms
    context_col = CONTEXT_COLORS.get(rd.context, vector(0.7, 0.7, 0.7))
    selected_motor = selected_motor_name(rd.chosen)
    total_so_far = 0

    # Neurons
    for name in ALL_NEURONS:
        count = rd.spike_counts.get(name, 0)
        now = spikes_so_far(rd, name, t)
        total_so_far += now
        max_count = max(1, count)
        active_level = clamp(now / max_count, 0.0, 1.0)

        if now > last_counts.get(name, 0):
            spawn_spike(name)
            for src, dst in LINKS:
                if src == name and link_weight(rd, src, dst) > 0.22:
                    spawn_pulse(src, dst, rd)

        spiked = just_spiked(rd, name, t)
        base = neuron_base_color(name)
        active = neuron_active_color(name)
        col = mix(base, active, active_level)

        # Context-selected pathways glow more strongly.
        involved_in_context = False
        for token in rd.strongest:
            if name in token:
                involved_in_context = True
                break

        if involved_in_context:
            col = mix(col, context_col, 0.32)

        if name == selected_motor:
            col = mix(col, vector(1.0, 0.95, 0.10), 0.35)

        if spiked:
            col = vector(1.0, 0.90, 0.12)

        neurons[name].color = col
        neurons[name].radius = 0.27 + 0.12 * active_level + (0.08 if spiked else 0)
        rings[name].color = col
        rings[name].opacity = 0.18 + 0.55 * active_level
        rings[name].radius = 0.44 + 0.06 * (1 if involved_in_context else 0) + 0.03 * sin(t / 50.0 + ALL_NEURONS.index(name))
        rings[name].thickness = 0.026 if involved_in_context else 0.018

        neuron_labels[name].text = f"{name}: {now}/{count}"

    last_counts = {name: spikes_so_far(rd, name, t) for name in ALL_NEURONS}

    # Links
    for (src, dst), c in link_curves.items():
        w = link_weight(rd, src, dst)
        token = f"{src}->{dst}"
        if token in rd.strongest:
            w = 1.0
        c.visible = show_links
        c.radius = 0.008 + 0.050 * w
        c.opacity = 0.08 + 0.76 * w
        c.color = context_col if token in rd.strongest else mix(vector(0.35, 0.35, 0.35), context_col, 0.45 * w)

    # Context display
    context_box.visible = show_context
    context_arrow.visible = show_context
    context_box.color = context_col
    context_arrow.color = context_col
    context_arrow.axis = vector(0, 0.45 + 0.25 * sin(t / 60.0), 0)

    if show_context and rng.random() < 0.18:
        spawn_context_particle(rd)

    # World bars
    for i, key in enumerate(["L", "R", "heat", "touch", "chem"]):
        val = rd.world[key]
        b = world_bars[key]
        b.visible = show_world
        b.size = vector(0.30, 0.10 + 0.90 * val, 0.25)
        b.pos.y = -2.10 + b.size.y / 2
        if key in ("L", "R"):
            b.color = vector(1.0, 0.85, 0.15)
        elif key == "heat":
            b.color = vector(1.0, 0.25, 0.12)
        elif key == "touch":
            b.color = vector(0.85, 0.45, 0.10)
        else:
            b.color = vector(0.20, 0.75, 0.25)

    # Motor arrows
    for motor, arr in motor_arrows.items():
        active = motor == selected_motor
        arr.visible = True
        arr.opacity = 1.0 if active else 0.18
        scale = 1.0 if active else 0.35
        if motor == "M_LEFT":
            arr.axis = vector(-0.75 * scale, 0, 0)
        elif motor == "M_RIGHT":
            arr.axis = vector(0.75 * scale, 0, 0)
        elif motor == "M_FORWARD":
            arr.axis = vector(0, 0.75 * scale, 0)
        elif motor == "M_BACKWARD":
            arr.axis = vector(0, -0.75 * scale, 0)

    # Round bars
    max_spikes = max(r.total_spikes for r in ROUNDS)
    for i, b in enumerate(round_bars):
        rr = ROUNDS[i]
        h = 0.12 + 1.35 * rr.total_spikes / max_spikes
        b.size = vector(0.34, h, 0.30)
        b.pos = vector(-5.75 + i * 1.25, -2.12 + h / 2, 2.55)
        if i == current_round:
            b.color = vector(1.0, 0.85, 0.12)
        else:
            b.color = CONTEXT_COLORS.get(rr.context, vector(0.5, 0.5, 0.6))

    network_light.color = mix(vector(0.25, 0.30, 0.45), context_col, 0.75)

    title_label.text = f"Round {rd.number}: {rd.scenario.upper()} — {rd.context}"
    state_label.text = (
        f"expected={rd.expected} | chosen={rd.chosen} | correct={rd.correct} | "
        f"spikes≈{total_so_far}/{rd.total_spikes} | relay={rd.relay}"
    )

    left_label.text = (
        f"World/context\n"
        f"scenario: {rd.scenario}\n"
        f"context: {rd.context}\n"
        f"light L/R: {rd.world['L']:.2f}/{rd.world['R']:.2f}\n"
        f"heat: {rd.world['heat']:.2f}\n"
        f"touch: {rd.world['touch']:.2f}\n"
        f"chemical: {rd.world['chem']:.2f}\n"
        f"obstacle: {rd.world['obstacle']}\n\n"
        f"gating note:\n{rd.note}"
    )

    right_label.text = (
        f"Round output\n"
        f"expected: {rd.expected}\n"
        f"chosen: {rd.chosen}\n"
        f"score: {rd.score:.2f}\n"
        f"reaction: {rd.reaction_ms:.0f} ms\n"
        f"sensory/inter/motor:\n"
        f"{rd.sensory_spikes}/{rd.inter_spikes}/{rd.motor_spikes}\n\n"
        f"Final run:\n"
        f"accuracy 8/8 = 100%\n"
        f"real motor output 8/8\n"
        f"relay activity 8/8"
    )

    status.text = (
        f"Round {rd.number}/8 | {rd.scenario} | ctx={rd.context} | "
        f"{rd.expected}->{rd.chosen} | spikes={total_so_far}/{rd.total_spikes} | auto={'on' if auto_cycle else 'off'}"
    )

    update_particles()


# -----------------------------------------------------------------------------
# Main loop
# -----------------------------------------------------------------------------

while True:
    rate(60)

    if not paused:
        round_time_ms += dt_ms
        if auto_cycle and round_time_ms >= ROUND_MS:
            set_round(current_round + 1)
        elif not auto_cycle and round_time_ms > ROUND_MS:
            round_time_ms = ROUND_MS

    update_scene()
