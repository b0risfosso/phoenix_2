#!/usr/bin/env python3
"""
VPython Homeostatic Neuron Network Self-Exploration
---------------------------------------------------

Visual simulation based on terminal output from:
terminal_neuron_network_self_exploration_homeostatic_v3.py

Run:
    python vpython_homeostatic_neuron_network_self_exploration.py

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
    H      - toggle homeostasis display
    M      - toggle motif glow

This is a visualization of the terminal results, not a full biophysical solver.
It maps round summaries into a VPython scene:

    - neurons grow from N=6 to N=10
    - synapses grow and prune
    - motifs are stored across rounds
    - homeostasis dampens overactivity by raising threshold and inhibition
    - underactivity lifting triggers Round 9 rebound
    - final network is smaller and less saturated than v2
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
    cylinder,
    label,
    local_light,
    rate,
    ring,
    sphere,
    vector,
    wtext,
)


# -----------------------------------------------------------------------------
# Terminal-output round data
# -----------------------------------------------------------------------------

@dataclass
class RoundData:
    number: int
    mode: str
    neurons_before: int
    neurons_after: int
    syn_before: int
    syn_after: int
    spikes: int
    hz_per_neuron: float
    active: int
    silent: int
    homeo: str
    threshold_shift: float
    inhibition_gain: float
    syn_scale: float
    growth_neurons: int
    growth_synapses: int
    prune_neurons: int
    prune_synapses: int
    strongest: str
    motif: str
    spike_counts: dict
    note: str


ROUNDS = [
    RoundData(
        1, "seed exploration", 6, 7, 11, 23, 75, 15.31, 6, 1,
        "initial", 0.00, 1.00, 1.000, 1, 12, 0, 0,
        "E1→E0 +0.580", "R1: I4-I5-E0-E1",
        {"E0":12, "E1":12, "E2":12, "E3":12, "I4":14, "I5":13, "E6":0},
        "first motif; growth begins"
    ),
    RoundData(
        2, "growth search", 7, 8, 23, 32, 115, 20.54, 7, 1,
        "balanced", 0.00, 1.00, 1.000, 1, 9, 0, 0,
        "E1→I5 +0.856", "R2: E6-E0-E2-I4",
        {"E0":16, "E1":14, "E2":16, "E3":15, "I4":16, "I5":16, "E6":22, "I7":0},
        "activity rises; inhibitory regulator I7 is born"
    ),
    RoundData(
        3, "self-test pulse", 8, 8, 32, 35, 70, 12.50, 8, 0,
        "dampening overactivity", 0.87, 1.18, 0.959, 0, 3, 0, 0,
        "E1→I5 +0.848", "R3: E2-E6-E0-E3",
        {"E0":8, "E1":7, "E2":12, "E3":8, "I4":7, "I5":8, "E6":12, "I7":8},
        "homeostasis reduces firing without silencing"
    ),
    RoundData(
        4, "prune quiet paths", 8, 8, 35, 37, 120, 21.43, 8, 0,
        "balanced", 0.78, 1.16, 1.000, 0, 2, 0, 0,
        "E1→E2 +0.889", "R4: E2-E6-E3-I4",
        {"E0":12, "E1":12, "E2":19, "E3":18, "I4":14, "I5":13, "E6":19, "I7":13},
        "active pruning round; still high activity"
    ),
    RoundData(
        5, "motif rehearsal", 8, 8, 37, 35, 102, 18.21, 8, 0,
        "dampening overactivity", 1.79, 1.38, 0.951, 0, 0, 0, 2,
        "E6→E2 +0.889", "R5: E2-E3-E6-I4",
        {"E0":4, "E1":2, "E2":23, "E3":23, "I4":20, "I5":3, "E6":23, "I7":4},
        "motif neurons dominate; two synapses pruned"
    ),
    RoundData(
        6, "novelty search", 8, 8, 35, 36, 47, 8.39, 8, 0,
        "dampening overactivity", 2.77, 1.59, 0.952, 0, 3, 0, 2,
        "E3→E2 +0.889", "R6: E2-E3-E6-E0",
        {"E0":4, "E1":4, "E2":10, "E3":9, "I4":4, "I5":4, "E6":8, "I7":4},
        "strong dampening lowers activity but keeps all neurons active"
    ),
    RoundData(
        7, "stability test", 8, 9, 36, 28, 7, 1.11, 3, 6,
        "dampening overactivity", 3.58, 1.76, 0.963, 1, 1, 0, 9,
        "E6→E3 +0.855", "R7: E6-E2-E3",
        {"E0":0, "E1":0, "E2":2, "E3":2, "I4":0, "I5":0, "E6":3, "I7":0, "I8":0},
        "homeostasis cuts activity; pruning removes nine synapses"
    ),
    RoundData(
        8, "branch growth", 9, 9, 28, 32, 33, 5.24, 9, 0,
        "dampening overactivity", 4.00, 1.85, 0.958, 0, 4, 0, 0,
        "E6→E3 +0.867", "R8: E6-E2-E3-I4",
        {"E0":1, "E1":1, "E2":7, "E3":7, "I4":3, "I5":1, "E6":8, "I7":3, "I8":2},
        "maximum dampening; all neurons still participate"
    ),
    RoundData(
        9, "deep self-exploration", 9, 10, 32, 44, 182, 26.00, 9, 1,
        "lifting underactivity", 3.49, 1.75, 1.019, 1, 12, 0, 0,
        "E1→I5 +0.889", "R9: E2-E3-E6-I4",
        {"E0":12, "E1":12, "E2":31, "E3":31, "I4":30, "I5":11, "E6":31, "I7":13, "I8":11, "E9":0},
        "underactivity lift causes rebound and new growth"
    ),
    RoundData(
        10, "consolidation", 10, 10, 44, 38, 0, 0.00, 0, 10,
        "dampening overactivity", 4.00, 1.85, 0.938, 0, 1, 0, 7,
        "I4→E1 -0.889", "none",
        {"E0":0, "E1":0, "E2":0, "E3":0, "I4":0, "I5":0, "E6":0, "I7":0, "I8":0, "E9":0},
        "over-dampened consolidation; seven synapses pruned"
    ),
]

ALL_NEURONS = ["E0", "E1", "E2", "E3", "I4", "I5", "E6", "I7", "I8", "E9"]
EXCITATORY = {"E0", "E1", "E2", "E3", "E6", "E9"}
INHIBITORY = {"I4", "I5", "I7", "I8"}

# Approximate final links / strong pathways from output.
LINKS = [
    ("E1", "E0"), ("E1", "I5"), ("E1", "E2"),
    ("E6", "E2"), ("E3", "E2"), ("E6", "E3"),
    ("I4", "E1"), ("I4", "I5"), ("I5", "E3"),
    ("I4", "E0"), ("I4", "E6"), ("I7", "E3"),
    ("I7", "I4"), ("I4", "E3"),
    ("E2", "E3"), ("E2", "I4"), ("E3", "I4"),
    ("E6", "I4"), ("E0", "E2"), ("E9", "E2"),
]

BASE_POS = {
    "E0": vector(-4.2, 2.05, 0.0),
    "E1": vector(-4.2, 0.85, 0.0),
    "E2": vector(-1.4, 2.20, 0.0),
    "E3": vector(-1.4, 0.70, 0.0),
    "I4": vector(1.6, 2.05, 0.0),
    "I5": vector(1.6, 0.85, 0.0),
    "E6": vector(-1.4, -0.65, 0.0),
    "I7": vector(3.8, 1.55, 0.0),
    "I8": vector(3.8, 0.10, 0.0),
    "E9": vector(-3.1, -0.95, 0.0),
}


# -----------------------------------------------------------------------------
# Scene setup
# -----------------------------------------------------------------------------

scene = canvas(
    title="Homeostatic Neuron Network Self-Exploration",
    width=1280,
    height=760,
    background=vector(0.86, 0.90, 0.96),
    center=vector(0, 1.0, 0),
)
scene.camera.pos = vector(0, 6.1, 11.5)
scene.camera.axis = vector(0, -4.8, -11.5)

status = wtext(text="")
wtext(text="\nControls: Space pause/resume | N next | B previous | R restart | A auto-cycle | S spikes | L links | H homeostasis | M motif\n\n")

rng = Random(4417)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def mix(a, b, u):
    u = clamp(u, 0.0, 1.0)
    return a * (1 - u) + b * u


def neuron_exists(rd: RoundData, name: str) -> bool:
    return name in rd.spike_counts


def neuron_color(name: str, active_level: float, motif: bool, homeo: str):
    if name in INHIBITORY:
        base = vector(0.95, 0.25, 0.75)
        active = vector(1.0, 0.25, 0.15)
    else:
        base = vector(0.20, 0.50, 1.0)
        active = vector(1.0, 0.85, 0.15)

    col = mix(base, active, active_level)
    if motif:
        col = mix(col, vector(0.65, 0.35, 1.0), 0.35)
    if homeo.startswith("dampening") and name in INHIBITORY:
        col = mix(col, vector(1.0, 0.1, 0.1), 0.25)
    return col


def motif_set(rd: RoundData):
    if rd.motif == "none":
        return set()
    tail = rd.motif.split(":", 1)[-1].strip()
    return set(tail.split("-"))


def round_spike_times(rd: RoundData, name: str):
    count = rd.spike_counts.get(name, 0)
    if count <= 0:
        return []
    duration = 700.0
    start = 45.0 + (ALL_NEURONS.index(name) % 5) * 8
    end = 690.0 - (ALL_NEURONS.index(name) % 3) * 9
    if count == 1:
        return [(start + end) / 2.0]
    return [start + i * (end - start) / max(1, count - 1) for i in range(count)]


def spikes_so_far(rd: RoundData, name: str, t_ms: float):
    return sum(1 for st in round_spike_times(rd, name) if st <= t_ms)


def just_spiked(rd: RoundData, name: str, t_ms: float):
    return any(0 <= t_ms - st <= 6.0 for st in round_spike_times(rd, name))


def link_strength(rd: RoundData, src: str, dst: str):
    # Based on synapse count, strong count, and strongest named link.
    if src not in rd.spike_counts or dst not in rd.spike_counts:
        return 0.0

    base = clamp(rd.syn_after / 50.0, 0.10, 0.85)
    strongest_srcdst = rd.strongest.replace("→", "->").replace(" ", "")
    token = f"{src}->{dst}"
    if token in strongest_srcdst:
        base = 1.0

    if src in INHIBITORY and rd.homeo.startswith("dampening"):
        base *= 1.25
    if rd.syn_scale < 1.0 and src in EXCITATORY:
        base *= rd.syn_scale

    return clamp(base, 0.0, 1.0)


def homeo_color(rd: RoundData):
    if rd.homeo.startswith("dampening"):
        return vector(1.0, 0.35, 0.10)
    if rd.homeo.startswith("lifting"):
        return vector(0.25, 0.70, 1.0)
    if rd.homeo == "balanced":
        return vector(0.25, 0.75, 0.25)
    return vector(0.65, 0.65, 0.65)


# -----------------------------------------------------------------------------
# Visual objects
# -----------------------------------------------------------------------------

floor = box(pos=vector(0, -1.28, 0), size=vector(13.0, 0.08, 6.4), color=vector(0.72, 0.76, 0.78))

neurons = {}
rings = {}
neuron_labels = {}
for name in ALL_NEURONS:
    s = sphere(pos=BASE_POS[name], radius=0.30, color=vector(0.4, 0.4, 0.4), opacity=0.12)
    r = ring(pos=BASE_POS[name], axis=vector(0, 1, 0), radius=0.45, thickness=0.018, color=s.color, opacity=0.10)
    lab = label(pos=BASE_POS[name] + vector(0, -0.52, 0), text=name, height=10, box=False, opacity=0, color=color.black)
    neurons[name] = s
    rings[name] = r
    neuron_labels[name] = lab

link_curves = {}
for src, dst in LINKS:
    start = BASE_POS[src]
    end = BASE_POS[dst]
    mid = (start + end) * 0.5 + vector(0, 0.25, 0.08)
    c = curve(pos=[start, mid, end], radius=0.012, color=vector(0.35, 0.35, 0.35), opacity=0.20)
    link_curves[(src, dst)] = c

# Growth and pruning bars.
n_bar = box(pos=vector(-5.7, -0.9, -2.4), size=vector(0.36, 0.2, 0.35), color=vector(0.2, 0.5, 1.0))
s_bar = box(pos=vector(-5.15, -0.9, -2.4), size=vector(0.36, 0.2, 0.35), color=vector(0.2, 0.75, 0.35))
strong_bar = box(pos=vector(-4.60, -0.9, -2.4), size=vector(0.36, 0.2, 0.35), color=vector(1.0, 0.65, 0.10))
label(pos=vector(-5.7, -1.20, -2.4), text="N", height=10, box=False, opacity=0, color=color.black)
label(pos=vector(-5.15, -1.20, -2.4), text="S", height=10, box=False, opacity=0, color=color.black)
label(pos=vector(-4.60, -1.20, -2.4), text="strong", height=10, box=False, opacity=0, color=color.black)

homeo_panel = box(pos=vector(5.05, 1.2, -2.35), size=vector(1.15, 1.0, 0.16), color=vector(0.65, 0.65, 0.65), opacity=0.45)
threshold_arrow = arrow(pos=vector(4.45, 0.38, -2.35), axis=vector(0, 0.6, 0), shaftwidth=0.045, color=color.red)
inhibition_arrow = arrow(pos=vector(5.10, 0.38, -2.35), axis=vector(0, 0.6, 0), shaftwidth=0.045, color=vector(1.0, 0.2, 0.2))
scale_arrow = arrow(pos=vector(5.75, 0.38, -2.35), axis=vector(0, 0.6, 0), shaftwidth=0.045, color=vector(0.2, 0.65, 1.0))

motif_cloud = []
for _ in range(40):
    p = sphere(pos=vector(0, -9, 0), radius=0.035, color=vector(0.65, 0.35, 1.0), opacity=0.0, emissive=True)
    p.v = vector(0, 0, 0)
    p.life = 0.0
    motif_cloud.append(p)

spike_particles = []
for _ in range(150):
    p = sphere(pos=vector(0, -9, 0), radius=0.045, color=color.yellow, opacity=0.0, emissive=True)
    p.v = vector(0, 0, 0)
    p.life = 0.0
    spike_particles.append(p)

pulse_particles = []
for _ in range(80):
    p = sphere(pos=vector(0, -9, 0), radius=0.04, color=color.yellow, opacity=0.0, emissive=True)
    p.life = 0.0
    p.start = vector(0, 0, 0)
    p.end = vector(0, 0, 0)
    p.u = 0.0
    p.speed = 0.0
    pulse_particles.append(p)

# Summary round bars.
round_bars = []
for i, rd in enumerate(ROUNDS):
    x = -5.75 + i * 1.15
    b = box(pos=vector(x, -1.05, 2.7), size=vector(0.34, 0.2, 0.32), color=vector(0.35, 0.45, 0.80))
    round_bars.append(b)
    label(pos=vector(x, -1.32, 2.7), text=str(rd.number), height=9, box=False, opacity=0, color=color.black)

title_label = label(pos=vector(0, 3.45, 0), text="", height=18, box=False, opacity=0, color=color.black)
state_label = label(pos=vector(0, 3.08, 0), text="", height=13, box=False, opacity=0, color=color.black)
left_label = label(pos=vector(-6.15, 2.35, 0), text="", height=11, box=False, opacity=0, color=color.black, align="left")
right_label = label(pos=vector(3.75, 2.55, 0), text="", height=11, box=False, opacity=0, color=color.black, align="left")

network_light = local_light(pos=vector(0, 2.7, 1.2), color=vector(0.3, 0.4, 0.8))


# -----------------------------------------------------------------------------
# State / controls
# -----------------------------------------------------------------------------

current_round = 0
round_time_ms = 0.0
paused = False
auto_cycle = True
show_spikes = True
show_links = True
show_homeostasis = True
show_motif = True
last_counts = {name: 0 for name in ALL_NEURONS}
dt_ms = 2.0
ROUND_MS = 700.0


def reset_round():
    global round_time_ms, last_counts
    round_time_ms = 0.0
    last_counts = {name: 0 for name in ALL_NEURONS}
    for p in spike_particles + pulse_particles + motif_cloud:
        p.pos = vector(0, -9, 0)
        p.opacity = 0.0
        p.life = 0.0


def set_round(idx):
    global current_round
    current_round = idx % len(ROUNDS)
    reset_round()


def on_key(evt):
    global paused, auto_cycle, show_spikes, show_links, show_homeostasis, show_motif
    key = evt.key.lower()
    if key == " ":
        paused = not paused
    elif key == "n":
        set_round(current_round + 1)
    elif key == "b":
        set_round(current_round - 1)
    elif key == "r":
        reset_round()
    elif key == "a":
        auto_cycle = not auto_cycle
    elif key == "s":
        show_spikes = not show_spikes
    elif key == "l":
        show_links = not show_links
        for c in link_curves.values():
            c.visible = show_links
    elif key == "h":
        show_homeostasis = not show_homeostasis
    elif key == "m":
        show_motif = not show_motif


scene.bind("keydown", on_key)


# -----------------------------------------------------------------------------
# Particles
# -----------------------------------------------------------------------------

def spawn_spike(name):
    if not show_spikes:
        return
    pos = BASE_POS[name]
    col = vector(1.0, 0.85, 0.1) if name in EXCITATORY else vector(1.0, 0.25, 0.18)
    for _ in range(8):
        for p in spike_particles:
            if p.life <= 0:
                a = rng.random() * 2 * pi
                p.pos = pos
                p.v = vector(0.025 * cos(a), rng.uniform(0.004, 0.035), 0.025 * sin(a))
                p.radius = rng.uniform(0.025, 0.055)
                p.color = col
                p.opacity = 0.85
                p.life = rng.uniform(0.45, 0.95)
                break


def spawn_pulse(src, dst):
    if not show_spikes or src not in BASE_POS or dst not in BASE_POS:
        return
    for p in pulse_particles:
        if p.life <= 0:
            p.start = BASE_POS[src]
            p.end = BASE_POS[dst]
            p.pos = p.start
            p.u = 0.0
            p.speed = rng.uniform(0.025, 0.055)
            p.color = vector(1.0, 0.85, 0.12) if src in EXCITATORY else vector(1.0, 0.20, 0.12)
            p.opacity = 0.80
            p.life = 1.0
            return


def spawn_motif_particle(rd):
    if not show_motif:
        return
    nodes = list(motif_set(rd))
    if not nodes:
        return
    for p in motif_cloud:
        if p.life <= 0:
            name = rng.choice(nodes)
            base = BASE_POS.get(name, vector(0, 0, 0))
            a = rng.random() * 2 * pi
            p.pos = base + vector(0.35 * cos(a), rng.uniform(0.1, 0.55), 0.35 * sin(a))
            p.v = vector(rng.uniform(-0.004, 0.004), rng.uniform(0.002, 0.012), rng.uniform(-0.004, 0.004))
            p.radius = rng.uniform(0.025, 0.060)
            p.color = vector(0.65, 0.35, 1.0)
            p.opacity = rng.uniform(0.15, 0.42)
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
            lift = vector(0, 0.22 * sin(pi * u), 0.10 * sin(pi * u))
            p.pos = p.start * (1 - u) + p.end * u + lift
            p.opacity = 0.85 * (1 - 0.4 * u)
            if p.u >= 1:
                p.life = 0
                p.opacity = 0
                p.pos = vector(0, -9, 0)

    for p in motif_cloud:
        if p.life > 0:
            p.pos += p.v
            p.life -= 0.018
            p.opacity *= 0.992
            if p.life <= 0:
                p.opacity = 0
                p.pos = vector(0, -9, 0)


# -----------------------------------------------------------------------------
# Main update
# -----------------------------------------------------------------------------

def update_scene():
    global last_counts

    rd = ROUNDS[current_round]
    t = round_time_ms
    mot = motif_set(rd)
    total_so_far = 0

    # Neurons
    for name in ALL_NEURONS:
        exists = neuron_exists(rd, name)
        s = neurons[name]
        r = rings[name]
        lab = neuron_labels[name]

        if not exists:
            s.opacity = 0.08
            r.opacity = 0.04
            lab.text = f"{name} absent"
            continue

        count_now = spikes_so_far(rd, name, t)
        total_so_far += count_now
        max_count = max(1, rd.spike_counts.get(name, 1))
        active_level = clamp(count_now / max_count, 0.0, 1.0)

        if count_now > last_counts.get(name, 0):
            spawn_spike(name)
            # Send pulses along active outgoing links.
            for src, dst in LINKS:
                if src == name and dst in rd.spike_counts and link_strength(rd, src, dst) > 0.2:
                    spawn_pulse(src, dst)

        spiked = just_spiked(rd, name, t)
        col = neuron_color(name, active_level, name in mot, rd.homeo)
        if spiked:
            col = vector(1.0, 0.9, 0.15) if name in EXCITATORY else vector(1.0, 0.25, 0.12)

        s.opacity = 0.92
        s.color = col
        s.radius = 0.28 + 0.10 * active_level + (0.08 if spiked else 0.0)
        r.opacity = 0.25 + 0.45 * active_level
        r.color = col
        if name in mot:
            r.radius = 0.52 + 0.05 * sin(t / 60.0)
            r.thickness = 0.025
        else:
            r.radius = 0.45
            r.thickness = 0.018

        lab.text = f"{name}: {count_now}/{rd.spike_counts.get(name, 0)}"

    last_counts = {name: spikes_so_far(rd, name, t) for name in ALL_NEURONS}

    # Links
    for (src, dst), c in link_curves.items():
        strength = link_strength(rd, src, dst)
        c.visible = show_links and strength > 0.02
        c.radius = 0.008 + 0.045 * strength
        c.opacity = 0.08 + 0.75 * strength
        if src in INHIBITORY:
            c.color = vector(1.0, 0.15, 0.10)
        else:
            c.color = vector(0.20, 0.55, 1.0)

    # Homeostatic panel/arrows
    hc = homeo_color(rd)
    homeo_panel.visible = show_homeostasis
    threshold_arrow.visible = show_homeostasis
    inhibition_arrow.visible = show_homeostasis
    scale_arrow.visible = show_homeostasis

    homeo_panel.color = hc
    threshold_arrow.axis = vector(0, 0.20 + 0.80 * clamp(rd.threshold_shift / 4.0, 0, 1), 0)
    inhibition_arrow.axis = vector(0, 0.20 + 0.80 * clamp((rd.inhibition_gain - 1.0) / 0.85, 0, 1), 0)
    scale_len = 0.20 + 0.80 * clamp(abs(rd.syn_scale - 1.0) / 0.08, 0, 1)
    scale_arrow.axis = vector(0, scale_len if rd.syn_scale >= 1 else -scale_len, 0)

    # Bars for N, S, strong estimate.
    n_h = 0.15 + 1.4 * rd.neurons_after / 10.0
    s_h = 0.15 + 1.4 * rd.syn_after / 44.0
    # Estimate strong count from final text trend.
    strong_est = int(round(rd.syn_after * (0.25 + 0.55 * clamp(rd.spikes / 120.0, 0, 1))))
    if rd.number == 10:
        strong_est = 27
    strong_h = 0.15 + 1.4 * clamp(strong_est / 38.0, 0, 1)
    for b, h in [(n_bar, n_h), (s_bar, s_h), (strong_bar, strong_h)]:
        b.size = vector(b.size.x, h, b.size.z)
        b.pos.y = -1.20 + h / 2

    # Round bars by spike totals.
    max_spikes = max(r.spikes for r in ROUNDS)
    for i, b in enumerate(round_bars):
        rr = ROUNDS[i]
        h = 0.12 + 1.45 * rr.spikes / max(1, max_spikes)
        b.size = vector(0.34, h, 0.32)
        b.pos = vector(-5.75 + i * 1.15, -1.25 + h / 2, 2.7)
        if i == current_round:
            b.color = vector(1.0, 0.65, 0.10)
        elif rr.homeo.startswith("dampening"):
            b.color = vector(1.0, 0.35, 0.10)
        elif rr.homeo.startswith("lifting"):
            b.color = vector(0.25, 0.70, 1.0)
        elif rr.homeo == "balanced":
            b.color = vector(0.25, 0.75, 0.25)
        else:
            b.color = vector(0.55, 0.55, 0.60)

    if show_motif and rd.motif != "none" and rng.random() < 0.25:
        spawn_motif_particle(rd)

    network_light.color = hc * 0.7 + vector(0.1, 0.1, 0.1)

    title_label.text = f"Round {rd.number}: {rd.mode.upper()} — {rd.homeo}"
    state_label.text = (
        f"t={t:5.1f} ms | spikes≈{total_so_far}/{rd.spikes} | "
        f"N {rd.neurons_before}->{rd.neurons_after} | S {rd.syn_before}->{rd.syn_after} | motif {rd.motif}"
    )

    left_label.text = (
        f"Round summary\n"
        f"mode: {rd.mode}\n"
        f"spikes: {rd.spikes}\n"
        f"Hz/neuron: {rd.hz_per_neuron:.2f}\n"
        f"active/silent: {rd.active}/{rd.silent}\n"
        f"growth: +{rd.growth_neurons} neurons, +{rd.growth_synapses} synapses\n"
        f"pruning: -{rd.prune_neurons} neurons, -{rd.prune_synapses} synapses\n"
        f"strongest: {rd.strongest}\n"
        f"motif: {rd.motif}\n"
        f"note: {rd.note}"
    )

    right_label.text = (
        f"Homeostatic control\n"
        f"state: {rd.homeo}\n"
        f"threshold shift: {rd.threshold_shift:+.2f} mV\n"
        f"inhibition gain: {rd.inhibition_gain:.2f}\n"
        f"synaptic scale: {rd.syn_scale:.3f}\n\n"
        f"Final comparison target:\n"
        f"v2 ended: N=11 S=64 strong=60\n"
        f"v3 ended: N=10 S=38 strong=27\n\n"
        f"Interpretation:\n"
        f"homeostasis prevents runaway\n"
        f"strong-synapse saturation."
    )

    status.text = (
        f"Round {rd.number}/10 | {rd.mode} | homeo={rd.homeo} | "
        f"spikes={total_so_far}/{rd.spikes} | N={rd.neurons_after} S={rd.syn_after} | "
        f"auto={'on' if auto_cycle else 'off'}"
    )

    update_particles()


# -----------------------------------------------------------------------------
# Loop
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
