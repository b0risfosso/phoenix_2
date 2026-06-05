#!/usr/bin/env python3
"""
VPython Neuron Network Computation and Dream Simulation
------------------------------------------------------

Visual simulation based on the terminal output from:
neuron_network_compute_dream_accuracy_tuned.py

Run:
    python vpython_neuron_network_compute_dream.py

Requires:
    pip install vpython

Controls:
    Space  - pause/resume
    N      - next round
    B      - previous round
    R      - restart current round
    A      - toggle auto-cycle
    S      - toggle spike particles
    L      - toggle learned-link display
    D      - toggle dream replay glow

This VPython script visualizes the terminal-output behavior:
    - 6-neuron excitatory/inhibitory spiking network
    - compute rounds for Pattern A and Pattern B
    - output A = E3 and output B = I5
    - dream rounds that replay stored traces
    - memory-link strengthening over rounds
    - wake-test classification after dream replay
    - final compute accuracy of 6/6 = 100%
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
# Round data summarized from the terminal output
# -----------------------------------------------------------------------------

@dataclass
class RoundData:
    number: int
    mode: str
    pattern: str
    description: str
    total_spikes: int
    hz_per_neuron: float
    a_out: int
    b_out: int
    decision: str
    correct: str
    dream_similarity: float | None
    ei_ratio: float
    memory_traces: int
    strongest_links: str
    spike_counts: tuple[int, int, int, int, int, int]
    avg_voltage: float
    synaptic_events: int


ROUNDS = [
    RoundData(
        number=1,
        mode="compute",
        pattern="A",
        description="Pattern A drives E0/E1; output A should dominate.",
        total_spikes=21,
        hz_per_neuron=5.00,
        a_out=7,
        b_out=2,
        decision="A",
        correct="True",
        dream_similarity=None,
        ei_ratio=3.20,
        memory_traces=1,
        strongest_links="E1→E3, E3→I4, E0→E1, I4→I5",
        spike_counts=(3, 5, 1, 7, 3, 2),
        avg_voltage=-61.07,
        synaptic_events=105,
    ),
    RoundData(
        number=2,
        mode="compute",
        pattern="B",
        description="Pattern B drives E2/E3; output B should dominate.",
        total_spikes=24,
        hz_per_neuron=5.71,
        a_out=4,
        b_out=10,
        decision="B",
        correct="True",
        dream_similarity=None,
        ei_ratio=0.71,
        memory_traces=2,
        strongest_links="E3→I4, E1→E3, I4→I5, E3→I5",
        spike_counts=(0, 1, 5, 4, 4, 10),
        avg_voltage=-61.04,
        synaptic_events=120,
    ),
    RoundData(
        number=3,
        mode="compute",
        pattern="A",
        description="Repeat Pattern A to reinforce a memory trace.",
        total_spikes=35,
        hz_per_neuron=8.33,
        a_out=9,
        b_out=5,
        decision="A",
        correct="True",
        dream_similarity=None,
        ei_ratio=2.18,
        memory_traces=3,
        strongest_links="E1→E3, E3→I4, E2→I4, I4→I5",
        spike_counts=(4, 6, 5, 9, 6, 5),
        avg_voltage=-61.58,
        synaptic_events=175,
    ),
    RoundData(
        number=4,
        mode="compute",
        pattern="B",
        description="Repeat Pattern B to reinforce a memory trace.",
        total_spikes=31,
        hz_per_neuron=7.38,
        a_out=4,
        b_out=11,
        decision="B",
        correct="True",
        dream_similarity=None,
        ei_ratio=0.94,
        memory_traces=4,
        strongest_links="E3→I4, I4→I5, E1→E3, E2→I4",
        spike_counts=(2, 3, 6, 4, 5, 11),
        avg_voltage=-61.50,
        synaptic_events=155,
    ),
    RoundData(
        number=5,
        mode="dream",
        pattern="DREAM",
        description="Reduce outside input and replay stored traces.",
        total_spikes=13,
        hz_per_neuron=3.10,
        a_out=3,
        b_out=6,
        decision="none",
        correct="--",
        dream_similarity=0.827,
        ei_ratio=0.44,
        memory_traces=4,
        strongest_links="E3→I4, I4→I5, E1→E3, E2→I4",
        spike_counts=(0, 0, 1, 3, 3, 6),
        avg_voltage=-63.06,
        synaptic_events=65,
    ),
    RoundData(
        number=6,
        mode="dream",
        pattern="DREAM",
        description="Replay with mild recombination between traces.",
        total_spikes=16,
        hz_per_neuron=3.81,
        a_out=3,
        b_out=7,
        decision="none",
        correct="--",
        dream_similarity=0.875,
        ei_ratio=0.60,
        memory_traces=4,
        strongest_links="E3→I4, I4→I5, E1→E3, E2→I4",
        spike_counts=(1, 0, 2, 3, 3, 7),
        avg_voltage=-63.04,
        synaptic_events=80,
    ),
    RoundData(
        number=7,
        mode="compute",
        pattern="A",
        description="Wake test: classify Pattern A after dream replay.",
        total_spikes=51,
        hz_per_neuron=12.14,
        a_out=11,
        b_out=8,
        decision="A",
        correct="True",
        dream_similarity=None,
        ei_ratio=2.19,
        memory_traces=5,
        strongest_links="I4→I5, E3→I4, E1→E3, E2→I4",
        spike_counts=(7, 10, 7, 11, 8, 8),
        avg_voltage=-62.17,
        synaptic_events=255,
    ),
    RoundData(
        number=8,
        mode="compute",
        pattern="B",
        description="Wake test: classify Pattern B after dream replay.",
        total_spikes=50,
        hz_per_neuron=11.90,
        a_out=7,
        b_out=14,
        decision="B",
        correct="True",
        dream_similarity=None,
        ei_ratio=1.27,
        memory_traces=6,
        strongest_links="I4→I5, E3→I4, E2→I4, E1→E3",
        spike_counts=(5, 6, 10, 7, 8, 14),
        avg_voltage=-62.44,
        synaptic_events=250,
    ),
]


NEURON_NAMES = ["E0", "E1", "E2", "E3", "I4", "I5"]
EXCITATORY = {"E0", "E1", "E2", "E3"}
INHIBITORY = {"I4", "I5"}
OUTPUT_A = "E3"
OUTPUT_B = "I5"

# Fixed layout.
NEURON_POS = {
    "E0": vector(-3.8, 2.0, 0.0),
    "E1": vector(-3.8, 0.8, 0.0),
    "E2": vector(-0.9, 2.0, 0.0),
    "E3": vector(-0.9, 0.8, 0.0),
    "I4": vector(2.2, 2.0, 0.0),
    "I5": vector(2.2, 0.8, 0.0),
}

# Learned links highlighted in terminal output.
LINKS = [
    ("E0", "E1"),
    ("E1", "E3"),
    ("E3", "I4"),
    ("I4", "I5"),
    ("E2", "I4"),
    ("E3", "I5"),
    ("E2", "E3"),
    ("E0", "E3"),
]


# -----------------------------------------------------------------------------
# Scene
# -----------------------------------------------------------------------------

scene = canvas(
    title="Neuron Network Computation and Dream Replay",
    width=1280,
    height=760,
    background=vector(0.86, 0.90, 0.96),
    center=vector(0, 1.0, 0),
)
scene.camera.pos = vector(0, 5.4, 9.2)
scene.camera.axis = vector(0, -4.1, -9.2)

status = wtext(text="")
wtext(text="\nControls: Space pause/resume | N next | B previous | R restart | A auto-cycle | S spikes | L links | D dream glow\n\n")

rng = Random(29)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def lerp(a: float, b: float, u: float) -> float:
    u = clamp(u, 0.0, 1.0)
    return a + (b - a) * u


def mix(a, b, u: float):
    u = clamp(u, 0.0, 1.0)
    return a * (1.0 - u) + b * u


def neuron_base_color(name: str):
    if name in EXCITATORY:
        return vector(0.25, 0.55, 1.0)
    return vector(0.85, 0.25, 0.75)


def neuron_active_color(name: str):
    if name in EXCITATORY:
        return vector(1.0, 0.85, 0.15)
    return vector(1.0, 0.35, 0.25)


def output_color(name: str):
    if name == OUTPUT_A:
        return vector(0.20, 0.80, 0.25)
    if name == OUTPUT_B:
        return vector(0.95, 0.35, 0.15)
    return neuron_base_color(name)


def round_spike_times(rd: RoundData, neuron_index: int):
    """
    Deterministic approximate spike times from total spike counts.
    More active neurons spike more often across the 700 ms round.
    """
    count = rd.spike_counts[neuron_index]
    if count <= 0:
        return []
    start = 90 + neuron_index * 9
    end = 680 - neuron_index * 3
    if count == 1:
        return [(start + end) / 2]
    return [start + i * (end - start) / max(1, count - 1) for i in range(count)]


def voltage_for_neuron(rd: RoundData, neuron_index: int, t_ms: float):
    """
    Reconstruct a stylized membrane-voltage trace:
    charge toward threshold, spike/reset to -75 mV, repeat.
    """
    spikes = round_spike_times(rd, neuron_index)
    threshold = -55.5
    reset = -75.0
    rest = -70.0

    prev_spike = None
    next_spike = None
    for st in spikes:
        if st <= t_ms:
            prev_spike = st
        elif st > t_ms and next_spike is None:
            next_spike = st

    if prev_spike is not None and 0 <= t_ms - prev_spike <= 7:
        return reset, True

    if next_spike is None:
        phase_start = 0.0 if prev_spike is None else prev_spike + 7
        u = clamp((t_ms - phase_start) / 130.0, 0, 1)
        v = lerp(reset if prev_spike is not None else rest, -58.0, 1 - (1 - u) ** 2)
        v += 0.55 * sin(t_ms / 85.0 + neuron_index)
        return v, False

    phase_start = 0.0 if prev_spike is None else prev_spike + 7
    u = clamp((t_ms - phase_start) / max(1, next_spike - phase_start), 0, 1)
    v = lerp(reset if prev_spike is not None else rest, threshold + 0.1, 1 - (1 - u) ** 2.25)
    v += 0.45 * sin(t_ms / 55.0 + neuron_index * 0.7)
    return v, False


def count_spikes_so_far(rd: RoundData, neuron_index: int, t_ms: float):
    return sum(1 for st in round_spike_times(rd, neuron_index) if st <= t_ms)


def active_input_neurons(rd: RoundData):
    if rd.pattern == "A":
        return {"E0", "E1"}
    if rd.pattern == "B":
        return {"E2", "E3"}
    # Dream: replay/mix stored traces.
    return {"E1", "E3", "I4", "I5"}


def link_strength_for_round(rd: RoundData, link):
    """
    Approximate visual link strength from memory trace count and named strong links.
    """
    src, dst = link
    token = f"{src}→{dst}"
    base = 0.08 + rd.memory_traces * 0.035
    if token in rd.strongest_links:
        base += 0.22
    if rd.number >= 7 and token in ("I4→I5", "E3→I4", "E1→E3", "E2→I4"):
        base += 0.16
    return clamp(base, 0.04, 0.75)


# -----------------------------------------------------------------------------
# Static objects
# -----------------------------------------------------------------------------

floor = box(pos=vector(0, -0.08, 0), size=vector(12.5, 0.08, 6.3), color=vector(0.72, 0.76, 0.78))

neurons = {}
neuron_labels = {}
output_rings = {}

for name, pos in NEURON_POS.items():
    soma = sphere(pos=pos, radius=0.32, color=neuron_base_color(name), opacity=0.92)
    neurons[name] = soma

    # Small dendrite ring to make each node neuron-like.
    ring_obj = ring(pos=pos, axis=vector(0, 1, 0), radius=0.45, thickness=0.018, color=neuron_base_color(name), opacity=0.35)
    output_rings[name] = ring_obj

    lab = label(
        pos=pos + vector(0, -0.55, 0),
        text=name,
        height=12,
        box=False,
        opacity=0,
        color=color.black,
    )
    neuron_labels[name] = lab

# Output labels.
label(pos=NEURON_POS["E3"] + vector(0.0, -0.85, 0), text="output A", height=10, box=False, opacity=0, color=vector(0.1, 0.5, 0.1))
label(pos=NEURON_POS["I5"] + vector(0.0, -0.85, 0), text="output B", height=10, box=False, opacity=0, color=vector(0.6, 0.1, 0.1))

# Link curves.
link_curves = {}
for src, dst in LINKS:
    start = NEURON_POS[src]
    end = NEURON_POS[dst]
    mid = (start + end) * 0.5 + vector(0, 0.28, 0.12)
    c = curve(pos=[start, mid, end], radius=0.018, color=vector(0.38, 0.40, 0.42), opacity=0.35)
    link_curves[(src, dst)] = c

# Pattern input arrows.
input_a_arrow = arrow(pos=vector(-5.8, 2.25, -0.4), axis=vector(1.2, -0.35, 0.4), shaftwidth=0.055, color=vector(0.2, 0.55, 1.0))
input_b_arrow = arrow(pos=vector(-2.8, 2.75, -0.4), axis=vector(1.4, -0.45, 0.4), shaftwidth=0.055, color=vector(0.2, 0.55, 1.0))
dream_arrow = arrow(pos=vector(0.0, 3.0, -0.55), axis=vector(0, -0.95, 0.55), shaftwidth=0.055, color=vector(0.65, 0.35, 1.0))

# Output decision meter.
a_meter = box(pos=vector(-0.9, -0.02, -2.1), size=vector(0.35, 0.12, 0.35), color=vector(0.20, 0.80, 0.25))
b_meter = box(pos=vector(2.2, -0.02, -2.1), size=vector(0.35, 0.12, 0.35), color=vector(0.95, 0.35, 0.15))
label(pos=vector(-0.9, -0.35, -2.1), text="A output E3", height=10, box=False, opacity=0, color=color.black)
label(pos=vector(2.2, -0.35, -2.1), text="B output I5", height=10, box=False, opacity=0, color=color.black)

# Memory trace blocks.
memory_blocks = []
for i in range(6):
    block = box(pos=vector(-5.5 + i * 0.42, 0.25, 2.45), size=vector(0.30, 0.25, 0.30), color=vector(0.35, 0.55, 0.95), opacity=0.18)
    memory_blocks.append(block)
label(pos=vector(-6.05, 0.25, 2.45), text="memory", height=10, box=False, opacity=0, color=color.black)

# Round summary bars.
round_bars = []
for i, rd in enumerate(ROUNDS):
    x = -5.45 + i * 1.35
    b = box(pos=vector(x, 0.1, -2.95), size=vector(0.38, 0.2, 0.32), color=vector(0.35, 0.45, 0.80))
    round_bars.append(b)
    label(pos=vector(x, 0.02, -3.33), text=str(rd.number), height=10, box=False, opacity=0, color=color.black)

# Text labels.
title_label = label(pos=vector(0, 3.55, 0), text="", height=18, box=False, opacity=0, color=color.black)
state_label = label(pos=vector(0, 3.15, 0), text="", height=13, box=False, opacity=0, color=color.black)
data_label = label(pos=vector(-6.05, 2.55, 0), text="", height=11, box=False, opacity=0, color=color.black, align="left")
memory_label = label(pos=vector(4.35, 2.65, 0), text="", height=11, box=False, opacity=0, color=color.black, align="left")

# Spike particles and traveling synaptic pulse particles.
particles = []
for _ in range(140):
    p = sphere(pos=vector(0, -8, 0), radius=0.045, color=color.yellow, opacity=0.0, emissive=True)
    p.v = vector(0, 0, 0)
    p.life = 0.0
    particles.append(p)

pulse_particles = []
for _ in range(70):
    p = sphere(pos=vector(0, -8, 0), radius=0.055, color=color.yellow, opacity=0.0, emissive=True)
    p.life = 0.0
    p.start = vector(0, 0, 0)
    p.end = vector(0, 0, 0)
    p.u = 0.0
    p.speed = 0.0
    pulse_particles.append(p)

dream_cloud = []
for _ in range(45):
    p = sphere(pos=vector(0, -8, 0), radius=0.035, color=vector(0.65, 0.35, 1.0), opacity=0.0)
    p.v = vector(0, 0, 0)
    p.life = 0.0
    dream_cloud.append(p)

network_light = local_light(pos=vector(0, 2.5, 1.2), color=vector(0.3, 0.35, 0.7))


# -----------------------------------------------------------------------------
# Runtime controls
# -----------------------------------------------------------------------------

current_round_index = 0
round_time_ms = 0.0
paused = False
auto_cycle = True
show_particles = True
show_links = True
show_dream_glow = True
dt_ms_visual = 2.0
ROUND_MS = 700.0

last_spike_counts = [0] * 6


def reset_round():
    global round_time_ms, last_spike_counts
    round_time_ms = 0.0
    last_spike_counts = [0] * 6
    for p in particles + pulse_particles + dream_cloud:
        p.pos = vector(0, -8, 0)
        p.opacity = 0.0
        p.life = 0.0


def set_round(index: int):
    global current_round_index
    current_round_index = index % len(ROUNDS)
    reset_round()


def on_key(evt):
    global paused, auto_cycle, show_particles, show_links, show_dream_glow
    key = evt.key.lower()
    if key == " ":
        paused = not paused
    elif key == "n":
        set_round(current_round_index + 1)
    elif key == "b":
        set_round(current_round_index - 1)
    elif key == "r":
        reset_round()
    elif key == "a":
        auto_cycle = not auto_cycle
    elif key == "s":
        show_particles = not show_particles
    elif key == "l":
        show_links = not show_links
        for c in link_curves.values():
            c.visible = show_links
    elif key == "d":
        show_dream_glow = not show_dream_glow


scene.bind("keydown", on_key)


# -----------------------------------------------------------------------------
# Particle functions
# -----------------------------------------------------------------------------

def spawn_spike_burst(name: str):
    if not show_particles:
        return
    pos = NEURON_POS[name]
    col = neuron_active_color(name)
    for _ in range(8):
        for p in particles:
            if p.life <= 0:
                a = rng.random() * 2 * pi
                p.pos = pos + vector(0, 0, 0)
                p.v = vector(
                    rng.uniform(-0.025, 0.025),
                    rng.uniform(0.005, 0.040),
                    rng.uniform(-0.025, 0.025),
                ) + vector(0.03 * cos(a), 0.0, 0.03 * sin(a))
                p.radius = rng.uniform(0.025, 0.055)
                p.color = col
                p.opacity = 0.85
                p.life = rng.uniform(0.45, 0.95)
                break


def spawn_synaptic_pulse(src: str, dst: str):
    if not show_particles:
        return
    for p in pulse_particles:
        if p.life <= 0:
            p.start = NEURON_POS[src]
            p.end = NEURON_POS[dst]
            p.pos = p.start
            p.u = 0.0
            p.speed = rng.uniform(0.025, 0.050)
            p.color = vector(1.0, 0.85, 0.1) if src in EXCITATORY else vector(1.0, 0.25, 0.18)
            p.opacity = 0.9
            p.life = 1.0
            return


def spawn_dream_particle():
    if not show_dream_glow:
        return
    for p in dream_cloud:
        if p.life <= 0:
            a = rng.random() * 2 * pi
            r = rng.uniform(0.4, 2.9)
            p.pos = vector(r * cos(a) - 0.2, 1.45 + rng.uniform(-0.4, 0.7), r * sin(a) * 0.35)
            p.v = vector(rng.uniform(-0.004, 0.004), rng.uniform(0.000, 0.010), rng.uniform(-0.004, 0.004))
            p.radius = rng.uniform(0.025, 0.060)
            p.opacity = rng.uniform(0.12, 0.35)
            p.life = rng.uniform(1.5, 3.5)
            return


def update_all_particles():
    for p in particles:
        if p.life > 0:
            p.pos += p.v
            p.v *= 0.985
            p.life -= 0.025
            p.opacity *= 0.955
            if p.life <= 0:
                p.opacity = 0
                p.pos = vector(0, -8, 0)

    for p in pulse_particles:
        if p.life > 0:
            p.u += p.speed
            u = clamp(p.u, 0, 1)
            mid_lift = vector(0, 0.22 * sin(pi * u), 0.10 * sin(pi * u))
            p.pos = p.start * (1 - u) + p.end * u + mid_lift
            p.opacity = 0.90 * (1.0 - 0.45 * u)
            if p.u >= 1:
                p.life = 0
                p.opacity = 0
                p.pos = vector(0, -8, 0)

    for p in dream_cloud:
        if p.life > 0:
            p.pos += p.v
            p.life -= 0.018
            p.opacity *= 0.993
            if p.life <= 0:
                p.opacity = 0
                p.pos = vector(0, -8, 0)


# -----------------------------------------------------------------------------
# Main visual update
# -----------------------------------------------------------------------------

def update_scene():
    global last_spike_counts

    rd = ROUNDS[current_round_index]
    t = round_time_ms
    active_inputs = active_input_neurons(rd)

    # Update neurons.
    current_counts = []
    for i, name in enumerate(NEURON_NAMES):
        v, just_reset = voltage_for_neuron(rd, i, t)
        count_now = count_spikes_so_far(rd, i, t)
        current_counts.append(count_now)

        if count_now > last_spike_counts[i]:
            spawn_spike_burst(name)
            # Trigger outgoing pulses from this neuron on strongest outgoing links.
            for src, dst in LINKS:
                if src == name and link_strength_for_round(rd, (src, dst)) > 0.20:
                    spawn_synaptic_pulse(src, dst)

        # Color by voltage/role/input.
        near_threshold = clamp((v + 75.0) / 19.5, 0.0, 1.0)
        base_col = neuron_base_color(name)
        active_col = neuron_active_color(name)
        col = mix(base_col, active_col, near_threshold)

        if name in active_inputs:
            col = mix(col, vector(1.0, 0.85, 0.15), 0.22)
        if just_reset:
            col = vector(0.20, 0.20, 0.22)

        if name == OUTPUT_A:
            col = mix(col, vector(0.20, 0.80, 0.25), 0.18 + 0.15 * (rd.pattern == "A"))
        if name == OUTPUT_B:
            col = mix(col, vector(0.95, 0.35, 0.15), 0.18 + 0.15 * (rd.pattern == "B"))

        neurons[name].color = col
        neurons[name].radius = 0.31 + 0.07 * near_threshold + (0.08 if count_now > last_spike_counts[i] else 0.0)
        output_rings[name].color = col
        output_rings[name].opacity = 0.22 + 0.40 * near_threshold

        # Label includes spike count.
        neuron_labels[name].text = f"{name}: {count_now}/{rd.spike_counts[i]}"

    last_spike_counts = current_counts

    # Update learned links.
    for link, c in link_curves.items():
        strength = link_strength_for_round(rd, link)
        c.radius = 0.010 + 0.045 * strength
        src, dst = link
        if src in INHIBITORY:
            c.color = vector(1.0, 0.25, 0.18)
        else:
            c.color = vector(0.20, 0.55, 1.0)
        c.opacity = 0.12 + 0.72 * strength
        c.visible = show_links

    # Input arrows.
    input_a_arrow.visible = rd.pattern == "A"
    input_b_arrow.visible = rd.pattern == "B"
    dream_arrow.visible = rd.mode == "dream"

    input_a_arrow.color = vector(1.0, 0.85, 0.15) if rd.pattern == "A" else vector(0.2, 0.55, 1.0)
    input_b_arrow.color = vector(1.0, 0.85, 0.15) if rd.pattern == "B" else vector(0.2, 0.55, 1.0)
    dream_arrow.color = vector(0.70, 0.35, 1.0)

    # Dream cloud.
    if rd.mode == "dream" and rng.random() < 0.35:
        spawn_dream_particle()

    # Output meters.
    max_out = max(1, rd.a_out, rd.b_out)
    a_h = 0.18 + 1.4 * rd.a_out / max_out
    b_h = 0.18 + 1.4 * rd.b_out / max_out
    a_meter.size = vector(0.35, a_h, 0.35)
    b_meter.size = vector(0.35, b_h, 0.35)
    a_meter.pos = vector(-0.9, a_h / 2, -2.1)
    b_meter.pos = vector(2.2, b_h / 2, -2.1)

    # Memory blocks.
    for i, block in enumerate(memory_blocks):
        if i < rd.memory_traces:
            block.opacity = 0.85
            block.color = vector(0.35, 0.55, 0.95) if i < 4 else vector(0.65, 0.35, 1.0)
        else:
            block.opacity = 0.18
            block.color = vector(0.35, 0.55, 0.95)

    # Round bars: spike total, green/orange depending correct/dream.
    max_total = max(r.total_spikes for r in ROUNDS)
    for i, rdi in enumerate(ROUNDS):
        h = 0.15 + 1.45 * rdi.total_spikes / max_total
        round_bars[i].size = vector(0.38, h, 0.32)
        round_bars[i].pos = vector(-5.45 + i * 1.35, h / 2, -2.95)
        if i == current_round_index:
            round_bars[i].color = vector(1.0, 0.65, 0.12)
        elif rdi.mode == "dream":
            round_bars[i].color = vector(0.65, 0.35, 1.0)
        elif rdi.correct == "True":
            round_bars[i].color = vector(0.25, 0.75, 0.25)
        else:
            round_bars[i].color = vector(1.0, 0.25, 0.12)

    # Network light.
    if rd.mode == "dream":
        network_light.color = vector(0.55, 0.30, 1.0)
    else:
        network_light.color = vector(0.25, 0.40, 0.95)

    # Text.
    correctness = "correct" if rd.correct == "True" else ("dream replay" if rd.mode == "dream" else "not scored")
    title_label.text = f"Round {rd.number}: {rd.mode.upper()} {rd.pattern} — {correctness}"
    state_label.text = (
        f"t={t:5.1f} ms | total spikes {sum(current_counts)}/{rd.total_spikes} | "
        f"A_out E3={current_counts[3]}/{rd.a_out} | B_out I5={current_counts[5]}/{rd.b_out}"
    )

    if rd.mode == "dream":
        dream_line = f"dream similarity: {rd.dream_similarity:.3f} to stored B-like trace"
    else:
        dream_line = f"decision: {rd.decision}, expected: {rd.pattern}, correct: {rd.correct}"

    data_label.text = (
        f"{rd.description}\n\n"
        f"mode: {rd.mode}\n"
        f"pattern: {rd.pattern}\n"
        f"total spikes: {rd.total_spikes}\n"
        f"Hz per neuron: {rd.hz_per_neuron:.2f}\n"
        f"output A E3: {rd.a_out}\n"
        f"output B I5: {rd.b_out}\n"
        f"{dream_line}\n"
        f"E/I spike ratio: {rd.ei_ratio:.2f}\n"
        f"synaptic events: {rd.synaptic_events}\n"
        f"avg voltage: {rd.avg_voltage:.2f} mV"
    )

    memory_label.text = (
        f"Memory / learning\n"
        f"traces stored: {rd.memory_traces}\n"
        f"strongest links:\n{rd.strongest_links}\n\n"
        f"Final behavior:\n"
        f"compute accuracy = 6/6\n"
        f"dream similarity avg = 0.851\n\n"
        f"Interpretation:\n"
        f"compute rounds classify A/B;\n"
        f"dream rounds replay stored traces."
    )

    status.text = (
        f"Round {rd.number}/8 | {rd.mode} {rd.pattern} | total={sum(current_counts)}/{rd.total_spikes} | "
        f"A={current_counts[3]}/{rd.a_out} B={current_counts[5]}/{rd.b_out} | "
        f"auto={'on' if auto_cycle else 'off'}"
    )

    update_all_particles()


# -----------------------------------------------------------------------------
# Main loop
# -----------------------------------------------------------------------------

while True:
    rate(60)

    if not paused:
        round_time_ms += dt_ms_visual
        if auto_cycle and round_time_ms >= ROUND_MS:
            set_round(current_round_index + 1)
        elif not auto_cycle and round_time_ms > ROUND_MS:
            round_time_ms = ROUND_MS

    update_scene()
