#!/usr/bin/env python3
"""
VPython Targeted Neuron Spike Controller Visualization
-----------------------------------------------------

Visual simulation based on the terminal output from:
neuron_ai_rounds_simulation_targeted.py

Run:
    python vpython_targeted_neuron_spike_controller.py

Requires:
    pip install vpython

Controls:
    Space  - pause/resume
    N      - next round
    B      - previous round
    R      - restart current round
    A      - toggle auto-cycle
    S      - toggle spike particles
    T      - toggle target guide

This is a VPython visualization of the terminal output. It shows:
    - a leaky integrate-and-fire neuron charging toward threshold
    - spike events and reset to -75 mV
    - adaptation current building after spikes
    - Round 1 over-firing at 7 spikes
    - AI correction after Round 1
    - Rounds 2-8 hitting the 6-spike target
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
    cone,
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
# Round data copied from the terminal-output summary
# -----------------------------------------------------------------------------

@dataclass
class NeuronRound:
    number: int
    spikes: int
    error: int
    hz: float
    avg_v: float
    max_v: float
    base_i: float
    pulse_i: float
    threshold: float
    refractory: float
    pulse_interval: float
    pulse_duration: float
    adapt_add: float
    avg_interval: float
    note: str


ROUNDS = [
    NeuronRound(1, 7, -1, 14.00, -61.38, -54.01, 1.200, 0.800, -54.00, 4.00, 100.0, 250.0, 0.180, 57.67, "over target; controller reduces excitability"),
    NeuronRound(2, 6, 0, 12.00, -60.78, -53.86, 1.170, 0.775, -53.86, 4.24, 106.0, 248.5, 0.191, 66.50, "target achieved"),
    NeuronRound(3, 6, 0, 12.00, -60.79, -53.86, 1.170, 0.775, -53.86, 4.24, 106.0, 248.5, 0.193, 66.75, "target maintained"),
    NeuronRound(4, 6, 0, 12.00, -60.80, -53.86, 1.170, 0.775, -53.86, 4.24, 106.0, 248.5, 0.195, 67.05, "target maintained"),
    NeuronRound(5, 6, 0, 12.00, -60.81, -53.87, 1.170, 0.775, -53.86, 4.24, 106.0, 248.5, 0.197, 67.35, "target maintained"),
    NeuronRound(6, 6, 0, 12.00, -60.82, -53.86, 1.170, 0.775, -53.86, 4.24, 106.0, 248.5, 0.199, 67.60, "target maintained"),
    NeuronRound(7, 6, 0, 12.00, -60.83, -53.86, 1.170, 0.775, -53.86, 4.24, 106.0, 248.5, 0.201, 67.85, "target maintained"),
    NeuronRound(8, 6, 0, 12.00, -60.84, -53.86, 1.170, 0.775, -53.86, 4.24, 106.0, 248.5, 0.203, 68.15, "target maintained"),
]


# -----------------------------------------------------------------------------
# Simulation constants
# -----------------------------------------------------------------------------

TARGET_SPIKES = 6
ROUND_DURATION_MS = 500.0
V_REST = -70.0
V_RESET = -75.0
V_LOW = -78.0
V_HIGH = -52.0

rng = Random(12)


# -----------------------------------------------------------------------------
# VPython scene
# -----------------------------------------------------------------------------

scene = canvas(
    title="Targeted Neuron Spike Controller VPython Simulation",
    width=1280,
    height=760,
    background=vector(0.86, 0.90, 0.96),
    center=vector(0, 1.0, 0),
)
scene.camera.pos = vector(0, 6.5, 12.5)
scene.camera.axis = vector(0, -5.4, -12.5)

status = wtext(text="")
wtext(text="\nControls: Space pause/resume | N next | B previous | R restart | A auto-cycle | S spike particles | T target guide\n\n")


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def lerp(a: float, b: float, u: float) -> float:
    u = clamp(u, 0.0, 1.0)
    return a + (b - a) * u


def voltage_to_y(v: float) -> float:
    return lerp(0.25, 2.25, (v - V_LOW) / (V_HIGH - V_LOW))


def voltage_color(v: float, threshold: float):
    closeness = clamp((v - V_RESET) / (threshold - V_RESET), 0.0, 1.0)
    return vector(0.20 + 0.80 * closeness, 0.35 + 0.40 * closeness, 1.0 - 0.85 * closeness)


def spike_times_for_round(rd: NeuronRound):
    """
    Approximate spike times from the output. Round 1 has 7 spikes; later rounds
    have 6 spikes. The first spike is near 95-100 ms, then repeated intervals.
    """
    if rd.spikes <= 0:
        return []
    first = 95.0
    interval = rd.avg_interval if rd.avg_interval > 0 else 66.0
    return [first + i * interval for i in range(rd.spikes) if first + i * interval <= ROUND_DURATION_MS]


def voltage_at_time(rd: NeuronRound, t_ms: float):
    """
    Toy reconstruction of leaky integrate-and-fire behavior:
    voltage ramps toward threshold, spikes reset to -75 mV, then charges again.
    """
    spikes = spike_times_for_round(rd)
    previous_spike = None
    next_spike = None

    for st in spikes:
        if st <= t_ms:
            previous_spike = st
        elif st > t_ms and next_spike is None:
            next_spike = st

    if previous_spike is not None and 0 <= t_ms - previous_spike <= rd.refractory:
        return V_RESET, True

    if next_spike is None:
        # After final spike, slowly climbs but usually does not fire again.
        if previous_spike is None:
            phase_start = 0.0
        else:
            phase_start = previous_spike + rd.refractory
        u = clamp((t_ms - phase_start) / max(40.0, rd.avg_interval), 0.0, 1.0)
        v = lerp(V_RESET if previous_spike is not None else V_REST, rd.max_v - 0.6, 1.0 - (1.0 - u) ** 2)
        return v, False

    phase_start = 0.0 if previous_spike is None else previous_spike + rd.refractory
    start_v = V_REST if previous_spike is None else V_RESET
    span = max(1.0, next_spike - phase_start)
    u = clamp((t_ms - phase_start) / span, 0.0, 1.0)

    # Curved charging profile, plus small deterministic wave like the terminal model.
    v = lerp(start_v, rd.threshold + 0.02, 1.0 - (1.0 - u) ** 2.2)
    v += 0.35 * sin(t_ms / 60.0 * 2.0 * pi)
    return v, False


def count_spikes_so_far(rd: NeuronRound, t_ms: float) -> int:
    return sum(1 for st in spike_times_for_round(rd) if st <= t_ms)


def adaptation_at_time(rd: NeuronRound, t_ms: float):
    spikes = spike_times_for_round(rd)
    adapt = 0.0
    decay = 122.0 if rd.number >= 2 else 120.0
    for st in spikes:
        if st <= t_ms:
            adapt += rd.adapt_add
            adapt *= 2.718281828 ** (-(t_ms - st) / decay)
    # Approximation, enough for display.
    return clamp(adapt, 0.0, 0.55)


def input_current_at_time(rd: NeuronRound, t_ms: float):
    current = rd.base_i
    if t_ms >= 80.0:
        phase = (t_ms - 80.0) % rd.pulse_interval
        if phase <= rd.pulse_duration:
            current += rd.pulse_i
    current += 0.05 * sin(2.0 * pi * t_ms / 60.0)
    return current


# -----------------------------------------------------------------------------
# Static objects
# -----------------------------------------------------------------------------

floor = box(pos=vector(0, -0.05, 0), size=vector(13.5, 0.08, 6.5), color=vector(0.72, 0.76, 0.78))

# Neuron body and dendrites.
soma = sphere(pos=vector(-4.2, 1.1, 0), radius=0.55, color=vector(0.28, 0.45, 0.95), opacity=0.92)
nucleus = sphere(pos=soma.pos, radius=0.22, color=vector(0.15, 0.25, 0.65), opacity=0.9)
axon = cylinder(pos=vector(-3.75, 1.1, 0), axis=vector(6.2, 0, 0), radius=0.075, color=vector(0.32, 0.36, 0.42))

dendrites = []
for i in range(9):
    angle = -1.2 + i * 0.3
    start = vector(-4.65, 1.1, 0)
    end = start + vector(-1.0 - 0.45 * rng.random(), 0.9 * sin(angle), 0.75 * cos(angle))
    d = curve(pos=[start, end], radius=0.035, color=vector(0.28, 0.45, 0.95))
    dendrites.append(d)

# Axon nodes.
nodes = []
for i in range(7):
    x = -3.25 + i * 0.82
    node = ring(pos=vector(x, 1.1, 0), axis=vector(1, 0, 0), radius=0.17, thickness=0.025, color=vector(0.25, 0.55, 1.0))
    nodes.append(node)

terminal = cone(pos=vector(2.7, 1.1, 0), axis=vector(0.75, 0, 0), radius=0.26, color=vector(0.25, 0.55, 1.0), opacity=0.85)

# Voltage gauge.
gauge_back = box(pos=vector(4.55, 1.25, 0), size=vector(0.14, 2.25, 0.14), color=vector(0.28, 0.30, 0.33))
gauge_fill = box(pos=vector(4.55, 0.35, 0), size=vector(0.22, 0.15, 0.22), color=color.cyan)
threshold_marker = box(pos=vector(4.55, voltage_to_y(-54.0), 0), size=vector(0.45, 0.035, 0.35), color=color.red)
reset_marker = box(pos=vector(4.55, voltage_to_y(V_RESET), 0), size=vector(0.35, 0.03, 0.25), color=vector(0.25, 0.25, 0.25))
target_guide = box(pos=vector(5.25, 1.1, 0), size=vector(0.06, 1.4, 0.06), color=vector(0.1, 0.65, 0.1))

# Current/adaptation arrows.
input_arrow = arrow(pos=vector(-6.1, 0.15, -2.0), axis=vector(1.0, 0, 0), shaftwidth=0.05, color=vector(0.15, 0.45, 1.0))
pulse_arrow = arrow(pos=vector(-6.1, 0.42, -2.0), axis=vector(1.0, 0, 0), shaftwidth=0.05, color=vector(1.0, 0.55, 0.1))
adapt_arrow = arrow(pos=vector(1.4, 0.28, -2.0), axis=vector(-1.0, 0, 0), shaftwidth=0.05, color=vector(0.75, 0.25, 0.9))

# Spike counter blocks.
target_blocks = []
actual_blocks = []
for i in range(TARGET_SPIKES):
    target_blocks.append(box(pos=vector(-2.8 + i * 0.38, 2.85, -2.2), size=vector(0.28, 0.22, 0.22), color=vector(0.25, 0.75, 0.25)))
for i in range(8):
    actual_blocks.append(box(pos=vector(-2.8 + i * 0.38, 2.48, -2.2), size=vector(0.28, 0.22, 0.22), color=vector(0.55, 0.58, 0.62), opacity=0.25))

label(pos=vector(-3.9, 2.85, -2.2), text="target", height=10, box=False, opacity=0, color=color.black)
label(pos=vector(-3.9, 2.48, -2.2), text="actual", height=10, box=False, opacity=0, color=color.black)

# Round comparison bars.
bars = []
for i, rd in enumerate(ROUNDS):
    x = -5.4 + i * 1.45
    b = box(pos=vector(x, 0.1, 2.45), size=vector(0.42, 0.2, 0.35), color=vector(0.35, 0.45, 0.80))
    bars.append(b)
    label(pos=vector(x, 0.02, 2.85), text=str(rd.number), height=11, box=False, opacity=0, color=color.black)

# Labels.
title_label = label(pos=vector(0, 3.55, 0), text="", height=18, box=False, opacity=0, color=color.black)
state_label = label(pos=vector(0, 3.15, 0), text="", height=13, box=False, opacity=0, color=color.black)
data_label = label(pos=vector(-6.3, 2.15, 0), text="", height=11, box=False, opacity=0, color=color.black, align="left")
controller_label = label(pos=vector(3.0, 2.35, 0), text="", height=11, box=False, opacity=0, color=color.black, align="left")

# Light for spikes.
spike_light = local_light(pos=vector(-1.0, 1.7, 0), color=vector(0.3, 0.5, 1.0))

# Particle pool.
particles = []
for _ in range(90):
    p = sphere(pos=vector(0, -8, 0), radius=0.045, color=color.yellow, opacity=0.0, emissive=True)
    p.v = vector(0, 0, 0)
    p.life = 0.0
    particles.append(p)

# Moving pulse on axon.
pulse_sphere = sphere(pos=soma.pos, radius=0.12, color=color.yellow, opacity=0.0, emissive=True)


# -----------------------------------------------------------------------------
# Runtime state
# -----------------------------------------------------------------------------

current_round_index = 0
round_time_ms = 0.0
paused = False
auto_cycle = True
show_particles = True
show_target_guide = True
dt_ms_visual = 1.5
cycle_ms = ROUND_DURATION_MS


def reset_round():
    global round_time_ms
    round_time_ms = 0.0
    for p in particles:
        p.pos = vector(0, -8, 0)
        p.opacity = 0.0
        p.life = 0.0
    pulse_sphere.opacity = 0.0


def set_round(index: int):
    global current_round_index
    current_round_index = index % len(ROUNDS)
    reset_round()


def on_key(evt):
    global paused, auto_cycle, show_particles, show_target_guide
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
    elif key == "t":
        show_target_guide = not show_target_guide
        target_guide.visible = show_target_guide


scene.bind("keydown", on_key)


# -----------------------------------------------------------------------------
# Animation functions
# -----------------------------------------------------------------------------

def spawn_spike_burst(pos, intensity=1.0):
    if not show_particles:
        return
    for _ in range(12):
        for p in particles:
            if p.life <= 0:
                a = rng.random() * 2 * pi
                p.pos = pos
                p.v = vector(0.03 + rng.random() * 0.05, rng.uniform(-0.018, 0.018), rng.uniform(-0.025, 0.025))
                p.v += vector(0, 0.025 * sin(a), 0.025 * cos(a))
                p.radius = rng.uniform(0.025, 0.06)
                p.color = vector(1.0, 0.85, 0.15)
                p.opacity = 0.85 * intensity
                p.life = rng.uniform(0.35, 0.9)
                break


def update_particles():
    for p in particles:
        if p.life > 0:
            p.pos += p.v
            p.v *= 0.985
            p.life -= 0.025
            p.opacity *= 0.955
            if p.life <= 0:
                p.opacity = 0
                p.pos = vector(0, -8, 0)


last_spike_count = 0


def update_scene():
    global last_spike_count

    rd = ROUNDS[current_round_index]
    t = round_time_ms
    v, in_refractory = voltage_at_time(rd, t)
    current = input_current_at_time(rd, t)
    adapt = adaptation_at_time(rd, t)
    spikes_now = count_spikes_so_far(rd, t)

    # Spike burst when count increases.
    if spikes_now > last_spike_count:
        spawn_spike_burst(vector(-3.2, 1.1, 0), 1.0)
    last_spike_count = spikes_now

    # Soma color follows voltage.
    soma.color = voltage_color(v, rd.threshold)
    nucleus.color = vector(0.10, 0.18, 0.50) if not in_refractory else vector(0.18, 0.18, 0.18)

    # Gauge.
    y = voltage_to_y(v)
    gauge_fill.pos = vector(4.55, lerp(0.25, y, 0.5), 0)
    gauge_fill.size = vector(0.22, max(0.05, y - 0.25), 0.22)
    gauge_fill.color = voltage_color(v, rd.threshold)
    threshold_marker.pos = vector(4.55, voltage_to_y(rd.threshold), 0)

    # Axon pulse: show most recent spike propagating.
    spike_times = spike_times_for_round(rd)
    recent = None
    for st in spike_times:
        if 0 <= t - st <= 70:
            recent = st
    if recent is not None:
        u = clamp((t - recent) / 70.0, 0.0, 1.0)
        pulse_sphere.pos = vector(lerp(-3.6, 2.6, u), 1.1, 0)
        pulse_sphere.opacity = 0.95 * (1.0 - u * 0.25)
        pulse_sphere.color = color.yellow
        spike_light.pos = pulse_sphere.pos + vector(0, 0.5, 0)
        spike_light.color = vector(1.0, 0.85, 0.25)
    else:
        pulse_sphere.opacity = 0.0
        spike_light.color = vector(0.15, 0.25, 0.45)

    # Node colors.
    for i, node in enumerate(nodes):
        node.color = vector(0.25, 0.55, 1.0)
        if recent is not None:
            node_x = node.pos.x
            if abs(pulse_sphere.pos.x - node_x) < 0.35:
                node.color = color.yellow

    # Input/adaptation arrows.
    input_arrow.axis = vector(0.55 + 0.55 * clamp(rd.base_i / 1.25, 0, 1.4), 0, 0)
    pulse_arrow.axis = vector(0.55 + 0.75 * clamp(rd.pulse_i / 0.9, 0, 1.4), 0, 0)
    adapt_arrow.axis = vector(-(0.45 + 1.0 * clamp(adapt / 0.45, 0, 1.2)), 0, 0)

    # Actual spike blocks.
    for i, block in enumerate(actual_blocks):
        if i < spikes_now:
            block.opacity = 1.0
            if i < TARGET_SPIKES:
                block.color = vector(0.25, 0.75, 0.25)
            else:
                block.color = vector(1.0, 0.20, 0.10)
        else:
            block.opacity = 0.25
            block.color = vector(0.55, 0.58, 0.62)

    # Comparison bars: height = spike count, green when exact target.
    for i, b in enumerate(bars):
        rdi = ROUNDS[i]
        h = 0.20 + 1.2 * (rdi.spikes / 8.0)
        b.size = vector(0.42, h, 0.35)
        b.pos = vector(-5.4 + i * 1.45, h / 2, 2.45)
        if rdi.number == rd.number:
            b.color = vector(1.0, 0.65, 0.12)
        elif rdi.spikes == TARGET_SPIKES:
            b.color = vector(0.25, 0.75, 0.25)
        else:
            b.color = vector(1.0, 0.25, 0.12)

    # Labels.
    title_label.text = f"Round {rd.number}: {rd.spikes} spikes / target {TARGET_SPIKES} — {rd.note}"
    state_label.text = (
        f"t={t:5.1f} ms | V={v:6.2f} mV | threshold={rd.threshold:.2f} mV | "
        f"spikes={spikes_now}/{TARGET_SPIKES} | {'refractory reset' if in_refractory else 'charging'}"
    )

    data_label.text = (
        f"Round summary\n"
        f"spikes: {rd.spikes}\n"
        f"error target-observed: {rd.error:+d}\n"
        f"firing rate: {rd.hz:.2f} Hz\n"
        f"average V: {rd.avg_v:.2f} mV\n"
        f"max V: {rd.max_v:.2f} mV\n"
        f"base current: {rd.base_i:.3f} nA\n"
        f"pulse current: {rd.pulse_i:.3f} nA\n"
        f"pulse interval: {rd.pulse_interval:.1f} ms\n"
        f"pulse duration: {rd.pulse_duration:.1f} ms\n"
        f"refractory: {rd.refractory:.2f} ms\n"
        f"adaptation add: {rd.adapt_add:.3f} nA"
    )

    if rd.number == 1:
        controller_text = (
            "AI controller after Round 1\n"
            "Observed: 7 spikes\n"
            "Target: 6 spikes\n"
            "Action: reduce excitability\n\n"
            "base current lowered\n"
            "pulse current lowered\n"
            "threshold raised slightly\n"
            "refractory lengthened\n"
            "pulse interval increased\n"
            "adaptation strengthened"
        )
    else:
        controller_text = (
            "AI controller status\n"
            "Target achieved\n"
            "Main parameters preserved\n\n"
            "Small stabilization:\n"
            "adaptation_add slowly increases\n"
            "spike count remains at 6\n"
            "firing rate remains 12 Hz"
        )
    controller_label.text = controller_text

    status.text = (
        f"Round {rd.number}/8 | spikes={spikes_now}/{TARGET_SPIKES} | "
        f"V={v:.2f} mV | I={current:.2f} nA | adapt={adapt:.2f} nA | "
        f"auto={'on' if auto_cycle else 'off'}"
    )

    update_particles()


# -----------------------------------------------------------------------------
# Main loop
# -----------------------------------------------------------------------------

while True:
    rate(60)

    if not paused:
        round_time_ms += dt_ms_visual
        if auto_cycle and round_time_ms >= cycle_ms:
            set_round(current_round_index + 1)
            last_spike_count = 0
        elif not auto_cycle and round_time_ms > cycle_ms:
            round_time_ms = cycle_ms

    update_scene()
