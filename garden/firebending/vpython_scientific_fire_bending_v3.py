#!/usr/bin/env python3
"""
VPython Scientific Fire Bending v3 Visualization
------------------------------------------------

Visual simulation based on the terminal output from:
scientific_fire_bending_ai_rounds_v3.py

Run:
    python vpython_scientific_fire_bending_v3.py

Requires:
    pip install vpython

Controls:
    Space  - pause/resume
    N      - next round
    B      - previous round
    R      - restart current round
    A      - toggle auto-cycle
    S      - toggle smoke/soot particles
    T      - toggle target heat-guide line

This is a VPython visualization of the terminal scientific model output. It is
not CFD and does not solve real combustion chemistry. It maps the terminal
measurements into a visual scene: local heat fields, flame-capable modes,
sub-ignition failures, blue long-range thermal contact, target remaining cold,
and final comparison bars.
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
# Terminal-output based round data
# -----------------------------------------------------------------------------

@dataclass
class RoundData:
    number: int
    mode: str
    result: str
    peak_k: float
    avg_k: float
    target_k: float
    dose: float
    energy_j: float
    loss_j: float
    fuel_burned: float
    o2_used: float
    peak_rate: float
    reach: float
    efficiency: float
    stability: float
    precision: float
    profile: str


ROUNDS = [
    RoundData(1, "jab", "sub-ignition heating", 601.2, 300.8, 300.0, 0.0, 2.0, 3483.6, 0.002, 0.004, 0.045, 0.00, 0.001, 0.02, -0.00, "brief_warm"),
    RoundData(2, "stream", "localized controlled burn", 1046.8, 304.2, 300.0, 0.0, 2564.1, 20866.5, 2.465, 4.931, 1.209, 28.00, 0.109, 5.34, -0.01, "stream_burn"),
    RoundData(3, "wave", "localized controlled burn", 1679.1, 309.4, 300.0, 0.0, 14483.8, 42835.1, 13.927, 27.853, 1.850, 9.06, 0.253, 22.22, -0.38, "wave_burn"),
    RoundData(4, "shield", "sub-ignition heating", 384.6, 301.6, 300.0, 0.0, 0.0, 7701.7, 0.000, 0.000, 0.000, 0.00, 0.000, 0.00, 0.00, "shield_warm"),
    RoundData(5, "whip", "sub-ignition heating", 573.7, 303.9, 300.0, 0.0, 0.0, 15155.4, 0.000, 0.000, 0.000, 0.00, 0.000, 0.00, 0.00, "whip_warm"),
    RoundData(6, "sparks", "sub-ignition heating", 336.0, 300.1, 300.0, 0.0, 0.0, 1209.9, 0.000, 0.000, 0.000, 0.00, 0.000, 0.00, 0.00, "sparks_warm"),
    RoundData(7, "blue", "long-range thermal contact", 2100.0, 307.4, 300.0, 0.0, 9739.7, 34150.8, 9.365, 18.730, 1.929, 37.00, 0.222, 40.13, -0.06, "blue_contact"),
    RoundData(8, "obstacle", "localized controlled burn", 756.5, 303.9, 300.0, 0.0, 1062.1, 14880.7, 1.021, 2.043, 0.568, 18.00, 0.067, 3.35, -0.00, "obstacle_burn"),
]


# -----------------------------------------------------------------------------
# Scene setup
# -----------------------------------------------------------------------------

scene = canvas(
    title="Scientific Fire Bending v3 VPython Visualization",
    width=1280,
    height=760,
    background=vector(0.86, 0.90, 0.96),
    center=vector(0, 1.1, 0),
)
scene.camera.pos = vector(0, 6.5, 14)
scene.camera.axis = vector(0, -4.8, -14)

rng = Random(52)

status = wtext(text="")
wtext(text="\nControls: Space pause/resume | N next | B previous | R restart | A auto-cycle | S smoke | T target guide\n\n")


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def lerp(a: float, b: float, u: float) -> float:
    u = clamp(u, 0.0, 1.0)
    return a + (b - a) * u


def lerp_vec(a, b, u: float):
    u = clamp(u, 0.0, 1.0)
    return a * (1.0 - u) + b * u


def heat_color(temp_k: float, mode: str):
    """Map terminal temperature to visual color."""
    hot = clamp((temp_k - 300.0) / 1500.0, 0.0, 1.0)
    if mode == "blue":
        return lerp_vec(vector(0.40, 0.70, 1.0), vector(0.85, 0.95, 1.0), hot)
    if temp_k < 590:
        return lerp_vec(vector(0.65, 0.70, 0.75), vector(1.0, 0.62, 0.18), hot)
    return lerp_vec(vector(1.0, 0.55, 0.08), vector(1.0, 0.92, 0.25), hot)


def terminal_temperature(rd: RoundData, t: float) -> float:
    """
    Approximate each round's time behavior from the printed v3 frames.
    """
    if rd.profile == "brief_warm":
        if t < 1.0:
            return lerp(310, rd.peak_k, t / 1.0)
        return lerp(rd.peak_k, 326.9, (t - 1.0) / 4.5)

    if rd.profile == "stream_burn":
        if t < 2.4:
            return lerp(320, 650, t / 2.4)
        if t < 3.0:
            return lerp(650, rd.peak_k, (t - 2.4) / 0.6)
        return lerp(rd.peak_k, 575.7, (t - 3.0) / 2.5)

    if rd.profile == "wave_burn":
        if t < 1.0:
            return lerp(316, 778, t / 1.0)
        if t < 2.0:
            return lerp(778, rd.peak_k, (t - 1.0) / 1.0)
        return lerp(rd.peak_k, 827.5, (t - 2.0) / 3.5)

    if rd.profile == "shield_warm":
        if t < 0.5:
            return lerp(305, 372, t / 0.5)
        return lerp(372, 384.6, 0.5 + 0.5 * sin(t * 1.6))

    if rd.profile == "whip_warm":
        if t < 0.7:
            return lerp(321, rd.peak_k, t / 0.7)
        return lerp(rd.peak_k, 417.7, (t - 0.7) / 4.8)

    if rd.profile == "sparks_warm":
        if t < 0.9:
            return lerp(302, rd.peak_k, t / 0.9)
        return lerp(rd.peak_k, 302, (t - 0.9) / 4.6)

    if rd.profile == "blue_contact":
        if t < 0.5:
            return lerp(335, 728, t / 0.5)
        if t < 2.5:
            return lerp(728, rd.peak_k, (t - 0.5) / 2.0)
        return lerp(rd.peak_k, 660.9, (t - 2.5) / 3.0)

    if rd.profile == "obstacle_burn":
        if t < 2.0:
            return lerp(325, 672, t / 2.0)
        if t < 3.0:
            return lerp(672, rd.peak_k, (t - 2.0) / 1.0)
        return lerp(rd.peak_k, 469.0, (t - 3.0) / 2.5)

    return 300.0


def hot_fraction(rd: RoundData, temp_k: float) -> float:
    base = clamp((temp_k - 590.0) / 900.0, 0.0, 1.0)
    if rd.mode in ("shield", "sparks", "jab", "whip") and rd.energy_j <= 3.0:
        base *= 0.35
    return base


def reach_at_time(rd: RoundData, t: float, temp_k: float) -> float:
    if temp_k < 590:
        return 0.0

    if rd.profile == "stream_burn":
        return 28.0 * clamp((t - 2.2) / 0.8, 0, 1) * clamp((5.7 - t) / 0.8, 0, 1)
    if rd.profile == "wave_burn":
        return 9.1 * clamp((t - 0.8) / 1.2, 0, 1)
    if rd.profile == "blue_contact":
        return 37.0 * clamp((t - 0.4) / 1.2, 0, 1)
    if rd.profile == "obstacle_burn":
        return 18.0 * clamp((t - 1.8) / 1.2, 0, 1) * clamp((4.7 - t) / 1.0, 0, 1)
    return rd.reach


def mode_description(rd: RoundData) -> str:
    if rd.mode == "jab":
        return "short impulse; nearly reaches extinction threshold"
    if rd.mode == "stream":
        return "focused local jet; burns briefly then cools"
    if rd.mode == "wave":
        return "broad plume; best heat efficiency"
    if rd.mode == "shield":
        return "defensive shell; warm but sub-ignition"
    if rd.mode == "whip":
        return "oscillating filament; too cool to ignite"
    if rd.mode == "sparks":
        return "scattered embers; weak heating"
    if rd.mode == "blue":
        return "oxygen-rich long-range thermal contact"
    if rd.mode == "obstacle":
        return "fuel-coupled path; localized combustion"
    return ""


# -----------------------------------------------------------------------------
# Static scene objects
# -----------------------------------------------------------------------------

floor = box(pos=vector(0, -0.06, 0), size=vector(14.5, 0.08, 6.4), color=vector(0.72, 0.76, 0.78))

# Coordinate mapping: bender at x=-5.4, target at x=5.4.
bender_pos = vector(-5.4, 0.25, 0)
target_pos = vector(5.4, 0.25, 0)

bender = sphere(pos=bender_pos + vector(0, 0.45, 0), radius=0.34, color=vector(0.15, 0.18, 0.22))
bender_body = cylinder(pos=bender_pos, axis=vector(0, 0.55, 0), radius=0.20, color=vector(0.18, 0.20, 0.26))
left_arm = cylinder(pos=bender_pos + vector(0.10, 0.55, 0), axis=vector(0.80, 0.05, 0), radius=0.045, color=vector(0.18, 0.20, 0.26))
right_arm = cylinder(pos=bender_pos + vector(0.10, 0.42, 0), axis=vector(0.75, -0.03, 0), radius=0.045, color=vector(0.18, 0.20, 0.26))

target_base = cylinder(pos=target_pos, axis=vector(0, 0.72, 0), radius=0.12, color=vector(0.24, 0.24, 0.24))
target_ring = ring(pos=target_pos + vector(0, 0.85, 0), axis=vector(0, 0, 1), radius=0.36, thickness=0.035, color=color.red)
target_core = sphere(pos=target_pos + vector(0, 0.85, 0), radius=0.08, color=color.red, emissive=True)

guide_line = curve(pos=[bender_pos + vector(0.5, 0.55, 0), target_pos + vector(0, 0.85, 0)], radius=0.01, color=vector(0.5, 0.5, 0.5), visible=True)

# Flame/heat primitives.
main_plume = cone(pos=bender_pos + vector(0.45, 0.55, 0), axis=vector(1, 0, 0), radius=0.25, color=color.orange, opacity=0.35)
inner_plume = cone(pos=bender_pos + vector(0.45, 0.55, 0), axis=vector(0.8, 0, 0), radius=0.12, color=color.yellow, opacity=0.55)
blue_core = cone(pos=bender_pos + vector(0.45, 0.55, 0), axis=vector(0.7, 0, 0), radius=0.07, color=vector(0.25, 0.55, 1.0), opacity=0.65)
pilot_sphere = sphere(pos=bender_pos + vector(0.6, 0.55, 0), radius=0.10, color=color.yellow, opacity=0.6, emissive=True)
heat_light = local_light(pos=bender_pos + vector(1.2, 0.9, 0), color=vector(1.0, 0.55, 0.18))

shield_ring = ring(pos=bender_pos + vector(0, 0.52, 0), axis=vector(0, 1, 0), radius=0.95, thickness=0.04, color=color.orange, opacity=0.35, visible=False)

whip_curve = curve(radius=0.055, color=color.orange, visible=False)
wave_ribs = []
for _ in range(9):
    c = curve(radius=0.025, color=color.orange, visible=False)
    wave_ribs.append(c)

obstacle_posts = []
for z in [-1.2, -0.8, -0.4, 0.0, 0.4, 0.8, 1.2]:
    post = cylinder(pos=vector(0.2, 0.0, z), axis=vector(0, 0.8, 0), radius=0.045, color=vector(0.48, 0.28, 0.10), visible=False)
    obstacle_posts.append(post)

fuel_arrow = arrow(pos=vector(-6.2, 0.18, -2.1), axis=vector(1.2, 0, 0), shaftwidth=0.05, color=vector(1.0, 0.45, 0.08))
oxygen_arrow = arrow(pos=vector(-6.2, 0.42, -2.1), axis=vector(1.2, 0, 0), shaftwidth=0.05, color=vector(0.25, 0.55, 1.0))
loss_arrow = arrow(pos=vector(0, 1.8, 2.1), axis=vector(0, 0.7, 0), shaftwidth=0.05, color=vector(0.55, 0.62, 0.68))

# Data labels.
title_label = label(pos=vector(0, 3.3, 0), text="", height=18, box=False, opacity=0, color=color.black)
state_label = label(pos=vector(0, 2.85, 0), text="", height=13, box=False, opacity=0, color=color.black)
data_label = label(pos=vector(-6.8, 2.65, 0), text="", height=11, box=False, opacity=0, color=color.black, align="left")
legend_label = label(
    pos=vector(4.2, 2.65, 0),
    text="v3 behavior\nLocal heat only\nTarget remains cold\nBlue = long reach\nWave = best efficiency",
    height=11,
    box=False,
    opacity=0,
    color=color.black,
    align="left",
)

# Final comparison bars.
bars = []
bar_text = []
max_energy = max(rd.energy_j for rd in ROUNDS)
max_stability = max(rd.stability for rd in ROUNDS)
for i, rd in enumerate(ROUNDS):
    x = -5.4 + i * 1.55
    bar = box(pos=vector(x, 0.1, -2.8), size=vector(0.42, 0.2, 0.36), color=vector(0.35, 0.45, 0.80))
    bars.append(bar)
    lab = label(pos=vector(x, 0.02, -3.18), text=str(rd.number), height=11, box=False, opacity=0, color=color.black)
    bar_text.append(lab)

# Heat particles.
particles = []
for _ in range(170):
    p = sphere(pos=vector(0, -8, 0), radius=0.035, color=color.orange, opacity=0.0, emissive=True)
    p.v = vector(0, 0, 0)
    p.life = 0.0
    p.kind = "heat"
    particles.append(p)

smoke_particles = []
for _ in range(80):
    p = sphere(pos=vector(0, -8, 0), radius=0.045, color=vector(0.25, 0.25, 0.25), opacity=0.0)
    p.v = vector(0, 0, 0)
    p.life = 0.0
    smoke_particles.append(p)


# -----------------------------------------------------------------------------
# State and controls
# -----------------------------------------------------------------------------

current_round = 0
round_time = 0.0
paused = False
auto_cycle = True
show_smoke = True
show_target_guide = True
dt = 0.015
cycle_seconds = 5.8


def reset_round() -> None:
    global round_time
    round_time = 0.0
    for p in particles + smoke_particles:
        p.pos = vector(0, -8, 0)
        p.opacity = 0.0
        p.life = 0.0


def set_round(index: int) -> None:
    global current_round
    current_round = index % len(ROUNDS)
    reset_round()


def on_key(evt) -> None:
    global paused, auto_cycle, show_smoke, show_target_guide
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
        show_smoke = not show_smoke
    elif key == "t":
        show_target_guide = not show_target_guide
        guide_line.visible = show_target_guide


scene.bind("keydown", on_key)


# -----------------------------------------------------------------------------
# Particle logic
# -----------------------------------------------------------------------------

def spawn_particle(pos, vel, col, radius, life, opacity=0.75):
    for p in particles:
        if p.life <= 0:
            p.pos = pos
            p.v = vel
            p.color = col
            p.radius = radius
            p.life = life
            p.opacity = opacity
            return


def spawn_smoke(pos, amount: float):
    if not show_smoke or amount <= 0:
        return
    if rng.random() > clamp(amount, 0.0, 0.9):
        return
    for p in smoke_particles:
        if p.life <= 0:
            p.pos = pos
            p.v = vector(rng.uniform(-0.012, 0.012), rng.uniform(0.010, 0.025), rng.uniform(-0.012, 0.012))
            p.radius = rng.uniform(0.035, 0.075)
            p.opacity = clamp(0.08 + amount * 0.25, 0.04, 0.35)
            p.color = vector(0.22, 0.22, 0.22)
            p.life = rng.uniform(1.5, 3.0)
            return


def update_particles():
    for p in particles:
        if p.life > 0:
            p.pos += p.v
            p.v *= 0.992
            p.v.y += 0.0006
            p.life -= dt
            p.opacity *= 0.982
            if p.life <= 0:
                p.opacity = 0
                p.pos = vector(0, -8, 0)

    for p in smoke_particles:
        if p.life > 0:
            p.pos += p.v
            p.v.y += 0.0004
            p.life -= dt
            p.opacity *= 0.990
            if p.life <= 0:
                p.opacity = 0
                p.pos = vector(0, -8, 0)


# -----------------------------------------------------------------------------
# Visual update
# -----------------------------------------------------------------------------

def update_mode_geometry(rd: RoundData, temp_k: float, reach: float, hf: float) -> None:
    # Hide optional geometries by default.
    shield_ring.visible = False
    whip_curve.visible = False
    for c in wave_ribs:
        c.visible = False
    for post in obstacle_posts:
        post.visible = (rd.mode == "obstacle")

    col = heat_color(temp_k, rd.mode)
    length = max(0.25, reach / 37.0 * 9.8)
    if rd.mode in ("jab", "shield", "whip", "sparks") and rd.energy_j < 5:
        length = clamp((temp_k - 300) / 350 * 1.3, 0.15, 1.4)

    plume_visible = rd.mode not in ("shield", "whip", "wave") or rd.energy_j > 1
    main_plume.visible = plume_visible
    inner_plume.visible = plume_visible
    blue_core.visible = plume_visible and rd.mode == "blue"

    radius = 0.14 + hf * 0.42
    if rd.mode == "blue":
        radius *= 0.45
    if rd.mode == "stream":
        radius *= 0.65
    if rd.mode == "wave":
        radius *= 1.25

    flicker = 1.0 + 0.05 * sin(round_time * 17.0) + 0.03 * sin(round_time * 29.0)

    main_plume.pos = bender_pos + vector(0.55, 0.56, 0)
    main_plume.axis = vector(length * flicker, 0.05 * sin(round_time * 4.0), 0.0)
    main_plume.radius = radius
    main_plume.color = col
    main_plume.opacity = 0.15 + 0.36 * hf

    inner_plume.pos = bender_pos + vector(0.55, 0.56, 0)
    inner_plume.axis = vector(length * 0.72, 0, 0)
    inner_plume.radius = radius * 0.45
    inner_plume.color = vector(1.0, 0.85, 0.20) if rd.mode != "blue" else vector(0.45, 0.75, 1.0)
    inner_plume.opacity = 0.18 + 0.48 * hf

    blue_core.pos = bender_pos + vector(0.55, 0.56, 0)
    blue_core.axis = vector(length * 0.95, 0, 0)
    blue_core.radius = max(0.035, radius * 0.35)
    blue_core.color = vector(0.25, 0.55, 1.0)
    blue_core.opacity = 0.65 * hf

    pilot_sphere.pos = bender_pos + vector(0.70, 0.58 + 0.03 * sin(round_time * 9), 0)
    pilot_sphere.radius = 0.05 + 0.18 * clamp((temp_k - 300) / 900, 0, 1)
    pilot_sphere.color = col
    pilot_sphere.opacity = 0.20 + 0.50 * clamp((temp_k - 300) / 800, 0, 1)

    heat_light.pos = bender_pos + vector(min(length, 5.5), 0.9, 0)
    heat_light.color = col * (0.35 + 0.65 * hf)

    if rd.mode == "shield":
        shield_ring.visible = True
        shield_ring.radius = 0.85 + 0.12 * sin(round_time * 2.0)
        shield_ring.color = col
        shield_ring.opacity = 0.16 + 0.30 * clamp((temp_k - 300) / 400, 0, 1)

    if rd.mode == "whip":
        whip_curve.visible = True
        pts = []
        amp = 0.55 + 0.25 * hf
        for i in range(35):
            u = i / 34
            x = bender_pos.x + 0.4 + u * 5.1
            y = 0.55 + amp * sin(u * 4.2 * pi + round_time * 2.5) * 0.35
            z = amp * sin(u * 2.2 * pi + round_time * 1.8) * 0.18
            pts.append(vector(x, y, z))
        whip_curve.clear()
        whip_curve.append(pts)
        whip_curve.color = col
        whip_curve.radius = 0.025 + 0.045 * hf

    if rd.mode == "wave":
        fan = clamp((temp_k - 500) / 900, 0, 1)
        for i, c in enumerate(wave_ribs):
            c.visible = True
            c.clear()
            angle = -0.75 + i * (1.5 / max(1, len(wave_ribs) - 1))
            pts = []
            for j in range(16):
                u = j / 15
                x = bender_pos.x + 0.5 + u * (1.2 + 2.2 * fan)
                y = 0.55 + 0.10 * sin(u * pi + round_time * 3)
                z = angle * u * (0.6 + 1.6 * fan)
                pts.append(vector(x, y, z))
            c.append(pts)
            c.color = col
            c.radius = 0.018 + 0.025 * fan


def update_bars(rd_active: RoundData) -> None:
    for i, rd in enumerate(ROUNDS):
        energy_h = 0.12 + 1.35 * (rd.energy_j / max_energy if max_energy > 0 else 0)
        stab_h = 0.12 + 1.05 * (rd.stability / max_stability if max_stability > 0 else 0)
        h = max(energy_h, stab_h)
        bars[i].size = vector(0.42, h, 0.36)
        bars[i].pos = vector(-5.4 + i * 1.55, h / 2, -2.8)

        if rd.number == rd_active.number:
            bars[i].color = vector(1.0, 0.62, 0.12)
        elif rd.mode == "blue":
            bars[i].color = vector(0.25, 0.55, 1.0)
        elif rd.result.startswith("sub"):
            bars[i].color = vector(0.55, 0.60, 0.68)
        else:
            bars[i].color = vector(0.95, 0.38, 0.12)


def spawn_mode_particles(rd: RoundData, temp_k: float, reach: float, hf: float) -> None:
    if hf <= 0.01 and temp_k < 340:
        return

    col = heat_color(temp_k, rd.mode)
    intensity = clamp((temp_k - 300.0) / 1200.0, 0.0, 1.0)

    if rd.mode == "sparks":
        spawn_count = 3 if rng.random() < 0.75 else 1
    elif rd.mode == "blue":
        spawn_count = 4 if hf > 0.2 else 1
    elif rd.mode == "wave":
        spawn_count = 3 if hf > 0.15 else 1
    else:
        spawn_count = 2 if hf > 0.1 else 1

    for _ in range(spawn_count):
        if rng.random() > 0.25 + 0.70 * max(intensity, hf):
            continue

        if rd.mode == "wave":
            x = bender_pos.x + rng.uniform(0.5, 2.9)
            z = rng.uniform(-1.5, 1.5)
            y = 0.55 + rng.uniform(-0.12, 0.25)
            v = vector(rng.uniform(0.015, 0.045), rng.uniform(0.000, 0.010), rng.uniform(-0.012, 0.012))
        elif rd.mode == "shield":
            a = rng.random() * 2 * pi
            x = bender_pos.x + cos(a) * 0.9
            z = sin(a) * 0.9
            y = 0.55
            v = vector(cos(a) * 0.015, rng.uniform(0.004, 0.018), sin(a) * 0.015)
        elif rd.mode == "whip":
            u = rng.random()
            x = bender_pos.x + 0.5 + u * 4.8
            z = sin(u * 4 * pi + round_time) * 0.35
            y = 0.55 + sin(u * 3 * pi + round_time * 2) * 0.15
            v = vector(0.025, rng.uniform(0.003, 0.014), rng.uniform(-0.010, 0.010))
        elif rd.mode == "sparks":
            x = bender_pos.x + rng.uniform(0.3, 3.5)
            z = rng.uniform(-1.5, 1.5)
            y = 0.55 + rng.uniform(-0.2, 0.4)
            v = vector(rng.uniform(0.02, 0.07), rng.uniform(-0.006, 0.020), rng.uniform(-0.03, 0.03))
        else:
            x = bender_pos.x + rng.uniform(0.5, max(0.7, reach / 37.0 * 9.4))
            z = rng.uniform(-0.12, 0.12) * (1.0 if rd.mode == "blue" else 3.0)
            y = 0.55 + rng.uniform(-0.08, 0.18)
            v = vector(rng.uniform(0.020, 0.060), rng.uniform(0.000, 0.014), rng.uniform(-0.008, 0.008))

        spawn_particle(
            vector(x, y, z),
            v,
            col,
            rng.uniform(0.018, 0.050) * (1.0 + 0.5 * hf),
            rng.uniform(0.6, 1.4),
            opacity=0.30 + 0.55 * hf,
        )

    smoke_amount = clamp((rd.fuel_burned / 14.0) * 0.5 + (rd.peak_k - 590) / 1800 * 0.15, 0.0, 0.35)
    if rd.mode == "blue":
        smoke_amount *= 0.25
    spawn_smoke(bender_pos + vector(0.9 + reach / 37.0 * 2.5, 0.65, 0), smoke_amount)


def update_scene() -> None:
    rd = ROUNDS[current_round]
    t = round_time

    temp_k = terminal_temperature(rd, t)
    hf = hot_fraction(rd, temp_k)
    reach = reach_at_time(rd, t, temp_k)

    update_mode_geometry(rd, temp_k, reach, hf)
    update_bars(rd)
    spawn_mode_particles(rd, temp_k, reach, hf)
    update_particles()

    # Target remains cold in the terminal output: keep it red, not glowing hot.
    target_core.color = color.red
    target_core.radius = 0.08 + 0.01 * sin(round_time * 3.0)
    target_ring.color = color.red

    # Fuel/O2/loss indicators.
    fuel_scale = clamp(rd.fuel_burned / 14.0, 0.05, 1.0)
    o2_scale = clamp(rd.o2_used / 28.0, 0.05, 1.0)
    loss_scale = clamp(rd.loss_j / 43000.0, 0.10, 1.0)
    fuel_arrow.axis = vector(0.55 + 1.10 * fuel_scale, 0, 0)
    oxygen_arrow.axis = vector(0.55 + 1.10 * o2_scale, 0, 0)
    loss_arrow.axis = vector(0, 0.35 + 0.90 * loss_scale, 0)

    title_label.text = f"Round {rd.number}: {rd.mode.upper()} — {rd.result}"
    state_label.text = (
        f"T≈{temp_k:.0f} K | reach≈{reach:.1f} cells | target remains {rd.target_k:.0f} K | "
        f"auto={'on' if auto_cycle else 'off'}"
    )
    data_label.text = (
        f"{mode_description(rd)}\n"
        f"Peak T: {rd.peak_k:.1f} K\n"
        f"Avg T: {rd.avg_k:.1f} K\n"
        f"Target T: {rd.target_k:.1f} K\n"
        f"Target dose: {rd.dose:.1f} K*s\n"
        f"Energy released: {rd.energy_j:.1f} J\n"
        f"Heat lost: {rd.loss_j:.1f} J\n"
        f"Fuel burned: {rd.fuel_burned:.3f}\n"
        f"O2 used: {rd.o2_used:.3f}\n"
        f"Peak rate: {rd.peak_rate:.3f}\n"
        f"Terminal reach: {rd.reach:.2f} cells\n"
        f"Efficiency: {rd.efficiency:.3f}\n"
        f"Stability: {rd.stability:.2f}"
    )

    status.text = (
        f"Round {rd.number}/8 | {rd.mode} | {rd.result} | "
        f"T={temp_k:.0f}K | reach={reach:.1f} | target={rd.target_k:.0f}K | "
        f"smoke={'on' if show_smoke else 'off'}"
    )


# -----------------------------------------------------------------------------
# Main loop
# -----------------------------------------------------------------------------

while True:
    rate(60)

    if not paused:
        round_time += dt
        if auto_cycle and round_time >= cycle_seconds:
            set_round(current_round + 1)
        elif not auto_cycle and round_time > cycle_seconds:
            round_time = cycle_seconds

    update_scene()
