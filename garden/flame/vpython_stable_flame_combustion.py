#!/usr/bin/env python3
"""
VPython Stable Flame Combustion Simulation
-----------------------------------------

Visual simulation based on the terminal output from:
simple_flame_combustion_simulation_stable_flames.py

Run:
    python vpython_stable_flame_combustion.py

Requires:
    pip install vpython

Controls:
    Space  - pause/resume
    R      - restart current round
    N      - next round
    B      - previous round
    A      - toggle auto-cycle
    S      - toggle smoke particles

This is a visual toy model. It is not CFD and does not solve real combustion
chemistry. It translates the terminal output behavior into a VPython scene:
ignition, flame-kernel growth, stable flames, soot/smoke differences, heat
balance, and round-by-round comparison.
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
    compound,
    cone,
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
# Data model based on the stable terminal run
# -----------------------------------------------------------------------------

@dataclass
class StableFlameRound:
    round_number: int
    name: str
    peak_K: float
    avg_K: float
    final_K: float
    burn_s: float
    stable_s: float
    energy_J: float
    lost_J: float
    fuel_burned: float
    o2_used: float
    efficiency: float
    soot: float
    final_smoke: float
    holder: float
    pilot_W: float
    final_height_m: float


ROUNDS = [
    StableFlameRound(1, "baseline stable methane-like flame", 1744.2, 1508.7, 1427.3, 10.01, 9.82, 9101.06, 8460.85, 1.8202, 3.6404, 0.530, 0.0562, 0.0789, 0.52, 22.0, 0.094),
    StableFlameRound(2, "clean blue stabilized flame", 1768.9, 1518.7, 1390.2, 10.01, 9.87, 8660.35, 8100.37, 1.7321, 3.4641, 0.526, 0.0097, 0.0104, 0.58, 20.0, 0.124),
    StableFlameRound(3, "fuel-rich yellow stable flame", 1701.1, 1393.6, 1337.3, 10.01, 9.71, 7612.78, 7163.39, 1.5226, 3.0451, 0.522, 0.2018, 0.2680, 0.55, 25.0, 0.161),
    StableFlameRound(4, "lean oxygen-supported flame", 1714.9, 1414.3, 1297.6, 10.01, 9.84, 7122.60, 6746.61, 1.4245, 3.2552, 0.520, 0.0097, 0.0101, 0.60, 24.0, 0.150),
    StableFlameRound(5, "airflow-anchored stable flame", 1697.5, 1506.1, 1346.9, 10.01, 9.88, 10297.43, 10004.01, 2.0595, 4.1190, 0.521, 0.0236, 0.0258, 0.74, 34.0, 0.163),
    StableFlameRound(6, "humid flame rescued by pilot", 1731.7, 1557.7, 1397.5, 10.01, 9.88, 10345.59, 10129.32, 2.0691, 4.1382, 0.523, 0.0698, 0.0720, 0.68, 46.0, 0.102),
    StableFlameRound(7, "high-cooling stabilized flame", 1649.7, 1511.9, 1336.4, 10.01, 9.94, 11846.21, 11907.44, 2.3692, 4.7385, 0.515, 0.0335, 0.0362, 0.86, 55.0, 0.124),
    StableFlameRound(8, "pulsed but stable burner flame", 1754.7, 1577.9, 1403.3, 10.01, 9.90, 10527.04, 10123.99, 2.1054, 4.2108, 0.523, 0.0281, 0.0291, 0.66, 32.0, 0.103),
]


# -----------------------------------------------------------------------------
# Scene setup
# -----------------------------------------------------------------------------

scene = canvas(
    title="Stable Flame Combustion VPython Simulation",
    width=1280,
    height=760,
    background=vector(0.86, 0.90, 0.96),
    center=vector(0, 1.2, 0),
)
scene.camera.pos = vector(0, 3.4, 8.5)
scene.camera.axis = vector(0, -1.0, -8.5)
scene.forward = vector(0, -0.15, -1)

rng = Random(7)

status = wtext(text="")
help_text = wtext(text="\nControls: Space pause/resume | R restart | N next | B previous | A auto-cycle | S smoke\n\n")


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def lerp(a: float, b: float, u: float) -> float:
    return a + (b - a) * max(0.0, min(1.0, u))


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def temp_to_color(temp_K: float, soot: float):
    """
    Map flame temperature and soot level to a VPython color.
    Blue-white = hot clean flame.
    Yellow-orange = rich/sooty flame.
    """
    clean = clamp(1.0 - soot * 5.0, 0.0, 1.0)
    hot = clamp((temp_K - 900.0) / 900.0, 0.0, 1.0)

    blue_clean = vector(0.18, 0.48, 1.00)
    white_hot = vector(1.00, 0.95, 0.72)
    yellow_rich = vector(1.00, 0.62, 0.08)
    orange_sooty = vector(1.00, 0.25, 0.04)

    clean_color = blue_clean * (1.0 - hot) + white_hot * hot
    rich_color = yellow_rich * (1.0 - hot * 0.35) + orange_sooty * (hot * 0.35)
    return clean_color * clean + rich_color * (1.0 - clean)


def temperature_profile(round_data: StableFlameRound, t: float) -> float:
    """
    Approximate the terminal-output shape:
    ignition -> rapid build -> peak/stable plateau -> cooling stable equilibrium.
    """
    if t < 0.55:
        return lerp(760.0, round_data.avg_K * 0.72, t / 0.55)
    if t < 3.0:
        return lerp(round_data.avg_K * 0.72, round_data.peak_K, (t - 0.55) / 2.45)
    if t < 5.7:
        return lerp(round_data.peak_K, round_data.peak_K * 0.98, (t - 3.0) / 2.7)

    # Pulsed round oscillates visibly after the stable plateau.
    pulse = 0.0
    if round_data.round_number == 8:
        pulse = 70.0 * sin(2.7 * t)

    return lerp(round_data.peak_K * 0.98, round_data.final_K + pulse, (t - 5.7) / 4.3)


def kernel_profile(t: float, holder: float) -> float:
    return clamp(0.08 + holder * 0.18 + t / 0.75, 0.0, 1.0)


def state_name(t: float, temp_K: float, kernel: float) -> str:
    if t < 0.08:
        return "active burning"
    if kernel < 0.98:
        return "building flame"
    if t > 5.8 and temp_K < 1500:
        return "cooling stable"
    return "stable flame"


# -----------------------------------------------------------------------------
# Objects
# -----------------------------------------------------------------------------

floor = box(pos=vector(0, -0.05, 0), size=vector(10.5, 0.08, 6.5), color=vector(0.72, 0.76, 0.78))

burner_base = cylinder(pos=vector(0, 0, 0), axis=vector(0, 0.18, 0), radius=0.55, color=vector(0.18, 0.18, 0.18))
burner_top = ring(pos=vector(0, 0.22, 0), axis=vector(0, 1, 0), radius=0.56, thickness=0.035, color=vector(0.05, 0.05, 0.05))
nozzle = cylinder(pos=vector(0, 0.18, 0), axis=vector(0, 0.18, 0), radius=0.18, color=vector(0.36, 0.36, 0.36))

# Flame-holder ring and pilot bead.
holder_ring = ring(pos=vector(0, 0.36, 0), axis=vector(0, 1, 0), radius=0.37, thickness=0.018, color=color.orange)
pilot = sphere(pos=vector(-0.42, 0.35, 0.0), radius=0.055, color=color.yellow, emissive=True)

# Layered flame body. Cones are transparent, so several layers give a flame-like body.
outer_flame = cone(pos=vector(0, 0.28, 0), axis=vector(0, 1.6, 0), radius=0.42, color=color.orange, opacity=0.34)
mid_flame = cone(pos=vector(0, 0.32, 0), axis=vector(0, 1.25, 0), radius=0.30, color=color.yellow, opacity=0.46)
inner_flame = cone(pos=vector(0, 0.35, 0), axis=vector(0, 0.95, 0), radius=0.18, color=vector(0.25, 0.55, 1.0), opacity=0.58)

kernel_sphere = sphere(pos=vector(0, 0.45, 0), radius=0.12, color=color.white, opacity=0.65, emissive=True)
flame_light = local_light(pos=vector(0, 1.3, 0), color=vector(1.0, 0.65, 0.25))

# Fuel and oxygen arrows.
fuel_arrow = arrow(pos=vector(-2.6, 0.35, 0), axis=vector(1.3, 0, 0), shaftwidth=0.06, color=vector(1.0, 0.42, 0.1))
oxygen_arrow = arrow(pos=vector(2.6, 0.35, 0), axis=vector(-1.3, 0, 0), shaftwidth=0.06, color=vector(0.2, 0.5, 1.0))
loss_arrow = arrow(pos=vector(0, 2.35, 0), axis=vector(0, 0.9, 0), shaftwidth=0.05, color=vector(0.6, 0.65, 0.7))

# Comparison bars for the eight rounds.
bars = []
bar_labels = []
x0 = -4.2
for i, rd in enumerate(ROUNDS):
    x = x0 + i * 1.2
    bar = box(pos=vector(x, 0.15, -2.7), size=vector(0.45, 0.30, 0.35), color=vector(0.35, 0.45, 0.70))
    bars.append(bar)
    lab = label(
        pos=vector(x, 0.05, -3.05),
        text=str(rd.round_number),
        height=12,
        box=False,
        opacity=0,
        color=color.black,
    )
    bar_labels.append(lab)

title_label = label(pos=vector(0, 3.25, 0), text="", height=18, box=False, opacity=0, color=color.black)
data_label = label(pos=vector(-4.8, 2.7, 0), text="", height=12, box=False, opacity=0, color=color.black, align="left")
state_label = label(pos=vector(0, 2.75, 0), text="", height=14, box=False, opacity=0, color=color.black)
legend_label = label(
    pos=vector(4.35, 2.55, 0),
    text="Blue = clean/hot\nYellow/orange = rich/sooty\nKernel = flame stability\nBars = energy by round",
    height=12,
    box=False,
    opacity=0,
    color=color.black,
    align="left",
)

# Smoke particles.
smoke_particles = []
for _ in range(70):
    p = sphere(pos=vector(0, -5, 0), radius=0.035, color=vector(0.25, 0.25, 0.25), opacity=0.0)
    p.v = vector(0, 0, 0)
    p.life = 0.0
    smoke_particles.append(p)

# Small heat/spark particles.
spark_particles = []
for _ in range(30):
    p = sphere(pos=vector(0, -5, 0), radius=0.018, color=color.yellow, opacity=0.0, emissive=True)
    p.v = vector(0, 0, 0)
    p.life = 0.0
    spark_particles.append(p)


# -----------------------------------------------------------------------------
# Simulation controls
# -----------------------------------------------------------------------------

current_round_index = 0
round_time = 0.0
paused = False
auto_cycle = True
show_smoke = True
sim_speed = 1.0
dt = 0.01


def reset_round() -> None:
    global round_time
    round_time = 0.0
    for p in smoke_particles + spark_particles:
        p.pos = vector(0, -5, 0)
        p.opacity = 0.0
        p.life = 0.0


def set_round(index: int) -> None:
    global current_round_index
    current_round_index = index % len(ROUNDS)
    reset_round()


def keydown(evt) -> None:
    global paused, auto_cycle, show_smoke
    key = evt.key.lower()
    if key == " ":
        paused = not paused
    elif key == "r":
        reset_round()
    elif key == "n":
        set_round(current_round_index + 1)
    elif key == "b":
        set_round(current_round_index - 1)
    elif key == "a":
        auto_cycle = not auto_cycle
    elif key == "s":
        show_smoke = not show_smoke


scene.bind("keydown", keydown)


# -----------------------------------------------------------------------------
# Particle animation
# -----------------------------------------------------------------------------

def spawn_smoke(rd: StableFlameRound, flame_height: float) -> None:
    if not show_smoke:
        return

    spawn_chance = clamp(0.08 + rd.final_smoke * 0.9 + rd.soot * 0.6, 0.02, 0.45)
    if rng.random() > spawn_chance:
        return

    for p in smoke_particles:
        if p.life <= 0:
            angle = rng.random() * 2 * pi
            r = rng.random() * 0.18
            p.pos = vector(r * cos(angle), 0.7 + rng.random() * flame_height * 0.4, r * sin(angle))
            p.v = vector((rng.random() - 0.5) * 0.012, 0.016 + rng.random() * 0.018, (rng.random() - 0.5) * 0.012)
            p.radius = 0.025 + rng.random() * 0.045 + rd.soot * 0.03
            p.opacity = clamp(0.10 + rd.final_smoke * 0.65 + rd.soot * 0.35, 0.06, 0.42)
            p.color = vector(0.22, 0.22, 0.22) * (1.0 - clamp(rd.soot, 0.0, 0.4))
            p.life = 2.2 + rng.random() * 1.4
            return


def update_particles(rd: StableFlameRound, flame_height: float, temp_K: float) -> None:
    spawn_smoke(rd, flame_height)

    for p in smoke_particles:
        if p.life > 0:
            p.pos += p.v
            p.v.y += 0.0007
            p.life -= dt * sim_speed
            p.opacity *= 0.993
            if p.life <= 0:
                p.opacity = 0
                p.pos = vector(0, -5, 0)

    # Sparks appear mainly during ignition or pulsed-burner oscillations.
    spark_activity = round_time < 0.8 or (rd.round_number == 8 and sin(round_time * 4.8) > 0.75)
    if spark_activity and rng.random() < 0.24:
        for p in spark_particles:
            if p.life <= 0:
                angle = rng.random() * 2 * pi
                p.pos = vector(0.04 * cos(angle), 0.42, 0.04 * sin(angle))
                p.v = vector((rng.random() - 0.5) * 0.035, 0.035 + rng.random() * 0.045, (rng.random() - 0.5) * 0.035)
                p.opacity = 0.75
                p.color = temp_to_color(temp_K, rd.soot)
                p.life = 0.4 + rng.random() * 0.4
                break

    for p in spark_particles:
        if p.life > 0:
            p.pos += p.v
            p.v.y -= 0.001
            p.life -= dt * sim_speed
            p.opacity *= 0.95
            if p.life <= 0:
                p.opacity = 0
                p.pos = vector(0, -5, 0)


# -----------------------------------------------------------------------------
# Main visual update
# -----------------------------------------------------------------------------

def update_visuals() -> None:
    rd = ROUNDS[current_round_index]
    t = round_time

    temp_K = temperature_profile(rd, t)
    kernel = kernel_profile(t, rd.holder)
    state = state_name(t, temp_K, kernel)

    energy_scale = rd.energy_J / max(x.energy_J for x in ROUNDS)
    soot_factor = clamp(rd.soot / 0.22, 0.0, 1.0)
    efficiency_factor = clamp(rd.efficiency / 0.55, 0.0, 1.0)

    # Height uses both terminal estimated final height and dynamic heat growth.
    build = clamp(t / 2.8, 0.0, 1.0)
    cool = 1.0 - 0.22 * clamp((t - 5.8) / 4.2, 0.0, 1.0)
    pulse = 1.0
    if rd.round_number == 8:
        pulse = 1.0 + 0.12 * sin(t * 5.2)
    flame_height = (0.55 + energy_scale * 1.45) * build * cool * pulse
    flame_height = clamp(flame_height, 0.28, 2.15)

    base_radius = 0.28 + 0.10 * energy_scale + 0.12 * soot_factor
    flicker = 1.0 + 0.05 * sin(12.0 * t) + 0.025 * sin(23.0 * t + rd.round_number)

    flame_col = temp_to_color(temp_K, rd.soot)

    outer_flame.axis = vector(0, flame_height * flicker, 0)
    outer_flame.radius = base_radius * (1.0 + 0.15 * soot_factor)
    outer_flame.color = flame_col
    outer_flame.opacity = 0.28 + 0.20 * soot_factor

    mid_flame.axis = vector(0, flame_height * 0.78 * (1.0 + 0.04 * sin(17 * t)), 0)
    mid_flame.radius = base_radius * 0.70
    mid_flame.color = flame_col * 0.8 + vector(1.0, 0.85, 0.12) * 0.2
    mid_flame.opacity = 0.38

    inner_flame.axis = vector(0, flame_height * 0.58 * (1.0 + 0.035 * cos(15 * t)), 0)
    inner_flame.radius = base_radius * 0.38
    inner_flame.color = vector(0.20, 0.48, 1.0) * efficiency_factor + vector(1.0, 0.65, 0.08) * (1.0 - efficiency_factor)
    inner_flame.opacity = 0.50 + 0.20 * efficiency_factor

    kernel_sphere.radius = 0.07 + 0.13 * kernel
    kernel_sphere.pos = vector(0, 0.42 + 0.10 * sin(7 * t), 0)
    kernel_sphere.opacity = 0.25 + 0.45 * kernel
    kernel_sphere.color = color.white * kernel + color.yellow * (1.0 - kernel)

    holder_ring.radius = 0.30 + 0.20 * rd.holder
    holder_ring.color = vector(1.0, 0.45 + 0.35 * rd.holder, 0.08)

    pilot.radius = 0.035 + 0.0016 * rd.pilot_W
    pilot.color = vector(1.0, 0.95, 0.20) if rd.pilot_W > 30 else vector(1.0, 0.55, 0.12)

    flame_light.pos = vector(0, 0.8 + flame_height * 0.45, 0)
    flame_light.color = flame_col * 0.8 + vector(0.35, 0.25, 0.1)

    # Arrows: fuel and oxygen strength.
    fuel_strength = clamp(rd.fuel_burned / 2.4, 0.0, 1.0)
    o2_strength = clamp(rd.o2_used / 4.8, 0.0, 1.0)
    fuel_arrow.axis = vector(0.6 + fuel_strength * 1.0, 0, 0)
    oxygen_arrow.axis = vector(-(0.6 + o2_strength * 1.0), 0, 0)
    loss_arrow.axis = vector(0, 0.45 + clamp(rd.lost_J / 12000.0, 0.0, 1.0) * 0.8, 0)

    update_particles(rd, flame_height, temp_K)

    # Comparison bars: energy height, soot tint; current round highlighted.
    for i, bar in enumerate(bars):
        rdi = ROUNDS[i]
        h = 0.18 + 1.45 * (rdi.energy_J / max(x.energy_J for x in ROUNDS))
        bar.size = vector(0.45, h, 0.35)
        bar.pos = vector(x0 + i * 1.2, h / 2, -2.7)
        so = clamp(rdi.soot / 0.22, 0.0, 1.0)
        if i == current_round_index:
            bar.color = vector(1.0, 0.68, 0.12)
        else:
            bar.color = vector(0.25, 0.48, 0.95) * (1.0 - so) + vector(0.75, 0.35, 0.10) * so

    title_label.text = f"Round {rd.round_number}: {rd.name}"
    state_label.text = f"{state} | kernel {kernel:.2f} | T {temp_K:.0f} K"

    data_label.text = (
        f"Peak: {rd.peak_K:.1f} K\n"
        f"Average: {rd.avg_K:.1f} K\n"
        f"Final: {rd.final_K:.1f} K\n"
        f"Burn time: {rd.burn_s:.2f} s\n"
        f"Stable time: {rd.stable_s:.2f} s\n"
        f"Energy: {rd.energy_J:.0f} J\n"
        f"Heat lost: {rd.lost_J:.0f} J\n"
        f"Efficiency: {rd.efficiency:.3f}\n"
        f"Soot: {rd.soot:.4f}\n"
        f"Pilot: {rd.pilot_W:.1f} W\n"
        f"Holder: {rd.holder:.2f}"
    )

    status.text = (
        f"Round {rd.round_number}/8 | {state} | "
        f"T={temp_K:.0f} K | kernel={kernel:.2f} | "
        f"auto={'on' if auto_cycle else 'off'} | smoke={'on' if show_smoke else 'off'}"
    )


# -----------------------------------------------------------------------------
# Main loop
# -----------------------------------------------------------------------------

while True:
    rate(60)

    if not paused:
        round_time += dt * sim_speed
        if auto_cycle and round_time >= 10.01:
            set_round(current_round_index + 1)
        elif not auto_cycle and round_time > 10.01:
            round_time = 10.01

    update_visuals()
