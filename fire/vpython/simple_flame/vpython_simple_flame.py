#!/usr/bin/env python3
"""
VPython Simple Flame Simulation
-------------------------------

Run:
    python vpython_simple_flame.py

Requires:
    pip install vpython

This is a visual VPython simulation of a simple candle-like flame. It uses
many lightweight particles to represent hot gas, flame glow, smoke, and sparks.
The simulation is stylized, but it includes simple physical ideas:

- hot particles rise because of buoyancy
- flame particles cool and fade as they rise
- oxygen and fuel affect flame strength
- smoke appears after particles cool
- sparks occasionally eject from the flame core
- a controller slowly changes airflow, fuel, and oxygen over time
"""

from vpython import (
    canvas, vector, sphere, box, cylinder, cone, ring,
    color, rate, mag, norm, random, label
)
from math import sin, cos, pi


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------

scene = canvas(
    title="VPython Simple Flame Simulation",
    width=1000,
    height=650,
    background=vector(0.82, 0.88, 0.96),
)

scene.caption = """
Simple Flame Simulation

The flame is made from moving particles:
- orange/yellow particles are hot flame gas
- gray particles are smoke
- tiny bright particles are sparks
- the controller changes fuel, oxygen, and airflow over time

Controls:
- Press F to add fuel
- Press O to add oxygen
- Press C to increase cooling
- Press R to reset
"""

scene.camera.pos = vector(0, 4.8, 11)
scene.camera.axis = vector(0, -1.9, -9)


# ---------------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------------

MAX_FLAME_PARTICLES = 180
MAX_SMOKE_PARTICLES = 100
MAX_SPARKS = 45

fuel_level = 1.0
oxygen_level = 1.0
cooling = 0.018
airflow = vector(0.012, 0.0, 0.0)
flame_power = 1.0
controller_phase = 0.0
paused = False


# ---------------------------------------------------------------------------
# Environment objects
# ---------------------------------------------------------------------------

floor = box(
    pos=vector(0, -0.08, 0),
    size=vector(7, 0.12, 5),
    color=vector(0.72, 0.76, 0.78),
)

wick = cylinder(
    pos=vector(0, 0, 0),
    axis=vector(0, 0.55, 0),
    radius=0.045,
    color=vector(0.05, 0.04, 0.035),
)

candle = cylinder(
    pos=vector(0, -0.18, 0),
    axis=vector(0, 0.35, 0),
    radius=0.38,
    color=vector(0.92, 0.86, 0.72),
)

base_ring = ring(
    pos=vector(0, 0.17, 0),
    axis=vector(0, 1, 0),
    radius=0.39,
    thickness=0.015,
    color=vector(0.82, 0.76, 0.62),
)

inner_glow = sphere(
    pos=vector(0, 0.68, 0),
    radius=0.32,
    color=vector(1.0, 0.70, 0.18),
    opacity=0.26,
    emissive=True,
)

outer_glow = sphere(
    pos=vector(0, 0.95, 0),
    radius=0.72,
    color=vector(1.0, 0.36, 0.08),
    opacity=0.10,
    emissive=True,
)

flame_core = cone(
    pos=vector(0, 0.28, 0),
    axis=vector(0, 1.25, 0),
    radius=0.34,
    color=vector(1.0, 0.48, 0.08),
    opacity=0.32,
    emissive=True,
)

hot_center = cone(
    pos=vector(0, 0.32, 0),
    axis=vector(0, 0.92, 0),
    radius=0.18,
    color=vector(1.0, 0.90, 0.28),
    opacity=0.42,
    emissive=True,
)

status = label(
    pos=vector(-3.2, 3.4, 0),
    text="",
    height=13,
    box=False,
    color=vector(0.05, 0.05, 0.05),
)


# ---------------------------------------------------------------------------
# Particle state
# ---------------------------------------------------------------------------

flame_particles = []
smoke_particles = []
sparks = []


def rand_between(a, b):
    return a + (b - a) * random()


def random_radial(radius):
    angle = rand_between(0, 2 * pi)
    r = radius * (random() ** 0.5)
    return vector(cos(angle) * r, 0, sin(angle) * r)


def flame_color(temp):
    """Map normalized temperature to flame color."""
    if temp > 0.78:
        return vector(1.0, 0.93, 0.35)
    if temp > 0.52:
        return vector(1.0, 0.52, 0.08)
    if temp > 0.28:
        return vector(0.95, 0.18, 0.04)
    return vector(0.40, 0.10, 0.07)


def make_flame_particle():
    start = vector(0, 0.28, 0) + random_radial(0.16)
    temp = rand_between(0.72, 1.0) * flame_power * oxygen_level
    size = rand_between(0.035, 0.075) * (0.7 + 0.5 * fuel_level)

    p = sphere(
        pos=start,
        radius=size,
        color=flame_color(temp),
        opacity=0.66,
        emissive=True,
    )
    p.velocity = vector(
        rand_between(-0.028, 0.028),
        rand_between(0.028, 0.060) * (0.75 + oxygen_level),
        rand_between(-0.028, 0.028),
    )
    p.temperature = temp
    p.age = 0.0
    p.life = rand_between(0.85, 1.45)
    return p


def make_smoke_particle(pos, strength=1.0):
    p = sphere(
        pos=pos + random_radial(0.05),
        radius=rand_between(0.045, 0.11) * strength,
        color=vector(0.38, 0.38, 0.38),
        opacity=0.22,
        emissive=False,
    )
    p.velocity = vector(
        rand_between(-0.015, 0.015) + airflow.x * 0.7,
        rand_between(0.014, 0.035),
        rand_between(-0.015, 0.015) + airflow.z * 0.7,
    )
    p.age = 0.0
    p.life = rand_between(1.3, 2.8)
    return p


def make_spark():
    angle = rand_between(0, 2 * pi)
    launch = vector(cos(angle), rand_between(1.2, 2.4), sin(angle))
    launch = norm(launch)

    p = sphere(
        pos=vector(0, 0.55, 0) + random_radial(0.08),
        radius=rand_between(0.015, 0.028),
        color=vector(1.0, 0.85, 0.20),
        opacity=0.95,
        emissive=True,
        make_trail=True,
        retain=12,
    )
    p.velocity = launch * rand_between(0.035, 0.075)
    p.temperature = rand_between(0.7, 1.0)
    p.age = 0.0
    p.life = rand_between(0.7, 1.3)
    return p


def clear_particles():
    global flame_particles, smoke_particles, sparks
    for group in (flame_particles, smoke_particles, sparks):
        for p in group:
            p.visible = False
            p.clear_trail() if hasattr(p, "clear_trail") else None
    flame_particles = []
    smoke_particles = []
    sparks = []


# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------

def keydown(evt):
    global fuel_level, oxygen_level, cooling, paused
    key = evt.key.lower()

    if key == "f":
        fuel_level = min(1.8, fuel_level + 0.12)
    elif key == "o":
        oxygen_level = min(1.8, oxygen_level + 0.12)
    elif key == "c":
        cooling = min(0.08, cooling + 0.006)
    elif key == "r":
        reset_simulation()
    elif key == " ":
        paused = not paused


scene.bind("keydown", keydown)


def reset_simulation():
    global fuel_level, oxygen_level, cooling, airflow, flame_power, controller_phase
    clear_particles()
    fuel_level = 1.0
    oxygen_level = 1.0
    cooling = 0.018
    airflow = vector(0.012, 0.0, 0.0)
    flame_power = 1.0
    controller_phase = 0.0


# ---------------------------------------------------------------------------
# Controller and update loop
# ---------------------------------------------------------------------------

def update_controller(dt):
    """Slowly changes flame conditions like a simple AI/environment controller."""
    global fuel_level, oxygen_level, cooling, airflow, flame_power, controller_phase

    controller_phase += dt

    # Smooth cycling values.
    fuel_level = 1.0 + 0.28 * sin(controller_phase * 0.55)
    oxygen_level = 1.0 + 0.22 * sin(controller_phase * 0.73 + 1.4)
    flame_power = 0.95 + 0.25 * sin(controller_phase * 0.42 + 0.7)

    # Airflow slowly changes direction.
    airflow = vector(
        0.018 * sin(controller_phase * 0.38),
        0,
        0.010 * cos(controller_phase * 0.31),
    )

    # Occasional cooler period.
    cooling = 0.018 + 0.008 * max(0, sin(controller_phase * 0.21 - 1.0))


def emit_particles():
    """Create new particles according to fuel/oxygen conditions."""
    burn_strength = max(0.0, fuel_level * oxygen_level * flame_power)
    emit_count = int(2 + 5 * burn_strength)

    for _ in range(emit_count):
        if len(flame_particles) < MAX_FLAME_PARTICLES:
            flame_particles.append(make_flame_particle())

    spark_probability = 0.018 + 0.035 * max(0.0, oxygen_level - 0.85)
    if random() < spark_probability and len(sparks) < MAX_SPARKS:
        sparks.append(make_spark())


def update_flame_particles(dt):
    global flame_particles, smoke_particles

    survivors = []
    for p in flame_particles:
        p.age += dt

        # Buoyancy: hot particles rise faster.
        buoyancy = vector(0, 0.020 + 0.035 * p.temperature, 0)

        # Airflow bends the flame sideways.
        p.velocity += buoyancy * dt * 8
        p.velocity += airflow * dt * 7

        # Small turbulent jitter.
        p.velocity += vector(
            rand_between(-0.006, 0.006),
            rand_between(-0.002, 0.006),
            rand_between(-0.006, 0.006),
        )

        p.pos += p.velocity

        # Cooling and fading.
        p.temperature -= cooling * (1.0 + p.pos.y * 0.20)
        p.temperature *= 0.992
        p.radius *= 0.997
        p.color = flame_color(p.temperature)
        p.opacity = max(0.0, min(0.75, p.temperature * 0.75))

        # When flame cools, convert some particles into smoke.
        if p.temperature < 0.22 or p.age > p.life:
            if len(smoke_particles) < MAX_SMOKE_PARTICLES and random() < 0.62:
                smoke_particles.append(make_smoke_particle(p.pos, strength=0.8 + fuel_level * 0.3))
            p.visible = False
        else:
            survivors.append(p)

    flame_particles = survivors


def update_smoke_particles(dt):
    global smoke_particles

    survivors = []
    for p in smoke_particles:
        p.age += dt
        p.velocity += vector(airflow.x * 0.25, 0.006, airflow.z * 0.25)
        p.velocity += vector(rand_between(-0.002, 0.002), 0, rand_between(-0.002, 0.002))
        p.pos += p.velocity
        p.radius *= 1.004
        p.opacity *= 0.986

        if p.age > p.life or p.opacity < 0.015:
            p.visible = False
        else:
            survivors.append(p)

    smoke_particles = survivors


def update_sparks(dt):
    global sparks

    survivors = []
    for p in sparks:
        p.age += dt

        # Sparks rise briefly, then cool and fall.
        gravity = vector(0, -0.0025, 0)
        p.velocity += gravity
        p.velocity += airflow * dt * 5
        p.pos += p.velocity

        p.temperature -= cooling * 1.8
        p.color = vector(1.0, max(0.18, 0.85 * p.temperature), 0.10)
        p.opacity = max(0, p.temperature)

        if p.age > p.life or p.temperature < 0.08:
            p.visible = False
            p.clear_trail()
        else:
            survivors.append(p)

    sparks = survivors


def update_core_shapes():
    """Adjust translucent core/glow objects based on current burn conditions."""
    burn_strength = max(0.15, fuel_level * oxygen_level * flame_power)

    inner_glow.radius = 0.26 + 0.08 * burn_strength
    inner_glow.opacity = 0.18 + 0.10 * min(1.5, burn_strength)

    outer_glow.radius = 0.55 + 0.25 * burn_strength
    outer_glow.opacity = 0.05 + 0.05 * min(1.4, burn_strength)

    flame_core.axis = vector(airflow.x * 8, 1.05 + 0.18 * burn_strength, airflow.z * 8)
    flame_core.radius = 0.23 + 0.10 * fuel_level
    flame_core.opacity = 0.20 + 0.12 * min(1.3, burn_strength)

    hot_center.axis = vector(airflow.x * 5, 0.76 + 0.15 * oxygen_level, airflow.z * 5)
    hot_center.radius = 0.13 + 0.05 * oxygen_level
    hot_center.opacity = 0.28 + 0.16 * oxygen_level


def update_status():
    status.text = (
        "Simple Flame Properties\n"
        f"fuel level:     {fuel_level:5.2f}\n"
        f"oxygen level:   {oxygen_level:5.2f}\n"
        f"cooling:        {cooling:5.3f}\n"
        f"airflow x/z:    {airflow.x:5.3f}, {airflow.z:5.3f}\n"
        f"flame particles:{len(flame_particles):5d}\n"
        f"smoke particles:{len(smoke_particles):5d}\n"
        f"sparks:         {len(sparks):5d}\n"
        "keys: F fuel | O oxygen | C cooling | R reset | Space pause"
    )


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

dt = 0.016

while True:
    rate(60)

    if paused:
        update_status()
        continue

    update_controller(dt)
    emit_particles()
    update_flame_particles(dt)
    update_smoke_particles(dt)
    update_sparks(dt)
    update_core_shapes()
    update_status()
