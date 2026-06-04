#!/usr/bin/env python3
"""
VPython Firebending Simulation
------------------------------

Run:
    python vpython_firebending.py

Requires:
    pip install vpython

A stylized VPython firebending simulation. A firebender emits controlled flame
streams, fire waves, shields, whips, sparks, and blue-fire beams. The simulation
uses particles and translucent glow objects, with simple rules for heat, cooling,
airflow, target hits, and controller-driven mode changes.

Controls:
    1  fire jab
    2  continuous stream
    3  fire wave
    4  fire shield
    5  fire whip
    6  spark storm
    7  blue flame beam
    F  increase fuel
    O  increase oxygen
    C  increase cooling
    R  reset particles
    Space pause/resume
"""

from vpython import (
    canvas, vector, sphere, box, cylinder, cone, ring,
    color, rate, mag, norm, random, label, curve
)
from math import sin, cos, pi, radians


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------

scene = canvas(
    title="VPython Firebending Simulation",
    width=1100,
    height=700,
    background=vector(0.80, 0.88, 0.96),
)

scene.caption = """
Firebending Simulation

A firebender controls flame as moving heat particles.
The controller cycles through fire styles, or use the number keys.

Controls:
1 jab | 2 stream | 3 wave | 4 shield | 5 whip | 6 sparks | 7 blue beam
F fuel | O oxygen | C cooling | R reset particles | Space pause
"""

scene.camera.pos = vector(0, 4.0, 11)
scene.camera.axis = vector(0, -1.4, -9)


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

MAX_FIRE_PARTICLES = 260
MAX_SMOKE_PARTICLES = 120
MAX_SPARKS = 80
MAX_TRAILS = 12

mode_names = {
    "1": "jab",
    "2": "stream",
    "3": "wave",
    "4": "shield",
    "5": "whip",
    "6": "sparks",
    "7": "blue",
}

mode = "stream"
fuel_level = 1.0
oxygen_level = 1.0
cooling = 0.016
airflow = vector(0.015, 0.0, 0.0)
intensity = 1.0
controller_time = 0.0
manual_mode_timer = 0.0
paused = False
target_hits = 0
emissions = 0


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

floor = box(
    pos=vector(0, -0.08, 0),
    size=vector(11, 0.12, 6),
    color=vector(0.68, 0.72, 0.75),
)

bender_body = cylinder(
    pos=vector(-3.7, 0.15, 0),
    axis=vector(0, 1.25, 0),
    radius=0.22,
    color=vector(0.18, 0.22, 0.28),
)

bender_head = sphere(
    pos=vector(-3.7, 1.58, 0),
    radius=0.26,
    color=vector(0.72, 0.52, 0.38),
)

front_arm = cylinder(
    pos=vector(-3.62, 1.10, 0),
    axis=vector(0.88, -0.10, 0),
    radius=0.055,
    color=vector(0.72, 0.52, 0.38),
)

back_arm = cylinder(
    pos=vector(-3.78, 1.02, 0),
    axis=vector(-0.45, -0.28, 0.08),
    radius=0.055,
    color=vector(0.72, 0.52, 0.38),
)

front_hand = sphere(
    pos=front_arm.pos + front_arm.axis,
    radius=0.09,
    color=vector(0.78, 0.58, 0.42),
)

target = sphere(
    pos=vector(3.9, 1.0, 0),
    radius=0.32,
    color=vector(0.25, 0.35, 0.95),
    opacity=0.55,
)

target_ring = ring(
    pos=target.pos,
    axis=vector(1, 0, 0),
    radius=0.45,
    thickness=0.025,
    color=vector(0.1, 0.2, 0.8),
)

glow_core = sphere(
    pos=front_hand.pos,
    radius=0.22,
    color=vector(1.0, 0.65, 0.10),
    opacity=0.20,
    emissive=True,
)

status = label(
    pos=vector(-5.2, 3.5, 0),
    text="",
    height=13,
    box=False,
    color=vector(0.05, 0.05, 0.05),
)


# ---------------------------------------------------------------------------
# Particle groups
# ---------------------------------------------------------------------------

fire_particles = []
smoke_particles = []
sparks = []
whip_trails = []


def rand_between(a, b):
    return a + (b - a) * random()


def random_perp_spread(direction, amount):
    # Flame mostly travels in x/y plane, with z variation for thickness.
    return vector(
        rand_between(-amount, amount),
        rand_between(-amount, amount),
        rand_between(-amount, amount),
    )


def fire_color(temp, blue=False):
    if blue:
        if temp > 0.65:
            return vector(0.28, 0.72, 1.0)
        return vector(0.12, 0.32, 1.0)
    if temp > 0.82:
        return vector(1.0, 0.92, 0.28)
    if temp > 0.55:
        return vector(1.0, 0.46, 0.06)
    if temp > 0.30:
        return vector(0.85, 0.12, 0.02)
    return vector(0.35, 0.06, 0.03)


def make_fire_particle(pos, velocity, temp=1.0, radius=0.055, blue=False, trail=False):
    p = sphere(
        pos=pos,
        radius=radius,
        color=fire_color(temp, blue),
        opacity=0.72,
        emissive=True,
        make_trail=trail,
        retain=10 if trail else 0,
    )
    p.velocity = velocity
    p.temperature = temp
    p.age = 0.0
    p.life = rand_between(0.75, 1.55)
    p.is_blue_fire = blue
    return p


def make_smoke_particle(pos, strength=1.0):
    p = sphere(
        pos=pos,
        radius=rand_between(0.05, 0.13) * strength,
        color=vector(0.35, 0.35, 0.35),
        opacity=0.20,
    )
    p.velocity = vector(
        rand_between(-0.010, 0.010) + airflow.x * 0.4,
        rand_between(0.010, 0.030),
        rand_between(-0.010, 0.010) + airflow.z * 0.4,
    )
    p.age = 0.0
    p.life = rand_between(1.3, 2.6)
    return p


def make_spark(pos, direction):
    p = sphere(
        pos=pos,
        radius=rand_between(0.012, 0.028),
        color=vector(1.0, 0.86, 0.18),
        opacity=0.95,
        emissive=True,
        make_trail=True,
        retain=14,
    )
    p.velocity = norm(direction + random_perp_spread(direction, 0.9)) * rand_between(0.055, 0.110)
    p.temperature = rand_between(0.65, 1.0)
    p.age = 0.0
    p.life = rand_between(0.55, 1.2)
    return p


def clear_particles():
    global fire_particles, smoke_particles, sparks, whip_trails, target_hits, emissions
    for group in (fire_particles, smoke_particles, sparks):
        for p in group:
            p.visible = False
            if hasattr(p, "clear_trail"):
                p.clear_trail()
    for tr in whip_trails:
        tr.visible = False
    fire_particles = []
    smoke_particles = []
    sparks = []
    whip_trails = []
    target_hits = 0
    emissions = 0


# ---------------------------------------------------------------------------
# Firebending emitters
# ---------------------------------------------------------------------------

def hand_pos():
    return front_hand.pos


def direction_to_target():
    return norm(target.pos - hand_pos())


def emit_jab():
    global emissions
    d = direction_to_target()
    for i in range(10):
        pos = hand_pos() + d * (0.10 + 0.035 * i)
        vel = d * rand_between(0.095, 0.145) + random_perp_spread(d, 0.010)
        fire_particles.append(make_fire_particle(pos, vel, temp=rand_between(0.8, 1.1) * intensity, radius=rand_between(0.04, 0.07)))
    emissions += 1


def emit_stream():
    global emissions
    d = direction_to_target()
    count = int(5 + 7 * fuel_level * oxygen_level)
    for _ in range(count):
        pos = hand_pos() + random_perp_spread(d, 0.08)
        vel = d * rand_between(0.055, 0.085) + random_perp_spread(d, 0.012)
        fire_particles.append(make_fire_particle(pos, vel, temp=rand_between(0.65, 1.0) * intensity, radius=rand_between(0.035, 0.065)))
    if random() < 0.08 * oxygen_level and len(sparks) < MAX_SPARKS:
        sparks.append(make_spark(hand_pos(), d))
    emissions += 1


def emit_wave():
    global emissions
    base_angle = 0.0
    for offset in range(-45, 46, 9):
        a = radians(base_angle + offset)
        d = norm(vector(cos(a), sin(a) * 0.45, rand_between(-0.16, 0.16)))
        pos = hand_pos() + d * 0.12
        vel = d * rand_between(0.055, 0.095)
        fire_particles.append(make_fire_particle(pos, vel, temp=rand_between(0.65, 0.95) * intensity, radius=rand_between(0.035, 0.06)))
    emissions += 1


def emit_shield(t):
    global emissions
    center = vector(-3.25, 1.08, 0)
    radius = 0.82 + 0.10 * sin(t * 5.0)
    for k in range(0, 360, 18):
        a = radians(k)
        pos = center + vector(0, cos(a) * radius, sin(a) * radius)
        outward = norm(pos - center)
        vel = outward * rand_between(0.010, 0.035) + vector(0.008, 0, 0)
        fire_particles.append(make_fire_particle(pos, vel, temp=rand_between(0.65, 0.95) * intensity, radius=0.045))
    emissions += 1


def emit_whip(t):
    global emissions
    pts = []
    for i in range(18):
        x = -3.05 + i * 0.24
        y = 1.12 + 0.38 * sin(i * 0.60 + t * 8.0)
        z = 0.18 * sin(i * 0.35 + t * 5.0)
        pts.append(vector(x, y, z))
        if len(fire_particles) < MAX_FIRE_PARTICLES:
            forward = vector(0.07, 0.02 * cos(i * 0.60 + t * 8.0), 0.01)
            fire_particles.append(make_fire_particle(pts[-1], forward, temp=rand_between(0.6, 1.0), radius=0.045))
    tr = curve(pos=pts, color=vector(1.0, 0.42, 0.06), radius=0.018)
    tr.age = 0.0
    tr.life = 0.20
    whip_trails.append(tr)
    emissions += 1


def emit_sparks():
    global emissions
    d = direction_to_target()
    for _ in range(7):
        if len(sparks) < MAX_SPARKS:
            sparks.append(make_spark(hand_pos(), d + vector(0, rand_between(-0.4, 0.5), rand_between(-0.7, 0.7))))
    emissions += 1


def emit_blue_beam():
    global emissions
    d = direction_to_target()
    for i in range(12):
        pos = hand_pos() + d * (0.08 + i * 0.045) + random_perp_spread(d, 0.025)
        vel = d * rand_between(0.105, 0.165) + random_perp_spread(d, 0.004)
        fire_particles.append(make_fire_particle(pos, vel, temp=rand_between(0.9, 1.25) * intensity, radius=rand_between(0.025, 0.045), blue=True))
    emissions += 1


def emit_by_mode(t):
    if len(fire_particles) > MAX_FIRE_PARTICLES:
        return

    if mode == "jab":
        if int(t * 5) % 8 == 0:
            emit_jab()
    elif mode == "stream":
        emit_stream()
    elif mode == "wave":
        if int(t * 4) % 5 == 0:
            emit_wave()
    elif mode == "shield":
        emit_shield(t)
    elif mode == "whip":
        emit_whip(t)
    elif mode == "sparks":
        emit_sparks()
    elif mode == "blue":
        emit_blue_beam()


# ---------------------------------------------------------------------------
# Updates
# ---------------------------------------------------------------------------

def update_controller(dt):
    global controller_time, mode, fuel_level, oxygen_level, cooling, airflow, intensity, manual_mode_timer

    controller_time += dt
    if manual_mode_timer > 0:
        manual_mode_timer -= dt
    else:
        cycle = int(controller_time // 7) % 7
        mode = ["jab", "stream", "wave", "shield", "whip", "sparks", "blue"][cycle]

    fuel_level = 1.0 + 0.22 * sin(controller_time * 0.53)
    oxygen_level = 1.0 + 0.20 * sin(controller_time * 0.71 + 0.8)
    intensity = 0.95 + 0.28 * sin(controller_time * 0.39 + 1.4)
    cooling = 0.014 + 0.008 * max(0, sin(controller_time * 0.20 - 0.7))
    airflow = vector(0.014 * sin(controller_time * 0.35), 0, 0.012 * cos(controller_time * 0.28))


def update_fire_particles(dt):
    global fire_particles, smoke_particles, target_hits

    survivors = []
    for p in fire_particles:
        p.age += dt

        buoyancy = vector(0, 0.018 + 0.030 * p.temperature, 0)
        p.velocity += buoyancy * dt * 4.5
        p.velocity += airflow * dt * 4.0
        p.velocity += random_perp_spread(vector(1, 0, 0), 0.004)

        p.pos += p.velocity

        p.temperature -= cooling * (1.0 + 0.05 * p.age)
        p.temperature *= 0.994
        p.radius *= 0.997
        p.color = fire_color(p.temperature, getattr(p, 'is_blue_fire', False))
        p.opacity = clamp_opacity(p.temperature * 0.78)

        if mag(p.pos - target.pos) < target.radius + 0.16 and p.temperature > 0.28:
            target_hits += 1
            target.color = vector(1.0, 0.25, 0.08)
            p.velocity *= -0.15
            p.temperature *= 0.55

        if p.temperature < 0.18 or p.age > p.life or p.pos.y < -0.05 or abs(p.pos.x) > 6:
            if len(smoke_particles) < MAX_SMOKE_PARTICLES and random() < 0.50 and not getattr(p, 'is_blue_fire', False):
                smoke_particles.append(make_smoke_particle(p.pos, strength=0.9 + fuel_level * 0.2))
            p.visible = False
            if hasattr(p, "clear_trail"):
                p.clear_trail()
        else:
            survivors.append(p)

    fire_particles = survivors


def update_smoke_particles(dt):
    global smoke_particles
    survivors = []
    for p in smoke_particles:
        p.age += dt
        p.velocity += vector(airflow.x * 0.22, 0.004, airflow.z * 0.22)
        p.velocity += random_perp_spread(vector(0, 1, 0), 0.0018)
        p.pos += p.velocity
        p.radius *= 1.004
        p.opacity *= 0.987

        if p.age > p.life or p.opacity < 0.014:
            p.visible = False
        else:
            survivors.append(p)
    smoke_particles = survivors


def update_sparks(dt):
    global sparks, target_hits
    survivors = []
    for p in sparks:
        p.age += dt
        p.velocity += vector(0, -0.0028, 0)
        p.velocity += airflow * dt * 4
        p.pos += p.velocity
        p.temperature -= cooling * 1.4
        p.color = vector(1.0, max(0.15, 0.82 * p.temperature), 0.08)
        p.opacity = clamp_opacity(p.temperature)

        if mag(p.pos - target.pos) < target.radius + 0.10 and p.temperature > 0.22:
            target_hits += 1
            target.color = vector(1.0, 0.45, 0.10)
            p.temperature *= 0.3

        if p.age > p.life or p.temperature < 0.08:
            p.visible = False
            p.clear_trail()
        else:
            survivors.append(p)
    sparks = survivors


def update_whip_trails(dt):
    global whip_trails
    survivors = []
    for tr in whip_trails:
        tr.age += dt
        tr.radius *= 0.96
        if tr.age > tr.life:
            tr.visible = False
        else:
            survivors.append(tr)
    whip_trails = survivors


def clamp_opacity(value):
    return max(0.0, min(0.85, value))


def update_body_pose(t):
    # Hand and arm motion changes with bending style.
    if mode == "shield":
        arm_axis = vector(0.55, 0.22 * sin(t * 5), 0.18 * cos(t * 4))
    elif mode == "whip":
        arm_axis = vector(0.88, 0.25 * sin(t * 7), 0.12 * sin(t * 5))
    elif mode == "wave":
        arm_axis = vector(0.76, 0.16 * sin(t * 6), 0.22 * sin(t * 6))
    else:
        arm_axis = vector(0.88, -0.05 + 0.06 * sin(t * 3), 0.04 * sin(t * 2))

    front_arm.axis = arm_axis
    front_hand.pos = front_arm.pos + front_arm.axis
    glow_core.pos = front_hand.pos


def update_glow():
    burn = max(0.2, fuel_level * oxygen_level * intensity)
    if mode == "blue":
        glow_core.color = vector(0.2, 0.65, 1.0)
    else:
        glow_core.color = vector(1.0, 0.58, 0.08)
    glow_core.radius = 0.16 + 0.10 * burn
    glow_core.opacity = 0.12 + 0.14 * min(1.6, burn)

    if target.color.x > 0.3:
        target.color = target.color * 0.94 + vector(0.25, 0.35, 0.95) * 0.06


def update_status():
    status.text = (
        "Firebending Simulation\n"
        f"mode:          {mode}\n"
        f"fuel:          {fuel_level:5.2f}\n"
        f"oxygen:        {oxygen_level:5.2f}\n"
        f"intensity:     {intensity:5.2f}\n"
        f"cooling:       {cooling:5.3f}\n"
        f"airflow x/z:   {airflow.x:5.3f}, {airflow.z:5.3f}\n"
        f"fire particles:{len(fire_particles):5d}\n"
        f"smoke:         {len(smoke_particles):5d}\n"
        f"sparks:        {len(sparks):5d}\n"
        f"target hits:   {target_hits:5d}\n"
        "keys: 1-7 style | F fuel | O oxygen | C cooling | R reset | Space pause"
    )


# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------

def keydown(evt):
    global mode, manual_mode_timer, fuel_level, oxygen_level, cooling, paused
    key = evt.key.lower()

    if key in mode_names:
        mode = mode_names[key]
        manual_mode_timer = 12.0
    elif key == "f":
        fuel_level = min(1.8, fuel_level + 0.15)
        manual_mode_timer = max(manual_mode_timer, 5.0)
    elif key == "o":
        oxygen_level = min(1.8, oxygen_level + 0.15)
        manual_mode_timer = max(manual_mode_timer, 5.0)
    elif key == "c":
        cooling = min(0.08, cooling + 0.006)
        manual_mode_timer = max(manual_mode_timer, 5.0)
    elif key == "r":
        clear_particles()
    elif key == " ":
        paused = not paused


scene.bind("keydown", keydown)


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
    update_body_pose(controller_time)
    emit_by_mode(controller_time)

    update_fire_particles(dt)
    update_smoke_particles(dt)
    update_sparks(dt)
    update_whip_trails(dt)
    update_glow()
    update_status()
