"""
VPython Volcano Eruption Simulation
-----------------------------------
Geology + pressure buildup + lava flow + ash spread

Run:
    python vpython_volcano_eruption_geology_pressure_lava_ash.py

Controls:
    Space : pause / resume
    r     : reset simulation
    e     : force eruption trigger
    + / = : increase pressure rate
    -     : decrease pressure rate
    a     : toggle ash visibility
    l     : toggle lava visibility
    c     : toggle automatic eruption cycling

Loop behavior:
    The simulation automatically restarts after the eruption reaches
    the ash-fall / cooling stage, creating a continuous volcano cycle.

Notes:
    - Uses VPython primitives only.
    - Uses ring(...) instead of torus(...).
    - No external files are needed.
"""

from vpython import *
import random
import math

# -----------------------------
# Scene setup
# -----------------------------
scene = canvas(
    title="Volcano Eruption: Geology, Pressure Buildup, Lava Flow, Ash Spread",
    width=1200,
    height=780,
    background=vec(0.78, 0.88, 0.98),
    center=vec(0, 5, 0),
)
scene.forward = vec(-1.3, -0.65, -1.2)
scene.range = 34

# -----------------------------
# Utility functions
# -----------------------------

def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def lerp(a, b, t):
    return a + (b - a) * t


def lerp_vec(a, b, t):
    return vec(lerp(a.x, b.x, t), lerp(a.y, b.y, t), lerp(a.z, b.z, t))


def rand_unit_horizontal():
    ang = random.uniform(0, 2 * math.pi)
    return vec(math.cos(ang), 0, math.sin(ang))


def make_label(text, pos, height=13, color_value=vec(0.05, 0.05, 0.05), box=False):
    return label(
        text=text,
        pos=pos,
        height=height,
        color=color_value,
        box=box,
        opacity=0,
        border=4,
        font="sans",
    )

# -----------------------------
# Materials
# -----------------------------
rock_dark = vec(0.28, 0.22, 0.18)
rock_mid = vec(0.45, 0.34, 0.25)
rock_light = vec(0.58, 0.48, 0.36)
soil = vec(0.30, 0.22, 0.13)
basalt = vec(0.12, 0.12, 0.13)
crust_color = vec(0.38, 0.26, 0.18)
magma_color = vec(1.0, 0.22, 0.02)
lava_hot = vec(1.0, 0.38, 0.02)
lava_cool = vec(0.28, 0.04, 0.02)
ash_color = vec(0.32, 0.32, 0.32)
steam_color = vec(0.72, 0.72, 0.72)

# -----------------------------
# Ground and geologic layers
# -----------------------------
ground = box(pos=vec(0, -0.45, 0), size=vec(58, 0.9, 58), color=soil)

layer_objs = []
layer_specs = [
    (vec(0, -1.3, 0), vec(54, 0.42, 54), vec(0.22, 0.16, 0.10), "sediment"),
    (vec(0, -2.0, 0), vec(54, 0.55, 54), vec(0.34, 0.25, 0.17), "older ash layer"),
    (vec(0, -2.85, 0), vec(54, 0.65, 54), vec(0.18, 0.18, 0.19), "basaltic bedrock"),
]
for p, s, c, _name in layer_specs:
    layer_objs.append(box(pos=p, size=s, color=c, opacity=0.82))

# distant terrain rings/plates
for radius, y, c in [(17, 0.02, vec(0.34, 0.27, 0.19)), (23, 0.01, vec(0.29, 0.24, 0.18)), (29, 0.0, vec(0.24, 0.22, 0.18))]:
    ring(pos=vec(0, y, 0), axis=vec(0, 1, 0), radius=radius, thickness=0.08, color=c, opacity=0.35)

# -----------------------------
# Volcano cone
# -----------------------------
# cone points upward by default when axis is positive y; radius at base.
cone_body = cone(pos=vec(0, 0, 0), axis=vec(0, 14, 0), radius=9.0, color=rock_mid, opacity=0.96)
cone_shadow = cone(pos=vec(0, 0.08, 0), axis=vec(0, 13.55, 0), radius=8.3, color=rock_dark, opacity=0.23)
crater_rim = ring(pos=vec(0, 14.15, 0), axis=vec(0, 1, 0), radius=2.25, thickness=0.28, color=rock_light)
crater_inner = cylinder(pos=vec(0, 13.65, 0), axis=vec(0, 0.25, 0), radius=2.05, color=basalt, opacity=0.92)
crater_glow = cylinder(pos=vec(0, 13.78, 0), axis=vec(0, 0.04, 0), radius=1.55, color=magma_color, opacity=0.45, emissive=True)

# side streaks / strata on cone
streaks = []
for i in range(32):
    ang = 2 * math.pi * i / 32
    radial = vec(math.cos(ang), 0, math.sin(ang))
    y = random.uniform(2.0, 11.5)
    base_radius = 9.0 * (1.0 - y / 14.0)
    p = radial * (base_radius + 0.05) + vec(0, y, 0)
    length = random.uniform(1.7, 4.2)
    streak = cylinder(
        pos=p,
        axis=radial * length + vec(0, -random.uniform(0.25, 0.75), 0),
        radius=random.uniform(0.035, 0.075),
        color=lerp_vec(rock_dark, rock_light, random.random()),
        opacity=0.55,
    )
    streaks.append(streak)

# -----------------------------
# Internal magma chamber and conduit
# -----------------------------
magma_chamber = sphere(pos=vec(0, -5.1, 0), radius=3.6, color=magma_color, opacity=0.62, emissive=True)
chamber_shell = sphere(pos=vec(0, -5.1, 0), radius=3.85, color=vec(0.16, 0.10, 0.08), opacity=0.18)
conduit = cylinder(pos=vec(0, -4.7, 0), axis=vec(0, 18.2, 0), radius=0.55, color=magma_color, opacity=0.34, emissive=True)
pressure_wave = sphere(pos=magma_chamber.pos, radius=4.2, color=vec(1.0, 0.55, 0.05), opacity=0.08, emissive=True)

# gas bubbles in conduit/chamber
bubbles = []
for _ in range(22):
    if random.random() < 0.55:
        pos = vec(random.uniform(-2.1, 2.1), random.uniform(-7.1, -3.0), random.uniform(-2.1, 2.1))
    else:
        pos = vec(random.uniform(-0.35, 0.35), random.uniform(-2.8, 12.5), random.uniform(-0.35, 0.35))
    b = sphere(pos=pos, radius=random.uniform(0.10, 0.22), color=vec(1.0, 0.75, 0.18), opacity=0.45, emissive=True)
    b.v = vec(0, random.uniform(0.018, 0.055), 0)
    bubbles.append(b)

# -----------------------------
# Lava flow paths
# -----------------------------
lava_paths = []
flow_dirs = [
    vec(1, 0, 0.15),
    vec(-0.55, 0, 0.88),
    vec(-0.85, 0, -0.55),
    vec(0.42, 0, -1.0),
]
for idx, d in enumerate(flow_dirs):
    d = norm(d)
    path = []
    previous = vec(0, 13.85, 0)
    for j in range(17):
        # flows descend cone, then flatten onto ground
        t = j / 16
        y = lerp(13.4, 0.18, t)
        radial_dist = lerp(2.0, 19.5, t)
        side_wander = vec(-d.z, 0, d.x) * math.sin(t * math.pi * 3 + idx) * 0.75
        p = d * radial_dist + side_wander + vec(0, y, 0)
        if j == 0:
            previous = p
            continue
        segment = cylinder(
            pos=previous,
            axis=p - previous,
            radius=lerp(0.18, 0.42, t),
            color=lava_hot,
            opacity=0.0,
            emissive=True,
        )
        segment.cool = random.uniform(0.0015, 0.0038)
        segment.heat = 1.0
        segment.active_after = idx * 38 + j * 5
        path.append(segment)
        previous = p
    lava_paths.append(path)

lava_fronts = []
for i, d in enumerate(flow_dirs):
    f = sphere(pos=vec(0, 13.5, 0), radius=0.35, color=lava_hot, opacity=0.0, emissive=True)
    f.direction = norm(d)
    f.path_index = i
    f.progress = 0.0
    lava_fronts.append(f)

# -----------------------------
# Ash plume and ballistic fragments
# -----------------------------
ash_particles = []
for _ in range(130):
    p = sphere(pos=vec(0, 14.2, 0), radius=random.uniform(0.06, 0.22), color=ash_color, opacity=0.0)
    p.v = vec(0, 0, 0)
    p.age = random.uniform(0, 180)
    p.max_age = random.uniform(170, 360)
    p.kind = "ash"
    ash_particles.append(p)

fragments = []
for _ in range(45):
    f = sphere(pos=vec(0, 14.25, 0), radius=random.uniform(0.07, 0.18), color=vec(0.16, 0.12, 0.10), opacity=0.0)
    f.v = vec(0, 0, 0)
    f.flight = 9999
    fragments.append(f)

# ash fall blanket on ground
ash_fall = []
for _ in range(70):
    patch = cylinder(
        pos=vec(random.uniform(-27, 27), 0.055, random.uniform(-27, 27)),
        axis=vec(0, 0.018, 0),
        radius=random.uniform(0.18, 0.75),
        color=vec(0.42, 0.42, 0.40),
        opacity=0.0,
    )
    patch.start_time = random.uniform(400, 1100)
    ash_fall.append(patch)

# gas/steam rings above crater
steam_rings = []
for i in range(9):
    rr = ring(
        pos=vec(0, 14.5 + i * 0.8, 0),
        axis=vec(0, 1, 0),
        radius=0.45 + i * 0.18,
        thickness=0.035,
        color=steam_color,
        opacity=0.0,
    )
    rr.vy = 0.018 + i * 0.002
    rr.age = i * 24
    steam_rings.append(rr)

# -----------------------------
# Monitoring dashboard
# -----------------------------
pressure_bar_bg = box(pos=vec(-20, 17.2, 0), size=vec(0.7, 8.0, 0.25), color=vec(0.16, 0.16, 0.16), opacity=0.35)
pressure_bar = box(pos=vec(-20, 13.2, 0), size=vec(0.72, 0.1, 0.32), color=vec(1, 0.25, 0.02), emissive=True)
pressure_text = make_label("Pressure", vec(-20, 22.0, 0), height=12)
stage_text = make_label("Stage: dormant", vec(0, 22.5, 0), height=16)
info_text = make_label("", vec(14, 18.5, 0), height=12)
legend_text = make_label(
    "Space pause | r reset | e erupt | +/- pressure rate | a ash | l lava | c auto-loop",
    vec(0, -3.8, 23),
    height=11,
    color_value=vec(0.08, 0.08, 0.08),
)

# simple seismometer needles
seismo_base = box(pos=vec(18, 7.5, 0), size=vec(7.5, 0.16, 0.3), color=vec(0.12, 0.12, 0.12))
seismo_needles = []
for i in range(11):
    needle = cylinder(pos=vec(14.6 + i * 0.68, 7.6, 0), axis=vec(0, 0.3, 0), radius=0.025, color=vec(0.95, 0.15, 0.05), emissive=True)
    seismo_needles.append(needle)
make_label("Seismic tremor", vec(18, 8.35, 0), height=11)

# -----------------------------
# Simulation state
# -----------------------------
paused = False
ash_visible = True
lava_visible = True
pressure = 0.0
pressure_rate = 0.0026
erupting = False
time_step = 0
eruption_clock = 0
wind = vec(0.018, 0, -0.006)
auto_cycle = True
cycle_count = 1
AUTO_RESET_AFTER = 1450

# -----------------------------
# Input handling
# -----------------------------

def reset_simulation(advance_cycle=False):
    global pressure, erupting, time_step, eruption_clock, wind, cycle_count
    pressure = 0.0
    erupting = False
    time_step = 0
    eruption_clock = 0
    if advance_cycle:
        cycle_count += 1
        # Each loop starts with a slightly different wind so the ash plume
        # and lava/ash visual rhythm do not look identical every cycle.
        wind = vec(random.uniform(0.010, 0.026), 0, random.uniform(-0.016, 0.012))
    else:
        cycle_count = 1
        wind = vec(0.018, 0, -0.006)
    crater_glow.opacity = 0.45
    magma_chamber.radius = 3.6
    pressure_wave.radius = 4.2
    pressure_wave.opacity = 0.08
    for path in lava_paths:
        for seg in path:
            seg.opacity = 0.0
            seg.heat = 1.0
            seg.color = lava_hot
    for f in lava_fronts:
        f.pos = vec(0, 13.5, 0)
        f.opacity = 0.0
        f.progress = 0.0
    for p in ash_particles:
        p.pos = vec(0, 14.2, 0)
        p.v = vec(0, 0, 0)
        p.opacity = 0.0
        p.age = random.uniform(0, 180)
    for f in fragments:
        f.pos = vec(0, 14.25, 0)
        f.v = vec(0, 0, 0)
        f.opacity = 0.0
        f.flight = 9999
    for patch in ash_fall:
        patch.opacity = 0.0
    for rr in steam_rings:
        rr.opacity = 0.0


def trigger_eruption():
    global erupting, eruption_clock, pressure
    erupting = True
    eruption_clock = 0
    pressure = max(pressure, 1.0)
    for p in ash_particles:
        direction = rand_unit_horizontal()
        p.pos = vec(random.uniform(-0.7, 0.7), 14.35 + random.uniform(0, 0.6), random.uniform(-0.7, 0.7))
        p.v = direction * random.uniform(0.02, 0.16) + vec(0, random.uniform(0.12, 0.32), 0) + wind * random.uniform(1, 5)
        p.opacity = random.uniform(0.22, 0.72) if ash_visible else 0.0
        p.age = 0
    for f in fragments:
        direction = rand_unit_horizontal()
        f.pos = vec(0, 14.5, 0)
        f.v = direction * random.uniform(0.08, 0.24) + vec(0, random.uniform(0.18, 0.38), 0)
        f.opacity = random.uniform(0.5, 0.9)
        f.flight = 0


def keydown(evt):
    global paused, pressure_rate, ash_visible, lava_visible, auto_cycle
    k = evt.key
    if k == " ":
        paused = not paused
    elif k == "r":
        reset_simulation()
    elif k == "e":
        trigger_eruption()
    elif k in ["+", "="]:
        pressure_rate = clamp(pressure_rate + 0.0006, 0.0004, 0.011)
    elif k == "-":
        pressure_rate = clamp(pressure_rate - 0.0006, 0.0004, 0.011)
    elif k == "a":
        ash_visible = not ash_visible
    elif k == "l":
        lava_visible = not lava_visible
    elif k == "c":
        auto_cycle = not auto_cycle

scene.bind("keydown", keydown)

# -----------------------------
# Update functions
# -----------------------------

def update_pressure():
    global pressure
    if not erupting:
        pulse = 0.0007 * math.sin(time_step * 0.09) + random.uniform(-0.0002, 0.00045)
        pressure = clamp(pressure + pressure_rate + pulse, 0, 1.05)
    else:
        # eruption vents pressure, then chamber begins recharging slowly
        pressure = clamp(pressure - 0.0042 + pressure_rate * 0.28, 0.16, 1.05)

    pressure_height = 0.2 + pressure * 7.75
    pressure_bar.size = vec(0.72, pressure_height, 0.32)
    pressure_bar.pos = vec(-20, 13.2 + pressure_height / 2, 0)
    pressure_bar.color = lerp_vec(vec(1.0, 0.72, 0.04), vec(1.0, 0.06, 0.01), pressure)
    pressure_bar.emissive = True

    magma_chamber.radius = 3.45 + pressure * 0.55 + 0.08 * math.sin(time_step * 0.15)
    crater_glow.opacity = 0.30 + pressure * 0.48
    conduit.opacity = 0.22 + pressure * 0.32
    pressure_wave.radius = 3.9 + pressure * 2.0 + 0.18 * math.sin(time_step * 0.11)
    pressure_wave.opacity = 0.04 + pressure * 0.13


def update_bubbles():
    for b in bubbles:
        b.pos += b.v * (0.4 + pressure * 1.2)
        b.radius = clamp(b.radius + 0.0007 * pressure, 0.08, 0.32)
        if b.pos.y > 13.5 or random.random() < 0.001:
            if random.random() < 0.45:
                b.pos = vec(random.uniform(-2.0, 2.0), random.uniform(-6.8, -3.5), random.uniform(-2.0, 2.0))
            else:
                b.pos = vec(random.uniform(-0.35, 0.35), random.uniform(-4.0, 9.5), random.uniform(-0.35, 0.35))
            b.radius = random.uniform(0.10, 0.20)
            b.v = vec(0, random.uniform(0.018, 0.055), 0)
        b.opacity = 0.25 + pressure * 0.35


def update_steam():
    for rr in steam_rings:
        rr.age += 1
        if rr.age > 180:
            rr.pos = vec(random.uniform(-0.15, 0.15), 14.5, random.uniform(-0.15, 0.15))
            rr.radius = random.uniform(0.35, 0.65)
            rr.age = 0
        rr.pos += vec(wind.x * 0.35, rr.vy * (0.4 + pressure), wind.z * 0.35)
        rr.radius += 0.006 + pressure * 0.006
        rr.opacity = clamp((pressure - 0.18) * 0.5 * (1 - rr.age / 185), 0, 0.38)


def update_ash():
    if not erupting:
        return
    for p in ash_particles:
        p.age += 1
        buoyancy = 0.0015 if p.age < 130 else -0.0016
        turbulence = vec(random.uniform(-0.006, 0.006), random.uniform(-0.002, 0.006), random.uniform(-0.006, 0.006))
        p.v += wind * 0.012 + vec(0, buoyancy, 0) + turbulence
        p.v *= 0.992
        p.pos += p.v
        spread = mag(vec(p.pos.x, 0, p.pos.z))
        p.radius = clamp(p.radius + 0.0008, 0.05, 0.34)
        if p.pos.y < 0.25 or p.age > p.max_age or spread > 45:
            # recycle some particles back into plume while eruption continues
            direction = rand_unit_horizontal()
            p.pos = vec(random.uniform(-0.8, 0.8), 14.2 + random.uniform(0, 1.1), random.uniform(-0.8, 0.8))
            p.v = direction * random.uniform(0.03, 0.14) + vec(0, random.uniform(0.08, 0.26), 0) + wind * random.uniform(1, 5)
            p.age = 0
            p.radius = random.uniform(0.06, 0.22)
        p.opacity = (0.12 + 0.55 * clamp(1 - p.age / p.max_age, 0, 1)) if ash_visible else 0.0


def update_fragments():
    if not erupting:
        return
    for f in fragments:
        if f.flight > 900:
            continue
        f.flight += 1
        f.v += vec(0, -0.009, 0)
        f.v *= 0.995
        f.pos += f.v
        if f.pos.y <= 0.16:
            f.pos.y = 0.16
            f.v = vec(0, 0, 0)
            f.opacity = clamp(f.opacity - 0.012, 0, 1)
        else:
            f.opacity = 0.65


def update_lava():
    if not erupting:
        return
    for path in lava_paths:
        for seg in path:
            if eruption_clock > seg.active_after:
                target_opacity = 0.86 if lava_visible else 0.0
                seg.opacity = lerp(seg.opacity, target_opacity, 0.06)
                seg.heat = clamp(seg.heat - seg.cool, 0.05, 1.0)
                seg.color = lerp_vec(lava_cool, lava_hot, seg.heat)
            elif lava_visible:
                seg.opacity = lerp(seg.opacity, 0.0, 0.04)
    for f in lava_fronts:
        if not lava_visible:
            f.opacity = 0.0
            continue
        f.progress = clamp((eruption_clock - f.path_index * 35) / 520, 0, 1)
        if f.progress > 0:
            y = lerp(13.2, 0.18, f.progress)
            radial_dist = lerp(2.0, 19.8, f.progress)
            f.pos = f.direction * radial_dist + vec(0, y, 0)
            f.radius = 0.25 + f.progress * 0.42
            f.opacity = 0.85


def update_ash_fall():
    if not erupting:
        return
    for patch in ash_fall:
        if eruption_clock > patch.start_time:
            patch.opacity = clamp(patch.opacity + 0.0025, 0, 0.42 if ash_visible else 0)
            patch.radius = clamp(patch.radius + 0.0012, 0.1, 1.25)
        elif not ash_visible:
            patch.opacity = 0.0


def update_seismometer():
    tremor = pressure * (1.0 if not erupting else 1.6) + random.uniform(0, 0.18)
    for i, needle in enumerate(seismo_needles):
        amp = 0.12 + tremor * random.uniform(0.35, 1.25)
        needle.axis = vec(0, amp * math.sin(time_step * 0.28 + i * 0.9), 0)


def update_text():
    if erupting:
        if eruption_clock < 220:
            stage = "explosive venting"
        elif eruption_clock < 620:
            stage = "lava flow expanding"
        else:
            stage = "ash fall and cooling lava"
    else:
        if pressure < 0.35:
            stage = "dormant recharge"
        elif pressure < 0.70:
            stage = "gas-rich pressure buildup"
        elif pressure < 0.95:
            stage = "critical swelling and tremor"
        else:
            stage = "eruption threshold"
    stage_text.text = f"Stage: {stage}"
    info_text.text = (
        f"Cycle: {cycle_count}\n"
        f"Auto-loop: {'on' if auto_cycle else 'off'}\n"
        f"Pressure: {pressure:0.2f}\n"
        f"Pressure rate: {pressure_rate:0.4f}\n"
        f"Wind drift: x {wind.x:0.3f}, z {wind.z:0.3f}\n"
        f"Ash: {'on' if ash_visible else 'off'} | Lava: {'on' if lava_visible else 'off'}\n"
        f"Paused: {'yes' if paused else 'no'}"
    )

# -----------------------------
# Main loop
# -----------------------------
while True:
    rate(60)
    if paused:
        update_text()
        continue

    time_step += 1
    if erupting:
        eruption_clock += 1
        # changing wind creates an asymmetric ash plume
        wind.x += 0.000015 * math.sin(eruption_clock * 0.017)
        wind.z += 0.000013 * math.cos(eruption_clock * 0.015)
    update_pressure()
    update_bubbles()
    update_steam()
    update_seismometer()

    if (not erupting) and pressure >= 1.0:
        trigger_eruption()

    update_ash()
    update_fragments()
    update_lava()
    update_ash_fall()

    # Automatic loop: after a full eruption has shown pressure release,
    # lava extension, ash spread, and cooling/ash fall, restart the geology
    # cycle so the scene continues indefinitely without user input.
    if auto_cycle and erupting and eruption_clock >= AUTO_RESET_AFTER:
        reset_simulation(advance_cycle=True)

    update_text()
