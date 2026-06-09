"""
VPython Rocket Launch Simulation
Topic: Rocket launch — aerospace physics, thrust, gravity, fuel use

Run:
    python vpython_rocket_launch_aerospace_physics_thrust_gravity_fuel.py

Controls:
    Space : pause / resume
    r     : reset simulation
    l     : launch / relaunch
    + / = : increase throttle
    -     : decrease throttle
    g     : toggle gravity
    d     : toggle drag
    f     : toggle flame / exhaust particles
    c     : toggle cinematic camera
    1     : launch pad view
    2     : follow rocket close
    3     : wide mission view
    4     : telemetry side view
    h     : toggle height-reset looping

Notes:
    - Uses VPython primitives only.
    - No CSV logging.
    - Uses ring(...), not torus(...).
"""

from vpython import *
import math
import random

# -----------------------------
# Scene setup
# -----------------------------
scene = canvas(
    title="Rocket Launch — Thrust, Gravity, Fuel Use",
    width=1200,
    height=760,
    background=vector(0.78, 0.88, 1.0),
    center=vector(0, 55, 0),
)
scene.forward = vector(-0.45, -0.22, -0.86)
scene.range = 110
scene.userspin = True
scene.userzoom = True

# -----------------------------
# Constants and tunables
# -----------------------------
g0 = 9.81                     # m/s^2
sea_level_density = 1.225     # kg/m^3
scale_y = 0.12                # visual meters-to-scene scale for altitude
scale_x = 0.18                # visual scale for horizontal drift

DRY_MASS = 12500.0            # kg
FUEL_MASS_INIT = 38000.0      # kg
MAX_THRUST = 820000.0         # N
MAX_BURN_RATE = 245.0         # kg/s at full throttle
ROCKET_AREA = 10.5            # m^2
CD = 0.42

PAD_Y = 2.0
GROUND_ALT = 0.0
MISSION_RESET_DELAY = 5.0
AUTO_RESET_ALTITUDE = 4000.0     # meters; rocket resets after reaching this altitude
ALTITUDE_RESET_DELAY = 1.2        # seconds to show the reached-height state before reset

paused = False
launched = False
gravity_enabled = True
drag_enabled = True
flame_enabled = True
cinematic_camera = True
camera_mode = 0
throttle = 0.86
sim_time = 0.0
phase = "PRELAUNCH"
phase_timer = 0.0
loop_mission = True
height_reset_enabled = True

# Rocket dynamic state in SI units
altitude = 0.0
velocity = 0.0
acceleration = 0.0
horizontal = 0.0
horizontal_velocity = 1.1
fuel_mass = FUEL_MASS_INIT
peak_altitude = 0.0
mission_elapsed = 0.0

# Object holders
objects = []
particles = []
smoke_particles = []
trail_points = []
stars = []
clouds = []

# -----------------------------
# Helpers
# -----------------------------
def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def altitude_to_y(a):
    return PAD_Y + a * scale_y


def horizontal_to_x(x):
    return x * scale_x


def air_density(a):
    # Simple exponential atmosphere
    return sea_level_density * math.exp(-max(0.0, a) / 8500.0)


def make_label(pos, text, height=12, color_value=color.black, box=False):
    return label(pos=pos, text=text, height=height, color=color_value, box=box, opacity=0.0)


def clear_world():
    global objects, particles, smoke_particles, trail_points, stars, clouds
    for group in (objects, particles, smoke_particles, trail_points, stars, clouds):
        for obj in group:
            try:
                obj.visible = False
            except Exception:
                pass
    objects = []
    particles = []
    smoke_particles = []
    trail_points = []
    stars = []
    clouds = []

# -----------------------------
# Build static scene
# -----------------------------
def build_world():
    global ground, pad, tower, tower_arm, rocket_body, nose, engine_bell
    global flame_core, flame_outer, fuel_bar_back, fuel_bar, thrust_arrow
    global gravity_arrow, drag_arrow, velocity_arrow, status_label, telemetry_label
    global phase_label, controls_label, altitude_axis, camera_label

    clear_world()

    # Ground and pad
    ground = box(pos=vector(0, -0.25, 0), size=vector(180, 0.5, 140), color=vector(0.50, 0.72, 0.45))
    pad = cylinder(pos=vector(0, 0.0, 0), axis=vector(0, 0.35, 0), radius=8.5, color=vector(0.32, 0.32, 0.34))
    flame_trench = box(pos=vector(0, 0.05, 0), size=vector(5.5, 0.15, 15), color=vector(0.16, 0.16, 0.17))
    objects.extend([ground, pad, flame_trench])

    # Launch tower
    tower = box(pos=vector(-9.5, 13.0, 0), size=vector(1.1, 26.0, 1.1), color=vector(0.45, 0.45, 0.48))
    tower_arm = box(pos=vector(-5.0, 20.0, 0), size=vector(8.5, 0.55, 0.65), color=vector(0.50, 0.50, 0.52))
    tower_base = box(pos=vector(-9.5, 1.1, 0), size=vector(4.0, 1.4, 4.0), color=vector(0.25, 0.25, 0.27))
    objects.extend([tower, tower_arm, tower_base])

    # Altitude axis / planning grid
    altitude_axis = curve(color=vector(0.1, 0.1, 0.1), radius=0.035)
    altitude_axis.append(vector(22, 0, -10))
    altitude_axis.append(vector(22, 122, -10))
    objects.append(altitude_axis)
    for meter in range(0, 1001, 100):
        y = altitude_to_y(meter)
        tick = curve(color=vector(0.18, 0.18, 0.18), radius=0.02)
        tick.append(vector(21.3, y, -10))
        tick.append(vector(23.0, y, -10))
        txt = label(pos=vector(25.5, y, -10), text=f"{meter} m", height=8, box=False, opacity=0.0, color=color.black)
        objects.extend([tick, txt])

    # Sky details
    for _ in range(32):
        cloud = ellipsoid(
            pos=vector(random.uniform(-85, 85), random.uniform(48, 118), random.uniform(-55, -35)),
            length=random.uniform(8, 18), height=random.uniform(2, 5), width=random.uniform(4, 8),
            color=vector(0.96, 0.96, 0.93), opacity=0.55,
        )
        clouds.append(cloud)
    for _ in range(45):
        star = sphere(pos=vector(random.uniform(-95, 95), random.uniform(120, 180), random.uniform(-70, 30)), radius=0.14, color=color.white, opacity=0.65)
        stars.append(star)

    # Rocket
    rocket_body = cylinder(pos=vector(0, PAD_Y, 0), axis=vector(0, 12, 0), radius=1.55, color=color.white)
    nose = cone(pos=vector(0, PAD_Y + 12, 0), axis=vector(0, 3.1, 0), radius=1.55, color=vector(0.92, 0.12, 0.08))
    engine_bell = cone(pos=vector(0, PAD_Y - 0.15, 0), axis=vector(0, -1.3, 0), radius=1.25, color=vector(0.18, 0.18, 0.20))
    band1 = ring(pos=vector(0, PAD_Y + 4.2, 0), axis=vector(0, 1, 0), radius=1.58, thickness=0.08, color=vector(0.08, 0.20, 0.75))
    band2 = ring(pos=vector(0, PAD_Y + 8.3, 0), axis=vector(0, 1, 0), radius=1.58, thickness=0.08, color=vector(0.08, 0.20, 0.75))
    fin_objs = []
    for angle in [0, math.pi / 2, math.pi, 3 * math.pi / 2]:
        x = math.cos(angle) * 1.75
        z = math.sin(angle) * 1.75
        fin = box(pos=vector(x, PAD_Y + 1.2, z), size=vector(0.26 if abs(x) > 0 else 1.4, 2.2, 1.4 if abs(z) > 0 else 0.26), color=vector(0.92, 0.12, 0.08))
        fin_objs.append(fin)
    rocket_parts = [rocket_body, nose, engine_bell, band1, band2] + fin_objs
    objects.extend(rocket_parts)

    # Flame visual
    flame_outer = cone(pos=vector(0, PAD_Y - 1.25, 0), axis=vector(0, -5.5, 0), radius=1.9, color=vector(1.0, 0.48, 0.08), opacity=0.35, visible=False)
    flame_core = cone(pos=vector(0, PAD_Y - 1.10, 0), axis=vector(0, -3.7, 0), radius=0.95, color=vector(1.0, 0.95, 0.25), opacity=0.55, visible=False)
    objects.extend([flame_outer, flame_core])

    # Vector arrows
    thrust_arrow = arrow(pos=vector(-20, 8, 0), axis=vector(0, 10, 0), shaftwidth=0.55, color=vector(1.0, 0.45, 0.05))
    gravity_arrow = arrow(pos=vector(-17, 18, 0), axis=vector(0, -7, 0), shaftwidth=0.45, color=vector(0.1, 0.1, 0.1))
    drag_arrow = arrow(pos=vector(-14, 18, 0), axis=vector(0, -2, 0), shaftwidth=0.35, color=vector(0.2, 0.45, 1.0))
    velocity_arrow = arrow(pos=vector(8, 12, 0), axis=vector(0, 6, 0), shaftwidth=0.4, color=vector(0.15, 0.75, 0.20))
    objects.extend([thrust_arrow, gravity_arrow, drag_arrow, velocity_arrow])

    # Fuel bar
    fuel_bar_back = box(pos=vector(-36, 38, 0), size=vector(3.0, 42, 0.5), color=vector(0.15, 0.15, 0.15))
    fuel_bar = box(pos=vector(-36, 38, 0.1), size=vector(2.35, 40, 0.55), color=vector(0.1, 0.55, 1.0))
    fuel_label = make_label(vector(-36, 62, 0), "FUEL", 10)
    objects.extend([fuel_bar_back, fuel_bar, fuel_label])

    # Labels
    status_label = make_label(vector(-48, 74, 0), "", 12)
    telemetry_label = make_label(vector(49, 74, 0), "", 11)
    phase_label = make_label(vector(0, 90, 0), "PRELAUNCH", 18, vector(0.05, 0.05, 0.05))
    camera_label = make_label(vector(0, 82, 0), "Camera: cinematic", 10)
    controls_label = make_label(
        vector(0, -5, 0),
        "Controls: Space pause | r reset | l launch | +/- throttle | g gravity | d drag | f flame | c camera | h height reset | 1-4 views",
        10,
    )
    objects.extend([status_label, telemetry_label, phase_label, camera_label, controls_label])

    return rocket_parts

rocket_parts = []

# -----------------------------
# Reset / launch
# -----------------------------
def reset_simulation(auto_launch=False):
    global altitude, velocity, acceleration, horizontal, horizontal_velocity, fuel_mass
    global peak_altitude, mission_elapsed, sim_time, phase, phase_timer, launched
    global rocket_parts

    build_world()
    rocket_parts = [obj for obj in objects if obj in [rocket_body, nose, engine_bell] or (hasattr(obj, 'pos') and obj.pos.y >= PAD_Y and obj.pos.y <= PAD_Y + 16 and abs(obj.pos.x) < 3 and abs(obj.pos.z) < 3)]

    altitude = 0.0
    velocity = 0.0
    acceleration = 0.0
    horizontal = 0.0
    horizontal_velocity = random.uniform(-0.8, 1.1)
    fuel_mass = FUEL_MASS_INIT
    peak_altitude = 0.0
    mission_elapsed = 0.0
    sim_time = 0.0
    phase = "PRELAUNCH"
    phase_timer = 0.0
    launched = auto_launch
    if auto_launch:
        phase = "IGNITION"


def start_launch():
    global launched, phase, phase_timer
    if not launched:
        launched = True
        phase = "IGNITION"
        phase_timer = 0.0

# -----------------------------
# Particle systems
# -----------------------------
def add_exhaust_particles(base_pos, amount, power):
    if not flame_enabled:
        return
    for _ in range(amount):
        angle = random.uniform(0, 2 * math.pi)
        radius = random.uniform(0, 1.2 + power * 0.7)
        p = sphere(
            pos=base_pos + vector(math.cos(angle) * radius, random.uniform(-0.5, 0.4), math.sin(angle) * radius),
            radius=random.uniform(0.18, 0.42),
            color=random.choice([vector(1.0, 0.78, 0.12), vector(1.0, 0.36, 0.05), vector(1.0, 0.95, 0.4)]),
            opacity=0.65,
        )
        p.vel = vector(random.uniform(-1.2, 1.2), random.uniform(-11, -5) * (0.6 + power), random.uniform(-1.2, 1.2))
        p.life = random.uniform(0.35, 0.75)
        particles.append(p)


def add_smoke(base_pos, amount, intensity):
    for _ in range(amount):
        angle = random.uniform(0, 2 * math.pi)
        radius = random.uniform(0.3, 3.5 + 2 * intensity)
        gray = random.uniform(0.58, 0.78)
        p = sphere(
            pos=base_pos + vector(math.cos(angle) * radius, random.uniform(-1.5, 1.2), math.sin(angle) * radius),
            radius=random.uniform(0.5, 1.4),
            color=vector(gray, gray, gray),
            opacity=0.22,
        )
        p.vel = vector(random.uniform(-1.7, 1.7), random.uniform(0.4, 2.1), random.uniform(-1.7, 1.7))
        p.life = random.uniform(2.5, 5.5)
        smoke_particles.append(p)


def add_trail(pos):
    dot = sphere(pos=pos, radius=0.16, color=vector(1.0, 1.0, 1.0), opacity=0.55)
    dot.life = 8.0
    trail_points.append(dot)
    if len(trail_points) > 220:
        old = trail_points.pop(0)
        old.visible = False


def update_particles(dt):
    for group in [particles, smoke_particles, trail_points]:
        survivors = []
        for p in group:
            if hasattr(p, 'vel'):
                p.pos += p.vel * dt
                if group is smoke_particles:
                    p.vel += vector(random.uniform(-0.08, 0.08), 0.10, random.uniform(-0.08, 0.08)) * dt
                else:
                    p.vel += vector(0, -2.0, 0) * dt
            p.life -= dt
            if group is smoke_particles:
                p.radius *= 1.0 + 0.20 * dt
                p.opacity = max(0.0, p.opacity - 0.05 * dt)
            elif group is trail_points:
                p.opacity = max(0.0, p.life / 8.0 * 0.55)
                p.radius *= 0.998
            else:
                p.opacity = max(0.0, p.life / 0.75 * 0.65)
                p.radius *= 0.99
            if p.life > 0 and p.opacity > 0.01:
                survivors.append(p)
            else:
                p.visible = False
        if group is particles:
            particles[:] = survivors
        elif group is smoke_particles:
            smoke_particles[:] = survivors
        else:
            trail_points[:] = survivors

# -----------------------------
# Physics update
# -----------------------------
def update_physics(dt):
    global altitude, velocity, acceleration, horizontal, horizontal_velocity, fuel_mass
    global peak_altitude, mission_elapsed, phase, phase_timer, launched

    if not launched:
        acceleration = 0.0
        return

    mission_elapsed += dt
    phase_timer += dt

    mass = DRY_MASS + fuel_mass
    thrust = 0.0
    burn_rate = 0.0

    if fuel_mass > 0 and phase not in ["COAST", "DESCENT", "LANDED", "TARGET ALTITUDE REACHED"]:
        ignition_factor = clamp(phase_timer / 2.0, 0.25, 1.0) if phase == "IGNITION" else 1.0
        thrust = MAX_THRUST * throttle * ignition_factor
        burn_rate = MAX_BURN_RATE * throttle * ignition_factor
        fuel_mass = max(0.0, fuel_mass - burn_rate * dt)

    gravity_force = mass * g0 if gravity_enabled else 0.0

    drag_force = 0.0
    if drag_enabled and abs(velocity) > 0.01:
        rho = air_density(altitude)
        drag_force = 0.5 * rho * velocity * velocity * CD * ROCKET_AREA
        if velocity > 0:
            drag_force *= 1
        else:
            drag_force *= -1

    net_force = thrust - gravity_force - drag_force
    acceleration = net_force / mass
    velocity += acceleration * dt
    altitude += velocity * dt

    # Horizontal weather / guidance drift
    horizontal_velocity += random.uniform(-0.08, 0.08) * dt
    horizontal_velocity *= 0.999
    horizontal += horizontal_velocity * dt

    if altitude <= GROUND_ALT and velocity <= 0:
        altitude = GROUND_ALT
        velocity = 0.0
        acceleration = 0.0
        if mission_elapsed > 2.0:
            phase = "LANDED"
            launched = False
            phase_timer = 0.0

    peak_altitude = max(peak_altitude, altitude)

    # Phase logic
    if fuel_mass <= 0 and phase not in ["COAST", "DESCENT", "LANDED"]:
        phase = "COAST"
        phase_timer = 0.0
    elif phase == "IGNITION" and phase_timer > 2.4:
        phase = "ASCENT"
        phase_timer = 0.0
    elif phase == "ASCENT" and fuel_mass < FUEL_MASS_INIT * 0.45:
        phase = "HIGH ASCENT"
        phase_timer = 0.0
    elif phase == "COAST" and velocity <= 0:
        phase = "DESCENT"
        phase_timer = 0.0

# -----------------------------
# Visual update
# -----------------------------
def update_rocket_visual(dt):
    rocket_x = horizontal_to_x(horizontal)
    rocket_y = altitude_to_y(altitude)
    rocket_z = 0

    # Move rocket as a group using known relative offsets from initial pad position.
    body_base_y = rocket_y
    rocket_body.pos = vector(rocket_x, body_base_y, rocket_z)
    nose.pos = vector(rocket_x, body_base_y + 12, rocket_z)
    engine_bell.pos = vector(rocket_x, body_base_y - 0.15, rocket_z)

    # Bands and fins are among objects; update by identifying approximate original categories.
    for obj in objects:
        if obj in [rocket_body, nose, engine_bell, flame_outer, flame_core]:
            continue
        # Move rings/bands/fins only if small rocket-relative objects, not labels or scene objects.
        if isinstance(obj, ring):
            # There are two rocket rings; keep them at body levels based on current y.
            if abs(obj.radius - 1.58) < 0.15:
                if obj.pos.y < rocket_y + 6:
                    obj.pos = vector(rocket_x, body_base_y + 4.2, rocket_z)
                else:
                    obj.pos = vector(rocket_x, body_base_y + 8.3, rocket_z)
        elif isinstance(obj, box) and obj.size.y < 3.0 and obj.size.x <= 1.5 and obj.size.z <= 1.5 and obj.pos.y > 1.0:
            # Fins only
            relx = 1.75 if obj.pos.x >= 0 else -1.75
            relz = 1.75 if obj.pos.z >= 0 else -1.75
            if abs(obj.pos.x) > abs(obj.pos.z):
                obj.pos = vector(rocket_x + relx, body_base_y + 1.2, rocket_z)
            else:
                obj.pos = vector(rocket_x, body_base_y + 1.2, rocket_z + relz)

    thrust_power = 0.0
    if launched and fuel_mass > 0 and phase != "LANDED":
        thrust_power = throttle

    flame_outer.visible = flame_enabled and thrust_power > 0.05
    flame_core.visible = flame_enabled and thrust_power > 0.05
    if flame_outer.visible:
        flicker = 0.75 + 0.35 * random.random()
        flame_outer.pos = vector(rocket_x, body_base_y - 1.15, rocket_z)
        flame_core.pos = vector(rocket_x, body_base_y - 1.0, rocket_z)
        flame_outer.axis = vector(0, -5.0 * flicker * (0.55 + thrust_power), 0)
        flame_core.axis = vector(0, -3.3 * flicker * (0.55 + thrust_power), 0)
        flame_outer.radius = 1.5 + 0.8 * thrust_power
        flame_core.radius = 0.75 + 0.45 * thrust_power

    if launched and fuel_mass > 0:
        add_exhaust_particles(vector(rocket_x, body_base_y - 2.0, rocket_z), 3 if altitude < 900 else 1, thrust_power)
        if altitude < 320:
            add_smoke(vector(rocket_x, max(0.5, body_base_y - 3.5), rocket_z), 2, thrust_power)

    if launched and int(mission_elapsed * 8) % 2 == 0:
        add_trail(vector(rocket_x, body_base_y + 2.0, rocket_z))

    # Arrows scale and positions
    thrust_arrow.axis = vector(0, max(0.1, 15 * thrust_power), 0)
    gravity_arrow.visible = gravity_enabled
    gravity_arrow.axis = vector(0, -7 if gravity_enabled else 0, 0)
    drag_arrow.visible = drag_enabled
    drag_arrow.axis = vector(0, -clamp(abs(velocity) / 45, 0.4, 8) * (1 if velocity > 0 else -1), 0)
    velocity_arrow.pos = vector(rocket_x + 8, body_base_y + 4, rocket_z)
    velocity_arrow.axis = vector(0, clamp(velocity / 10, -10, 12), 0)

    # Fuel bar update
    fuel_frac = fuel_mass / FUEL_MASS_INIT
    fuel_height = max(0.05, 40 * fuel_frac)
    fuel_bar.size = vector(2.35, fuel_height, 0.55)
    fuel_bar.pos = vector(-36, 18 + fuel_height / 2, 0.1)
    if fuel_frac > 0.5:
        fuel_bar.color = vector(0.1, 0.55, 1.0)
    elif fuel_frac > 0.18:
        fuel_bar.color = vector(1.0, 0.65, 0.08)
    else:
        fuel_bar.color = vector(1.0, 0.12, 0.08)


def update_labels():
    mass = DRY_MASS + fuel_mass
    thrust = MAX_THRUST * throttle if launched and fuel_mass > 0 else 0.0
    twr = thrust / (mass * g0) if mass > 0 else 0
    fuel_frac = fuel_mass / FUEL_MASS_INIT

    phase_label.text = phase
    status_label.text = (
        f"MISSION STATE\n"
        f"Phase: {phase}\n"
        f"Throttle: {throttle*100:5.0f}%\n"
        f"Fuel: {fuel_frac*100:6.1f}%\n"
        f"Mass: {mass:,.0f} kg\n"
        f"T/W ratio: {twr:4.2f}\n"
        f"Gravity: {'on' if gravity_enabled else 'off'}\n"
        f"Drag: {'on' if drag_enabled else 'off'}\n"
        f"Height reset: {'on' if height_reset_enabled else 'off'}"
    )
    telemetry_label.text = (
        f"TELEMETRY\n"
        f"Time: {mission_elapsed:6.1f} s\n"
        f"Altitude: {altitude:8.1f} m\n"
        f"Velocity: {velocity:8.1f} m/s\n"
        f"Acceleration: {acceleration:7.2f} m/s²\n"
        f"Peak altitude: {peak_altitude:8.1f} m\n"
        f"Horizontal drift: {horizontal:7.1f} m\n"
        f"Fuel mass: {fuel_mass:8.0f} kg\n"
        f"Reset height: {AUTO_RESET_ALTITUDE:7.0f} m"
    )
    mode_name = ["cinematic", "pad", "follow", "wide", "telemetry"][camera_mode] if camera_mode < 5 else "custom"
    camera_label.text = f"Camera: {'auto ' if cinematic_camera else ''}{mode_name}"

# -----------------------------
# Camera
# -----------------------------
def update_camera(dt):
    global camera_mode
    rocket_pos = vector(horizontal_to_x(horizontal), altitude_to_y(altitude) + 6, 0)

    if cinematic_camera:
        cycle = (mission_elapsed if launched else sim_time) % 28.0
        if cycle < 7:
            active = 1   # pad / orbit
        elif cycle < 14:
            active = 2   # close follow
        elif cycle < 21:
            active = 3   # wide mission
        else:
            active = 4   # telemetry side
    else:
        active = camera_mode

    if active == 1:
        angle = sim_time * 0.18
        target = vector(horizontal_to_x(horizontal) * 0.4, max(28, altitude_to_y(altitude) * 0.28), 0)
        scene.center = scene.center * 0.94 + target * 0.06
        scene.forward = scene.forward * 0.94 + vector(math.sin(angle) * -0.65, -0.20, math.cos(angle) * -0.65).norm() * 0.06
        scene.range = scene.range * 0.96 + 80 * 0.04
    elif active == 2:
        target = rocket_pos
        scene.center = scene.center * 0.86 + target * 0.14
        scene.forward = scene.forward * 0.90 + vector(-0.55, -0.10, -0.75).norm() * 0.10
        scene.range = scene.range * 0.88 + 22 * 0.12
    elif active == 3:
        target = vector(horizontal_to_x(horizontal) * 0.25, max(50, altitude_to_y(peak_altitude) * 0.46), 0)
        scene.center = scene.center * 0.94 + target * 0.06
        scene.forward = scene.forward * 0.95 + vector(-0.35, -0.35, -0.88).norm() * 0.05
        scene.range = scene.range * 0.94 + max(110, altitude_to_y(max(altitude, peak_altitude)) * 0.80) * 0.06
    elif active == 4:
        target = vector(8, max(35, altitude_to_y(altitude) * 0.65), 0)
        scene.center = scene.center * 0.92 + target * 0.08
        scene.forward = scene.forward * 0.92 + vector(-1.0, -0.12, -0.05).norm() * 0.08
        scene.range = scene.range * 0.94 + 70 * 0.06

# -----------------------------
# Keyboard controls
# -----------------------------
def keydown(evt):
    global paused, throttle, gravity_enabled, drag_enabled, flame_enabled
    global cinematic_camera, camera_mode, loop_mission, height_reset_enabled

    key = evt.key
    if key == ' ':
        paused = not paused
    elif key == 'r':
        reset_simulation(auto_launch=False)
    elif key == 'l':
        start_launch()
    elif key in ['+', '=']:
        throttle = clamp(throttle + 0.05, 0.1, 1.15)
    elif key == '-':
        throttle = clamp(throttle - 0.05, 0.1, 1.15)
    elif key == 'g':
        gravity_enabled = not gravity_enabled
    elif key == 'd':
        drag_enabled = not drag_enabled
    elif key == 'f':
        flame_enabled = not flame_enabled
    elif key == 'c':
        cinematic_camera = not cinematic_camera
    elif key in ['1', '2', '3', '4']:
        cinematic_camera = False
        camera_mode = int(key)
    elif key == 'a':
        loop_mission = not loop_mission
    elif key == 'h':
        height_reset_enabled = not height_reset_enabled

scene.bind('keydown', keydown)

# -----------------------------
# Main loop
# -----------------------------
reset_simulation(auto_launch=False)

# Auto-start after a short prelaunch display
prelaunch_auto_timer = 0.0
landed_timer = 0.0
altitude_reset_timer = 0.0

while True:
    rate(60)
    dt = 1.0 / 60.0
    if paused:
        update_labels()
        continue

    sim_time += dt

    if not launched and phase == "PRELAUNCH":
        prelaunch_auto_timer += dt
        if prelaunch_auto_timer > 2.0:
            start_launch()

    update_physics(dt)
    update_rocket_visual(dt)
    update_particles(dt)
    update_labels()
    update_camera(dt)

    if height_reset_enabled and loop_mission and launched and altitude >= AUTO_RESET_ALTITUDE:
        phase = "TARGET ALTITUDE REACHED"
        altitude_reset_timer += dt
        if altitude_reset_timer > ALTITUDE_RESET_DELAY:
            prelaunch_auto_timer = 0.0
            landed_timer = 0.0
            altitude_reset_timer = 0.0
            reset_simulation(auto_launch=False)
            continue
    else:
        altitude_reset_timer = 0.0

    if phase == "LANDED":
        landed_timer += dt
        if loop_mission and landed_timer > MISSION_RESET_DELAY:
            prelaunch_auto_timer = 0.0
            landed_timer = 0.0
            altitude_reset_timer = 0.0
            reset_simulation(auto_launch=False)
    else:
        landed_timer = 0.0
