"""
VPython Rocket Launch Simulation Based on Terminal Output

This file visualizes the terminal rocket launch output that was provided.

It recreates the same main system:
- multiple AI-controlled launch rounds
- thrust, fuel burn, changing mass, gravity, drag, wind drift
- powered ascent, coasting upward, near apogee, ballistic descent, parachute descent
- parachute deployment at the round's deploy altitude
- final round summaries and comparison table

Run:
    python vpython_rocket_launch_from_terminal_output.py

Requires:
    pip install vpython

No CSV logging.
"""

from vpython import (
    canvas, vector, color, box, cylinder, cone, sphere, curve, label, rate,
    mag, norm, arrow
)
import math
import time


# ---------------------------------------------------------------------------
# Reconstructed round data from the provided terminal output
# ---------------------------------------------------------------------------

ROUNDS = [{'round': 1, 'goal': 'baseline launch', 'thrust': 3600.0, 'fuel': 95.0, 'dry_mass': 55.0, 'burn_rate': 5.8, 'cd': 0.55, 'area': 0.22, 'parachute_area': 5.0, 'deploy_altitude': 650.0, 'expected_max_altitude': 2579.67, 'expected_max_speed': 195.87, 'expected_max_acceleration': 537.4, 'expected_burnout_time': 16.4, 'expected_flight_time': 110.75, 'expected_landing_speed': -11.39, 'expected_drift': 233.16, 'expected_result': 'rough parachute landing'}, {'round': 2, 'goal': 'maximize altitude', 'thrust': 4436.6, 'fuel': 111.6, 'dry_mass': 55.0, 'burn_rate': 6.15, 'cd': 0.522, 'area': 0.209, 'parachute_area': 5.0, 'deploy_altitude': 650.0, 'expected_max_altitude': 3477.26, 'expected_max_speed': 235.5, 'expected_max_acceleration': 597.91, 'expected_burnout_time': 18.2, 'expected_flight_time': 122.55, 'expected_landing_speed': -11.41, 'expected_drift': 247.05, 'expected_result': 'rough parachute landing'}, {'round': 3, 'goal': 'reduce landing speed', 'thrust': 4070.6, 'fuel': 106.3, 'dry_mass': 55.0, 'burn_rate': 6.1, 'cd': 0.522, 'area': 0.209, 'parachute_area': 7.75, 'deploy_altitude': 780.0, 'expected_max_altitude': 3125.29, 'expected_max_speed': 222.6, 'expected_max_acceleration': 916.65, 'expected_burnout_time': 17.45, 'expected_flight_time': 144.15, 'expected_landing_speed': -9.23, 'expected_drift': 323.19, 'expected_result': 'rough parachute landing'}, {'round': 4, 'goal': 'high thrust short burn', 'thrust': 5693.6, 'fuel': 95.7, 'dry_mass': 56.6, 'burn_rate': 9.78, 'cd': 0.522, 'area': 0.209, 'parachute_area': 7.75, 'deploy_altitude': 780.0, 'expected_max_altitude': 2325.92, 'expected_max_speed': 258.06, 'expected_max_acceleration': 895.86, 'expected_burnout_time': 9.8, 'expected_flight_time': 126.7, 'expected_landing_speed': -9.36, 'expected_drift': 300.0, 'expected_result': 'rough parachute landing'}, {'round': 5, 'goal': 'low thrust long burn', 'thrust': 4077.7, 'fuel': 116.5, 'dry_mass': 56.6, 'burn_rate': 4.77, 'cd': 0.522, 'area': 0.209, 'parachute_area': 7.75, 'deploy_altitude': 780.0, 'expected_max_altitude': 4388.36, 'expected_max_speed': 225.48, 'expected_max_acceleration': 919.49, 'expected_burnout_time': 24.45, 'expected_flight_time': 163.45, 'expected_landing_speed': -9.36, 'expected_drift': 343.74, 'expected_result': 'rough parachute landing'}, {'round': 6, 'goal': 'heavy payload test', 'thrust': 4863.6, 'fuel': 130.7, 'dry_mass': 93.5, 'burn_rate': 4.66, 'cd': 0.522, 'area': 0.209, 'parachute_area': 7.75, 'deploy_altitude': 780.0, 'expected_max_altitude': 5302.87, 'expected_max_speed': 236.34, 'expected_max_acceleration': 918.27, 'expected_burnout_time': 28.05, 'expected_flight_time': 151.3, 'expected_landing_speed': -12.03, 'expected_drift': 295.23, 'expected_result': 'rough parachute landing'}, {'round': 7, 'goal': 'low drag test', 'thrust': 4660.8, 'fuel': 131.7, 'dry_mass': 93.5, 'burn_rate': 4.79, 'cd': 0.34, 'area': 0.146, 'parachute_area': 7.75, 'deploy_altitude': 780.0, 'expected_max_altitude': 7007.61, 'expected_max_speed': 320.93, 'expected_max_acceleration': 1982.18, 'expected_burnout_time': 27.55, 'expected_flight_time': 157.15, 'expected_landing_speed': -12.08, 'expected_drift': 301.17, 'expected_result': 'rough parachute landing'}, {'round': 8, 'goal': 'wind recovery test', 'thrust': 4689.6, 'fuel': 130.9, 'dry_mass': 93.5, 'burn_rate': 4.92, 'cd': 0.34, 'area': 0.146, 'parachute_area': 9.69, 'deploy_altitude': 858.0, 'expected_max_altitude': 6793.27, 'expected_max_speed': 320.11, 'expected_max_acceleration': 2461.67, 'expected_burnout_time': 26.6, 'expected_flight_time': 170.0, 'expected_landing_speed': -10.82, 'expected_drift': 947.6, 'expected_result': 'rough parachute landing'}]

GRAVITY = 9.81
AIR_DENSITY = 1.225

# Physics timestep. Smaller = more accurate, larger = faster.
DT = 0.05

# Visual rate.
RATE = 90

# Visual scaling.
ALTITUDE_SCALE = 0.00115   # meters to VPython vertical units
DRIFT_SCALE = 0.0060       # meters to VPython horizontal units

# Visual limits.
MAX_VISUAL_ALTITUDE = 8.5
GROUND_Y = 0.0

# Keeps long parachute descents from taking too long visually.
# Physics still uses repeated integration steps, but stable parachute descent
# can be advanced with more substeps per visual frame.
NORMAL_STEPS_PER_FRAME = 2
FAST_DESCENT_STEPS_PER_FRAME = 12


# ---------------------------------------------------------------------------
# Physics helpers
# ---------------------------------------------------------------------------

def drag_force_against_velocity(velocity, drag_magnitude):
    if velocity > 0:
        return -drag_magnitude
    if velocity < 0:
        return drag_magnitude
    return 0.0


def choose_phase(altitude, velocity, engine_on, parachute_deployed, time_s):
    if altitude <= 0.01 and time_s < 0.2:
        return "on pad"
    if engine_on and velocity >= 0:
        return "powered ascent"
    if (not engine_on) and velocity > 2.0:
        return "coasting upward"
    if abs(velocity) <= 2.0 and altitude > 1.0:
        return "near apogee"
    if parachute_deployed and velocity < 0:
        return "parachute descent"
    if velocity < 0:
        return "ballistic descent"
    return "flight"


def visual_position(altitude_m, drift_m):
    y = min(MAX_VISUAL_ALTITUDE, max(0.0, altitude_m * ALTITUDE_SCALE))
    x = drift_m * DRIFT_SCALE
    return vector(x, y, 0)


def bar(value, max_value, width=20):
    if max_value <= 0:
        filled = 0
    else:
        filled = int(width * max(0.0, min(1.0, value / max_value)))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


# ---------------------------------------------------------------------------
# Scene setup
# ---------------------------------------------------------------------------

scene = canvas(
    title="VPython Rocket Launch Simulation Based on Terminal Output",
    width=1250,
    height=760,
    background=vector(0.88, 0.94, 1.0),
    center=vector(0.8, 4.2, 0),
)

scene.caption = """
Controls:
Space / P = pause or play
N / Right = next round
B / Left = previous round
R = restart current round
1-8 = jump to round
A = auto-advance on/off
T = trail on/off
F = fast descent on/off
I = labels on/off
Q / Esc = quit
"""

ground = box(
    pos=vector(0, GROUND_Y - 0.05, 0),
    size=vector(16, 0.1, 3.0),
    color=vector(0.35, 0.48, 0.42),
)

pad = box(
    pos=vector(0, 0.03, 0),
    size=vector(1.0, 0.08, 1.0),
    color=vector(0.38, 0.38, 0.40),
)

# Altitude guide.
guide_lines = []
for meters in range(1000, 8000, 1000):
    y = meters * ALTITUDE_SCALE
    guide = curve(
        pos=[vector(-3.5, y, -0.2), vector(8.5, y, -0.2)],
        color=vector(0.72, 0.78, 0.82),
        radius=0.006,
    )
    guide_lines.append(guide)
    label(
        pos=vector(-3.8, y, -0.2),
        text=f"{meters} m",
        height=9,
        box=False,
        color=vector(0.25, 0.30, 0.35),
        align="right",
    )

# Rocket body. Built as separate objects to avoid relying on compound().
rocket_body = cylinder(
    pos=vector(0, 0.25, 0),
    axis=vector(0, 0.55, 0),
    radius=0.12,
    color=vector(0.92, 0.94, 0.98),
)
rocket_nose = cone(
    pos=vector(0, 0.80, 0),
    axis=vector(0, 0.23, 0),
    radius=0.12,
    color=vector(0.95, 0.25, 0.16),
)
rocket_window = sphere(
    pos=vector(0, 0.62, 0.105),
    radius=0.045,
    color=vector(0.20, 0.55, 0.95),
)
flame = cone(
    pos=vector(0, 0.22, 0),
    axis=vector(0, -0.42, 0),
    radius=0.10,
    color=vector(1.0, 0.48, 0.06),
    visible=False,
)

# Parachute visual: canopy plus strings. It is hidden until deployment.
parachute_canopy = sphere(
    pos=vector(0, 1.35, 0),
    radius=0.42,
    color=vector(1.0, 0.18, 0.18),
    opacity=0.35,
    visible=False,
)
parachute_cut = box(
    pos=vector(0, 1.12, 0),
    size=vector(1.0, 0.45, 1.0),
    color=scene.background,
    visible=False,
)
string_left = curve(color=vector(0.95, 0.95, 0.95), radius=0.008, visible=False)
string_right = curve(color=vector(0.95, 0.95, 0.95), radius=0.008, visible=False)

trail = curve(color=vector(0.10, 0.28, 0.95), radius=0.012)

velocity_arrow = arrow(
    pos=vector(0, 0.8, 0.2),
    axis=vector(0, 0.4, 0),
    shaftwidth=0.035,
    color=vector(0.1, 0.45, 1.0),
)

info_label = label(
    pos=vector(-5.6, 8.6, 0),
    text="",
    height=12,
    box=False,
    align="left",
    color=color.black,
)

summary_label = label(
    pos=vector(-5.6, 2.2, 0),
    text="",
    height=11,
    box=False,
    align="left",
    color=color.black,
)

goal_label = label(
    pos=vector(2.5, 8.85, 0),
    text="",
    height=15,
    box=False,
    align="center",
    color=vector(0.05, 0.05, 0.05),
)

controls_label = label(
    pos=vector(6.8, 8.2, 0),
    text="",
    height=10,
    box=False,
    align="left",
    color=color.black,
)


# ---------------------------------------------------------------------------
# Simulation state
# ---------------------------------------------------------------------------

class SimState:
    def __init__(self):
        self.round_index = 0
        self.paused = False
        self.quit_requested = False
        self.auto_advance = True
        self.show_trail = True
        self.fast_descent = True
        self.show_labels = True
        self.summaries = []
        self.load_round(0)

    def load_round(self, index):
        self.round_index = max(0, min(len(ROUNDS) - 1, index))
        self.params = dict(ROUNDS[self.round_index])

        self.time_s = 0.0
        self.altitude = 0.0
        self.velocity = 0.0
        self.acceleration = 0.0
        self.fuel = self.params["fuel"]
        self.mass = self.params["dry_mass"] + self.fuel
        self.drift = 0.0
        self.phase = "on pad"

        self.engine_on = True
        self.parachute_deployed = False
        self.launched = False
        self.landed = False

        self.max_altitude = 0.0
        self.max_speed = 0.0
        self.max_acceleration = 0.0
        self.burnout_time = None
        self.landing_speed = 0.0
        self.final_drift = 0.0
        self.result = "in flight"

        self.frame_count = 0
        self.message = f"Loaded round {self.params['round']}: {self.params['goal']}"
        trail.clear()

    def restart(self):
        self.load_round(self.round_index)

    def next_round(self):
        self.load_round((self.round_index + 1) % len(ROUNDS))

    def previous_round(self):
        self.load_round((self.round_index - 1) % len(ROUNDS))

    def jump_round(self, number):
        self.load_round(number - 1)


state = SimState()


# ---------------------------------------------------------------------------
# Visual updates
# ---------------------------------------------------------------------------

def set_object_group_position(base_pos):
    # Rocket is vertical. Parts are positioned relative to base_pos.
    rocket_body.pos = base_pos + vector(0, 0.16, 0)
    rocket_body.axis = vector(0, 0.55, 0)

    rocket_nose.pos = base_pos + vector(0, 0.71, 0)
    rocket_nose.axis = vector(0, 0.23, 0)

    rocket_window.pos = base_pos + vector(0, 0.53, 0.105)

    flame.pos = base_pos + vector(0, 0.12, 0)
    flame.axis = vector(0, -0.42, 0)

    parachute_canopy.pos = base_pos + vector(0, 1.25, 0)
    parachute_cut.pos = base_pos + vector(0, 1.04, 0)

    string_left.clear()
    string_right.clear()
    string_left.append(pos=base_pos + vector(-0.28, 1.08, 0))
    string_left.append(pos=base_pos + vector(0, 0.72, 0))
    string_right.append(pos=base_pos + vector(0.28, 1.08, 0))
    string_right.append(pos=base_pos + vector(0, 0.72, 0))

    velocity_arrow.pos = base_pos + vector(0.35, 0.50, 0.15)
    arrow_scale = max(-0.9, min(0.9, state.velocity / 260.0))
    velocity_arrow.axis = vector(0, arrow_scale, 0)


def update_visuals():
    base_pos = visual_position(state.altitude, state.drift)
    set_object_group_position(base_pos)

    flame.visible = state.engine_on and not state.landed and not state.paused
    parachute_canopy.visible = state.parachute_deployed
    parachute_cut.visible = state.parachute_deployed
    string_left.visible = state.parachute_deployed
    string_right.visible = state.parachute_deployed

    if state.show_trail and state.frame_count % 3 == 0:
        trail.visible = True
        trail.append(pos=base_pos + vector(0, 0.4, 0))
    else:
        trail.visible = state.show_trail

    # Bob/rocket color by phase.
    if state.phase == "powered ascent":
        rocket_body.color = vector(0.92, 0.94, 0.98)
        velocity_arrow.color = vector(0.1, 0.45, 1.0)
    elif state.phase == "coasting upward":
        rocket_body.color = vector(0.82, 0.92, 1.0)
        velocity_arrow.color = vector(0.2, 0.4, 0.8)
    elif state.phase == "near apogee":
        rocket_body.color = vector(0.95, 0.95, 0.65)
        velocity_arrow.color = vector(0.95, 0.70, 0.15)
    elif state.phase == "ballistic descent":
        rocket_body.color = vector(1.0, 0.78, 0.55)
        velocity_arrow.color = vector(0.95, 0.25, 0.10)
    elif state.phase == "parachute descent":
        rocket_body.color = vector(0.90, 1.0, 0.86)
        velocity_arrow.color = vector(0.15, 0.60, 0.20)
    else:
        rocket_body.color = vector(0.92, 0.94, 0.98)

    if not state.show_labels:
        info_label.visible = False
        summary_label.visible = False
        goal_label.visible = False
        controls_label.visible = False
        return

    info_label.visible = True
    summary_label.visible = True
    goal_label.visible = True
    controls_label.visible = True

    p = state.params
    expected = (
        f"Expected from terminal output:\n"
        f"max_alt={p['expected_max_altitude']:.1f} m\n"
        f"max_speed={p['expected_max_speed']:.1f} m/s\n"
        f"burnout={p['expected_burnout_time']:.1f} s\n"
        f"flight={p['expected_flight_time']:.1f} s\n"
        f"landing_v={p['expected_landing_speed']:.2f} m/s\n"
        f"drift={p['expected_drift']:.1f} m\n"
        f"result={p['expected_result']}"
    )

    fuel_bar = bar(state.fuel, max(p["fuel"], 1), 16)
    alt_bar = bar(state.altitude, max(p["expected_max_altitude"], 1), 16)
    speed_bar = bar(abs(state.velocity), max(p["expected_max_speed"], 1), 16)

    info_label.text = (
        f"Round {p['round']} / {len(ROUNDS)}\n"
        f"goal: {p['goal']}\n"
        f"status: {'PAUSED' if state.paused else ('LANDED' if state.landed else 'RUNNING')}\n\n"
        f"t = {state.time_s:06.2f} s\n"
        f"altitude = {state.altitude:9.2f} m {alt_bar}\n"
        f"velocity = {state.velocity:9.2f} m/s {speed_bar}\n"
        f"accel = {state.acceleration:9.2f} m/s^2\n"
        f"fuel = {state.fuel:9.2f} kg {fuel_bar}\n"
        f"mass = {state.mass:9.2f} kg\n"
        f"drift = {state.drift:9.2f} m\n"
        f"phase = {state.phase}\n"
        f"parachute = {'deployed' if state.parachute_deployed else 'armed'}"
    )

    summary_label.text = (
        f"Live summary\n"
        f"max_altitude = {state.max_altitude:.2f} m\n"
        f"max_speed = {state.max_speed:.2f} m/s\n"
        f"max_accel = {state.max_acceleration:.2f} m/s^2\n"
        f"burnout_time = {0.0 if state.burnout_time is None else state.burnout_time:.2f} s\n"
        f"landing_speed = {state.landing_speed:.2f} m/s\n"
        f"final_drift = {state.final_drift:.2f} m\n"
        f"result = {state.result}\n\n"
        f"{expected}"
    )

    goal_label.text = (
        f"ROUND {p['round']}: {p['goal'].upper()}\n"
        f"thrust={p['thrust']:.0f} N   fuel={p['fuel']:.1f} kg   "
        f"burn={p['burn_rate']:.2f} kg/s   chute={p['parachute_area']:.2f} m^2"
    )

    controls_label.text = (
        "Controls\n"
        "Space/P pause-play\n"
        "N/Right next round\n"
        "B/Left previous round\n"
        "R restart\n"
        "1-8 jump round\n"
        "A auto-advance\n"
        "T trail\n"
        "F fast descent\n"
        "I labels\n"
        "Q/Esc quit\n\n"
        f"auto-advance: {'on' if state.auto_advance else 'off'}\n"
        f"fast descent: {'on' if state.fast_descent else 'off'}"
    )


# ---------------------------------------------------------------------------
# Physics update
# ---------------------------------------------------------------------------

def physics_step():
    if state.landed:
        return

    p = state.params

    state.mass = p["dry_mass"] + state.fuel

    if state.fuel > 0.0:
        fuel_to_burn = min(p["burn_rate"] * DT, state.fuel)
        state.fuel -= fuel_to_burn
        thrust = p["thrust"]
        state.engine_on = True
    else:
        thrust = 0.0
        state.engine_on = False
        if state.burnout_time is None:
            state.burnout_time = state.time_s

    if state.altitude > 0.1:
        state.launched = True

    if (
        not state.parachute_deployed
        and state.launched
        and state.velocity < 0
        and state.altitude <= p["deploy_altitude"]
    ):
        state.parachute_deployed = True
        state.message = "Parachute deployed"

    active_area = p["area"]
    active_cd = p["cd"]

    if state.parachute_deployed:
        active_area += p["parachute_area"]
        active_cd = max(active_cd, 1.30)

    drag_magnitude = 0.5 * AIR_DENSITY * active_cd * active_area * state.velocity * state.velocity
    drag = drag_force_against_velocity(state.velocity, drag_magnitude)

    gravity_force = state.mass * GRAVITY
    net_force = thrust + drag - gravity_force
    state.acceleration = net_force / max(state.mass, 0.001)

    state.velocity += state.acceleration * DT
    state.altitude += state.velocity * DT

    drift_multiplier = 2.5 if state.parachute_deployed else 1.0
    # The terminal output uses wind drift. The original baseline drift is about 1.2 m/s.
    # Reconstruct a round-consistent wind from expected drift / flight time.
    expected_time = max(p.get("expected_flight_time", 1.0), 1.0)
    expected_drift = p.get("expected_drift", 0.0)
    base_wind = expected_drift / expected_time / 1.65
    state.drift += base_wind * drift_multiplier * DT

    state.time_s += DT

    state.max_altitude = max(state.max_altitude, state.altitude)
    state.max_speed = max(state.max_speed, abs(state.velocity))
    state.max_acceleration = max(state.max_acceleration, abs(state.acceleration))

    state.phase = choose_phase(
        state.altitude,
        state.velocity,
        state.engine_on,
        state.parachute_deployed,
        state.time_s,
    )

    if state.launched and state.altitude <= 0.0:
        state.altitude = 0.0
        state.landing_speed = state.velocity
        state.final_drift = state.drift
        state.landed = True

        if abs(state.landing_speed) <= 8.0:
            state.result = "soft recovery"
        elif state.parachute_deployed and abs(state.landing_speed) <= 18.0:
            state.result = "rough parachute landing"
        elif state.max_altitude > 3000:
            state.result = "high altitude, hard landing"
        else:
            state.result = "hard landing"

        print_round_summary()

        if state.auto_advance and state.round_index < len(ROUNDS) - 1:
            time.sleep(0.6)
            state.next_round()
        elif state.auto_advance and state.round_index == len(ROUNDS) - 1:
            print_final_comparison()
            state.paused = True


def steps_per_frame():
    if not state.fast_descent:
        return NORMAL_STEPS_PER_FRAME

    stable_parachute_descent = (
        state.parachute_deployed
        and state.altitude < 900
        and abs(state.acceleration) < 0.30
        and abs(state.velocity) < 18
    )

    return FAST_DESCENT_STEPS_PER_FRAME if stable_parachute_descent else NORMAL_STEPS_PER_FRAME


# ---------------------------------------------------------------------------
# Terminal printing
# ---------------------------------------------------------------------------

def print_round_summary():
    p = state.params
    summary = {
        "round": p["round"],
        "goal": p["goal"],
        "max_altitude": state.max_altitude,
        "max_speed": state.max_speed,
        "max_acceleration": state.max_acceleration,
        "burnout_time": 0.0 if state.burnout_time is None else state.burnout_time,
        "flight_time": state.time_s,
        "landing_speed": state.landing_speed,
        "drift": state.final_drift,
        "result": state.result,
    }

    # Replace earlier summary for same round.
    state.summaries = [s for s in state.summaries if s["round"] != p["round"]]
    state.summaries.append(summary)

    print(
        f"Round {p['round']} summary: "
        f"max_altitude={state.max_altitude:.2f} m, "
        f"max_speed={state.max_speed:.2f} m/s, "
        f"max_acceleration={state.max_acceleration:.2f} m/s^2, "
        f"burnout_time={summary['burnout_time']:.2f} s, "
        f"flight_time={state.time_s:.2f} s, "
        f"landing_speed={state.landing_speed:.2f} m/s, "
        f"drift={state.final_drift:.2f} m, "
        f"result={state.result}"
    )


def print_final_comparison():
    print()
    print("=" * 120)
    print("VISUAL SIMULATION FINAL COMPARISON")
    print("=" * 120)
    print(
        f"{'round':>5} {'goal':<22} {'max_alt':>10} {'max_v':>9} "
        f"{'burnout':>8} {'flight':>8} {'land_v':>9} {'drift':>8} {'result':<24}"
    )
    for s in sorted(state.summaries, key=lambda item: item["round"]):
        print(
            f"{s['round']:5d} "
            f"{s['goal']:<22.22} "
            f"{s['max_altitude']:10.1f} "
            f"{s['max_speed']:9.1f} "
            f"{s['burnout_time']:8.1f} "
            f"{s['flight_time']:8.1f} "
            f"{s['landing_speed']:9.1f} "
            f"{s['drift']:8.1f} "
            f"{s['result']:<24}"
        )
    print("=" * 120)


# ---------------------------------------------------------------------------
# Keyboard controls
# ---------------------------------------------------------------------------

def handle_keydown(evt):
    key = evt.key

    if key in (" ", "p", "P"):
        state.paused = not state.paused

    elif key in ("n", "N", "right"):
        state.next_round()

    elif key in ("b", "B", "left"):
        state.previous_round()

    elif key in ("r", "R"):
        state.restart()

    elif key in tuple(str(i) for i in range(1, 9)):
        state.jump_round(int(key))

    elif key in ("a", "A"):
        state.auto_advance = not state.auto_advance

    elif key in ("t", "T"):
        state.show_trail = not state.show_trail

    elif key in ("f", "F"):
        state.fast_descent = not state.fast_descent

    elif key in ("i", "I"):
        state.show_labels = not state.show_labels

    elif key in ("q", "Q", "esc"):
        state.quit_requested = True


scene.bind("keydown", handle_keydown)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    print("VPython Rocket Launch Simulation Based on Terminal Output")
    print("No CSV logging.")
    print("Rounds loaded:", len(ROUNDS))
    print()

    while not state.quit_requested:
        rate(RATE)

        if not state.paused:
            for _ in range(steps_per_frame()):
                physics_step()
                if state.landed:
                    break

        state.frame_count += 1
        update_visuals()


if __name__ == "__main__":
    main()
