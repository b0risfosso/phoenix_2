"""
VPython Pendulum Simulation Based on Terminal AI-Controlled Rounds
Keyboard-Controlled Version

This visual version recreates the behavior shown in the terminal output:
- 8 pendulum rounds
- AI-selected goals between rounds
- changing gravity, length, damping, starting angle, and starting angular velocity
- visible damping and energy loss
- live labels for angle, angular velocity, speed, energy, zero crossings, and estimated period
- keyboard controls for pause/play, restart, and round navigation

Run with:

    python vpython_pendulum_ai_rounds_keyboard.py

Requires:
    pip install vpython

No CSV logging.

Keyboard controls:
    Space / P      Pause or play
    N / Right      Jump to next round
    B / Left       Jump to previous round
    R              Restart current round
    1-8            Jump directly to a round
    S              Single-step while paused
    A              Toggle auto-advance after each round
    T              Toggle trail visibility
    C              Clear current trail
    I              Toggle info labels
    Up             Increase damping for current run
    Down           Decrease damping for current run
    G              Increase gravity for current run
    H              Decrease gravity for current run
    Esc / Q        Quit
"""

from vpython import (
    canvas, vector, color, sphere, cylinder, box, curve, label,
    rate, sin, cos, radians, degrees
)
import math


# ---------------------------------------------------------------------------
# Simulation configuration
# ---------------------------------------------------------------------------

TITLE = "VPython Pendulum Simulation with AI-Controlled Rounds and Keyboard Controls"
TIME_STEP = 0.01
ROUND_DURATION = 12.0
DISPLAY_RATE = 100
TRACE_EVERY_STEPS = 4

# Visual scaling. Physical length in meters is multiplied by this value.
LENGTH_SCALE = 3.0

# Manual controls
DAMPING_STEP = 0.01
GRAVITY_STEP = 0.25

ROUNDS = [
    {
        "round": 1,
        "goal": "baseline stable swing",
        "next_goal": "fast short swing",
        "g": 9.810,
        "length": 1.250,
        "mass": 1.000,
        "damping": 0.0250,
        "initial_angle_deg": 35.00,
        "initial_omega": 0.000,
    },
    {
        "round": 2,
        "goal": "fast short swing",
        "next_goal": "high damping test",
        "g": 10.103,
        "length": 0.528,
        "mass": 1.000,
        "damping": 0.0278,
        "initial_angle_deg": 16.81,
        "initial_omega": 0.043,
    },
    {
        "round": 3,
        "goal": "high damping test",
        "next_goal": "low gravity test",
        "g": 9.087,
        "length": 1.129,
        "mass": 1.000,
        "damping": 0.1357,
        "initial_angle_deg": 31.31,
        "initial_omega": -0.040,
    },
    {
        "round": 4,
        "goal": "low gravity test",
        "next_goal": "energy boost test",
        "g": 1.818,
        "length": 0.870,
        "mass": 1.000,
        "damping": 0.0199,
        "initial_angle_deg": 53.94,
        "initial_omega": -0.226,
    },
    {
        "round": 5,
        "goal": "energy boost test",
        "next_goal": "near-settle test",
        "g": 10.255,
        "length": 0.717,
        "mass": 1.000,
        "damping": 0.0237,
        "initial_angle_deg": 78.08,
        "initial_omega": -0.248,
    },
    {
        "round": 6,
        "goal": "near-settle test",
        "next_goal": "slow wide swing",
        "g": 9.047,
        "length": 1.486,
        "mass": 1.000,
        "damping": 0.2058,
        "initial_angle_deg": 12.92,
        "initial_omega": -0.071,
    },
    {
        "round": 7,
        "goal": "slow wide swing",
        "next_goal": "fast short swing",
        "g": 9.139,
        "length": 1.694,
        "mass": 1.000,
        "damping": 0.0254,
        "initial_angle_deg": 50.42,
        "initial_omega": 0.033,
    },
    {
        "round": 8,
        "goal": "fast short swing",
        "next_goal": "complete",
        "g": 10.990,
        "length": 0.701,
        "mass": 1.000,
        "damping": 0.0242,
        "initial_angle_deg": 16.57,
        "initial_omega": -0.528,
    },
]


# ---------------------------------------------------------------------------
# Physics helpers
# ---------------------------------------------------------------------------

def pendulum_position(pivot, length_m, angle_rad):
    """Return bob position for angle measured from vertical."""
    visual_length = length_m * LENGTH_SCALE
    x = visual_length * math.sin(angle_rad)
    y = -visual_length * math.cos(angle_rad)
    return pivot + vector(x, y, 0)


def mechanical_energy(mass, g, length_m, angle_rad, omega):
    """Pendulum mechanical energy in joules."""
    height = length_m * (1.0 - math.cos(angle_rad))
    potential = mass * g * height
    speed = abs(omega) * length_m
    kinetic = 0.5 * mass * speed * speed
    return potential + kinetic


def classify_round(damping):
    if damping >= 0.10:
        return "strongly damped"
    return "stable oscillation"


def make_bar(value, max_value, width=26):
    """Return a text progress bar for labels."""
    if max_value <= 0:
        filled = 0
    else:
        filled = max(0, min(width, int(width * value / max_value)))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def clone_round_params(index):
    """Make a mutable copy of one round."""
    return dict(ROUNDS[index])


def compute_period_from_crossings(crossing_times):
    """Estimate period from center crossings."""
    if len(crossing_times) < 3:
        return 0.0
    half_periods = [
        crossing_times[i] - crossing_times[i - 1]
        for i in range(1, len(crossing_times))
    ]
    return 2.0 * sum(half_periods) / len(half_periods)


# ---------------------------------------------------------------------------
# Keyboard-controlled simulation state
# ---------------------------------------------------------------------------

class SimulationState:
    def __init__(self):
        self.round_index = 0
        self.params = clone_round_params(self.round_index)
        self.theta = radians(self.params["initial_angle_deg"])
        self.omega = self.params["initial_omega"]
        self.t = 0.0
        self.step_count = 0

        self.paused = False
        self.single_step = False
        self.auto_advance = True
        self.show_trail = True
        self.show_info = True
        self.quit_requested = False

        self.round_complete = False
        self.message = "Running"

        self.max_angle = 0.0
        self.max_speed = 0.0
        self.min_energy = 0.0
        self.max_energy = 0.0
        self.start_energy = 0.0
        self.zero_crossings = 0
        self.crossing_times = []
        self.previous_theta = self.theta

        self.summaries = {}
        self.reset_metrics()

    def reset_metrics(self):
        energy = mechanical_energy(
            self.params["mass"],
            self.params["g"],
            self.params["length"],
            self.theta,
            self.omega,
        )
        self.max_angle = abs(degrees(self.theta))
        self.max_speed = abs(self.omega * self.params["length"])
        self.min_energy = energy
        self.max_energy = energy
        self.start_energy = energy
        self.zero_crossings = 0
        self.crossing_times = []
        self.previous_theta = self.theta

    def load_round(self, index, paused=None):
        self.round_index = max(0, min(len(ROUNDS) - 1, index))
        self.params = clone_round_params(self.round_index)
        self.theta = radians(self.params["initial_angle_deg"])
        self.omega = self.params["initial_omega"]
        self.t = 0.0
        self.step_count = 0
        self.round_complete = False
        self.single_step = False
        self.message = f"Loaded round {self.params['round']}: {self.params['goal']}"
        if paused is not None:
            self.paused = paused
        self.reset_metrics()
        clear_trail()

    def restart_round(self):
        self.load_round(self.round_index, paused=self.paused)
        self.message = f"Restarted round {self.params['round']}"

    def next_round(self):
        next_index = (self.round_index + 1) % len(ROUNDS)
        self.load_round(next_index, paused=self.paused)
        self.message = f"Moved to round {self.params['round']}"

    def previous_round(self):
        previous_index = (self.round_index - 1) % len(ROUNDS)
        self.load_round(previous_index, paused=self.paused)
        self.message = f"Moved to round {self.params['round']}"

    def jump_to_round_number(self, number):
        self.load_round(number - 1, paused=self.paused)
        self.message = f"Jumped to round {number}"

    def finalize_current_round(self):
        estimated_period = compute_period_from_crossings(self.crossing_times)
        summary = {
            "round": self.params["round"],
            "g": self.params["g"],
            "length": self.params["length"],
            "damping": self.params["damping"],
            "start_deg": self.params["initial_angle_deg"],
            "max_deg": self.max_angle,
            "max_speed": self.max_speed,
            "period": estimated_period,
            "energy_min": self.min_energy,
            "energy_max": self.max_energy,
            "zero_crossings": self.zero_crossings,
            "note": classify_round(self.params["damping"]),
        }
        self.summaries[self.params["round"]] = summary
        self.round_complete = True

        print(
            f"Round {summary['round']}: max_angle={summary['max_deg']:.2f} deg, "
            f"max_speed={summary['max_speed']:.3f} m/s, "
            f"energy_range={summary['energy_min']:.4f}..{summary['energy_max']:.4f} J, "
            f"zero_crossings={summary['zero_crossings']}, "
            f"estimated_period={summary['period']:.3f} s, note={summary['note']}"
        )

        if self.auto_advance:
            if self.round_index < len(ROUNDS) - 1:
                self.load_round(self.round_index + 1, paused=False)
            else:
                self.paused = True
                self.message = "All rounds complete. Press 1-8, R, B, or N."
        else:
            self.paused = True
            self.message = "Round complete. Auto-advance off. Press N for next round."


# ---------------------------------------------------------------------------
# Scene setup
# ---------------------------------------------------------------------------

scene = canvas(
    title=TITLE,
    width=1200,
    height=760,
    background=vector(0.88, 0.94, 1.0),
    center=vector(0, -1.7, 0),
)

scene.caption = """
Keyboard controls:
Space/P = pause or play | N/Right = next round | B/Left = previous round | R = restart round
1-8 = jump to round | S = single-step while paused | A = toggle auto-advance
T = toggle trail | C = clear trail | I = toggle info labels
Up/Down = increase/decrease damping | G/H = increase/decrease gravity | Q/Esc = quit
"""

pivot = vector(0, 2.0, 0)

support = box(
    pos=pivot + vector(0, 0.12, 0),
    size=vector(6.5, 0.16, 0.20),
    color=vector(0.45, 0.45, 0.45),
)

stand_left = cylinder(
    pos=vector(-3.2, -3.8, 0),
    axis=vector(0, 5.9, 0),
    radius=0.04,
    color=vector(0.45, 0.45, 0.45),
)

stand_right = cylinder(
    pos=vector(3.2, -3.8, 0),
    axis=vector(0, 5.9, 0),
    radius=0.04,
    color=vector(0.45, 0.45, 0.45),
)

floor = box(
    pos=vector(0, -3.9, 0),
    size=vector(8.0, 0.08, 0.25),
    color=vector(0.60, 0.70, 0.75),
)

center_line = cylinder(
    pos=pivot,
    axis=vector(0, -5.4, 0),
    radius=0.01,
    color=vector(0.65, 0.65, 0.65),
)

rod = cylinder(
    pos=pivot,
    axis=vector(0, -2.0, 0),
    radius=0.025,
    color=vector(0.15, 0.15, 0.15),
)

bob = sphere(
    pos=pivot + vector(0, -2.0, 0),
    radius=0.18,
    color=vector(0.10, 0.35, 0.95),
    shininess=0.6,
)

trail = curve(color=vector(0.10, 0.35, 0.95), radius=0.012)

info = label(
    pos=vector(-5.35, 2.1, 0),
    text="",
    height=12,
    box=False,
    align="left",
    color=color.black,
)

summary_label = label(
    pos=vector(-5.35, -2.55, 0),
    text="",
    height=11,
    box=False,
    align="left",
    color=color.black,
)

goal_label = label(
    pos=vector(0, 2.65, 0),
    text="",
    height=15,
    box=False,
    align="center",
    color=vector(0.1, 0.1, 0.1),
)

controls_label = label(
    pos=vector(3.65, 2.05, 0),
    text="",
    height=10,
    box=False,
    align="left",
    color=color.black,
)

energy_marker = box(
    pos=vector(4.2, -3.4, 0),
    size=vector(0.35, 0.01, 0.12),
    color=vector(0.95, 0.55, 0.10),
)

energy_back = box(
    pos=vector(4.2, -1.9, 0),
    size=vector(0.12, 3.0, 0.08),
    color=vector(0.75, 0.78, 0.80),
)

energy_label = label(
    pos=vector(4.65, -1.9, 0),
    text="Energy",
    height=11,
    box=False,
    align="left",
    color=color.black,
)


def clear_trail():
    try:
        trail.clear()
    except Exception:
        pass


state = SimulationState()


# ---------------------------------------------------------------------------
# Visual update helpers
# ---------------------------------------------------------------------------

def set_bob_color(goal):
    if goal in ("high damping test", "near-settle test"):
        bob.color = vector(0.80, 0.30, 0.15)
    elif goal == "low gravity test":
        bob.color = vector(0.45, 0.25, 0.90)
    elif goal == "energy boost test":
        bob.color = vector(0.95, 0.25, 0.10)
    elif goal == "slow wide swing":
        bob.color = vector(0.10, 0.55, 0.75)
    else:
        bob.color = vector(0.10, 0.35, 0.95)
    trail.color = bob.color


def update_visuals():
    params = state.params
    set_bob_color(params["goal"])

    bob_pos = pendulum_position(pivot, params["length"], state.theta)
    bob.pos = bob_pos
    rod.pos = pivot
    rod.axis = bob.pos - pivot

    if state.step_count % TRACE_EVERY_STEPS == 0 and state.show_trail and not state.paused:
        trail.append(pos=bob.pos)

    if state.show_trail:
        trail.visible = True
    else:
        trail.visible = False

    energy = mechanical_energy(
        params["mass"],
        params["g"],
        params["length"],
        state.theta,
        state.omega,
    )

    energy_ratio = 0.0 if state.start_energy <= 0 else max(0.0, min(1.0, energy / state.start_energy))
    energy_marker.pos = vector(4.2, -3.4 + 3.0 * energy_ratio, 0)

    estimated_period = compute_period_from_crossings(state.crossing_times)
    angle_deg = degrees(state.theta)
    speed = abs(state.omega * params["length"])

    angle_bar = make_bar(abs(angle_deg), 90.0)
    speed_bar = make_bar(speed, 3.6)
    energy_bar = make_bar(energy, max(state.start_energy, 0.001))

    status = "PAUSED" if state.paused else "RUNNING"
    if state.single_step:
        status = "SINGLE STEP"

    goal_label.text = (
        f"ROUND {params['round']} / {len(ROUNDS)}: {params['goal'].upper()}\n"
        f"AI next goal: {params['next_goal']}"
    )

    if state.show_info:
        info.visible = True
        summary_label.visible = True
        controls_label.visible = True
        energy_label.visible = True

        info.text = (
            f"Status: {status}\n"
            f"Message: {state.message}\n\n"
            f"g = {params['g']:.3f} m/s^2\n"
            f"length = {params['length']:.3f} m\n"
            f"damping = {params['damping']:.4f}\n"
            f"start angle = {params['initial_angle_deg']:.2f} deg\n"
            f"start omega = {params['initial_omega']:.3f} rad/s\n\n"
            f"t = {state.t:05.2f} s / {ROUND_DURATION:.1f} s\n"
            f"angle = {angle_deg:8.3f} deg {angle_bar}\n"
            f"omega = {state.omega:8.3f} rad/s\n"
            f"speed = {speed:8.3f} m/s {speed_bar}\n"
            f"energy = {energy:8.4f} J {energy_bar}\n"
            f"zero crossings = {state.zero_crossings}\n"
            f"estimated period = {estimated_period:6.3f} s"
        )

        summary_label.text = (
            f"Current round summary\n"
            f"max_angle = {state.max_angle:.2f} deg\n"
            f"max_speed = {state.max_speed:.3f} m/s\n"
            f"energy_range = {state.min_energy:.4f}..{state.max_energy:.4f} J\n"
            f"zero_crossings = {state.zero_crossings}\n"
            f"estimated_period = {estimated_period:.3f} s\n"
            f"note = {classify_round(params['damping'])}"
        )

        controls_label.text = (
            "Keys\n"
            "Space/P pause-play\n"
            "N/Right next round\n"
            "B/Left previous round\n"
            "R restart current\n"
            "1-8 jump to round\n"
            "S single step\n"
            "A auto-advance on/off\n"
            "T trail on/off\n"
            "C clear trail\n"
            "I labels on/off\n"
            "Up/Down damping\n"
            "G/H gravity\n"
            "Q/Esc quit\n\n"
            f"auto-advance: {'on' if state.auto_advance else 'off'}\n"
            f"trail: {'on' if state.show_trail else 'off'}"
        )
    else:
        info.visible = False
        summary_label.visible = False
        controls_label.visible = False
        energy_label.visible = False


# ---------------------------------------------------------------------------
# Keyboard input
# ---------------------------------------------------------------------------

def handle_keydown(evt):
    key = evt.key

    if key in (" ", "p", "P"):
        state.paused = not state.paused
        state.message = "Paused" if state.paused else "Playing"

    elif key in ("n", "N", "right"):
        state.next_round()

    elif key in ("b", "B", "left"):
        state.previous_round()

    elif key in ("r", "R"):
        state.restart_round()

    elif key in ("s", "S"):
        state.single_step = True
        state.paused = True
        state.message = "Single step"

    elif key in ("a", "A"):
        state.auto_advance = not state.auto_advance
        state.message = f"Auto-advance {'on' if state.auto_advance else 'off'}"

    elif key in ("t", "T"):
        state.show_trail = not state.show_trail
        state.message = f"Trail {'on' if state.show_trail else 'off'}"

    elif key in ("c", "C"):
        clear_trail()
        state.message = "Trail cleared"

    elif key in ("i", "I"):
        state.show_info = not state.show_info
        state.message = f"Info labels {'on' if state.show_info else 'off'}"

    elif key == "up":
        state.params["damping"] = max(0.0, state.params["damping"] + DAMPING_STEP)
        state.message = f"Damping increased to {state.params['damping']:.4f}"

    elif key == "down":
        state.params["damping"] = max(0.0, state.params["damping"] - DAMPING_STEP)
        state.message = f"Damping decreased to {state.params['damping']:.4f}"

    elif key in ("g", "G"):
        state.params["g"] = min(25.0, state.params["g"] + GRAVITY_STEP)
        state.message = f"Gravity increased to {state.params['g']:.3f}"

    elif key in ("h", "H"):
        state.params["g"] = max(0.1, state.params["g"] - GRAVITY_STEP)
        state.message = f"Gravity decreased to {state.params['g']:.3f}"

    elif key in ("q", "Q", "esc"):
        state.quit_requested = True
        state.message = "Quit requested"

    elif key in tuple(str(i) for i in range(1, 9)):
        state.jump_to_round_number(int(key))

    update_visuals()


scene.bind("keydown", handle_keydown)


# ---------------------------------------------------------------------------
# Simulation loop
# ---------------------------------------------------------------------------

def physics_step():
    params = state.params

    alpha = -(params["g"] / params["length"]) * math.sin(state.theta) - params["damping"] * state.omega
    state.omega += alpha * TIME_STEP
    state.theta += state.omega * TIME_STEP
    state.t += TIME_STEP
    state.step_count += 1

    angle_deg = degrees(state.theta)
    speed = abs(state.omega * params["length"])
    energy = mechanical_energy(
        params["mass"],
        params["g"],
        params["length"],
        state.theta,
        state.omega,
    )

    state.max_angle = max(state.max_angle, abs(angle_deg))
    state.max_speed = max(state.max_speed, speed)
    state.min_energy = min(state.min_energy, energy)
    state.max_energy = max(state.max_energy, energy)

    if state.previous_theta != 0:
        crossed_positive = state.previous_theta < 0 and state.theta >= 0
        crossed_negative = state.previous_theta > 0 and state.theta <= 0
        if crossed_positive or crossed_negative:
            state.zero_crossings += 1
            state.crossing_times.append(state.t)
    state.previous_theta = state.theta

    if state.t >= ROUND_DURATION and not state.round_complete:
        state.finalize_current_round()


def print_startup():
    print(TITLE)
    print("Keyboard-controlled VPython version based on the terminal pendulum output.")
    print("No CSV logging.")
    print()
    print("Keyboard controls:")
    print("  Space/P      Pause or play")
    print("  N/Right      Next round")
    print("  B/Left       Previous round")
    print("  R            Restart current round")
    print("  1-8          Jump directly to a round")
    print("  S            Single-step while paused")
    print("  A            Toggle auto-advance")
    print("  T            Toggle trail visibility")
    print("  C            Clear current trail")
    print("  I            Toggle info labels")
    print("  Up/Down      Increase/decrease damping")
    print("  G/H          Increase/decrease gravity")
    print("  Q/Esc        Quit")
    print()


def print_final_comparison():
    print()
    print("=" * 96)
    print("FINAL ROUND COMPARISON")
    print("round       g       L     damp  start_deg   max_deg     max_v    period  note")
    for round_number in sorted(state.summaries):
        s = state.summaries[round_number]
        print(
            f"{s['round']:>5} "
            f"{s['g']:>7.2f} "
            f"{s['length']:>7.2f} "
            f"{s['damping']:>8.3f} "
            f"{s['start_deg']:>10.2f} "
            f"{s['max_deg']:>9.2f} "
            f"{s['max_speed']:>9.3f} "
            f"{s['period']:>9.3f}  "
            f"{s['note']}"
        )
    print("=" * 96)


def main():
    print_startup()
    state.load_round(0, paused=False)
    update_visuals()

    while not state.quit_requested:
        rate(DISPLAY_RATE)

        should_step = (not state.paused and not state.round_complete) or state.single_step
        if should_step:
            physics_step()
            state.single_step = False

        update_visuals()

    print_final_comparison()


if __name__ == "__main__":
    main()
