"""
Terminal Rocket Launch Simulation with AI-Controlled Rounds

A Python-only terminal simulation of a vertical rocket launch.

Features:
- No external packages
- Multiple launch rounds
- AI controller changes parameters after each round
- Simulates thrust, gravity, changing mass, fuel burn, air drag, coast, descent, and parachute recovery
- Prints live state to the terminal
- Ends with a final comparison table

Run:
    python terminal_rocket_launch_ai_rounds.py
"""

import math
import random
from dataclasses import dataclass, asdict


# ---------------------------------------------------------------------------
# User-adjustable base settings
# ---------------------------------------------------------------------------

RANDOM_SEED = 7

ROUNDS = 8
DT = 0.05
MAX_TIME = 180.0
PRINT_EVERY = 0.50

GRAVITY = 9.81
AIR_DENSITY = 1.225

# Stop if the rocket reaches the ground after launch.
GROUND_ALTITUDE = 0.0

# Safety clamps
MIN_THRUST = 800.0
MAX_THRUST = 12000.0
MIN_FUEL = 5.0
MAX_FUEL = 300.0
MIN_BURN_RATE = 0.5
MAX_BURN_RATE = 20.0
MIN_DRY_MASS = 10.0
MAX_DRY_MASS = 180.0


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RocketParams:
    thrust_newtons: float
    fuel_kg: float
    dry_mass_kg: float
    burn_rate_kg_s: float
    drag_coefficient: float
    cross_section_area_m2: float
    parachute_area_m2: float
    parachute_deploy_altitude_m: float
    parachute_deploy_when_descending: bool
    wind_drift_m_s: float
    goal: str


@dataclass
class RocketState:
    time_s: float
    altitude_m: float
    velocity_m_s: float
    acceleration_m_s2: float
    fuel_kg: float
    mass_kg: float
    horizontal_drift_m: float
    phase: str
    parachute_deployed: bool
    engine_on: bool


@dataclass
class RoundSummary:
    round_number: int
    goal: str
    thrust_newtons: float
    fuel_kg: float
    burn_rate_kg_s: float
    dry_mass_kg: float
    max_altitude_m: float
    max_speed_m_s: float
    max_acceleration_m_s2: float
    burnout_time_s: float
    flight_time_s: float
    landing_speed_m_s: float
    final_drift_m: float
    result: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def clamp(value, low, high):
    return max(low, min(high, value))


def format_bar(value, max_value, width=24):
    if max_value <= 0:
        filled = 0
    else:
        filled = int(width * clamp(value / max_value, 0.0, 1.0))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def sign_drag_force(velocity, drag_magnitude):
    """
    Drag always opposes motion.
    If velocity is upward, drag is downward.
    If velocity is downward, drag is upward.
    """
    if velocity > 0:
        return -drag_magnitude
    if velocity < 0:
        return drag_magnitude
    return 0.0


def choose_phase(state):
    if state.altitude_m <= 0.01 and state.time_s < 0.2:
        return "on pad"
    if state.engine_on and state.velocity_m_s >= 0:
        return "powered ascent"
    if not state.engine_on and state.velocity_m_s > 2.0:
        return "coasting upward"
    if abs(state.velocity_m_s) <= 2.0 and state.altitude_m > 1.0:
        return "near apogee"
    if state.parachute_deployed and state.velocity_m_s < 0:
        return "parachute descent"
    if state.velocity_m_s < 0:
        return "ballistic descent"
    return "flight"


def print_round_header(round_number, params):
    print()
    print("=" * 100)
    print(f"ROUND {round_number}")
    print("-" * 100)
    print(f"goal: {params.goal}")
    print(
        "parameters: "
        f"thrust={params.thrust_newtons:.1f} N, "
        f"fuel={params.fuel_kg:.1f} kg, "
        f"dry_mass={params.dry_mass_kg:.1f} kg, "
        f"burn_rate={params.burn_rate_kg_s:.2f} kg/s, "
        f"Cd={params.drag_coefficient:.3f}, "
        f"area={params.cross_section_area_m2:.3f} m^2, "
        f"parachute_area={params.parachute_area_m2:.2f} m^2, "
        f"deploy_altitude={params.parachute_deploy_altitude_m:.1f} m"
    )
    print("-" * 100)


def print_state_line(round_number, state, max_altitude_so_far):
    altitude_bar = format_bar(max(state.altitude_m, 0.0), max(max_altitude_so_far, 1.0), 20)
    fuel_bar = format_bar(state.fuel_kg, max(state.mass_kg, 1.0), 16)

    print(
        f"R{round_number:02d} "
        f"t={state.time_s:06.2f}s  "
        f"alt={state.altitude_m:9.2f} m {altitude_bar}  "
        f"vel={state.velocity_m_s:8.2f} m/s  "
        f"acc={state.acceleration_m_s2:8.2f} m/s^2  "
        f"fuel={state.fuel_kg:7.2f} kg {fuel_bar}  "
        f"mass={state.mass_kg:7.2f} kg  "
        f"drift={state.horizontal_drift_m:7.2f} m  "
        f"phase={state.phase}"
    )


# ---------------------------------------------------------------------------
# Physics simulation
# ---------------------------------------------------------------------------

def simulate_round(round_number, params):
    print_round_header(round_number, params)

    state = RocketState(
        time_s=0.0,
        altitude_m=0.0,
        velocity_m_s=0.0,
        acceleration_m_s2=0.0,
        fuel_kg=params.fuel_kg,
        mass_kg=params.dry_mass_kg + params.fuel_kg,
        horizontal_drift_m=0.0,
        phase="on pad",
        parachute_deployed=False,
        engine_on=True,
    )

    max_altitude = 0.0
    max_speed = 0.0
    max_acceleration = 0.0
    burnout_time = None
    landing_speed = 0.0
    launched = False
    previous_print_time = -PRINT_EVERY

    while state.time_s <= MAX_TIME:
        state.mass_kg = params.dry_mass_kg + state.fuel_kg

        # Engine burns fuel only while fuel remains.
        if state.fuel_kg > 0.0:
            fuel_to_burn = min(params.burn_rate_kg_s * DT, state.fuel_kg)
            state.fuel_kg -= fuel_to_burn
            thrust = params.thrust_newtons
            state.engine_on = True
        else:
            thrust = 0.0
            state.engine_on = False
            if burnout_time is None:
                burnout_time = state.time_s

        if state.altitude_m > 0.1:
            launched = True

        # Parachute deploys only on descent if enabled.
        if (
            params.parachute_deploy_when_descending
            and not state.parachute_deployed
            and launched
            and state.velocity_m_s < 0
            and state.altitude_m <= params.parachute_deploy_altitude_m
        ):
            state.parachute_deployed = True

        # Drag area changes after parachute deployment.
        active_area = params.cross_section_area_m2
        active_cd = params.drag_coefficient
        if state.parachute_deployed:
            active_area += params.parachute_area_m2
            active_cd = max(active_cd, 1.30)

        drag_magnitude = 0.5 * AIR_DENSITY * active_cd * active_area * state.velocity_m_s * state.velocity_m_s
        drag_force = sign_drag_force(state.velocity_m_s, drag_magnitude)

        gravity_force = state.mass_kg * GRAVITY
        net_force = thrust + drag_force - gravity_force

        state.acceleration_m_s2 = net_force / max(state.mass_kg, 0.001)

        # Integrate velocity and altitude.
        state.velocity_m_s += state.acceleration_m_s2 * DT
        state.altitude_m += state.velocity_m_s * DT

        # Simple horizontal drift from wind. Parachute increases drift effect.
        drift_multiplier = 2.5 if state.parachute_deployed else 1.0
        state.horizontal_drift_m += params.wind_drift_m_s * drift_multiplier * DT

        state.time_s += DT

        max_altitude = max(max_altitude, state.altitude_m)
        max_speed = max(max_speed, abs(state.velocity_m_s))
        max_acceleration = max(max_acceleration, abs(state.acceleration_m_s2))

        state.phase = choose_phase(state)

        if state.time_s - previous_print_time >= PRINT_EVERY:
            previous_print_time = state.time_s
            print_state_line(round_number, state, max_altitude)

        # Landing condition.
        if launched and state.altitude_m <= GROUND_ALTITUDE:
            landing_speed = state.velocity_m_s
            state.altitude_m = GROUND_ALTITUDE
            break

        # Failed launch condition.
        if not launched and state.fuel_kg <= 0.0 and state.velocity_m_s <= 0.0:
            landing_speed = 0.0
            break

    if burnout_time is None:
        burnout_time = state.time_s

    flight_time = state.time_s

    if not launched:
        result = "failed to lift off"
    elif abs(landing_speed) <= 8.0:
        result = "soft recovery"
    elif state.parachute_deployed and abs(landing_speed) <= 18.0:
        result = "rough parachute landing"
    elif max_altitude > 3000:
        result = "high altitude, hard landing"
    else:
        result = "hard landing"

    summary = RoundSummary(
        round_number=round_number,
        goal=params.goal,
        thrust_newtons=params.thrust_newtons,
        fuel_kg=params.fuel_kg,
        burn_rate_kg_s=params.burn_rate_kg_s,
        dry_mass_kg=params.dry_mass_kg,
        max_altitude_m=max_altitude,
        max_speed_m_s=max_speed,
        max_acceleration_m_s2=max_acceleration,
        burnout_time_s=burnout_time,
        flight_time_s=flight_time,
        landing_speed_m_s=landing_speed,
        final_drift_m=state.horizontal_drift_m,
        result=result,
    )

    print("-" * 100)
    print(
        f"summary: max_altitude={summary.max_altitude_m:.2f} m, "
        f"max_speed={summary.max_speed_m_s:.2f} m/s, "
        f"max_acceleration={summary.max_acceleration_m_s2:.2f} m/s^2, "
        f"burnout_time={summary.burnout_time_s:.2f} s, "
        f"flight_time={summary.flight_time_s:.2f} s, "
        f"landing_speed={summary.landing_speed_m_s:.2f} m/s, "
        f"drift={summary.final_drift_m:.2f} m, "
        f"result={summary.result}"
    )

    return summary


# ---------------------------------------------------------------------------
# AI controller
# ---------------------------------------------------------------------------

def initial_params():
    return RocketParams(
        thrust_newtons=3600.0,
        fuel_kg=95.0,
        dry_mass_kg=55.0,
        burn_rate_kg_s=5.8,
        drag_coefficient=0.55,
        cross_section_area_m2=0.22,
        parachute_area_m2=5.0,
        parachute_deploy_altitude_m=650.0,
        parachute_deploy_when_descending=True,
        wind_drift_m_s=1.2,
        goal="baseline launch",
    )


def ai_controller(previous_params, summary, round_number):
    """
    Simple rule-based AI controller.

    It reads the previous round summary and chooses the next experiment.
    The controller is intentionally transparent and deterministic enough to inspect.
    """
    next_params = RocketParams(**asdict(previous_params))

    goal_cycle = [
        "maximize altitude",
        "reduce landing speed",
        "high thrust short burn",
        "low thrust long burn",
        "heavy payload test",
        "low drag test",
        "wind recovery test",
    ]

    next_goal = goal_cycle[(round_number - 1) % len(goal_cycle)]
    next_params.goal = next_goal

    if next_goal == "maximize altitude":
        next_params.thrust_newtons *= 1.25
        next_params.fuel_kg *= 1.20
        next_params.burn_rate_kg_s *= 1.05
        next_params.cross_section_area_m2 *= 0.95
        next_params.drag_coefficient *= 0.95

    elif next_goal == "reduce landing speed":
        next_params.parachute_area_m2 *= 1.55
        next_params.parachute_deploy_altitude_m *= 1.20
        next_params.thrust_newtons *= 0.95
        next_params.fuel_kg *= 0.95

    elif next_goal == "high thrust short burn":
        next_params.thrust_newtons *= 1.45
        next_params.burn_rate_kg_s *= 1.65
        next_params.fuel_kg *= 0.90
        next_params.dry_mass_kg *= 1.03

    elif next_goal == "low thrust long burn":
        next_params.thrust_newtons *= 0.72
        next_params.burn_rate_kg_s *= 0.50
        next_params.fuel_kg *= 1.25

    elif next_goal == "heavy payload test":
        next_params.dry_mass_kg *= 1.65
        next_params.thrust_newtons *= 1.20
        next_params.fuel_kg *= 1.10

    elif next_goal == "low drag test":
        next_params.drag_coefficient *= 0.65
        next_params.cross_section_area_m2 *= 0.70
        next_params.thrust_newtons *= 0.98

    elif next_goal == "wind recovery test":
        next_params.wind_drift_m_s *= 2.75
        next_params.parachute_area_m2 *= 1.25
        next_params.parachute_deploy_altitude_m *= 1.10

    # Reactive corrections based on prior performance.
    if summary.result == "failed to lift off":
        next_params.thrust_newtons *= 1.40
        next_params.dry_mass_kg *= 0.90

    if summary.landing_speed_m_s < -25.0:
        next_params.parachute_area_m2 *= 1.20
        next_params.parachute_deploy_altitude_m *= 1.10

    if summary.max_altitude_m < 500.0:
        next_params.thrust_newtons *= 1.15
        next_params.fuel_kg *= 1.10

    # Add small deterministic variation.
    random_factor = 1.0 + random.uniform(-0.04, 0.04)
    next_params.thrust_newtons *= random_factor
    next_params.fuel_kg *= 1.0 + random.uniform(-0.03, 0.03)
    next_params.burn_rate_kg_s *= 1.0 + random.uniform(-0.03, 0.03)

    # Clamp all physical values.
    next_params.thrust_newtons = clamp(next_params.thrust_newtons, MIN_THRUST, MAX_THRUST)
    next_params.fuel_kg = clamp(next_params.fuel_kg, MIN_FUEL, MAX_FUEL)
    next_params.dry_mass_kg = clamp(next_params.dry_mass_kg, MIN_DRY_MASS, MAX_DRY_MASS)
    next_params.burn_rate_kg_s = clamp(next_params.burn_rate_kg_s, MIN_BURN_RATE, MAX_BURN_RATE)
    next_params.drag_coefficient = clamp(next_params.drag_coefficient, 0.10, 2.50)
    next_params.cross_section_area_m2 = clamp(next_params.cross_section_area_m2, 0.03, 3.00)
    next_params.parachute_area_m2 = clamp(next_params.parachute_area_m2, 0.0, 80.0)
    next_params.parachute_deploy_altitude_m = clamp(next_params.parachute_deploy_altitude_m, 50.0, 5000.0)
    next_params.wind_drift_m_s = clamp(next_params.wind_drift_m_s, -20.0, 20.0)

    print(f"AI controller selected next goal: {next_params.goal}")
    return next_params


# ---------------------------------------------------------------------------
# Final report
# ---------------------------------------------------------------------------

def print_final_comparison(summaries):
    print()
    print("=" * 120)
    print("FINAL ROUND COMPARISON")
    print("=" * 120)
    print(
        f"{'round':>5} "
        f"{'goal':<22} "
        f"{'thrust':>9} "
        f"{'fuel':>8} "
        f"{'burn':>7} "
        f"{'dry':>7} "
        f"{'max_alt':>10} "
        f"{'max_v':>9} "
        f"{'burnout':>8} "
        f"{'flight':>8} "
        f"{'land_v':>9} "
        f"{'drift':>8} "
        f"{'result':<24}"
    )

    for s in summaries:
        print(
            f"{s.round_number:5d} "
            f"{s.goal:<22.22} "
            f"{s.thrust_newtons:9.0f} "
            f"{s.fuel_kg:8.1f} "
            f"{s.burn_rate_kg_s:7.2f} "
            f"{s.dry_mass_kg:7.1f} "
            f"{s.max_altitude_m:10.1f} "
            f"{s.max_speed_m_s:9.1f} "
            f"{s.burnout_time_s:8.1f} "
            f"{s.flight_time_s:8.1f} "
            f"{s.landing_speed_m_s:9.1f} "
            f"{s.final_drift_m:8.1f} "
            f"{s.result:<24}"
        )

    best_altitude = max(summaries, key=lambda s: s.max_altitude_m)
    softest_landing = min(summaries, key=lambda s: abs(s.landing_speed_m_s))
    fastest = max(summaries, key=lambda s: s.max_speed_m_s)

    print("-" * 120)
    print(
        f"highest altitude: round {best_altitude.round_number} "
        f"({best_altitude.max_altitude_m:.1f} m, goal={best_altitude.goal})"
    )
    print(
        f"softest landing: round {softest_landing.round_number} "
        f"({softest_landing.landing_speed_m_s:.1f} m/s, goal={softest_landing.goal})"
    )
    print(
        f"fastest flight: round {fastest.round_number} "
        f"({fastest.max_speed_m_s:.1f} m/s, goal={fastest.goal})"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    random.seed(RANDOM_SEED)

    print("Terminal Rocket Launch Simulation with AI-Controlled Rounds")
    print("No graphics. No external packages. State is printed directly to the terminal.")
    print("The model includes thrust, gravity, fuel burn, changing mass, drag, wind drift, and parachute recovery.")

    params = initial_params()
    summaries = []

    for round_number in range(1, ROUNDS + 1):
        summary = simulate_round(round_number, params)
        summaries.append(summary)

        if round_number < ROUNDS:
            params = ai_controller(params, summary, round_number)

    print_final_comparison(summaries)


if __name__ == "__main__":
    main()
