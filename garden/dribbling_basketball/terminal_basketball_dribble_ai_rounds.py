"""
Terminal Basketball Dribbling Simulation with AI-Controlled Rounds

A Python-only terminal simulation of a basketball being dribbled.

Features:
- No external packages
- Multiple dribbling rounds
- AI controller changes the strategy after each round
- Simulates gravity, floor bounce, hand pushes, spin, horizontal drift, fatigue, rhythm timing, and control loss
- Prints state directly to the terminal
- Ends with a final comparison table

Run:
    python terminal_basketball_dribble_ai_rounds.py
"""

import math
import random
from dataclasses import dataclass, asdict


# ---------------------------------------------------------------------------
# User-adjustable settings
# ---------------------------------------------------------------------------

RANDOM_SEED = 13

ROUNDS = 8
DT = 0.01
ROUND_DURATION = 14.0
PRINT_EVERY = 0.25

GRAVITY = -9.81

BALL_RADIUS_M = 0.12
HAND_HEIGHT_MIN_M = 0.55
CONTROL_LOSS_HEIGHT_M = 1.85
CONTROL_LOSS_SIDE_M = 1.20

# A dribble is counted when the ball hits the floor and bounces back up.
MIN_BOUNCE_INTERVAL = 0.12


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DribbleParams:
    goal: str
    target_height_m: float
    hand_push_strength: float
    hand_timing_s: float
    hand_contact_window_m: float
    ball_elasticity: float
    floor_friction: float
    wrist_spin_impulse: float
    lateral_control_strength: float
    crossover_strength: float
    defender_pressure: float
    fatigue_rate: float
    weak_hand_penalty: float


@dataclass
class DribbleState:
    time_s: float
    height_m: float
    vertical_velocity_m_s: float
    horizontal_x_m: float
    horizontal_velocity_m_s: float
    spin_rad_s: float
    fatigue: float
    phase: str
    dribble_count: int
    lost_control_count: int
    last_bounce_time_s: float
    last_hand_contact_time_s: float


@dataclass
class RoundSummary:
    round_number: int
    goal: str
    target_height_m: float
    hand_push_strength: float
    elasticity: float
    avg_height_m: float
    max_height_m: float
    max_speed_m_s: float
    max_spin_rad_s: float
    dribbles: int
    hand_contacts: int
    floor_bounces: int
    rhythm_error_s: float
    lost_control_count: int
    final_fatigue: float
    final_side_offset_m: float
    control_score: float
    result: str


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def clamp(value, low, high):
    return max(low, min(high, value))


def format_bar(value, max_value, width=20):
    if max_value <= 0:
        filled = 0
    else:
        filled = int(width * clamp(value / max_value, 0.0, 1.0))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def sign(value):
    if value > 0:
        return 1.0
    if value < 0:
        return -1.0
    return 0.0


def choose_phase(state):
    if state.height_m <= BALL_RADIUS_M + 0.01 and state.vertical_velocity_m_s > 0:
        return "floor bounce"
    if state.vertical_velocity_m_s < -0.15:
        return "falling"
    if state.vertical_velocity_m_s > 0.15:
        return "rising"
    if state.height_m > 1.0:
        return "near peak"
    return "floating"


def ascii_ball_position(height_m, x_m):
    """
    Compact text visualization:
    - vertical bar represents rough height
    - x offset shows left/right control
    """
    max_h = 1.8
    levels = 8
    level = int(clamp(height_m / max_h, 0.0, 1.0) * levels)
    side_slots = 17
    center = side_slots // 2
    x_slot = center + int(clamp(x_m / CONTROL_LOSS_SIDE_M, -1.0, 1.0) * center)
    chars = [" "] * side_slots
    chars[max(0, min(side_slots - 1, x_slot))] = "O"
    chars[center] = "|" if chars[center] == " " else chars[center]
    return f"h{level:02d} " + "".join(chars)


def print_round_header(round_number, params):
    print()
    print("=" * 110)
    print(f"ROUND {round_number}")
    print("-" * 110)
    print(f"goal: {params.goal}")
    print(
        "parameters: "
        f"target_height={params.target_height_m:.2f} m, "
        f"hand_push={params.hand_push_strength:.2f}, "
        f"timing={params.hand_timing_s:.2f} s, "
        f"contact_window={params.hand_contact_window_m:.2f} m, "
        f"elasticity={params.ball_elasticity:.3f}, "
        f"floor_friction={params.floor_friction:.3f}, "
        f"spin_impulse={params.wrist_spin_impulse:.2f}, "
        f"crossover={params.crossover_strength:.2f}, "
        f"defender_pressure={params.defender_pressure:.2f}, "
        f"fatigue_rate={params.fatigue_rate:.3f}, "
        f"weak_hand_penalty={params.weak_hand_penalty:.2f}"
    )
    print("-" * 110)


def print_state_line(round_number, state, target_height, control_score_estimate):
    height_bar = format_bar(state.height_m, 1.8, 18)
    spin_bar = format_bar(abs(state.spin_rad_s), 12.0, 12)
    fatigue_bar = format_bar(state.fatigue, 1.0, 10)
    visual = ascii_ball_position(state.height_m, state.horizontal_x_m)

    print(
        f"R{round_number:02d} "
        f"t={state.time_s:05.2f}s  "
        f"h={state.height_m:5.3f} m {height_bar}  "
        f"vy={state.vertical_velocity_m_s:7.3f} m/s  "
        f"x={state.horizontal_x_m:6.3f} m  "
        f"vx={state.horizontal_velocity_m_s:7.3f} m/s  "
        f"spin={state.spin_rad_s:7.3f} rad/s {spin_bar}  "
        f"fatigue={state.fatigue:4.2f} {fatigue_bar}  "
        f"dribbles={state.dribble_count:2d}  "
        f"score~={control_score_estimate:5.1f}  "
        f"phase={state.phase:<16} {visual}"
    )


# ---------------------------------------------------------------------------
# Physics simulation
# ---------------------------------------------------------------------------

def simulate_round(round_number, params):
    print_round_header(round_number, params)

    state = DribbleState(
        time_s=0.0,
        height_m=params.target_height_m,
        vertical_velocity_m_s=0.0,
        horizontal_x_m=0.0,
        horizontal_velocity_m_s=0.0,
        spin_rad_s=0.0,
        fatigue=0.0,
        phase="release",
        dribble_count=0,
        lost_control_count=0,
        last_bounce_time_s=-99.0,
        last_hand_contact_time_s=-99.0,
    )

    total_height = 0.0
    samples = 0
    max_height = state.height_m
    max_speed = 0.0
    max_spin = 0.0
    hand_contacts = 0
    floor_bounces = 0
    rhythm_errors = []

    previous_print_time = -PRINT_EVERY
    last_dribble_period_start = None

    while state.time_s <= ROUND_DURATION:
        # Fatigue grows over time and weakens control/push.
        state.fatigue = clamp(state.fatigue + params.fatigue_rate * DT, 0.0, 1.0)
        fatigue_multiplier = 1.0 - 0.45 * state.fatigue
        weak_hand_multiplier = 1.0 - params.weak_hand_penalty

        # Gravity.
        state.vertical_velocity_m_s += GRAVITY * DT

        # Defender pressure adds small random side disturbance.
        defender_jab = random.uniform(-1.0, 1.0) * params.defender_pressure * DT
        state.horizontal_velocity_m_s += defender_jab

        # Crossover target alternates left and right every half rhythm.
        if params.crossover_strength > 0:
            crossover_phase = math.sin(2.0 * math.pi * state.time_s / max(params.hand_timing_s * 2.0, 0.1))
            target_x = params.crossover_strength * crossover_phase
        else:
            target_x = 0.0

        # Lateral hand correction.
        lateral_error = target_x - state.horizontal_x_m
        state.horizontal_velocity_m_s += lateral_error * params.lateral_control_strength * fatigue_multiplier * DT

        # Air-like horizontal damping.
        state.horizontal_velocity_m_s *= (1.0 - 0.35 * DT)

        # Integrate position.
        state.height_m += state.vertical_velocity_m_s * DT
        state.horizontal_x_m += state.horizontal_velocity_m_s * DT

        # Floor collision.
        if state.height_m <= BALL_RADIUS_M:
            state.height_m = BALL_RADIUS_M

            if state.vertical_velocity_m_s < 0:
                impact_speed = abs(state.vertical_velocity_m_s)

                # Floor bounce loses energy. Spin can convert into slight lateral motion.
                state.vertical_velocity_m_s = impact_speed * params.ball_elasticity

                # Spin creates small sideways drift after bounce.
                state.horizontal_velocity_m_s += 0.012 * state.spin_rad_s

                # Floor friction reduces horizontal sliding and spin.
                state.horizontal_velocity_m_s *= (1.0 - params.floor_friction)
                state.spin_rad_s *= (1.0 - 0.25 * params.floor_friction)

                if state.time_s - state.last_bounce_time_s >= MIN_BOUNCE_INTERVAL:
                    floor_bounces += 1
                    state.dribble_count += 1

                    if last_dribble_period_start is not None:
                        observed_period = state.time_s - last_dribble_period_start
                        rhythm_errors.append(abs(observed_period - params.hand_timing_s))
                    last_dribble_period_start = state.time_s

                    state.last_bounce_time_s = state.time_s

        # Hand contact condition:
        # The hand tries to catch/push the ball near the target height while it is rising or near the top.
        near_target = abs(state.height_m - params.target_height_m) <= params.hand_contact_window_m
        enough_time_since_contact = state.time_s - state.last_hand_contact_time_s >= params.hand_timing_s * 0.55
        hand_should_act = near_target and state.vertical_velocity_m_s > -0.8 and enough_time_since_contact

        if hand_should_act:
            hand_push = params.hand_push_strength * fatigue_multiplier * weak_hand_multiplier

            # Push downward to continue the dribble.
            state.vertical_velocity_m_s = -abs(hand_push)

            # Wrist snap adds spin.
            state.spin_rad_s += params.wrist_spin_impulse * fatigue_multiplier

            # Hand also corrects sideways control.
            state.horizontal_velocity_m_s += -state.horizontal_x_m * params.lateral_control_strength * 0.08

            hand_contacts += 1
            state.last_hand_contact_time_s = state.time_s
            state.phase = "hand push"

        # Spin decays.
        state.spin_rad_s *= (1.0 - 0.05 * DT)

        # Control loss checks.
        too_high = state.height_m > CONTROL_LOSS_HEIGHT_M
        too_sideways = abs(state.horizontal_x_m) > CONTROL_LOSS_SIDE_M
        too_dead = state.dribble_count > 1 and state.height_m <= BALL_RADIUS_M and abs(state.vertical_velocity_m_s) < 0.35

        if too_high or too_sideways or too_dead:
            state.lost_control_count += 1

            # Recover by pulling the ball back toward a controllable state.
            state.height_m = min(state.height_m, params.target_height_m)
            state.horizontal_x_m *= 0.35
            state.horizontal_velocity_m_s *= -0.25
            state.vertical_velocity_m_s = -abs(params.hand_push_strength * 0.65)
            state.spin_rad_s *= 0.50
            state.phase = "control recovery"

        if state.phase != "hand push" and state.phase != "control recovery":
            state.phase = choose_phase(state)

        # Metrics.
        speed = math.sqrt(state.vertical_velocity_m_s ** 2 + state.horizontal_velocity_m_s ** 2)
        total_height += state.height_m
        samples += 1
        max_height = max(max_height, state.height_m)
        max_speed = max(max_speed, speed)
        max_spin = max(max_spin, abs(state.spin_rad_s))

        avg_height_so_far = total_height / max(samples, 1)
        rhythm_error_so_far = sum(rhythm_errors) / len(rhythm_errors) if rhythm_errors else 0.0
        height_error = abs(avg_height_so_far - params.target_height_m)
        side_error = abs(state.horizontal_x_m)
        control_score_estimate = 100.0
        control_score_estimate -= 32.0 * height_error
        control_score_estimate -= 20.0 * rhythm_error_so_far
        control_score_estimate -= 18.0 * side_error
        control_score_estimate -= 12.0 * state.lost_control_count
        control_score_estimate -= 8.0 * state.fatigue
        control_score_estimate = clamp(control_score_estimate, 0.0, 100.0)

        if state.time_s - previous_print_time >= PRINT_EVERY:
            previous_print_time = state.time_s
            print_state_line(round_number, state, params.target_height_m, control_score_estimate)

        state.time_s += DT

    avg_height = total_height / max(samples, 1)
    rhythm_error = sum(rhythm_errors) / len(rhythm_errors) if rhythm_errors else params.hand_timing_s

    control_score = 100.0
    control_score -= 35.0 * abs(avg_height - params.target_height_m)
    control_score -= 25.0 * rhythm_error
    control_score -= 15.0 * abs(state.horizontal_x_m)
    control_score -= 14.0 * state.lost_control_count
    control_score -= 10.0 * state.fatigue
    control_score += min(state.dribble_count, 30) * 0.8
    control_score = clamp(control_score, 0.0, 100.0)

    if state.lost_control_count >= 5:
        result = "unstable handle"
    elif control_score >= 88:
        result = "clean controlled dribble"
    elif control_score >= 72:
        result = "usable dribble"
    elif control_score >= 55:
        result = "rough but maintained"
    else:
        result = "poor control"

    summary = RoundSummary(
        round_number=round_number,
        goal=params.goal,
        target_height_m=params.target_height_m,
        hand_push_strength=params.hand_push_strength,
        elasticity=params.ball_elasticity,
        avg_height_m=avg_height,
        max_height_m=max_height,
        max_speed_m_s=max_speed,
        max_spin_rad_s=max_spin,
        dribbles=state.dribble_count,
        hand_contacts=hand_contacts,
        floor_bounces=floor_bounces,
        rhythm_error_s=rhythm_error,
        lost_control_count=state.lost_control_count,
        final_fatigue=state.fatigue,
        final_side_offset_m=state.horizontal_x_m,
        control_score=control_score,
        result=result,
    )

    print("-" * 110)
    print(
        f"summary: avg_height={summary.avg_height_m:.3f} m, "
        f"max_height={summary.max_height_m:.3f} m, "
        f"max_speed={summary.max_speed_m_s:.3f} m/s, "
        f"max_spin={summary.max_spin_rad_s:.3f} rad/s, "
        f"dribbles={summary.dribbles}, "
        f"hand_contacts={summary.hand_contacts}, "
        f"floor_bounces={summary.floor_bounces}, "
        f"rhythm_error={summary.rhythm_error_s:.3f} s, "
        f"lost_control={summary.lost_control_count}, "
        f"fatigue={summary.final_fatigue:.3f}, "
        f"side_offset={summary.final_side_offset_m:.3f} m, "
        f"control_score={summary.control_score:.1f}, "
        f"result={summary.result}"
    )

    return summary


# ---------------------------------------------------------------------------
# AI controller
# ---------------------------------------------------------------------------

def initial_params():
    return DribbleParams(
        goal="baseline control",
        target_height_m=1.00,
        hand_push_strength=4.30,
        hand_timing_s=0.72,
        hand_contact_window_m=0.12,
        ball_elasticity=0.76,
        floor_friction=0.14,
        wrist_spin_impulse=0.55,
        lateral_control_strength=4.20,
        crossover_strength=0.00,
        defender_pressure=0.00,
        fatigue_rate=0.012,
        weak_hand_penalty=0.00,
    )


def ai_controller(previous_params, summary, round_number):
    next_params = DribbleParams(**asdict(previous_params))

    goal_cycle = [
        "low fast dribble",
        "high power dribble",
        "crossover rhythm",
        "fatigue test",
        "slippery floor test",
        "weak-hand control",
        "defender pressure test",
    ]

    next_goal = goal_cycle[(round_number - 1) % len(goal_cycle)]
    next_params.goal = next_goal

    if next_goal == "low fast dribble":
        next_params.target_height_m = 0.62
        next_params.hand_push_strength *= 0.82
        next_params.hand_timing_s *= 0.72
        next_params.hand_contact_window_m *= 1.10
        next_params.lateral_control_strength *= 1.10
        next_params.wrist_spin_impulse *= 0.90
        next_params.crossover_strength = 0.00
        next_params.defender_pressure = 0.00
        next_params.weak_hand_penalty = 0.00

    elif next_goal == "high power dribble":
        next_params.target_height_m = 1.30
        next_params.hand_push_strength *= 1.45
        next_params.hand_timing_s *= 1.18
        next_params.hand_contact_window_m *= 1.05
        next_params.wrist_spin_impulse *= 1.35
        next_params.crossover_strength = 0.00
        next_params.defender_pressure = 0.02
        next_params.weak_hand_penalty = 0.00

    elif next_goal == "crossover rhythm":
        next_params.target_height_m = 0.88
        next_params.hand_push_strength *= 1.02
        next_params.hand_timing_s *= 0.92
        next_params.hand_contact_window_m *= 1.15
        next_params.lateral_control_strength *= 1.20
        next_params.crossover_strength = 0.42
        next_params.wrist_spin_impulse *= 1.15
        next_params.defender_pressure = 0.03
        next_params.weak_hand_penalty = 0.00

    elif next_goal == "fatigue test":
        next_params.target_height_m = 0.95
        next_params.hand_push_strength *= 1.05
        next_params.hand_timing_s *= 1.00
        next_params.fatigue_rate *= 3.50
        next_params.crossover_strength = 0.18
        next_params.defender_pressure = 0.02
        next_params.weak_hand_penalty = 0.00

    elif next_goal == "slippery floor test":
        next_params.target_height_m = 0.90
        next_params.hand_push_strength *= 1.00
        next_params.hand_timing_s *= 0.96
        next_params.floor_friction *= 0.30
        next_params.lateral_control_strength *= 0.85
        next_params.crossover_strength = 0.30
        next_params.defender_pressure = 0.04
        next_params.weak_hand_penalty = 0.00

    elif next_goal == "weak-hand control":
        next_params.target_height_m = 0.82
        next_params.hand_push_strength *= 1.12
        next_params.hand_timing_s *= 1.03
        next_params.hand_contact_window_m *= 1.20
        next_params.lateral_control_strength *= 0.88
        next_params.crossover_strength = 0.22
        next_params.defender_pressure = 0.03
        next_params.weak_hand_penalty = 0.22

    elif next_goal == "defender pressure test":
        next_params.target_height_m = 0.72
        next_params.hand_push_strength *= 1.08
        next_params.hand_timing_s *= 0.86
        next_params.hand_contact_window_m *= 1.25
        next_params.lateral_control_strength *= 1.35
        next_params.crossover_strength = 0.50
        next_params.defender_pressure = 0.13
        next_params.weak_hand_penalty = 0.05

    # Reactive tuning based on prior outcome.
    if summary.lost_control_count > 0:
        next_params.lateral_control_strength *= 1.15
        next_params.hand_contact_window_m *= 1.08
        next_params.hand_push_strength *= 0.96

    if summary.rhythm_error_s > 0.10:
        next_params.hand_timing_s *= 0.96
        next_params.hand_contact_window_m *= 1.08

    if summary.avg_height_m > summary.target_height_m + 0.20:
        next_params.hand_push_strength *= 0.94

    if summary.avg_height_m < summary.target_height_m - 0.20:
        next_params.hand_push_strength *= 1.08

    if summary.control_score < 65:
        next_params.lateral_control_strength *= 1.20
        next_params.defender_pressure *= 0.85
        next_params.crossover_strength *= 0.85

    # Small variation between rounds.
    next_params.hand_push_strength *= 1.0 + random.uniform(-0.035, 0.035)
    next_params.ball_elasticity *= 1.0 + random.uniform(-0.015, 0.015)
    next_params.hand_timing_s *= 1.0 + random.uniform(-0.025, 0.025)

    # Clamp values.
    next_params.target_height_m = clamp(next_params.target_height_m, 0.45, 1.55)
    next_params.hand_push_strength = clamp(next_params.hand_push_strength, 1.2, 9.0)
    next_params.hand_timing_s = clamp(next_params.hand_timing_s, 0.28, 1.25)
    next_params.hand_contact_window_m = clamp(next_params.hand_contact_window_m, 0.05, 0.35)
    next_params.ball_elasticity = clamp(next_params.ball_elasticity, 0.45, 0.92)
    next_params.floor_friction = clamp(next_params.floor_friction, 0.01, 0.65)
    next_params.wrist_spin_impulse = clamp(next_params.wrist_spin_impulse, 0.0, 5.0)
    next_params.lateral_control_strength = clamp(next_params.lateral_control_strength, 0.5, 12.0)
    next_params.crossover_strength = clamp(next_params.crossover_strength, 0.0, 0.85)
    next_params.defender_pressure = clamp(next_params.defender_pressure, 0.0, 0.45)
    next_params.fatigue_rate = clamp(next_params.fatigue_rate, 0.0, 0.10)
    next_params.weak_hand_penalty = clamp(next_params.weak_hand_penalty, 0.0, 0.45)

    print(f"AI controller selected next goal: {next_params.goal}")
    return next_params


# ---------------------------------------------------------------------------
# Final report
# ---------------------------------------------------------------------------

def print_final_comparison(summaries):
    print()
    print("=" * 130)
    print("FINAL ROUND COMPARISON")
    print("=" * 130)
    print(
        f"{'round':>5} "
        f"{'goal':<22} "
        f"{'target':>7} "
        f"{'avg_h':>7} "
        f"{'max_h':>7} "
        f"{'max_v':>7} "
        f"{'spin':>7} "
        f"{'drib':>5} "
        f"{'hand':>5} "
        f"{'bounce':>7} "
        f"{'rhythm':>8} "
        f"{'lost':>5} "
        f"{'fatigue':>8} "
        f"{'side':>7} "
        f"{'score':>7} "
        f"{'result':<24}"
    )

    for s in summaries:
        print(
            f"{s.round_number:5d} "
            f"{s.goal:<22.22} "
            f"{s.target_height_m:7.2f} "
            f"{s.avg_height_m:7.2f} "
            f"{s.max_height_m:7.2f} "
            f"{s.max_speed_m_s:7.2f} "
            f"{s.max_spin_rad_s:7.2f} "
            f"{s.dribbles:5d} "
            f"{s.hand_contacts:5d} "
            f"{s.floor_bounces:7d} "
            f"{s.rhythm_error_s:8.3f} "
            f"{s.lost_control_count:5d} "
            f"{s.final_fatigue:8.3f} "
            f"{s.final_side_offset_m:7.3f} "
            f"{s.control_score:7.1f} "
            f"{s.result:<24}"
        )

    best_control = max(summaries, key=lambda s: s.control_score)
    most_dribbles = max(summaries, key=lambda s: s.dribbles)
    lowest_rhythm_error = min(summaries, key=lambda s: s.rhythm_error_s)

    print("-" * 130)
    print(
        f"best control: round {best_control.round_number} "
        f"({best_control.control_score:.1f}, goal={best_control.goal})"
    )
    print(
        f"most dribbles: round {most_dribbles.round_number} "
        f"({most_dribbles.dribbles}, goal={most_dribbles.goal})"
    )
    print(
        f"lowest rhythm error: round {lowest_rhythm_error.round_number} "
        f"({lowest_rhythm_error.rhythm_error_s:.3f} s, goal={lowest_rhythm_error.goal})"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    random.seed(RANDOM_SEED)

    print("Terminal Basketball Dribbling Simulation with AI-Controlled Rounds")
    print("No graphics. No external packages. State is printed directly to the terminal.")
    print("The model includes gravity, floor bounce, hand pushes, spin, side drift, fatigue, rhythm, and control loss.")

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
