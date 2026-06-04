"""
Terminal Pendulum Simulation with AI-Controlled Rounds

No VPython. No external packages.

The simulation runs several pendulum-swing rounds. After each round, a simple
AI controller inspects the previous motion and changes parameters for the next
round: gravity, pendulum length, damping, initial angle, and initial angular
velocity.

Run:
    python pendulum_ai_rounds_simulation.py
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import sin, cos, radians, degrees, sqrt
import random
import time


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

@dataclass
class SimConfig:
    rounds: int = 8
    duration: float = 12.0
    dt: float = 0.02
    print_every_steps: int = 20
    sleep_between_prints: float = 0.0
    seed: int = 7


@dataclass
class PendulumParams:
    gravity: float = 9.81              # m/s^2
    length: float = 1.2                # meters
    mass: float = 1.0                  # kg
    damping: float = 0.035             # angular damping coefficient
    initial_angle_deg: float = 35.0    # degrees
    initial_omega: float = 0.0         # rad/s


@dataclass
class PendulumState:
    t: float
    theta: float       # radians
    omega: float       # rad/s
    alpha: float       # rad/s^2
    x: float           # bob x position
    y: float           # bob y position
    kinetic: float
    potential: float
    total_energy: float


@dataclass
class RoundSummary:
    round_number: int
    params: PendulumParams
    max_angle_deg: float
    max_speed: float
    max_energy: float
    min_energy: float
    final_angle_deg: float
    final_omega: float
    zero_crossings: int
    estimated_period: float | None
    stability_note: str


# -----------------------------------------------------------------------------
# Physics
# -----------------------------------------------------------------------------

def angular_acceleration(theta: float, omega: float, params: PendulumParams) -> float:
    """
    Nonlinear pendulum equation with simple damping:
        theta'' = -(g/L) sin(theta) - damping * omega
    """
    return -(params.gravity / params.length) * sin(theta) - params.damping * omega


def energy(theta: float, omega: float, params: PendulumParams) -> tuple[float, float, float]:
    """
    Kinetic and gravitational potential energy.
    Potential energy is zero at the lowest point.
    """
    v = params.length * omega
    kinetic = 0.5 * params.mass * v * v
    potential = params.mass * params.gravity * params.length * (1.0 - cos(theta))
    return kinetic, potential, kinetic + potential


def pendulum_position(theta: float, length: float) -> tuple[float, float]:
    x = length * sin(theta)
    y = -length * cos(theta)
    return x, y


def rk4_step(theta: float, omega: float, dt: float, params: PendulumParams) -> tuple[float, float, float]:
    """Fourth-order Runge-Kutta integration for smoother physics."""

    def f(th: float, om: float) -> tuple[float, float]:
        return om, angular_acceleration(th, om, params)

    k1_theta, k1_omega = f(theta, omega)
    k2_theta, k2_omega = f(theta + 0.5 * dt * k1_theta, omega + 0.5 * dt * k1_omega)
    k3_theta, k3_omega = f(theta + 0.5 * dt * k2_theta, omega + 0.5 * dt * k2_omega)
    k4_theta, k4_omega = f(theta + dt * k3_theta, omega + dt * k3_omega)

    theta_next = theta + (dt / 6.0) * (k1_theta + 2 * k2_theta + 2 * k3_theta + k4_theta)
    omega_next = omega + (dt / 6.0) * (k1_omega + 2 * k2_omega + 2 * k3_omega + k4_omega)
    alpha_next = angular_acceleration(theta_next, omega_next, params)
    return theta_next, omega_next, alpha_next


# -----------------------------------------------------------------------------
# Terminal display
# -----------------------------------------------------------------------------

def pendulum_ascii(theta: float, width: int = 41) -> str:
    """Small single-line display of bob position across a horizontal range."""
    center = width // 2
    normalized = max(-1.0, min(1.0, sin(theta)))
    bob_index = int(center + normalized * (center - 1))
    chars = [" "] * width
    chars[center] = "|"
    chars[bob_index] = "O"
    return "".join(chars)


def print_state(round_number: int, state: PendulumState, params: PendulumParams) -> None:
    angle_deg = degrees(state.theta)
    speed = abs(params.length * state.omega)
    print(
        f"R{round_number:02d} "
        f"t={state.t:05.2f}s  "
        f"angle={angle_deg:8.3f} deg  "
        f"omega={state.omega:8.3f} rad/s  "
        f"speed={speed:7.3f} m/s  "
        f"E={state.total_energy:8.4f} J  "
        f"{pendulum_ascii(state.theta)}"
    )


# -----------------------------------------------------------------------------
# Round simulation
# -----------------------------------------------------------------------------

def simulate_round(round_number: int, params: PendulumParams, config: SimConfig) -> RoundSummary:
    theta = radians(params.initial_angle_deg)
    omega = params.initial_omega
    alpha = angular_acceleration(theta, omega, params)

    max_abs_theta = abs(theta)
    max_speed = 0.0
    max_energy = float("-inf")
    min_energy = float("inf")
    zero_crossing_times: list[float] = []
    previous_theta = theta

    print("\n" + "=" * 96)
    print(f"ROUND {round_number}")
    print(
        "parameters: "
        f"g={params.gravity:.3f} m/s^2, "
        f"length={params.length:.3f} m, "
        f"mass={params.mass:.3f} kg, "
        f"damping={params.damping:.4f}, "
        f"initial_angle={params.initial_angle_deg:.2f} deg, "
        f"initial_omega={params.initial_omega:.3f} rad/s"
    )
    print("-" * 96)

    steps = int(config.duration / config.dt)
    for step in range(steps + 1):
        t = step * config.dt
        x, y = pendulum_position(theta, params.length)
        kinetic, potential, total = energy(theta, omega, params)
        state = PendulumState(t, theta, omega, alpha, x, y, kinetic, potential, total)

        max_abs_theta = max(max_abs_theta, abs(theta))
        max_speed = max(max_speed, abs(params.length * omega))
        max_energy = max(max_energy, total)
        min_energy = min(min_energy, total)

        if step > 0 and previous_theta * theta < 0:
            zero_crossing_times.append(t)
        previous_theta = theta

        if step % config.print_every_steps == 0:
            print_state(round_number, state, params)
            if config.sleep_between_prints > 0:
                time.sleep(config.sleep_between_prints)

        theta, omega, alpha = rk4_step(theta, omega, config.dt, params)

    estimated_period = None
    if len(zero_crossing_times) >= 3:
        half_periods = [
            zero_crossing_times[i + 1] - zero_crossing_times[i]
            for i in range(len(zero_crossing_times) - 1)
        ]
        estimated_period = 2.0 * sum(half_periods) / len(half_periods)

    final_angle_deg = degrees(theta)
    final_omega = omega

    if abs(final_angle_deg) < 2 and abs(final_omega) < 0.15:
        stability_note = "nearly settled"
    elif max_abs_theta > radians(120):
        stability_note = "large swing"
    elif params.damping > 0.09:
        stability_note = "strongly damped"
    else:
        stability_note = "stable oscillation"

    summary = RoundSummary(
        round_number=round_number,
        params=params,
        max_angle_deg=degrees(max_abs_theta),
        max_speed=max_speed,
        max_energy=max_energy,
        min_energy=min_energy,
        final_angle_deg=final_angle_deg,
        final_omega=final_omega,
        zero_crossings=len(zero_crossing_times),
        estimated_period=estimated_period,
        stability_note=stability_note,
    )

    print("-" * 96)
    print_round_summary(summary)
    return summary


def print_round_summary(summary: RoundSummary) -> None:
    period_text = "not enough crossings"
    if summary.estimated_period is not None:
        period_text = f"{summary.estimated_period:.3f} s"

    print(
        f"summary: max_angle={summary.max_angle_deg:.2f} deg, "
        f"max_speed={summary.max_speed:.3f} m/s, "
        f"energy_range={summary.min_energy:.4f}..{summary.max_energy:.4f} J, "
        f"zero_crossings={summary.zero_crossings}, "
        f"estimated_period={period_text}, "
        f"note={summary.stability_note}"
    )


# -----------------------------------------------------------------------------
# AI controller
# -----------------------------------------------------------------------------

class PendulumAIController:
    """
    Simple parameter-changing controller.

    It is not a machine-learning model. It is a rule-based controller that tries
    different pendulum conditions and reacts to the previous round's behavior.
    """

    def __init__(self, rng: random.Random):
        self.rng = rng
        self.goal_cycle = [
            "slow wide swing",
            "fast short swing",
            "high damping test",
            "low gravity test",
            "energy boost test",
            "near-settle test",
        ]

    def choose_next(self, previous: RoundSummary | None, current: PendulumParams, round_number: int) -> PendulumParams:
        goal = self.goal_cycle[(round_number - 1) % len(self.goal_cycle)]

        # Start from current values, then adjust.
        p = replace(current)

        if previous is None:
            p.gravity = 9.81
            p.length = 1.25
            p.damping = 0.025
            p.initial_angle_deg = 35.0
            p.initial_omega = 0.0
            return p

        if goal == "slow wide swing":
            p.length = self.rng.uniform(1.6, 2.4)
            p.gravity = self.rng.uniform(8.8, 9.9)
            p.damping = self.rng.uniform(0.005, 0.030)
            p.initial_angle_deg = self.rng.uniform(45, 75)
            p.initial_omega = self.rng.uniform(-0.2, 0.2)

        elif goal == "fast short swing":
            p.length = self.rng.uniform(0.35, 0.9)
            p.gravity = self.rng.uniform(9.5, 13.5)
            p.damping = self.rng.uniform(0.005, 0.040)
            p.initial_angle_deg = self.rng.uniform(15, 40)
            p.initial_omega = self.rng.uniform(-0.6, 0.6)

        elif goal == "high damping test":
            p.length = self.rng.uniform(0.8, 1.7)
            p.gravity = self.rng.uniform(9.0, 10.5)
            p.damping = self.rng.uniform(0.090, 0.180)
            p.initial_angle_deg = self.rng.uniform(30, 65)
            p.initial_omega = self.rng.uniform(-0.3, 0.3)

        elif goal == "low gravity test":
            p.length = self.rng.uniform(0.8, 1.8)
            p.gravity = self.rng.uniform(1.6, 4.0)
            p.damping = self.rng.uniform(0.005, 0.040)
            p.initial_angle_deg = self.rng.uniform(25, 60)
            p.initial_omega = self.rng.uniform(-0.3, 0.3)

        elif goal == "energy boost test":
            p.length = self.rng.uniform(0.8, 1.5)
            p.gravity = self.rng.uniform(9.0, 11.0)
            p.damping = self.rng.uniform(0.000, 0.025)
            p.initial_angle_deg = self.rng.uniform(55, 95)
            p.initial_omega = self.rng.uniform(-1.2, 1.2)

        elif goal == "near-settle test":
            p.length = self.rng.uniform(0.9, 1.5)
            p.gravity = self.rng.uniform(9.0, 10.0)
            p.damping = self.rng.uniform(0.120, 0.220)
            p.initial_angle_deg = self.rng.uniform(8, 25)
            p.initial_omega = self.rng.uniform(-0.1, 0.1)

        # Reactive correction based on previous round.
        if previous.max_angle_deg > 120:
            # Prevent extremely wild next round.
            p.initial_angle_deg *= 0.65
            p.initial_omega *= 0.5
            p.damping = max(p.damping, 0.04)

        if previous.zero_crossings < 2:
            # It did not swing much; give it a stronger starting angle.
            p.initial_angle_deg = max(p.initial_angle_deg, 35.0)
            p.damping *= 0.7

        if previous.estimated_period is not None and previous.estimated_period > 4.0:
            # Try a shorter pendulum next time.
            p.length *= 0.75

        # Keep parameters inside safe ranges.
        p.gravity = clamp(p.gravity, 1.0, 15.0)
        p.length = clamp(p.length, 0.25, 3.0)
        p.mass = clamp(p.mass, 0.1, 10.0)
        p.damping = clamp(p.damping, 0.0, 0.25)
        p.initial_angle_deg = clamp(p.initial_angle_deg, -120.0, 120.0)
        p.initial_omega = clamp(p.initial_omega, -2.5, 2.5)

        print(f"AI controller selected next goal: {goal}")
        return p


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    config = SimConfig()
    rng = random.Random(config.seed)
    ai = PendulumAIController(rng)

    params = PendulumParams()
    previous_summary: RoundSummary | None = None
    summaries: list[RoundSummary] = []

    print("Terminal Pendulum Simulation with AI-Controlled Rounds")
    print("No graphics. No external packages. State is printed directly to the terminal.")

    for round_number in range(1, config.rounds + 1):
        params = ai.choose_next(previous_summary, params, round_number)
        summary = simulate_round(round_number, params, config)
        summaries.append(summary)
        previous_summary = summary

    print("\n" + "=" * 96)
    print("FINAL ROUND COMPARISON")
    print("=" * 96)
    print(
        f"{'round':>5}  {'g':>6}  {'L':>6}  {'damp':>7}  "
        f"{'start_deg':>9}  {'max_deg':>8}  {'max_v':>8}  {'period':>10}  note"
    )
    for s in summaries:
        period = "--"
        if s.estimated_period is not None:
            period = f"{s.estimated_period:.3f}"
        print(
            f"{s.round_number:5d}  "
            f"{s.params.gravity:6.2f}  "
            f"{s.params.length:6.2f}  "
            f"{s.params.damping:7.3f}  "
            f"{s.params.initial_angle_deg:9.2f}  "
            f"{s.max_angle_deg:8.2f}  "
            f"{s.max_speed:8.3f}  "
            f"{period:>10}  "
            f"{s.stability_note}"
        )


if __name__ == "__main__":
    main()
