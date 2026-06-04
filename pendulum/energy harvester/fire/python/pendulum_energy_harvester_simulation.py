"""
Terminal Pendulum Energy Harvester Simulation with AI-Controlled Rounds
-----------------------------------------------------------------------

No VPython. No external packages.

This modifies the pendulum rounds simulation into a simple energy-harvesting
pendulum. The pendulum can drive a generator near the bottom of its swing. The
extra generator load extracts mechanical energy, converts part of it into stored
electrical energy, and loses the rest as heat.

Important physical rule:
    The harvester does not create infinite energy. Electrical output comes from
    the pendulum's mechanical energy, so stronger generator loading slows and
    damps the swing faster.

Run:
    python pendulum_energy_harvester_simulation.py
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
    damping: float = 0.035             # non-generator angular damping coefficient
    initial_angle_deg: float = 35.0    # degrees
    initial_omega: float = 0.0         # rad/s

    # Energy harvester / generator properties
    generator_load: float = 0.050      # angular damping coefficient added by generator
    generator_efficiency: float = 0.62 # fraction of extracted mechanical energy stored
    engagement_angle_deg: float = 18.0 # generator engages near bottom of arc
    cut_in_speed: float = 0.20         # m/s; below this, generator produces no useful power
    storage_capacity_j: float = 8.0    # capacitor/battery capacity used in this model
    storage_initial_j: float = 0.0     # starting stored electrical energy
    storage_capacitance_f: float = 1.0 # for display-only capacitor voltage estimate


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
    generator_active: bool
    mechanical_power_out: float
    electrical_power_stored: float
    stored_energy: float
    storage_voltage: float


@dataclass
class HarvesterStats:
    stored_energy_j: float
    harvested_j: float
    heat_loss_j: float
    mechanical_extracted_j: float
    generator_on_time_s: float
    peak_electrical_power_w: float
    peak_mechanical_power_w: float
    storage_full_time_s: float | None


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
    harvested_j: float
    stored_final_j: float
    heat_loss_j: float
    mechanical_extracted_j: float
    average_electrical_power_w: float
    peak_electrical_power_w: float
    generator_on_time_s: float
    storage_full_time_s: float | None
    harvest_efficiency_observed: float


# -----------------------------------------------------------------------------
# Physics and harvester model
# -----------------------------------------------------------------------------

def generator_is_active(theta: float, omega: float, params: PendulumParams) -> bool:
    """Generator engages near the bottom of the swing after cut-in speed."""
    angle_ok = abs(degrees(theta)) <= params.engagement_angle_deg
    speed_ok = abs(params.length * omega) >= params.cut_in_speed
    load_ok = params.generator_load > 0.0 and params.generator_efficiency > 0.0
    return angle_ok and speed_ok and load_ok


def angular_acceleration(theta: float, omega: float, params: PendulumParams) -> float:
    """
    Nonlinear pendulum equation with physical damping and optional generator load:
        theta'' = -(g/L) sin(theta) - damping*omega - generator_load*omega

    The generator acts like additional speed-proportional drag only while engaged.
    """
    generator_drag = params.generator_load if generator_is_active(theta, omega, params) else 0.0
    return -(params.gravity / params.length) * sin(theta) - (params.damping + generator_drag) * omega


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


def generator_power(theta: float, omega: float, params: PendulumParams, stored_energy_j: float) -> tuple[bool, float, float, float]:
    """
    Return generator status and powers.

    mechanical_power_out is the mechanical power extracted from the pendulum.
    electrical_power_stored is the usable power delivered to storage.
    heat_power is conversion loss.
    """
    active = generator_is_active(theta, omega, params)
    if not active or stored_energy_j >= params.storage_capacity_j:
        return False, 0.0, 0.0, 0.0

    # Damping torque magnitude is generator_load * |omega|. Power is torque * omega.
    # Units are simplified but consistent for this terminal model.
    mechanical_power_out = params.generator_load * omega * omega
    electrical_power_stored = mechanical_power_out * params.generator_efficiency
    heat_power = mechanical_power_out - electrical_power_stored
    return True, mechanical_power_out, electrical_power_stored, heat_power


def storage_voltage(stored_energy_j: float, capacitance_f: float) -> float:
    """Display-only capacitor voltage estimate using E = 1/2 C V^2."""
    if capacitance_f <= 0.0 or stored_energy_j <= 0.0:
        return 0.0
    return sqrt(2.0 * stored_energy_j / capacitance_f)


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

def pendulum_ascii(theta: float, generator_active: bool, width: int = 41) -> str:
    """Small single-line display of bob position and generator zone."""
    center = width // 2
    normalized = max(-1.0, min(1.0, sin(theta)))
    bob_index = int(center + normalized * (center - 1))
    chars = [" "] * width
    chars[center] = "|"

    # Mark a small generator pickup coil around bottom/center.
    for offset in (-2, -1, 1, 2):
        idx = center + offset
        if 0 <= idx < width and chars[idx] == " ":
            chars[idx] = "="

    chars[bob_index] = "G" if generator_active else "O"
    return "".join(chars)


def battery_bar(stored: float, capacity: float, width: int = 20) -> str:
    if capacity <= 0:
        return "[no storage]"
    fill = int(round(max(0.0, min(1.0, stored / capacity)) * width))
    return "[" + "#" * fill + "." * (width - fill) + "]"


def print_state(round_number: int, state: PendulumState, params: PendulumParams) -> None:
    angle_deg = degrees(state.theta)
    speed = abs(params.length * state.omega)
    gen = "ON " if state.generator_active else "off"
    print(
        f"R{round_number:02d} "
        f"t={state.t:05.2f}s  "
        f"angle={angle_deg:8.3f} deg  "
        f"omega={state.omega:8.3f} rad/s  "
        f"speed={speed:7.3f} m/s  "
        f"mechE={state.total_energy:8.4f} J  "
        f"P_elec={state.electrical_power_stored:7.4f} W  "
        f"store={state.stored_energy:6.3f} J {battery_bar(state.stored_energy, params.storage_capacity_j, 12)} "
        f"Vcap={state.storage_voltage:5.2f} V  "
        f"GEN={gen}  "
        f"{pendulum_ascii(state.theta, state.generator_active)}"
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

    stats = HarvesterStats(
        stored_energy_j=params.storage_initial_j,
        harvested_j=0.0,
        heat_loss_j=0.0,
        mechanical_extracted_j=0.0,
        generator_on_time_s=0.0,
        peak_electrical_power_w=0.0,
        peak_mechanical_power_w=0.0,
        storage_full_time_s=None,
    )

    print("\n" + "=" * 120)
    print(f"ROUND {round_number}: PENDULUM ENERGY HARVESTER")
    print(
        "parameters: "
        f"g={params.gravity:.3f} m/s^2, "
        f"length={params.length:.3f} m, "
        f"mass={params.mass:.3f} kg, "
        f"damping={params.damping:.4f}, "
        f"initial_angle={params.initial_angle_deg:.2f} deg, "
        f"initial_omega={params.initial_omega:.3f} rad/s"
    )
    print(
        "harvester: "
        f"generator_load={params.generator_load:.4f}, "
        f"efficiency={params.generator_efficiency:.2f}, "
        f"engagement=±{params.engagement_angle_deg:.1f} deg, "
        f"cut_in_speed={params.cut_in_speed:.2f} m/s, "
        f"storage={params.storage_initial_j:.2f}/{params.storage_capacity_j:.2f} J"
    )
    print("-" * 120)

    steps = int(config.duration / config.dt)
    for step in range(steps + 1):
        t = step * config.dt
        x, y = pendulum_position(theta, params.length)
        kinetic, potential, total = energy(theta, omega, params)
        active, mechanical_power, electrical_power, heat_power = generator_power(theta, omega, params, stats.stored_energy_j)

        if step > 0:
            stored_before = stats.stored_energy_j
            requested_harvest = electrical_power * config.dt
            remaining_capacity = max(0.0, params.storage_capacity_j - stored_before)
            actual_harvest = min(requested_harvest, remaining_capacity)

            # If storage fills during the step, scale losses and extracted energy by delivered fraction.
            delivery_fraction = 0.0 if requested_harvest <= 0 else actual_harvest / requested_harvest
            actual_mechanical = mechanical_power * config.dt * delivery_fraction
            actual_heat = heat_power * config.dt * delivery_fraction

            stats.stored_energy_j += actual_harvest
            stats.harvested_j += actual_harvest
            stats.mechanical_extracted_j += actual_mechanical
            stats.heat_loss_j += actual_heat

            if active and delivery_fraction > 0.0:
                stats.generator_on_time_s += config.dt * delivery_fraction
            if stats.storage_full_time_s is None and stats.stored_energy_j >= params.storage_capacity_j:
                stats.storage_full_time_s = t

        stats.peak_electrical_power_w = max(stats.peak_electrical_power_w, electrical_power)
        stats.peak_mechanical_power_w = max(stats.peak_mechanical_power_w, mechanical_power)

        state = PendulumState(
            t=t,
            theta=theta,
            omega=omega,
            alpha=alpha,
            x=x,
            y=y,
            kinetic=kinetic,
            potential=potential,
            total_energy=total,
            generator_active=active,
            mechanical_power_out=mechanical_power,
            electrical_power_stored=electrical_power,
            stored_energy=stats.stored_energy_j,
            storage_voltage=storage_voltage(stats.stored_energy_j, params.storage_capacitance_f),
        )

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
        stability_note = "harvested and nearly settled"
    elif max_abs_theta > radians(120):
        stability_note = "large swing"
    elif params.damping + params.generator_load > 0.16:
        stability_note = "heavily loaded harvester"
    elif stats.harvested_j <= 0.001:
        stability_note = "little useful harvest"
    else:
        stability_note = "stable harvesting swing"

    observed_efficiency = 0.0
    if stats.mechanical_extracted_j > 0.0:
        observed_efficiency = stats.harvested_j / stats.mechanical_extracted_j

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
        harvested_j=stats.harvested_j,
        stored_final_j=stats.stored_energy_j,
        heat_loss_j=stats.heat_loss_j,
        mechanical_extracted_j=stats.mechanical_extracted_j,
        average_electrical_power_w=stats.harvested_j / max(config.duration, 1e-9),
        peak_electrical_power_w=stats.peak_electrical_power_w,
        generator_on_time_s=stats.generator_on_time_s,
        storage_full_time_s=stats.storage_full_time_s,
        harvest_efficiency_observed=observed_efficiency,
    )

    print("-" * 120)
    print_round_summary(summary)
    return summary


def print_round_summary(summary: RoundSummary) -> None:
    period_text = "not enough crossings"
    if summary.estimated_period is not None:
        period_text = f"{summary.estimated_period:.3f} s"

    full_text = "not full"
    if summary.storage_full_time_s is not None:
        full_text = f"full at {summary.storage_full_time_s:.2f} s"

    print(
        f"summary: max_angle={summary.max_angle_deg:.2f} deg, "
        f"max_speed={summary.max_speed:.3f} m/s, "
        f"mech_energy_range={summary.min_energy:.4f}..{summary.max_energy:.4f} J, "
        f"harvested={summary.harvested_j:.4f} J, "
        f"stored_final={summary.stored_final_j:.4f} J, "
        f"avg_power={summary.average_electrical_power_w:.5f} W, "
        f"peak_power={summary.peak_electrical_power_w:.5f} W, "
        f"heat_loss={summary.heat_loss_j:.4f} J, "
        f"generator_on={summary.generator_on_time_s:.2f} s, "
        f"storage={full_text}, "
        f"period={period_text}, "
        f"note={summary.stability_note}"
    )


# -----------------------------------------------------------------------------
# AI controller
# -----------------------------------------------------------------------------

class PendulumAIController:
    """
    Rule-based controller for harvester experiments.

    It searches for practical energy-harvesting conditions while showing the
    real tradeoff: more generator load produces more electrical output for a
    short time but also damps the pendulum faster.
    """

    def __init__(self, rng: random.Random):
        self.rng = rng
        self.goal_cycle = [
            "baseline light harvester",
            "wide swing high energy",
            "strong generator load",
            "low load long run",
            "narrow pickup coil",
            "wide pickup coil",
            "fast short harvester",
            "storage fill test",
        ]

    def choose_next(self, previous: RoundSummary | None, current: PendulumParams, round_number: int) -> PendulumParams:
        goal = self.goal_cycle[(round_number - 1) % len(self.goal_cycle)]
        p = replace(current)

        if previous is None:
            p.gravity = 9.81
            p.length = 1.25
            p.mass = 1.0
            p.damping = 0.025
            p.initial_angle_deg = 45.0
            p.initial_omega = 0.0
            p.generator_load = 0.045
            p.generator_efficiency = 0.62
            p.engagement_angle_deg = 20.0
            p.cut_in_speed = 0.20
            p.storage_initial_j = 0.0
            p.storage_capacity_j = 8.0
            return p

        # Each round starts with an empty storage element so harvested energy is comparable.
        p.storage_initial_j = 0.0

        if goal == "baseline light harvester":
            p.gravity = 9.81
            p.length = self.rng.uniform(1.0, 1.5)
            p.damping = self.rng.uniform(0.015, 0.035)
            p.initial_angle_deg = self.rng.uniform(35, 55)
            p.initial_omega = self.rng.uniform(-0.2, 0.2)
            p.generator_load = self.rng.uniform(0.025, 0.060)
            p.engagement_angle_deg = self.rng.uniform(15, 25)
            p.cut_in_speed = self.rng.uniform(0.15, 0.28)

        elif goal == "wide swing high energy":
            p.gravity = self.rng.uniform(9.0, 10.5)
            p.length = self.rng.uniform(1.4, 2.2)
            p.damping = self.rng.uniform(0.005, 0.025)
            p.initial_angle_deg = self.rng.uniform(60, 90)
            p.initial_omega = self.rng.uniform(-0.3, 0.3)
            p.generator_load = self.rng.uniform(0.035, 0.080)
            p.engagement_angle_deg = self.rng.uniform(18, 32)
            p.cut_in_speed = self.rng.uniform(0.18, 0.35)

        elif goal == "strong generator load":
            p.gravity = self.rng.uniform(9.0, 10.5)
            p.length = self.rng.uniform(0.9, 1.7)
            p.damping = self.rng.uniform(0.010, 0.035)
            p.initial_angle_deg = self.rng.uniform(45, 75)
            p.initial_omega = self.rng.uniform(-0.5, 0.5)
            p.generator_load = self.rng.uniform(0.100, 0.220)
            p.engagement_angle_deg = self.rng.uniform(18, 35)
            p.cut_in_speed = self.rng.uniform(0.20, 0.45)

        elif goal == "low load long run":
            p.gravity = self.rng.uniform(9.0, 10.2)
            p.length = self.rng.uniform(1.3, 2.4)
            p.damping = self.rng.uniform(0.002, 0.018)
            p.initial_angle_deg = self.rng.uniform(35, 65)
            p.initial_omega = self.rng.uniform(-0.2, 0.2)
            p.generator_load = self.rng.uniform(0.010, 0.035)
            p.engagement_angle_deg = self.rng.uniform(12, 22)
            p.cut_in_speed = self.rng.uniform(0.12, 0.25)

        elif goal == "narrow pickup coil":
            p.gravity = self.rng.uniform(9.3, 10.3)
            p.length = self.rng.uniform(0.8, 1.5)
            p.damping = self.rng.uniform(0.005, 0.030)
            p.initial_angle_deg = self.rng.uniform(45, 75)
            p.initial_omega = self.rng.uniform(-0.4, 0.4)
            p.generator_load = self.rng.uniform(0.060, 0.130)
            p.engagement_angle_deg = self.rng.uniform(5, 12)
            p.cut_in_speed = self.rng.uniform(0.20, 0.40)

        elif goal == "wide pickup coil":
            p.gravity = self.rng.uniform(9.3, 10.3)
            p.length = self.rng.uniform(0.8, 1.6)
            p.damping = self.rng.uniform(0.005, 0.035)
            p.initial_angle_deg = self.rng.uniform(40, 70)
            p.initial_omega = self.rng.uniform(-0.4, 0.4)
            p.generator_load = self.rng.uniform(0.040, 0.110)
            p.engagement_angle_deg = self.rng.uniform(28, 45)
            p.cut_in_speed = self.rng.uniform(0.15, 0.35)

        elif goal == "fast short harvester":
            p.gravity = self.rng.uniform(10.0, 13.5)
            p.length = self.rng.uniform(0.35, 0.9)
            p.damping = self.rng.uniform(0.005, 0.035)
            p.initial_angle_deg = self.rng.uniform(30, 65)
            p.initial_omega = self.rng.uniform(-0.8, 0.8)
            p.generator_load = self.rng.uniform(0.040, 0.130)
            p.engagement_angle_deg = self.rng.uniform(15, 32)
            p.cut_in_speed = self.rng.uniform(0.18, 0.38)

        elif goal == "storage fill test":
            p.gravity = self.rng.uniform(9.5, 10.5)
            p.length = self.rng.uniform(1.0, 1.8)
            p.damping = self.rng.uniform(0.005, 0.025)
            p.initial_angle_deg = self.rng.uniform(65, 95)
            p.initial_omega = self.rng.uniform(-0.6, 0.6)
            p.generator_load = self.rng.uniform(0.080, 0.180)
            p.engagement_angle_deg = self.rng.uniform(20, 38)
            p.cut_in_speed = self.rng.uniform(0.15, 0.30)
            p.storage_capacity_j = self.rng.uniform(0.20, 0.80)

        # Reactive correction based on previous round.
        if previous.harvested_j < 0.01:
            p.initial_angle_deg = max(p.initial_angle_deg, 45.0)
            p.generator_load = max(p.generator_load, 0.050)
            p.cut_in_speed *= 0.85

        if previous.generator_on_time_s < 0.8:
            p.engagement_angle_deg += 5.0
            p.cut_in_speed *= 0.9

        if previous.stability_note in {"harvested and nearly settled", "heavily loaded harvester"}:
            # Reduce generator drag if it killed the swing too quickly.
            p.generator_load *= 0.70
            p.initial_angle_deg = max(p.initial_angle_deg, 50.0)

        if previous.max_angle_deg > 120:
            p.initial_angle_deg *= 0.65
            p.initial_omega *= 0.5
            p.damping = max(p.damping, 0.04)

        if previous.estimated_period is not None and previous.estimated_period > 4.0:
            p.length *= 0.75

        # Keep parameters inside safe ranges.
        p.gravity = clamp(p.gravity, 1.0, 15.0)
        p.length = clamp(p.length, 0.25, 3.0)
        p.mass = clamp(p.mass, 0.1, 10.0)
        p.damping = clamp(p.damping, 0.0, 0.25)
        p.initial_angle_deg = clamp(p.initial_angle_deg, -120.0, 120.0)
        p.initial_omega = clamp(p.initial_omega, -2.5, 2.5)
        p.generator_load = clamp(p.generator_load, 0.0, 0.35)
        p.generator_efficiency = clamp(p.generator_efficiency, 0.05, 0.95)
        p.engagement_angle_deg = clamp(p.engagement_angle_deg, 3.0, 65.0)
        p.cut_in_speed = clamp(p.cut_in_speed, 0.01, 1.5)
        p.storage_capacity_j = clamp(p.storage_capacity_j, 0.05, 20.0)
        p.storage_initial_j = clamp(p.storage_initial_j, 0.0, p.storage_capacity_j)
        p.storage_capacitance_f = clamp(p.storage_capacitance_f, 0.05, 100.0)

        print(f"AI controller selected next harvester goal: {goal}")
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

    print("Terminal Pendulum Energy Harvester Simulation with AI-Controlled Rounds")
    print("No graphics. No external packages. State is printed directly to the terminal.")
    print("Physical rule: harvested electricity is removed from pendulum motion; no infinite energy is created.")

    for round_number in range(1, config.rounds + 1):
        params = ai.choose_next(previous_summary, params, round_number)
        summary = simulate_round(round_number, params, config)
        summaries.append(summary)
        previous_summary = summary

    print("\n" + "=" * 132)
    print("FINAL HARVESTER COMPARISON")
    print("=" * 132)
    print(
        f"{'round':>5}  {'g':>6}  {'L':>6}  {'damp':>7}  {'gen_load':>8}  "
        f"{'engage':>7}  {'start':>8}  {'max_v':>8}  {'harvest_J':>10}  "
        f"{'avg_W':>9}  {'peak_W':>9}  {'gen_s':>7}  {'period':>8}  note"
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
            f"{s.params.generator_load:8.3f}  "
            f"{s.params.engagement_angle_deg:7.1f}  "
            f"{s.params.initial_angle_deg:8.2f}  "
            f"{s.max_speed:8.3f}  "
            f"{s.harvested_j:10.4f}  "
            f"{s.average_electrical_power_w:9.5f}  "
            f"{s.peak_electrical_power_w:9.5f}  "
            f"{s.generator_on_time_s:7.2f}  "
            f"{period:>8}  "
            f"{s.stability_note}"
        )

    best = max(summaries, key=lambda s: s.harvested_j)
    print("\nBest energy harvest:")
    print(
        f"  Round {best.round_number}: harvested {best.harvested_j:.4f} J, "
        f"average {best.average_electrical_power_w:.5f} W, peak {best.peak_electrical_power_w:.5f} W, "
        f"generator active {best.generator_on_time_s:.2f} s."
    )
    print("  Interpretation: higher harvest means the generator extracted more pendulum energy, not free energy.")


if __name__ == "__main__":
    main()
