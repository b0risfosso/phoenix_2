#!/usr/bin/env python3
"""
Simple Flame Combustion Property Simulation
-------------------------------------------

A terminal-only scientific simulation of a simple flame.

Run:
    python simple_flame_combustion_simulation.py

No external packages are required.

This is not a visual flame simulation. It models the time evolution of flame
properties: fuel, oxygen, temperature, reaction rate, heat release, heat losses,
smoke/soot formation, burn efficiency, and extinction cause.

The model is intentionally simplified. It is a zero-dimensional, well-mixed
reactor-style toy model suitable for terminal experiments, not a full CFD or
detailed chemical kinetics solver.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import exp
import random
from typing import Dict, List


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

@dataclass
class SimConfig:
    rounds: int = 8
    duration_s: float = 6.0
    dt_s: float = 0.01
    print_every_s: float = 0.20
    seed: int = 23


@dataclass
class FlameParams:
    fuel_name: str = "methane-like gas"

    # Reservoir/supply parameters, normalized to simple units.
    initial_fuel: float = 1.00
    initial_oxygen: float = 1.00
    fuel_supply_rate: float = 0.10       # normalized fuel units per second
    oxygen_supply_rate: float = 0.55     # normalized oxygen units per second
    max_fuel: float = 1.80
    max_oxygen: float = 1.80

    # Ignition and reaction behavior.
    initial_temperature_K: float = 300.0
    spark_energy_J: float = 18.0
    ignition_temperature_K: float = 720.0
    extinction_temperature_K: float = 620.0
    reaction_strength: float = 2.4
    activation_temperature_K: float = 900.0
    oxygen_per_fuel: float = 2.0

    # Thermodynamic scale.
    heat_of_combustion_J_per_unit: float = 520.0
    heat_capacity_J_per_K: float = 1.20
    ambient_temperature_K: float = 300.0

    # Heat loss terms.
    convective_loss_coeff: float = 0.75
    radiative_loss_coeff: float = 0.000000026
    mixing_loss_coeff: float = 0.12

    # Byproduct behavior.
    soot_rate: float = 0.020
    smoke_decay_rate: float = 0.18
    humidity: float = 0.20
    chamber_volume: float = 1.00


@dataclass
class FlameState:
    t_s: float = 0.0
    temperature_K: float = 300.0
    fuel: float = 1.0
    oxygen: float = 1.0
    burned_gas: float = 0.0
    smoke: float = 0.0
    soot: float = 0.0
    total_energy_released_J: float = 0.0
    total_heat_lost_J: float = 0.0
    total_fuel_burned: float = 0.0
    state_name: str = "unignited"


@dataclass
class RoundSummary:
    round_number: int
    params: FlameParams
    peak_temperature_K: float
    final_temperature_K: float
    burn_time_s: float
    total_energy_released_J: float
    total_heat_lost_J: float
    fuel_burned: float
    oxygen_used: float
    peak_reaction_rate: float
    average_efficiency: float
    final_smoke: float
    final_soot: float
    extinction_cause: str


# -----------------------------------------------------------------------------
# Core flame model
# -----------------------------------------------------------------------------

def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def ignition_boost_temperature(params: FlameParams) -> float:
    """Temperature increase caused by the initial spark."""
    return params.spark_energy_J / max(0.001, params.heat_capacity_J_per_K)


def temperature_factor(temperature_K: float, params: FlameParams) -> float:
    """
    Smooth thermal activation factor.

    Below ignition temperature, reaction is weak.
    Above ignition temperature, reaction grows toward 1.
    """
    if temperature_K < params.ignition_temperature_K:
        return 0.04 * temperature_K / max(1.0, params.ignition_temperature_K)

    excess = temperature_K - params.ignition_temperature_K
    return clamp(1.0 - exp(-excess / max(1.0, params.activation_temperature_K)), 0.0, 1.0)


def reaction_rate(state: FlameState, params: FlameParams) -> float:
    """
    Simplified reaction rate.

    Rate depends on available fuel, oxygen, temperature activation, humidity,
    and chamber volume. This is a toy expression, not a detailed Arrhenius law.
    """
    if state.temperature_K < params.extinction_temperature_K:
        return 0.0
    if state.fuel <= 0.0001 or state.oxygen <= 0.0001:
        return 0.0

    tf = temperature_factor(state.temperature_K, params)
    fuel_factor = state.fuel / max(0.001, params.chamber_volume)
    oxygen_factor = state.oxygen / max(0.001, params.chamber_volume)
    humidity_penalty = clamp(1.0 - 0.45 * params.humidity, 0.25, 1.0)

    return params.reaction_strength * fuel_factor * oxygen_factor * tf * humidity_penalty


def heat_losses_W(state: FlameState, params: FlameParams) -> Dict[str, float]:
    """Compute convective, radiative, and mixing heat losses in watts."""
    delta_T = max(0.0, state.temperature_K - params.ambient_temperature_K)

    convective = params.convective_loss_coeff * delta_T
    radiative = params.radiative_loss_coeff * (
        max(0.0, state.temperature_K ** 4 - params.ambient_temperature_K ** 4)
    )
    mixing = params.mixing_loss_coeff * delta_T * (params.oxygen_supply_rate + params.fuel_supply_rate)

    return {
        "convective": convective,
        "radiative": radiative,
        "mixing": mixing,
        "total": convective + radiative + mixing,
    }


def classify_state(state: FlameState, params: FlameParams, rate: float) -> str:
    if state.temperature_K < params.ignition_temperature_K and state.total_energy_released_J <= 0.0:
        return "pre-ignition"
    if state.fuel <= 0.005:
        return "fuel-limited"
    if state.oxygen <= 0.005:
        return "oxygen-limited"
    if state.temperature_K < params.extinction_temperature_K and state.total_energy_released_J > 0.0:
        return "extinguishing"
    if rate > 0.40:
        return "active burning"
    if rate > 0.05:
        return "weak burning"
    return "hot but slow"


def step_flame(state: FlameState, params: FlameParams, dt_s: float, is_first_step: bool) -> tuple[FlameState, Dict[str, float]]:
    s = replace(state)

    if is_first_step:
        s.temperature_K += ignition_boost_temperature(params)

    # Supply fresh reactants.
    s.fuel = clamp(s.fuel + params.fuel_supply_rate * dt_s, 0.0, params.max_fuel)
    s.oxygen = clamp(s.oxygen + params.oxygen_supply_rate * dt_s, 0.0, params.max_oxygen)

    # Combustion.
    rate = reaction_rate(s, params)
    requested_fuel_burn = rate * dt_s
    possible_fuel_burn = min(s.fuel, s.oxygen / max(0.001, params.oxygen_per_fuel))
    fuel_burned = min(requested_fuel_burn, possible_fuel_burn)

    oxygen_used = fuel_burned * params.oxygen_per_fuel
    heat_generated_J = fuel_burned * params.heat_of_combustion_J_per_unit

    s.fuel -= fuel_burned
    s.oxygen -= oxygen_used
    s.burned_gas += fuel_burned + oxygen_used
    s.total_fuel_burned += fuel_burned
    s.total_energy_released_J += heat_generated_J

    # Smoke/soot: richer, oxygen-poor, humid, or cooler burns produce more.
    equivalence_indicator = s.fuel / max(0.001, s.oxygen)
    oxygen_starvation = clamp((equivalence_indicator - 0.45) * 0.6, 0.0, 2.0)
    cool_penalty = clamp((params.ignition_temperature_K + 350.0 - s.temperature_K) / 900.0, 0.0, 1.0)
    soot_created = fuel_burned * params.soot_rate * (1.0 + oxygen_starvation + cool_penalty + params.humidity)
    smoke_created = soot_created * 5.0 + fuel_burned * 0.02 * oxygen_starvation

    s.soot += soot_created
    s.smoke += smoke_created
    s.smoke *= exp(-params.smoke_decay_rate * dt_s)

    # Heat losses.
    losses = heat_losses_W(s, params)
    heat_lost_J = losses["total"] * dt_s
    s.total_heat_lost_J += heat_lost_J

    # Temperature update.
    net_heat_J = heat_generated_J - heat_lost_J
    s.temperature_K += net_heat_J / max(0.001, params.heat_capacity_J_per_K)

    # Passive cooling cannot go below ambient in this simple model.
    if s.temperature_K < params.ambient_temperature_K:
        s.temperature_K = params.ambient_temperature_K

    s.t_s += dt_s
    s.state_name = classify_state(s, params, rate)

    diagnostics = {
        "reaction_rate": rate,
        "fuel_burned": fuel_burned,
        "oxygen_used": oxygen_used,
        "heat_generated_J": heat_generated_J,
        "heat_lost_J": heat_lost_J,
        "convective_loss_W": losses["convective"],
        "radiative_loss_W": losses["radiative"],
        "mixing_loss_W": losses["mixing"],
        "total_loss_W": losses["total"],
        "power_W": heat_generated_J / dt_s if dt_s > 0 else 0.0,
        "efficiency": heat_generated_J / max(0.001, heat_generated_J + heat_lost_J),
    }
    return s, diagnostics


def flame_height_estimate_m(state: FlameState, diagnostics: Dict[str, float], params: FlameParams) -> float:
    """
    Crude estimated flame height.

    It grows with heat-release power and oxygen, and shrinks with high cooling
    or low fuel. Units are illustrative meters.
    """
    power = diagnostics["power_W"]
    oxygen_factor = clamp(state.oxygen / max(0.001, params.max_oxygen), 0.0, 1.0)
    fuel_factor = clamp(state.fuel / max(0.001, params.max_fuel), 0.0, 1.0)
    return clamp(0.04 + 0.010 * (power ** 0.5) * (0.4 + oxygen_factor) * (0.3 + fuel_factor), 0.0, 1.5)


# -----------------------------------------------------------------------------
# Round simulation
# -----------------------------------------------------------------------------

def run_round(round_number: int, params: FlameParams, config: SimConfig) -> RoundSummary:
    state = FlameState(
        temperature_K=params.initial_temperature_K,
        fuel=params.initial_fuel,
        oxygen=params.initial_oxygen,
    )

    print("\n" + "=" * 120)
    print(f"ROUND {round_number}: {params.fuel_name}")
    print("=" * 120)
    print(
        f"initial fuel={params.initial_fuel:.3f}, initial O2={params.initial_oxygen:.3f}, "
        f"fuel_supply={params.fuel_supply_rate:.3f}/s, O2_supply={params.oxygen_supply_rate:.3f}/s"
    )
    print(
        f"spark={params.spark_energy_J:.2f} J, ignition={params.ignition_temperature_K:.1f} K, "
        f"extinction={params.extinction_temperature_K:.1f} K, reaction_strength={params.reaction_strength:.2f}"
    )
    print(
        f"cooling: conv={params.convective_loss_coeff:.3f}, rad={params.radiative_loss_coeff:.2e}, "
        f"mix={params.mixing_loss_coeff:.3f}, humidity={params.humidity:.2f}, volume={params.chamber_volume:.2f}"
    )
    print("-" * 120)

    print(
        f"{'t(s)':>6} | {'state':>15} | {'T(K)':>8} | {'fuel':>7} | {'O2':>7} | "
        f"{'rate':>8} | {'power(W)':>9} | {'loss(W)':>8} | {'smoke':>7} | {'soot':>7} | {'h(m)':>6}"
    )
    print("-" * 120)

    peak_temperature = state.temperature_K
    peak_rate = 0.0
    burn_time = 0.0
    efficiency_sum = 0.0
    efficiency_samples = 0
    oxygen_initial_plus_supply = params.initial_oxygen
    next_print = 0.0

    steps = int(config.duration_s / config.dt_s)
    diagnostics = {
        "reaction_rate": 0.0,
        "power_W": 0.0,
        "total_loss_W": 0.0,
        "efficiency": 0.0,
    }

    for step in range(steps + 1):
        state, diagnostics = step_flame(state, params, config.dt_s, is_first_step=(step == 0))

        peak_temperature = max(peak_temperature, state.temperature_K)
        peak_rate = max(peak_rate, diagnostics["reaction_rate"])

        if diagnostics["reaction_rate"] > 0.02:
            burn_time += config.dt_s

        efficiency_sum += diagnostics["efficiency"]
        efficiency_samples += 1
        oxygen_initial_plus_supply += params.oxygen_supply_rate * config.dt_s

        if state.t_s + 1e-9 >= next_print or step == steps:
            h = flame_height_estimate_m(state, diagnostics, params)
            print(
                f"{state.t_s:6.2f} | {state.state_name:>15} | {state.temperature_K:8.1f} | "
                f"{state.fuel:7.3f} | {state.oxygen:7.3f} | "
                f"{diagnostics['reaction_rate']:8.4f} | {diagnostics['power_W']:9.2f} | "
                f"{diagnostics['total_loss_W']:8.2f} | {state.smoke:7.4f} | "
                f"{state.soot:7.4f} | {h:6.3f}"
            )
            next_print += config.print_every_s

    oxygen_used = max(0.0, oxygen_initial_plus_supply - state.oxygen)
    avg_efficiency = efficiency_sum / max(1, efficiency_samples)

    extinction_cause = determine_extinction_cause(state, params, diagnostics)

    summary = RoundSummary(
        round_number=round_number,
        params=params,
        peak_temperature_K=peak_temperature,
        final_temperature_K=state.temperature_K,
        burn_time_s=burn_time,
        total_energy_released_J=state.total_energy_released_J,
        total_heat_lost_J=state.total_heat_lost_J,
        fuel_burned=state.total_fuel_burned,
        oxygen_used=oxygen_used,
        peak_reaction_rate=peak_rate,
        average_efficiency=avg_efficiency,
        final_smoke=state.smoke,
        final_soot=state.soot,
        extinction_cause=extinction_cause,
    )

    print("-" * 120)
    print_round_summary(summary)
    return summary


def determine_extinction_cause(state: FlameState, params: FlameParams, diagnostics: Dict[str, float]) -> str:
    if state.fuel <= 0.01:
        return "fuel exhausted"
    if state.oxygen <= 0.01:
        return "oxygen starved"
    if state.temperature_K < params.extinction_temperature_K:
        return "thermal extinction"
    if diagnostics["reaction_rate"] < 0.03:
        return "weak reaction"
    if state.total_energy_released_J > 0 and state.temperature_K >= params.extinction_temperature_K:
        return "still burning at end"
    return "not ignited"


def print_round_summary(summary: RoundSummary) -> None:
    print("Round result:")
    print(f"  peak temperature: {summary.peak_temperature_K:.1f} K")
    print(f"  final temperature: {summary.final_temperature_K:.1f} K")
    print(f"  burn time: {summary.burn_time_s:.2f} s")
    print(f"  total energy released: {summary.total_energy_released_J:.2f} J")
    print(f"  total heat lost: {summary.total_heat_lost_J:.2f} J")
    print(f"  fuel burned: {summary.fuel_burned:.4f} normalized units")
    print(f"  oxygen used: {summary.oxygen_used:.4f} normalized units")
    print(f"  peak reaction rate: {summary.peak_reaction_rate:.4f} units/s")
    print(f"  average heat efficiency: {summary.average_efficiency:.3f}")
    print(f"  final smoke: {summary.final_smoke:.4f}")
    print(f"  final soot: {summary.final_soot:.4f}")
    print(f"  extinction/end state: {summary.extinction_cause}")


# -----------------------------------------------------------------------------
# AI controller
# -----------------------------------------------------------------------------

class FlameAIController:
    """Rule-based controller that changes scientific flame parameters by round."""

    def __init__(self, rng: random.Random):
        self.rng = rng
        self.goal_cycle = [
            "baseline balanced flame",
            "fuel rich flame",
            "oxygen poor flame",
            "strong airflow flame",
            "humid weak ignition",
            "high cooling test",
            "hot efficient flame",
            "small chamber flash",
        ]

    def choose_next(self, previous: RoundSummary | None, current: FlameParams, round_number: int) -> FlameParams:
        goal = self.goal_cycle[(round_number - 1) % len(self.goal_cycle)]
        p = replace(current)

        if goal == "baseline balanced flame":
            p.fuel_name = "baseline methane-like gas"
            p.initial_fuel = 1.00
            p.initial_oxygen = 1.20
            p.fuel_supply_rate = 0.10
            p.oxygen_supply_rate = 0.55
            p.spark_energy_J = 650.0
            p.reaction_strength = 2.4
            p.convective_loss_coeff = 0.75
            p.mixing_loss_coeff = 0.12
            p.humidity = 0.20
            p.chamber_volume = 1.00

        elif goal == "fuel rich flame":
            p.fuel_name = "fuel-rich yellow flame"
            p.initial_fuel = 1.35
            p.initial_oxygen = 0.85
            p.fuel_supply_rate = 0.18
            p.oxygen_supply_rate = 0.33
            p.spark_energy_J = 700.0
            p.reaction_strength = 2.1
            p.convective_loss_coeff = 0.68
            p.mixing_loss_coeff = 0.10
            p.humidity = 0.18
            p.chamber_volume = 1.05
            p.soot_rate = 0.040

        elif goal == "oxygen poor flame":
            p.fuel_name = "oxygen-starved flame"
            p.initial_fuel = 1.05
            p.initial_oxygen = 0.42
            p.fuel_supply_rate = 0.12
            p.oxygen_supply_rate = 0.14
            p.spark_energy_J = 640.0
            p.reaction_strength = 1.9
            p.convective_loss_coeff = 0.72
            p.mixing_loss_coeff = 0.07
            p.humidity = 0.24
            p.chamber_volume = 1.00
            p.soot_rate = 0.055

        elif goal == "strong airflow flame":
            p.fuel_name = "airflow-stretched flame"
            p.initial_fuel = 0.95
            p.initial_oxygen = 1.35
            p.fuel_supply_rate = 0.11
            p.oxygen_supply_rate = 0.85
            p.spark_energy_J = 680.0
            p.reaction_strength = 2.8
            p.convective_loss_coeff = 1.05
            p.mixing_loss_coeff = 0.28
            p.humidity = 0.16
            p.chamber_volume = 1.10
            p.soot_rate = 0.018

        elif goal == "humid weak ignition":
            p.fuel_name = "humid weak ignition"
            p.initial_fuel = 0.90
            p.initial_oxygen = 1.00
            p.fuel_supply_rate = 0.08
            p.oxygen_supply_rate = 0.45
            p.spark_energy_J = 430.0
            p.reaction_strength = 1.85
            p.convective_loss_coeff = 0.82
            p.mixing_loss_coeff = 0.15
            p.humidity = 0.70
            p.chamber_volume = 1.00
            p.soot_rate = 0.026

        elif goal == "high cooling test":
            p.fuel_name = "high cooling flame"
            p.initial_fuel = 1.10
            p.initial_oxygen = 1.20
            p.fuel_supply_rate = 0.10
            p.oxygen_supply_rate = 0.55
            p.spark_energy_J = 710.0
            p.reaction_strength = 2.3
            p.convective_loss_coeff = 1.45
            p.mixing_loss_coeff = 0.35
            p.humidity = 0.25
            p.chamber_volume = 1.15
            p.soot_rate = 0.022

        elif goal == "hot efficient flame":
            p.fuel_name = "hot blue efficient flame"
            p.initial_fuel = 1.05
            p.initial_oxygen = 1.60
            p.fuel_supply_rate = 0.13
            p.oxygen_supply_rate = 0.95
            p.spark_energy_J = 760.0
            p.reaction_strength = 3.2
            p.convective_loss_coeff = 0.70
            p.mixing_loss_coeff = 0.16
            p.humidity = 0.08
            p.chamber_volume = 1.00
            p.soot_rate = 0.010

        elif goal == "small chamber flash":
            p.fuel_name = "small chamber flash"
            p.initial_fuel = 1.25
            p.initial_oxygen = 1.25
            p.fuel_supply_rate = 0.02
            p.oxygen_supply_rate = 0.03
            p.spark_energy_J = 820.0
            p.reaction_strength = 4.2
            p.convective_loss_coeff = 0.95
            p.mixing_loss_coeff = 0.05
            p.humidity = 0.10
            p.chamber_volume = 0.55
            p.soot_rate = 0.028

        # Reactive correction from previous round.
        if previous is not None:
            if previous.peak_temperature_K < p.ignition_temperature_K + 100:
                p.spark_energy_J += 120.0
                p.oxygen_supply_rate += 0.08
            if previous.extinction_cause == "oxygen starved":
                p.oxygen_supply_rate += 0.15
                p.initial_oxygen += 0.20
            if previous.final_soot > 0.02:
                p.oxygen_supply_rate += 0.10
                p.soot_rate *= 0.85
            if previous.peak_temperature_K > 2600:
                p.reaction_strength *= 0.9
                p.convective_loss_coeff += 0.1

        # Keep values in reasonable ranges.
        p.initial_fuel = clamp(p.initial_fuel, 0.05, 2.0)
        p.initial_oxygen = clamp(p.initial_oxygen, 0.05, 2.0)
        p.fuel_supply_rate = clamp(p.fuel_supply_rate, 0.0, 0.5)
        p.oxygen_supply_rate = clamp(p.oxygen_supply_rate, 0.0, 1.4)
        p.spark_energy_J = clamp(p.spark_energy_J, 0.0, 1200.0)
        p.reaction_strength = clamp(p.reaction_strength, 0.1, 6.0)
        p.convective_loss_coeff = clamp(p.convective_loss_coeff, 0.05, 3.0)
        p.mixing_loss_coeff = clamp(p.mixing_loss_coeff, 0.0, 0.8)
        p.humidity = clamp(p.humidity, 0.0, 1.0)
        p.chamber_volume = clamp(p.chamber_volume, 0.3, 3.0)
        p.soot_rate = clamp(p.soot_rate, 0.0, 0.12)

        print(f"\nAI controller selected next experiment: {goal}")
        return p


# -----------------------------------------------------------------------------
# Final summary
# -----------------------------------------------------------------------------

def print_final_summary(summaries: List[RoundSummary]) -> None:
    print("\n" + "=" * 132)
    print("FINAL SIMPLE FLAME COMBUSTION COMPARISON")
    print("=" * 132)
    header = (
        f"{'Round':>5} | {'Fuel model':>24} | {'Peak K':>8} | {'Final K':>8} | "
        f"{'Burn s':>7} | {'Energy J':>9} | {'Lost J':>9} | {'Fuel':>7} | "
        f"{'O2 used':>8} | {'PeakRate':>8} | {'Eff':>6} | {'Soot':>7} | End state"
    )
    print(header)
    print("-" * len(header))

    for s in summaries:
        print(
            f"{s.round_number:5d} | {s.params.fuel_name[:24]:>24} | "
            f"{s.peak_temperature_K:8.1f} | {s.final_temperature_K:8.1f} | "
            f"{s.burn_time_s:7.2f} | {s.total_energy_released_J:9.2f} | "
            f"{s.total_heat_lost_J:9.2f} | {s.fuel_burned:7.4f} | "
            f"{s.oxygen_used:8.4f} | {s.peak_reaction_rate:8.4f} | "
            f"{s.average_efficiency:6.3f} | {s.final_soot:7.4f} | {s.extinction_cause}"
        )

    hottest = max(summaries, key=lambda s: s.peak_temperature_K)
    most_energy = max(summaries, key=lambda s: s.total_energy_released_J)
    cleanest = min(summaries, key=lambda s: s.final_soot)
    longest = max(summaries, key=lambda s: s.burn_time_s)

    print("\nBest scientific observations:")
    print(f"  Hottest flame: Round {hottest.round_number}, {hottest.params.fuel_name}, peak {hottest.peak_temperature_K:.1f} K")
    print(f"  Most energy released: Round {most_energy.round_number}, {most_energy.total_energy_released_J:.2f} J")
    print(f"  Cleanest burn: Round {cleanest.round_number}, soot {cleanest.final_soot:.4f}")
    print(f"  Longest active burn: Round {longest.round_number}, {longest.burn_time_s:.2f} s")


def main() -> None:
    config = SimConfig()
    rng = random.Random(config.seed)
    controller = FlameAIController(rng)

    params = FlameParams()
    previous_summary: RoundSummary | None = None
    summaries: List[RoundSummary] = []

    print("Simple Flame Combustion Property Simulation")
    print("Terminal-only scientific model. No external packages.")
    print("Model: well-mixed flame cell with fuel, oxygen, heat release, losses, smoke, soot, efficiency, and extinction.")

    for round_number in range(1, config.rounds + 1):
        params = controller.choose_next(previous_summary, params, round_number)
        summary = run_round(round_number, params, config)
        summaries.append(summary)
        previous_summary = summary

    print_final_summary(summaries)


if __name__ == "__main__":
    main()
