#!/usr/bin/env python3
"""
Stable Flame Combustion Property Simulation
-------------------------------------------

A terminal-only scientific simulation of a simple flame that can now develop
stable flames instead of only brief flash/extinction behavior.

Run:
    python simple_flame_combustion_simulation_stable_flames.py

No external packages are required.

This is not a visual flame simulation. It models the time evolution of flame
properties: fuel, oxygen, temperature, reaction rate, heat release, heat losses,
smoke/soot formation, burn efficiency, flame-kernel stability, and extinction
or sustained burning cause.

The model is intentionally simplified. It is a zero-dimensional, well-mixed
reactor-style toy model suitable for terminal experiments, not a full CFD or
detailed chemical kinetics solver.

What evolved from the source version:
    - Adds flame-kernel memory so a recently established flame can stabilize.
    - Adds pilot/holder heat recirculation to represent a burner/flame holder.
    - Separates fresh reactant supply from thermal dilution.
    - Adds supply-ratio control for stable blue, yellow, lean, rich, humid,
      high-airflow, and pulsed flames.
    - Adds sustained-burn reporting instead of treating every round as a flash.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import exp, sin, pi
import random
from typing import Dict, List


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

@dataclass
class SimConfig:
    rounds: int = 8
    duration_s: float = 10.0
    dt_s: float = 0.01
    print_every_s: float = 0.25
    seed: int = 23


@dataclass
class FlameParams:
    fuel_name: str = "stable methane-like burner"

    # Reservoir/supply parameters, normalized to simple units.
    initial_fuel: float = 0.85
    initial_oxygen: float = 1.35
    fuel_supply_rate: float = 0.115       # normalized fuel units per second
    oxygen_supply_rate: float = 0.245     # normalized oxygen units per second
    max_fuel: float = 1.60
    max_oxygen: float = 1.90

    # Ignition and reaction behavior.
    initial_temperature_K: float = 300.0
    spark_energy_J: float = 660.0
    ignition_temperature_K: float = 720.0
    extinction_temperature_K: float = 620.0
    reaction_strength: float = 8.0
    activation_temperature_K: float = 180.0
    oxygen_per_fuel: float = 2.0

    # Thermodynamic scale.
    heat_of_combustion_J_per_unit: float = 5000.0
    heat_capacity_J_per_K: float = 1.45
    ambient_temperature_K: float = 300.0

    # Heat loss terms.
    convective_loss_coeff: float = 0.16
    radiative_loss_coeff: float = 0.00000000018
    mixing_loss_coeff: float = 0.018

    # Stabilization terms.
    flame_holder_strength: float = 0.52       # heat recirculation from hot burned gas
    pilot_heat_W: float = 22.0                # small burner/pilot support while flame exists
    kernel_growth_rate: float = 1.45
    kernel_decay_rate: float = 0.20
    stable_kernel_threshold: float = 0.36
    target_temperature_K: float = 1180.0      # soft cap from expansion/dilution, not a hard clamp

    # Byproduct behavior.
    soot_rate: float = 0.014
    smoke_decay_rate: float = 0.30
    humidity: float = 0.16
    chamber_volume: float = 1.10

    # Optional forcing for unstable experiments.
    fuel_pulse_fraction: float = 0.0
    oxygen_pulse_fraction: float = 0.0
    pulse_period_s: float = 2.5


@dataclass
class FlameState:
    t_s: float = 0.0
    temperature_K: float = 300.0
    fuel: float = 1.0
    oxygen: float = 1.0
    burned_gas: float = 0.0
    smoke: float = 0.0
    soot: float = 0.0
    flame_kernel: float = 0.0
    total_energy_released_J: float = 0.0
    total_heat_lost_J: float = 0.0
    total_recirculated_heat_J: float = 0.0
    total_fuel_burned: float = 0.0
    state_name: str = "unignited"


@dataclass
class RoundSummary:
    round_number: int
    params: FlameParams
    peak_temperature_K: float
    final_temperature_K: float
    average_temperature_K: float
    burn_time_s: float
    stable_time_s: float
    total_energy_released_J: float
    total_heat_lost_J: float
    total_recirculated_heat_J: float
    fuel_burned: float
    oxygen_used: float
    peak_reaction_rate: float
    average_efficiency: float
    final_smoke: float
    final_soot: float
    final_kernel: float
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

    The evolved model has a steeper activation curve and does not drop to zero
    immediately when a flame kernel has already formed.
    """
    if temperature_K < params.extinction_temperature_K - 80.0:
        return 0.0

    center = params.ignition_temperature_K
    slope = max(20.0, params.activation_temperature_K)
    return clamp(1.0 / (1.0 + exp(-(temperature_K - center) / slope)), 0.0, 1.0)


def supply_multiplier(t_s: float, fraction: float, period_s: float) -> float:
    """Optional smooth pulsing for test rounds."""
    if fraction <= 0.0:
        return 1.0
    wave = 0.5 + 0.5 * sin(2.0 * pi * t_s / max(0.1, period_s))
    return clamp(1.0 - fraction + 2.0 * fraction * wave, 0.05, 2.25)


def mixture_quality(state: FlameState, params: FlameParams) -> float:
    """
    1.0 is near stoichiometric. Low values represent too rich or too lean.
    """
    stoich_o2 = params.oxygen_per_fuel * max(0.001, state.fuel)
    ratio = state.oxygen / max(0.001, stoich_o2)
    rich_penalty = clamp(ratio / 0.75, 0.0, 1.0)
    lean_penalty = clamp(1.60 / max(0.001, ratio), 0.0, 1.0)
    return clamp(min(rich_penalty, lean_penalty), 0.0, 1.0)


def reaction_rate(state: FlameState, params: FlameParams) -> float:
    """
    Simplified reaction rate.

    Rate depends on available fuel, oxygen, temperature activation, mixture
    quality, humidity, chamber volume, and flame-kernel stabilization.
    """
    if state.fuel <= 0.0001 or state.oxygen <= 0.0001:
        return 0.0

    thermal_support = temperature_factor(state.temperature_K, params)
    kernel_support = 0.35 * state.flame_kernel
    activation = clamp(thermal_support + kernel_support, 0.0, 1.25)

    fuel_factor = state.fuel / max(0.001, params.chamber_volume)
    oxygen_factor = state.oxygen / max(0.001, params.chamber_volume)
    mix_quality = mixture_quality(state, params)
    humidity_penalty = clamp(1.0 - 0.36 * params.humidity, 0.35, 1.0)

    # In a stable burner, the flame consumes only part of the reservoir each
    # time step. This avoids the whole chamber flashing at once.
    reservoir_damping = 1.0 / (1.0 + 0.42 * (state.fuel + state.oxygen))

    return (
        params.reaction_strength
        * fuel_factor
        * oxygen_factor
        * activation
        * mix_quality
        * humidity_penalty
        * reservoir_damping
    )


def heat_losses_W(state: FlameState, params: FlameParams) -> Dict[str, float]:
    """Compute convective, radiative, mixing, and stabilization-adjusted losses."""
    delta_T = max(0.0, state.temperature_K - params.ambient_temperature_K)

    convective = params.convective_loss_coeff * delta_T
    radiative = params.radiative_loss_coeff * (
        max(0.0, state.temperature_K ** 4 - params.ambient_temperature_K ** 4)
    )
    supply_flow = params.oxygen_supply_rate + params.fuel_supply_rate
    mixing = params.mixing_loss_coeff * delta_T * supply_flow

    # A stronger flame kernel shields/anchors the flame and reduces effective
    # losses, like a simplified flame holder.
    stabilization = clamp(1.0 - 0.34 * state.flame_kernel, 0.58, 1.0)

    # Very high temperatures push extra heat out through expansion and dilution.
    overheat = max(0.0, state.temperature_K - params.target_temperature_K)
    expansion = 0.018 * overheat * overheat / max(1.0, params.target_temperature_K - params.ambient_temperature_K)

    subtotal = convective + radiative + mixing + expansion
    return {
        "convective": convective,
        "radiative": radiative,
        "mixing": mixing,
        "expansion": expansion,
        "stabilized_total": subtotal * stabilization,
        "raw_total": subtotal,
    }


def classify_state(state: FlameState, params: FlameParams, rate: float, net_heat_J: float) -> str:
    if state.fuel <= 0.005:
        return "fuel-limited"
    if state.oxygen <= 0.005:
        return "oxygen-limited"
    if state.temperature_K < params.extinction_temperature_K and state.flame_kernel < 0.12:
        if state.total_energy_released_J > 0.0:
            return "extinguishing"
        return "pre-ignition"
    if state.flame_kernel >= params.stable_kernel_threshold and rate > 0.08:
        if abs(net_heat_J) < 0.35:
            return "stable flame"
        if net_heat_J > 0.0:
            return "building flame"
        return "cooling stable"
    if rate > 0.55:
        return "active burning"
    if rate > 0.08:
        return "weak burning"
    return "hot but slow"


def step_flame(state: FlameState, params: FlameParams, dt_s: float, is_first_step: bool) -> tuple[FlameState, Dict[str, float]]:
    s = replace(state)

    if is_first_step:
        s.temperature_K += ignition_boost_temperature(params)
        s.flame_kernel = max(s.flame_kernel, 0.10)

    # Supply fresh reactants. Optional pulsing lets one round show oscillation.
    fuel_supply = params.fuel_supply_rate * supply_multiplier(s.t_s, params.fuel_pulse_fraction, params.pulse_period_s)
    oxygen_supply = params.oxygen_supply_rate * supply_multiplier(s.t_s, params.oxygen_pulse_fraction, params.pulse_period_s)

    s.fuel = clamp(s.fuel + fuel_supply * dt_s, 0.0, params.max_fuel)
    s.oxygen = clamp(s.oxygen + oxygen_supply * dt_s, 0.0, params.max_oxygen)

    # Combustion.
    rate = reaction_rate(s, params)
    requested_fuel_burn = rate * dt_s

    # The flame front can only burn what local mixing can feed. This creates
    # sustained consumption instead of a single all-at-once flash.
    local_feed_limit = (0.90 * params.fuel_supply_rate + 0.030 + 0.12 * s.flame_kernel) * dt_s
    possible_fuel_burn = min(s.fuel, s.oxygen / max(0.001, params.oxygen_per_fuel), local_feed_limit)
    fuel_burned = min(requested_fuel_burn, possible_fuel_burn)

    oxygen_used = fuel_burned * params.oxygen_per_fuel
    heat_generated_J = fuel_burned * params.heat_of_combustion_J_per_unit

    s.fuel -= fuel_burned
    s.oxygen -= oxygen_used
    s.burned_gas = max(0.0, s.burned_gas * exp(-0.18 * dt_s) + fuel_burned + oxygen_used)
    s.total_fuel_burned += fuel_burned
    s.total_energy_released_J += heat_generated_J

    # Flame kernel evolves. It grows when burn rate and temperature support
    # combustion, then slowly decays when the flame weakens.
    thermal_margin = clamp((s.temperature_K - params.extinction_temperature_K) / 450.0, 0.0, 1.0)
    kernel_growth = params.kernel_growth_rate * rate * thermal_margin * dt_s
    kernel_decay = params.kernel_decay_rate * (1.0 - thermal_margin) * dt_s
    if rate < 0.04:
        kernel_decay += 0.22 * dt_s
    s.flame_kernel = clamp(s.flame_kernel + kernel_growth - kernel_decay, 0.0, 1.0)

    # Smoke/soot: richer, oxygen-poor, humid, or cooler burns produce more.
    stoich_o2 = params.oxygen_per_fuel * max(0.001, s.fuel)
    ratio = s.oxygen / max(0.001, stoich_o2)
    oxygen_starvation = clamp((1.0 - ratio) * 1.8, 0.0, 2.0)
    rich_indicator = clamp((s.fuel / max(0.001, s.oxygen) - 0.45) * 0.6, 0.0, 2.0)
    cool_penalty = clamp((params.ignition_temperature_K + 250.0 - s.temperature_K) / 800.0, 0.0, 1.0)
    soot_created = fuel_burned * params.soot_rate * (1.0 + oxygen_starvation + rich_indicator + cool_penalty + params.humidity)
    smoke_created = soot_created * 4.5 + fuel_burned * 0.018 * oxygen_starvation

    s.soot += soot_created
    s.smoke += smoke_created
    s.smoke *= exp(-params.smoke_decay_rate * dt_s)

    # Heat losses and stabilizing recirculation.
    losses = heat_losses_W(s, params)
    heat_lost_J = losses["stabilized_total"] * dt_s
    recirculated_heat_J = (
        params.flame_holder_strength
        * s.flame_kernel
        * min(1.0, s.burned_gas)
        * max(0.0, s.temperature_K - params.ambient_temperature_K)
        * 0.020
        * dt_s
    )
    pilot_heat_J = params.pilot_heat_W * s.flame_kernel * dt_s

    s.total_heat_lost_J += heat_lost_J
    s.total_recirculated_heat_J += recirculated_heat_J + pilot_heat_J

    # Temperature update.
    net_heat_J = heat_generated_J + recirculated_heat_J + pilot_heat_J - heat_lost_J
    s.temperature_K += net_heat_J / max(0.001, params.heat_capacity_J_per_K)

    # Passive cooling cannot go below ambient in this simple model.
    if s.temperature_K < params.ambient_temperature_K:
        s.temperature_K = params.ambient_temperature_K

    s.t_s += dt_s
    s.state_name = classify_state(s, params, rate, net_heat_J)

    diagnostics = {
        "reaction_rate": rate,
        "fuel_burned": fuel_burned,
        "oxygen_used": oxygen_used,
        "heat_generated_J": heat_generated_J,
        "heat_lost_J": heat_lost_J,
        "recirculated_heat_J": recirculated_heat_J + pilot_heat_J,
        "convective_loss_W": losses["convective"],
        "radiative_loss_W": losses["radiative"],
        "mixing_loss_W": losses["mixing"],
        "expansion_loss_W": losses["expansion"],
        "raw_loss_W": losses["raw_total"],
        "total_loss_W": losses["stabilized_total"],
        "power_W": heat_generated_J / dt_s if dt_s > 0 else 0.0,
        "net_heat_J": net_heat_J,
        "efficiency": (heat_generated_J + recirculated_heat_J + pilot_heat_J)
        / max(0.001, heat_generated_J + recirculated_heat_J + pilot_heat_J + heat_lost_J),
    }
    return s, diagnostics


def flame_height_estimate_m(state: FlameState, diagnostics: Dict[str, float], params: FlameParams) -> float:
    """
    Crude estimated flame height.

    It grows with heat-release power, oxygen, and kernel stability. Units are
    illustrative meters.
    """
    power = diagnostics["power_W"]
    oxygen_factor = clamp(state.oxygen / max(0.001, params.max_oxygen), 0.0, 1.0)
    fuel_factor = clamp(state.fuel / max(0.001, params.max_fuel), 0.0, 1.0)
    kernel_factor = 0.65 + 0.55 * state.flame_kernel
    return clamp(0.04 + 0.009 * (power ** 0.5) * (0.4 + oxygen_factor) * (0.3 + fuel_factor) * kernel_factor, 0.0, 1.5)


# -----------------------------------------------------------------------------
# Round simulation
# -----------------------------------------------------------------------------

def run_round(round_number: int, params: FlameParams, config: SimConfig) -> RoundSummary:
    state = FlameState(
        temperature_K=params.initial_temperature_K,
        fuel=params.initial_fuel,
        oxygen=params.initial_oxygen,
    )

    print("\n" + "=" * 132)
    print(f"ROUND {round_number}: {params.fuel_name}")
    print("=" * 132)
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
        f"mix={params.mixing_loss_coeff:.3f}, humidity={params.humidity:.2f}, volume={params.chamber_volume:.2f}, "
        f"holder={params.flame_holder_strength:.2f}, pilot={params.pilot_heat_W:.1f} W"
    )
    print("-" * 132)

    print(
        f"{'t(s)':>6} | {'state':>15} | {'T(K)':>8} | {'fuel':>7} | {'O2':>7} | "
        f"{'rate':>8} | {'power(W)':>9} | {'loss(W)':>8} | {'kernel':>6} | "
        f"{'smoke':>7} | {'soot':>7} | {'h(m)':>6}"
    )
    print("-" * 132)

    peak_temperature = state.temperature_K
    peak_rate = 0.0
    burn_time = 0.0
    stable_time = 0.0
    temp_sum = 0.0
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
        "net_heat_J": 0.0,
    }

    for step in range(steps + 1):
        state, diagnostics = step_flame(state, params, config.dt_s, is_first_step=(step == 0))

        peak_temperature = max(peak_temperature, state.temperature_K)
        peak_rate = max(peak_rate, diagnostics["reaction_rate"])
        temp_sum += state.temperature_K

        if diagnostics["reaction_rate"] > 0.02:
            burn_time += config.dt_s

        if state.flame_kernel >= params.stable_kernel_threshold and state.temperature_K >= params.extinction_temperature_K:
            stable_time += config.dt_s

        efficiency_sum += diagnostics["efficiency"]
        efficiency_samples += 1
        oxygen_initial_plus_supply += params.oxygen_supply_rate * config.dt_s

        if state.t_s + 1e-9 >= next_print or step == steps:
            h = flame_height_estimate_m(state, diagnostics, params)
            print(
                f"{state.t_s:6.2f} | {state.state_name:>15} | {state.temperature_K:8.1f} | "
                f"{state.fuel:7.3f} | {state.oxygen:7.3f} | "
                f"{diagnostics['reaction_rate']:8.4f} | {diagnostics['power_W']:9.2f} | "
                f"{diagnostics['total_loss_W']:8.2f} | {state.flame_kernel:6.3f} | "
                f"{state.smoke:7.4f} | {state.soot:7.4f} | {h:6.3f}"
            )
            next_print += config.print_every_s

    oxygen_used = max(0.0, oxygen_initial_plus_supply - state.oxygen)
    avg_efficiency = efficiency_sum / max(1, efficiency_samples)
    avg_temperature = temp_sum / max(1, efficiency_samples)

    extinction_cause = determine_extinction_cause(state, params, diagnostics, stable_time, config.duration_s)

    summary = RoundSummary(
        round_number=round_number,
        params=params,
        peak_temperature_K=peak_temperature,
        final_temperature_K=state.temperature_K,
        average_temperature_K=avg_temperature,
        burn_time_s=burn_time,
        stable_time_s=stable_time,
        total_energy_released_J=state.total_energy_released_J,
        total_heat_lost_J=state.total_heat_lost_J,
        total_recirculated_heat_J=state.total_recirculated_heat_J,
        fuel_burned=state.total_fuel_burned,
        oxygen_used=oxygen_used,
        peak_reaction_rate=peak_rate,
        average_efficiency=avg_efficiency,
        final_smoke=state.smoke,
        final_soot=state.soot,
        final_kernel=state.flame_kernel,
        extinction_cause=extinction_cause,
    )

    print("-" * 132)
    print_round_summary(summary)
    return summary


def determine_extinction_cause(
    state: FlameState,
    params: FlameParams,
    diagnostics: Dict[str, float],
    stable_time_s: float,
    duration_s: float,
) -> str:
    if state.fuel <= 0.01:
        return "fuel exhausted"
    if state.oxygen <= 0.01:
        return "oxygen starved"
    if state.temperature_K < params.extinction_temperature_K and state.flame_kernel < 0.12:
        return "thermal extinction"
    if stable_time_s >= 0.65 * duration_s and state.temperature_K >= params.extinction_temperature_K:
        return "sustained stable flame"
    if stable_time_s > 0.0 and state.temperature_K >= params.extinction_temperature_K:
        return "partly stabilized flame"
    if diagnostics["reaction_rate"] < 0.03:
        return "weak reaction"
    if state.total_energy_released_J > 0 and state.temperature_K >= params.extinction_temperature_K:
        return "still burning at end"
    return "not ignited"


def print_round_summary(summary: RoundSummary) -> None:
    print("Round result:")
    print(f"  peak temperature: {summary.peak_temperature_K:.1f} K")
    print(f"  average temperature: {summary.average_temperature_K:.1f} K")
    print(f"  final temperature: {summary.final_temperature_K:.1f} K")
    print(f"  burn time: {summary.burn_time_s:.2f} s")
    print(f"  stable flame time: {summary.stable_time_s:.2f} s")
    print(f"  total energy released: {summary.total_energy_released_J:.2f} J")
    print(f"  total heat lost: {summary.total_heat_lost_J:.2f} J")
    print(f"  heat recirculated/pilot support: {summary.total_recirculated_heat_J:.2f} J")
    print(f"  fuel burned: {summary.fuel_burned:.4f} normalized units")
    print(f"  oxygen used: {summary.oxygen_used:.4f} normalized units")
    print(f"  peak reaction rate: {summary.peak_reaction_rate:.4f} units/s")
    print(f"  average heat efficiency: {summary.average_efficiency:.3f}")
    print(f"  final flame kernel: {summary.final_kernel:.3f}")
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
            "baseline stable flame",
            "clean blue flame",
            "fuel rich yellow stable flame",
            "lean oxygen supported flame",
            "airflow anchored flame",
            "humid rescued flame",
            "high cooling stabilized flame",
            "pulsed burner flame",
        ]

    def choose_next(self, previous: RoundSummary | None, current: FlameParams, round_number: int) -> FlameParams:
        goal = self.goal_cycle[(round_number - 1) % len(self.goal_cycle)]
        p = replace(current)

        # Reset optional forcing unless a round uses it.
        p.fuel_pulse_fraction = 0.0
        p.oxygen_pulse_fraction = 0.0
        p.pulse_period_s = 2.5

        if goal == "baseline stable flame":
            p.fuel_name = "baseline stable methane-like flame"
            p.initial_fuel = 0.85
            p.initial_oxygen = 1.35
            p.fuel_supply_rate = 0.115
            p.oxygen_supply_rate = 0.245
            p.spark_energy_J = 660.0
            p.reaction_strength = 8.0
            p.convective_loss_coeff = 0.20
            p.mixing_loss_coeff = 0.025
            p.radiative_loss_coeff = 0.00000000018
            p.humidity = 0.16
            p.chamber_volume = 1.10
            p.flame_holder_strength = 0.52
            p.pilot_heat_W = 22.0
            p.soot_rate = 0.014
            p.target_temperature_K = 1180.0

        elif goal == "clean blue flame":
            p.fuel_name = "clean blue stabilized flame"
            p.initial_fuel = 0.75
            p.initial_oxygen = 1.60
            p.fuel_supply_rate = 0.105
            p.oxygen_supply_rate = 0.260
            p.spark_energy_J = 690.0
            p.reaction_strength = 8.8
            p.convective_loss_coeff = 0.18
            p.mixing_loss_coeff = 0.020
            p.radiative_loss_coeff = 0.00000000016
            p.humidity = 0.06
            p.chamber_volume = 1.00
            p.flame_holder_strength = 0.58
            p.pilot_heat_W = 20.0
            p.soot_rate = 0.006
            p.target_temperature_K = 1280.0

        elif goal == "fuel rich yellow stable flame":
            p.fuel_name = "fuel-rich yellow stable flame"
            p.initial_fuel = 1.20
            p.initial_oxygen = 1.05
            p.fuel_supply_rate = 0.135
            p.oxygen_supply_rate = 0.220
            p.spark_energy_J = 700.0
            p.reaction_strength = 7.2
            p.convective_loss_coeff = 0.19
            p.mixing_loss_coeff = 0.022
            p.radiative_loss_coeff = 0.00000000020
            p.humidity = 0.14
            p.chamber_volume = 1.18
            p.flame_holder_strength = 0.55
            p.pilot_heat_W = 25.0
            p.soot_rate = 0.033
            p.target_temperature_K = 1120.0

        elif goal == "lean oxygen supported flame":
            p.fuel_name = "lean oxygen-supported flame"
            p.initial_fuel = 0.62
            p.initial_oxygen = 1.75
            p.fuel_supply_rate = 0.088
            p.oxygen_supply_rate = 0.300
            p.spark_energy_J = 710.0
            p.reaction_strength = 8.3
            p.convective_loss_coeff = 0.22
            p.mixing_loss_coeff = 0.030
            p.radiative_loss_coeff = 0.00000000016
            p.humidity = 0.10
            p.chamber_volume = 1.05
            p.flame_holder_strength = 0.60
            p.pilot_heat_W = 24.0
            p.soot_rate = 0.007
            p.target_temperature_K = 1210.0

        elif goal == "airflow anchored flame":
            p.fuel_name = "airflow-anchored stable flame"
            p.initial_fuel = 0.90
            p.initial_oxygen = 1.60
            p.fuel_supply_rate = 0.125
            p.oxygen_supply_rate = 0.420
            p.spark_energy_J = 720.0
            p.reaction_strength = 9.4
            p.convective_loss_coeff = 0.34
            p.mixing_loss_coeff = 0.055
            p.radiative_loss_coeff = 0.00000000018
            p.humidity = 0.12
            p.chamber_volume = 1.16
            p.flame_holder_strength = 0.74
            p.pilot_heat_W = 34.0
            p.soot_rate = 0.010
            p.target_temperature_K = 1230.0

        elif goal == "humid rescued flame":
            p.fuel_name = "humid flame rescued by pilot"
            p.initial_fuel = 0.95
            p.initial_oxygen = 1.40
            p.fuel_supply_rate = 0.120
            p.oxygen_supply_rate = 0.285
            p.spark_energy_J = 760.0
            p.reaction_strength = 8.5
            p.convective_loss_coeff = 0.24
            p.mixing_loss_coeff = 0.035
            p.radiative_loss_coeff = 0.00000000018
            p.humidity = 0.58
            p.chamber_volume = 1.08
            p.flame_holder_strength = 0.68
            p.pilot_heat_W = 46.0
            p.soot_rate = 0.020
            p.target_temperature_K = 1050.0

        elif goal == "high cooling stabilized flame":
            p.fuel_name = "high-cooling stabilized flame"
            p.initial_fuel = 1.00
            p.initial_oxygen = 1.55
            p.fuel_supply_rate = 0.145
            p.oxygen_supply_rate = 0.360
            p.spark_energy_J = 820.0
            p.reaction_strength = 10.2
            p.convective_loss_coeff = 0.48
            p.mixing_loss_coeff = 0.075
            p.radiative_loss_coeff = 0.00000000020
            p.humidity = 0.18
            p.chamber_volume = 1.12
            p.flame_holder_strength = 0.86
            p.pilot_heat_W = 55.0
            p.soot_rate = 0.012
            p.target_temperature_K = 1140.0

        elif goal == "pulsed burner flame":
            p.fuel_name = "pulsed but stable burner flame"
            p.initial_fuel = 0.92
            p.initial_oxygen = 1.50
            p.fuel_supply_rate = 0.125
            p.oxygen_supply_rate = 0.295
            p.spark_energy_J = 720.0
            p.reaction_strength = 8.9
            p.convective_loss_coeff = 0.23
            p.mixing_loss_coeff = 0.030
            p.radiative_loss_coeff = 0.00000000018
            p.humidity = 0.12
            p.chamber_volume = 1.05
            p.flame_holder_strength = 0.66
            p.pilot_heat_W = 32.0
            p.soot_rate = 0.012
            p.target_temperature_K = 1190.0
            p.fuel_pulse_fraction = 0.28
            p.oxygen_pulse_fraction = 0.18
            p.pulse_period_s = 2.0

        # Reactive correction from previous round: make the controller learn
        # toward stable flames if a test underperformed.
        if previous is not None:
            if previous.stable_time_s < 0.50 * SimConfig.duration_s:
                p.spark_energy_J += 40.0
                p.pilot_heat_W += 8.0
                p.flame_holder_strength += 0.05
                p.convective_loss_coeff *= 0.94
            if previous.extinction_cause == "oxygen starved":
                p.oxygen_supply_rate += 0.06
                p.initial_oxygen += 0.12
            if previous.final_soot > 0.015:
                p.oxygen_supply_rate += 0.04
                p.soot_rate *= 0.88
            if previous.peak_temperature_K > 1650:
                p.reaction_strength *= 0.93
                p.convective_loss_coeff += 0.04
                p.target_temperature_K -= 40.0

        # Keep values in reasonable ranges.
        p.initial_fuel = clamp(p.initial_fuel, 0.05, 2.0)
        p.initial_oxygen = clamp(p.initial_oxygen, 0.05, 2.0)
        p.fuel_supply_rate = clamp(p.fuel_supply_rate, 0.0, 0.5)
        p.oxygen_supply_rate = clamp(p.oxygen_supply_rate, 0.0, 1.4)
        p.spark_energy_J = clamp(p.spark_energy_J, 0.0, 1200.0)
        p.reaction_strength = clamp(p.reaction_strength, 0.1, 14.0)
        p.convective_loss_coeff = clamp(p.convective_loss_coeff, 0.03, 3.0)
        p.mixing_loss_coeff = clamp(p.mixing_loss_coeff, 0.0, 0.8)
        p.radiative_loss_coeff = clamp(p.radiative_loss_coeff, 0.0, 0.000000002)
        p.humidity = clamp(p.humidity, 0.0, 1.0)
        p.chamber_volume = clamp(p.chamber_volume, 0.3, 3.0)
        p.soot_rate = clamp(p.soot_rate, 0.0, 0.12)
        p.flame_holder_strength = clamp(p.flame_holder_strength, 0.0, 1.2)
        p.pilot_heat_W = clamp(p.pilot_heat_W, 0.0, 90.0)
        p.target_temperature_K = clamp(p.target_temperature_K, 850.0, 1600.0)

        print(f"\nAI controller selected next experiment: {goal}")
        return p


# -----------------------------------------------------------------------------
# Final summary
# -----------------------------------------------------------------------------

def print_final_summary(summaries: List[RoundSummary]) -> None:
    print("\n" + "=" * 152)
    print("FINAL STABLE FLAME COMBUSTION COMPARISON")
    print("=" * 152)
    header = (
        f"{'Round':>5} | {'Fuel model':>30} | {'Peak K':>8} | {'Avg K':>8} | {'Final K':>8} | "
        f"{'Burn s':>7} | {'Stable s':>8} | {'Energy J':>9} | {'Lost J':>9} | {'Fuel':>7} | "
        f"{'O2 used':>8} | {'Kernel':>6} | {'Eff':>6} | {'Soot':>7} | End state"
    )
    print(header)
    print("-" * len(header))

    for s in summaries:
        print(
            f"{s.round_number:5d} | {s.params.fuel_name[:30]:>30} | "
            f"{s.peak_temperature_K:8.1f} | {s.average_temperature_K:8.1f} | {s.final_temperature_K:8.1f} | "
            f"{s.burn_time_s:7.2f} | {s.stable_time_s:8.2f} | {s.total_energy_released_J:9.2f} | "
            f"{s.total_heat_lost_J:9.2f} | {s.fuel_burned:7.4f} | {s.oxygen_used:8.4f} | "
            f"{s.final_kernel:6.3f} | {s.average_efficiency:6.3f} | {s.final_soot:7.4f} | {s.extinction_cause}"
        )

    hottest = max(summaries, key=lambda s: s.peak_temperature_K)
    most_energy = max(summaries, key=lambda s: s.total_energy_released_J)
    cleanest = min(summaries, key=lambda s: s.final_soot)
    longest = max(summaries, key=lambda s: s.burn_time_s)
    most_stable = max(summaries, key=lambda s: s.stable_time_s)

    stable_count = sum(1 for s in summaries if "stable" in s.extinction_cause)

    print("\nBest scientific observations:")
    print(f"  Stable outcomes: {stable_count}/{len(summaries)} rounds reached stable-flame behavior")
    print(f"  Most stable flame: Round {most_stable.round_number}, {most_stable.params.fuel_name}, stable {most_stable.stable_time_s:.2f} s")
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

    print("Stable Flame Combustion Property Simulation")
    print("Terminal-only scientific model. No external packages.")
    print("Model: well-mixed flame cell with fuel, oxygen, heat release, losses, smoke, soot, efficiency, flame kernel, and stability.")
    print("Evolution goal: build stable flames instead of immediate thermal extinction.")

    for round_number in range(1, config.rounds + 1):
        params = controller.choose_next(previous_summary, params, round_number)
        summary = run_round(round_number, params, config)
        summaries.append(summary)
        previous_summary = summary

    print_final_summary(summaries)


if __name__ == "__main__":
    main()
