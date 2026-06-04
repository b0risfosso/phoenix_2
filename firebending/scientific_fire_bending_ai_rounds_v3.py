#!/usr/bin/env python3
"""
Scientific Fire Bending AI Rounds Simulation v3
-----------------------------------------------

A calibrated terminal-only scientific/mathematical simulation of stylized fire
bending.

Run:
    python scientific_fire_bending_ai_rounds_v3.py

No external packages are required.

Version 3 balances the v1/v2 extremes:
    - ambient temperature remains stable near 300 K
    - pilot ignition lets selected modes cross ignition threshold
    - heat remains local instead of saturating the full grid
    - styles produce failed, scorching, stable, and ignition outcomes
    - shield remains local, sparks remain scattered, blue remains narrow

This is a toy terminal model. It is not CFD, a real combustion solver, or a
safety model.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import cos, sin, radians, sqrt, exp
import random
from typing import Dict, List


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

@dataclass
class SimConfig:
    rounds: int = 8
    width: int = 64
    height: int = 22
    steps_per_round: int = 220
    dt_s: float = 0.025
    print_every_steps: int = 20
    seed: int = 52

    ambient_temperature_K: float = 300.0
    cell_heat_capacity_J_per_K: float = 2.65
    thermal_diffusivity: float = 0.015
    scalar_diffusivity: float = 0.012
    oxygen_recovery_rate: float = 0.010
    velocity_decay: float = 0.68
    smoke_decay: float = 0.982
    max_temperature_K: float = 2100.0

    # Reporting thresholds.
    hot_threshold_K: float = 620.0
    scorch_threshold_K: float = 620.0
    ignition_threshold_K: float = 820.0


@dataclass
class FireParams:
    style: str = "jab"

    # Controlled emission parameters.
    injection_power_W: float = 520.0
    fuel_injection_rate: float = 0.32
    oxygen_injection_rate: float = 0.28
    momentum: float = 0.70
    duration_steps: int = 55
    direction_deg: float = 0.0
    cone_half_angle_deg: float = 8.0
    turbulence: float = 0.05

    # Combustion model.
    reaction_strength: float = 2.35
    ignition_temperature_K: float = 680.0
    extinction_temperature_K: float = 590.0
    activation_temperature_K: float = 650.0
    oxygen_per_fuel: float = 2.0
    heat_of_combustion_J_per_fuel: float = 1040.0

    # Heat losses and byproducts.
    convective_loss_coeff: float = 0.52
    radiative_loss_coeff: float = 1.4e-10
    mixing_loss_coeff: float = 0.040
    soot_yield: float = 0.020
    smoke_yield: float = 0.060

    # Geometry.
    bender_x: int = 8
    bender_y: int = 11
    target_x: int = 54
    target_y: int = 11


@dataclass
class FireCell:
    temperature_K: float = 300.0
    fuel: float = 0.0
    oxygen: float = 0.21
    smoke: float = 0.0
    soot: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    combustion_rate: float = 0.0


@dataclass
class RoundSummary:
    round_number: int
    params: FireParams
    peak_temperature_K: float
    average_temperature_K: float
    final_average_temperature_K: float
    total_energy_released_J: float
    total_heat_lost_J: float
    fuel_burned: float
    oxygen_consumed: float
    peak_combustion_rate: float
    average_combustion_rate: float
    target_peak_temperature_K: float
    target_average_temperature_K: float
    target_heat_dose_Ks: float
    thermal_reach_cells: float
    hot_cell_time: float
    smoke_index: float
    soot_index: float
    efficiency: float
    stability_score: float
    precision_score: float
    result: str


Grid = List[List[FireCell]]


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def clamp_float(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def clamp_int(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def distance(ax: int, ay: int, bx: int, by: int) -> float:
    return sqrt((ax - bx) ** 2 + (ay - by) ** 2)


def total_field(grid: Grid, attr: str) -> float:
    return sum(getattr(c, attr) for row in grid for c in row)


def average_temperature(grid: Grid) -> float:
    n = len(grid) * len(grid[0])
    return sum(c.temperature_K for row in grid for c in row) / max(1, n)


def peak_temperature(grid: Grid) -> float:
    return max(c.temperature_K for row in grid for c in row)


def make_grid(config: SimConfig, params: FireParams) -> Grid:
    grid = [
        [
            FireCell(
                temperature_K=config.ambient_temperature_K,
                fuel=0.0,
                oxygen=0.21,
            )
            for _ in range(config.width)
        ]
        for _ in range(config.height)
    ]

    # Target: thermally sensitive, lightly burnable region.
    tx = clamp_int(params.target_x, 0, config.width - 1)
    ty = clamp_int(params.target_y, 0, config.height - 1)
    for yy in range(ty - 1, ty + 2):
        for xx in range(tx - 1, tx + 2):
            if 0 <= xx < config.width and 0 <= yy < config.height:
                grid[yy][xx].fuel = max(grid[yy][xx].fuel, 0.75)

    # Environmental fuel patches for obstacle round. Kept modest so they do not
    # blow up the entire grid.
    if params.style == "obstacle":
        wall_x = config.width // 2
        for y in range(5, config.height - 4):
            if y not in (9, 10, 11, 12):
                grid[y][wall_x].fuel = 1.15
        for x in range(42, 58):
            grid[15][x].fuel = 0.95
    else:
        for x in range(43, 56):
            grid[16][x].fuel = 0.35
        for x in range(45, 51):
            grid[9][x].fuel = 0.25

    return grid


# -----------------------------------------------------------------------------
# Physics model
# -----------------------------------------------------------------------------

def temperature_factor(temperature_K: float, params: FireParams) -> float:
    if temperature_K < params.extinction_temperature_K:
        return 0.0
    if temperature_K < params.ignition_temperature_K:
        span = params.ignition_temperature_K - params.extinction_temperature_K
        # v3 allows weak pilot chemistry in the pre-ignition band. This lets
        # blue/stream/obstacle cross into active flame without making every
        # style explode into full-grid saturation.
        return 0.18 * (temperature_K - params.extinction_temperature_K) / max(1.0, span)

    excess = temperature_K - params.ignition_temperature_K
    return clamp_float(0.18 + 0.82 * (1.0 - exp(-excess / max(1.0, params.activation_temperature_K))), 0.0, 1.0)


def combustion_rate(cell: FireCell, params: FireParams) -> float:
    tf = temperature_factor(cell.temperature_K, params)
    if tf <= 0.0 or cell.fuel <= 1e-7 or cell.oxygen <= 1e-7:
        return 0.0

    oxygen_limiting = cell.oxygen / (cell.oxygen + 0.18)
    fuel_limiting = cell.fuel / (cell.fuel + 0.12)
    return params.reaction_strength * oxygen_limiting * fuel_limiting * tf


def heat_loss_J(cell: FireCell, params: FireParams, config: SimConfig) -> float:
    delta_T = max(0.0, cell.temperature_K - config.ambient_temperature_K)
    convective_W = params.convective_loss_coeff * delta_T
    radiative_W = params.radiative_loss_coeff * max(
        0.0, cell.temperature_K ** 4 - config.ambient_temperature_K ** 4
    )
    mixing_W = params.mixing_loss_coeff * delta_T * (abs(cell.vx) + abs(cell.vy))
    return (convective_W + radiative_W + mixing_W) * config.dt_s


def diffuse_temperature(grid: Grid, x: int, y: int, config: SimConfig) -> float:
    center_excess = grid[y][x].temperature_K - config.ambient_temperature_K
    h = len(grid)
    w = len(grid[0])
    neighbor_sum = 0.0
    count = 0

    for dy, dx in ((0, -1), (0, 1), (-1, 0), (1, 0)):
        xx = x + dx
        yy = y + dy
        if 0 <= xx < w and 0 <= yy < h:
            neighbor_sum += grid[yy][xx].temperature_K - config.ambient_temperature_K
            count += 1
        else:
            neighbor_sum += 0.0
            count += 1

    return config.thermal_diffusivity * ((neighbor_sum / max(1, count)) - center_excess)


def diffuse_scalar(grid: Grid, x: int, y: int, attr: str, config: SimConfig) -> float:
    center = getattr(grid[y][x], attr)
    h = len(grid)
    w = len(grid[0])
    neighbor_sum = 0.0
    count = 0

    for dy, dx in ((0, -1), (0, 1), (-1, 0), (1, 0)):
        xx = x + dx
        yy = y + dy
        if 0 <= xx < w and 0 <= yy < h:
            neighbor_sum += getattr(grid[yy][xx], attr)
            count += 1

    return config.scalar_diffusivity * ((neighbor_sum / max(1, count)) - center)


def add_to_cell(
    grid: Grid,
    x: int,
    y: int,
    heat_J: float,
    fuel: float,
    oxygen: float,
    vx: float,
    vy: float,
    config: SimConfig,
) -> None:
    h = len(grid)
    w = len(grid[0])
    if not (0 <= x < w and 0 <= y < h):
        return

    c = grid[y][x]
    c.temperature_K += heat_J / max(1e-6, config.cell_heat_capacity_J_per_K)
    c.temperature_K = clamp_float(c.temperature_K, config.ambient_temperature_K, config.max_temperature_K)
    c.fuel = clamp_float(c.fuel + fuel, 0.0, 3.0)
    c.oxygen = clamp_float(c.oxygen + oxygen, 0.0, 2.0)
    c.vx += vx
    c.vy += vy


def apply_pilot_kernel(grid: Grid, params: FireParams, config: SimConfig, step: int) -> None:
    """Add a small local ignition kernel near the bender for flame-capable modes.

    v2 stayed too cold because injection heat was spread along the path before any
    cell crossed ignition. The pilot kernel is short-lived and local, so it starts
    chemistry without reheating the whole domain.
    """
    if step > min(28, params.duration_steps):
        return

    style_scale = {
        "jab": 0.42,
        "stream": 0.90,
        "wave": 0.68,
        "shield": 0.34,
        "whip": 0.72,
        "sparks": 0.18,
        "blue": 1.28,
        "obstacle": 0.95,
    }.get(params.style, 0.55)

    if style_scale <= 0.0:
        return

    bx = clamp_int(params.bender_x, 1, config.width - 2)
    by = clamp_int(params.bender_y, 1, config.height - 2)
    angle = radians(params.direction_deg)
    dx = cos(angle)
    dy = sin(angle)

    # Center just in front of the bender; decay over the first pilot frames.
    cx = int(round(bx + dx * 2))
    cy = int(round(by + dy * 2))
    time_decay = max(0.0, 1.0 - step / max(1.0, min(28, params.duration_steps)))
    heat_J = params.injection_power_W * config.dt_s * (1.8 + 2.4 * time_decay) * style_scale
    fuel = params.fuel_injection_rate * config.dt_s * 1.25 * style_scale
    oxygen = params.oxygen_injection_rate * config.dt_s * 1.35 * style_scale

    for yy in range(cy - 1, cy + 2):
        for xx in range(cx - 1, cx + 2):
            weight = 1.0 if (xx == cx and yy == cy) else 0.45
            add_to_cell(
                grid,
                xx,
                yy,
                heat_J * weight,
                fuel * weight,
                oxygen * weight,
                dx * params.momentum * 0.25,
                dy * params.momentum * 0.25,
                config,
            )


def inject_control(grid: Grid, params: FireParams, config: SimConfig, step: int, rng: random.Random) -> None:
    apply_pilot_kernel(grid, params, config, step)

    if step > params.duration_steps and params.style not in ("shield", "whip"):
        return

    bx = clamp_int(params.bender_x, 1, config.width - 2)
    by = clamp_int(params.bender_y, 1, config.height - 2)

    def ray(angle_deg: float, length: int, width_base: float, power_scale: float, fuel_scale: float, o2_scale: float, mom_scale: float) -> None:
        angle = radians(angle_deg)
        dx = cos(angle)
        dy = sin(angle)

        for r in range(1, length + 1):
            decay = 1.0 / (1.0 + 0.10 * r)
            width = int(max(0, round(width_base + 0.025 * r)))
            for off in range(-width, width + 1):
                side_weight = 1.0 / (1.0 + abs(off) * 1.8)
                jitter = rng.uniform(-params.turbulence, params.turbulence) * r * 0.4
                x = int(round(bx + dx * r - dy * (off + jitter)))
                y = int(round(by + dy * r + dx * (off + jitter)))

                heat = params.injection_power_W * config.dt_s * power_scale * decay * side_weight
                fuel = params.fuel_injection_rate * config.dt_s * fuel_scale * decay * side_weight
                oxygen = params.oxygen_injection_rate * config.dt_s * o2_scale * decay * side_weight

                add_to_cell(
                    grid,
                    x,
                    y,
                    heat,
                    fuel,
                    oxygen,
                    dx * params.momentum * mom_scale * decay,
                    dy * params.momentum * mom_scale * decay,
                    config,
                )

    if params.style == "jab":
        # Brief impulse; should be hot near source but not sustained.
        length = min(16, 4 + step // 5)
        ray(params.direction_deg, length, 0.0, 0.95, 0.65, 0.45, 1.35)

    elif params.style == "stream":
        ray(params.direction_deg, 27, 0.35, 0.80, 0.82, 0.75, 1.10)

    elif params.style == "wave":
        for off_deg in range(-int(params.cone_half_angle_deg), int(params.cone_half_angle_deg) + 1, 10):
            ray(params.direction_deg + off_deg, 19, 0.25, 0.38, 0.55, 0.50, 0.70)

    elif params.style == "shield":
        radius = 4 + int(1.2 * sin(step * 0.15))
        for a_deg in range(0, 360, 18):
            a = radians(a_deg)
            x = int(round(bx + cos(a) * radius))
            y = int(round(by + sin(a) * radius))
            heat = params.injection_power_W * config.dt_s * 0.36
            fuel = params.fuel_injection_rate * config.dt_s * 0.25
            oxygen = params.oxygen_injection_rate * config.dt_s * 0.30
            add_to_cell(grid, x, y, heat, fuel, oxygen, cos(a) * 0.18, sin(a) * 0.18, config)

    elif params.style == "whip":
        phase = step * 0.17
        for r in range(1, 27):
            curve = sin(r * 0.38 + phase) * 2.7
            x = int(round(bx + r))
            y = int(round(by + curve))
            decay = 1.0 / (1.0 + 0.09 * r)
            add_to_cell(
                grid,
                x,
                y,
                params.injection_power_W * config.dt_s * 0.58 * decay,
                params.fuel_injection_rate * config.dt_s * 0.55 * decay,
                params.oxygen_injection_rate * config.dt_s * 0.50 * decay,
                0.72 * decay,
                cos(r * 0.38 + phase) * 0.45 * decay,
                config,
            )

    elif params.style == "sparks":
        for _ in range(9):
            a = radians(params.direction_deg + rng.uniform(-48, 48))
            r = rng.randint(2, 22)
            x = int(round(bx + cos(a) * r))
            y = int(round(by + sin(a) * r))
            add_to_cell(
                grid,
                x,
                y,
                params.injection_power_W * config.dt_s * rng.uniform(0.11, 0.28),
                params.fuel_injection_rate * config.dt_s * rng.uniform(0.02, 0.07),
                params.oxygen_injection_rate * config.dt_s * rng.uniform(0.04, 0.12),
                cos(a) * params.momentum * 0.35,
                sin(a) * params.momentum * 0.35,
                config,
            )

    elif params.style == "blue":
        # Narrow oxygen-rich jet.
        ray(params.direction_deg, 36, 0.0, 1.05, 0.48, 1.35, 1.55)

    elif params.style == "obstacle":
        bend = sin(step * 0.11) * 0.22
        for r in range(1, 31):
            angle = params.direction_deg + bend * r
            dx = cos(radians(angle))
            dy = sin(radians(angle))
            decay = 1.0 / (1.0 + 0.085 * r)
            x = int(round(bx + r))
            y = int(round(by + dy * r))
            add_to_cell(
                grid,
                x,
                y,
                params.injection_power_W * config.dt_s * 0.72 * decay,
                params.fuel_injection_rate * config.dt_s * 0.74 * decay,
                params.oxygen_injection_rate * config.dt_s * 0.78 * decay,
                dx * params.momentum * decay,
                dy * params.momentum * decay,
                config,
            )


def update_physics(grid: Grid, params: FireParams, config: SimConfig, rng: random.Random) -> Dict[str, float]:
    h = len(grid)
    w = len(grid[0])

    # Combustion and losses happen in-place.
    energy_J = 0.0
    loss_J = 0.0
    fuel_burned_total = 0.0
    oxygen_used_total = 0.0
    peak_rate = 0.0
    hot_cells = 0

    for y in range(h):
        for x in range(w):
            c = grid[y][x]

            rate = combustion_rate(c, params)
            requested = rate * config.dt_s
            possible = min(c.fuel, c.oxygen / max(1e-6, params.oxygen_per_fuel))
            burned = min(requested, possible)
            oxygen_used = burned * params.oxygen_per_fuel
            heat_generated = burned * params.heat_of_combustion_J_per_fuel

            c.fuel -= burned
            c.oxygen -= oxygen_used
            c.temperature_K += heat_generated / config.cell_heat_capacity_J_per_K
            c.combustion_rate = rate

            equivalence = c.fuel / max(1e-6, c.oxygen)
            rich_penalty = clamp_float((equivalence - 0.5) * 0.35, 0.0, 2.0)
            cool_penalty = clamp_float((params.ignition_temperature_K + 250.0 - c.temperature_K) / 850.0, 0.0, 1.0)
            c.smoke += burned * params.smoke_yield * (1.0 + rich_penalty + cool_penalty)
            c.soot += burned * params.soot_yield * (1.0 + rich_penalty + cool_penalty)

            loss = heat_loss_J(c, params, config)
            c.temperature_K -= loss / config.cell_heat_capacity_J_per_K
            c.temperature_K = clamp_float(c.temperature_K, config.ambient_temperature_K, config.max_temperature_K)

            # Ambient oxygen slowly replenishes cells that are not fully oxygen-rich.
            c.oxygen += config.oxygen_recovery_rate * (0.21 - min(c.oxygen, 0.21)) * config.dt_s
            c.oxygen = clamp_float(c.oxygen, 0.0, 2.0)

            energy_J += heat_generated
            loss_J += loss
            fuel_burned_total += burned
            oxygen_used_total += oxygen_used
            peak_rate = max(peak_rate, rate)

            if c.temperature_K >= config.hot_threshold_K:
                hot_cells += 1

    # Conservative update of excess temperature and scalars. This is the main v2
    # fix: only excess above ambient is transported, so ambient is not doubled.
    next_grid = [
        [
            FireCell(
                temperature_K=config.ambient_temperature_K,
                fuel=0.0,
                oxygen=0.21,
                smoke=0.0,
                soot=0.0,
                vx=0.0,
                vy=0.0,
                combustion_rate=0.0,
            )
            for _ in range(w)
        ]
        for _ in range(h)
    ]

    def add_fraction(dest: FireCell, source: FireCell, frac: float, dT: float, df: float, do: float, ds: float, dsoot: float) -> None:
        excess = max(0.0, source.temperature_K - config.ambient_temperature_K + dT)
        dest.temperature_K += excess * frac
        dest.fuel += max(0.0, source.fuel + df) * frac
        dest.oxygen += max(0.0, source.oxygen + do) * frac
        dest.smoke += max(0.0, source.smoke + ds) * frac
        dest.soot += max(0.0, source.soot + dsoot) * frac
        dest.vx += source.vx * config.velocity_decay * frac
        dest.vy += source.vy * config.velocity_decay * frac
        dest.combustion_rate += source.combustion_rate * frac

    for y in range(h):
        for x in range(w):
            c = grid[y][x]
            dT = diffuse_temperature(grid, x, y, config)
            df = diffuse_scalar(grid, x, y, "fuel", config)
            do = diffuse_scalar(grid, x, y, "oxygen", config)
            ds = diffuse_scalar(grid, x, y, "smoke", config)
            dso = diffuse_scalar(grid, x, y, "soot", config)

            buoyancy = -0.020 * clamp_float((c.temperature_K - config.ambient_temperature_K) / 1000.0, 0.0, 1.0)
            adv_x = c.vx + rng.uniform(-params.turbulence, params.turbulence) * 0.04
            adv_y = c.vy + buoyancy + rng.uniform(-params.turbulence, params.turbulence) * 0.04

            mx = clamp_int(int(round(x + adv_x)), 0, w - 1)
            my = clamp_int(int(round(y + adv_y)), 0, h - 1)

            # Most heat stays local; a small fraction advects. This prevents
            # the whole grid from saturating instantly.
            add_fraction(next_grid[y][x], c, 0.82, dT, df, do, ds, dso)
            add_fraction(next_grid[my][mx], c, 0.18, dT, df, do, ds, dso)

    for y in range(h):
        for x in range(w):
            c = next_grid[y][x]
            c.temperature_K = config.ambient_temperature_K + max(0.0, c.temperature_K - config.ambient_temperature_K)
            c.temperature_K = clamp_float(c.temperature_K, config.ambient_temperature_K, config.max_temperature_K)
            c.fuel = clamp_float(c.fuel, 0.0, 3.0)
            c.oxygen = clamp_float(c.oxygen, 0.0, 2.0)
            c.smoke = max(0.0, c.smoke * config.smoke_decay)
            c.soot = max(0.0, c.soot)
            c.vx *= config.velocity_decay
            c.vy *= config.velocity_decay
            grid[y][x] = c

    return {
        "energy_J": energy_J,
        "loss_J": loss_J,
        "fuel_burned": fuel_burned_total,
        "oxygen_used": oxygen_used_total,
        "peak_rate": peak_rate,
        "hot_cells": float(hot_cells),
    }


# -----------------------------------------------------------------------------
# Measurement and classification
# -----------------------------------------------------------------------------

def target_metrics(grid: Grid, params: FireParams, config: SimConfig) -> Dict[str, float]:
    tx = clamp_int(params.target_x, 0, config.width - 1)
    ty = clamp_int(params.target_y, 0, config.height - 1)
    temps = []
    fuels = []
    for yy in range(ty - 1, ty + 2):
        for xx in range(tx - 1, tx + 2):
            if 0 <= xx < config.width and 0 <= yy < config.height:
                temps.append(grid[yy][xx].temperature_K)
                fuels.append(grid[yy][xx].fuel)
    return {
        "peak_temperature_K": max(temps) if temps else config.ambient_temperature_K,
        "average_temperature_K": sum(temps) / max(1, len(temps)),
        "fuel": sum(fuels),
    }


def thermal_reach(grid: Grid, params: FireParams, threshold_K: float) -> float:
    reach = 0.0
    for y, row in enumerate(grid):
        for x, c in enumerate(row):
            if c.temperature_K >= threshold_K:
                reach = max(reach, distance(params.bender_x, params.bender_y, x, y))
    return reach


def count_cells_above(grid: Grid, threshold_K: float) -> int:
    return sum(1 for row in grid for c in row if c.temperature_K >= threshold_K)


def classify_result(summary: RoundSummary, config: SimConfig) -> str:
    if summary.target_peak_temperature_K >= config.ignition_threshold_K or summary.target_heat_dose_Ks >= 950:
        return "target ignition"
    if summary.target_peak_temperature_K >= config.scorch_threshold_K or summary.target_heat_dose_Ks >= 300:
        return "target scorching"
    if summary.thermal_reach_cells >= 32 and summary.peak_temperature_K > 850:
        return "long-range thermal contact"
    if summary.stability_score >= 30 and summary.hot_cell_time > 35:
        return "stable controlled plume"
    if summary.smoke_index > 15 and summary.smoke_index > summary.soot_index * 2:
        return "smoke dominated plume"
    if summary.peak_temperature_K < 720:
        return "sub-ignition heating"
    return "localized controlled burn"


def print_frame(step: int, config: SimConfig, params: FireParams, grid: Grid, totals: Dict[str, float], target_dose: float) -> None:
    t = step * config.dt_s
    target = target_metrics(grid, params, config)
    reach = thermal_reach(grid, params, config.hot_threshold_K)
    print(
        f"t={t:5.2f}s | "
        f"Tmax={peak_temperature(grid):7.1f}K | Tavg={average_temperature(grid):6.1f}K | "
        f"hot={count_cells_above(grid, config.hot_threshold_K):4d} | "
        f"fuel={total_field(grid, 'fuel'):6.2f} | O2={total_field(grid, 'oxygen'):6.2f} | "
        f"rate={totals['current_rate']:6.3f} | "
        f"Erel={totals['energy_J']:8.1f}J | Eloss={totals['loss_J']:8.1f}J | "
        f"reach={reach:5.1f} | targetT={target['peak_temperature_K']:7.1f}K | "
        f"dose={target_dose:7.1f} | smoke={total_field(grid, 'smoke'):6.2f} | soot={total_field(grid, 'soot'):6.2f}"
    )


def print_round_summary(summary: RoundSummary) -> None:
    print("\nRound result:")
    print(f"  style/control mode: {summary.params.style}")
    print(f"  peak temperature: {summary.peak_temperature_K:.1f} K")
    print(f"  average temperature: {summary.average_temperature_K:.1f} K")
    print(f"  final average temperature: {summary.final_average_temperature_K:.1f} K")
    print(f"  total energy released: {summary.total_energy_released_J:.2f} J")
    print(f"  total heat lost: {summary.total_heat_lost_J:.2f} J")
    print(f"  fuel burned: {summary.fuel_burned:.4f} normalized units")
    print(f"  oxygen consumed: {summary.oxygen_consumed:.4f} normalized units")
    print(f"  peak combustion rate: {summary.peak_combustion_rate:.4f} units/s")
    print(f"  average combustion rate: {summary.average_combustion_rate:.4f} units/s")
    print(f"  target peak temperature: {summary.target_peak_temperature_K:.1f} K")
    print(f"  target average temperature: {summary.target_average_temperature_K:.1f} K")
    print(f"  target heat dose: {summary.target_heat_dose_Ks:.1f} K*s")
    print(f"  thermal reach: {summary.thermal_reach_cells:.2f} cells")
    print(f"  hot-cell time: {summary.hot_cell_time:.2f} cell*s")
    print(f"  smoke index: {summary.smoke_index:.2f}")
    print(f"  soot index: {summary.soot_index:.2f}")
    print(f"  heat efficiency: {summary.efficiency:.3f}")
    print(f"  stability score: {summary.stability_score:.2f}")
    print(f"  precision score: {summary.precision_score:.2f}")
    print(f"  scientific result: {summary.result}")


# -----------------------------------------------------------------------------
# AI controller
# -----------------------------------------------------------------------------

class ScientificFireBendingAIController:
    def __init__(self, rng: random.Random, config: SimConfig):
        self.rng = rng
        self.config = config
        self.goals = [
            "short thermal impulse",
            "continuous combustion jet",
            "wide turbulent thermal wave",
            "radial defensive heat shell",
            "oscillating flame filament",
            "distributed ember field",
            "oxygen-rich blue precision jet",
            "obstacle-coupled ignition jet",
        ]

    def choose_next(self, previous: RoundSummary | None, current: FireParams, round_number: int) -> FireParams:
        goal = self.goals[(round_number - 1) % len(self.goals)]
        p = replace(current)

        p.bender_x = 8
        p.bender_y = self.config.height // 2
        p.target_x = self.config.width - 10
        p.target_y = self.config.height // 2

        if goal == "short thermal impulse":
            p.style = "jab"
            p.injection_power_W = self.rng.uniform(430, 560)
            p.fuel_injection_rate = self.rng.uniform(0.42, 0.66)
            p.oxygen_injection_rate = self.rng.uniform(0.20, 0.34)
            p.momentum = self.rng.uniform(1.1, 1.5)
            p.duration_steps = self.rng.randint(28, 48)
            p.direction_deg = self.rng.uniform(-3, 3)
            p.cone_half_angle_deg = self.rng.uniform(2, 5)
            p.turbulence = self.rng.uniform(0.02, 0.06)
            p.reaction_strength = self.rng.uniform(2.0, 2.8)
            p.convective_loss_coeff = self.rng.uniform(0.48, 0.68)
            p.soot_yield = 0.018

        elif goal == "continuous combustion jet":
            p.style = "stream"
            p.injection_power_W = self.rng.uniform(520, 720)
            p.fuel_injection_rate = self.rng.uniform(0.30, 0.48)
            p.oxygen_injection_rate = self.rng.uniform(0.28, 0.48)
            p.momentum = self.rng.uniform(0.9, 1.25)
            p.duration_steps = self.rng.randint(110, 160)
            p.direction_deg = self.rng.uniform(-4, 4)
            p.cone_half_angle_deg = self.rng.uniform(6, 12)
            p.turbulence = self.rng.uniform(0.04, 0.10)
            p.reaction_strength = self.rng.uniform(2.3, 3.2)
            p.convective_loss_coeff = self.rng.uniform(0.42, 0.62)
            p.soot_yield = 0.020

        elif goal == "wide turbulent thermal wave":
            p.style = "wave"
            p.injection_power_W = self.rng.uniform(460, 640)
            p.fuel_injection_rate = self.rng.uniform(0.36, 0.60)
            p.oxygen_injection_rate = self.rng.uniform(0.26, 0.42)
            p.momentum = self.rng.uniform(0.55, 0.90)
            p.duration_steps = self.rng.randint(70, 105)
            p.direction_deg = self.rng.uniform(-4, 4)
            p.cone_half_angle_deg = self.rng.uniform(25, 42)
            p.turbulence = self.rng.uniform(0.10, 0.20)
            p.reaction_strength = self.rng.uniform(1.9, 2.8)
            p.convective_loss_coeff = self.rng.uniform(0.50, 0.72)
            p.soot_yield = self.rng.uniform(0.025, 0.045)

        elif goal == "radial defensive heat shell":
            p.style = "shield"
            p.injection_power_W = self.rng.uniform(330, 470)
            p.fuel_injection_rate = self.rng.uniform(0.18, 0.30)
            p.oxygen_injection_rate = self.rng.uniform(0.20, 0.34)
            p.momentum = self.rng.uniform(0.20, 0.45)
            p.duration_steps = self.config.steps_per_round
            p.direction_deg = 0.0
            p.cone_half_angle_deg = 180.0
            p.turbulence = self.rng.uniform(0.03, 0.07)
            p.reaction_strength = self.rng.uniform(2.0, 2.8)
            p.convective_loss_coeff = self.rng.uniform(0.48, 0.70)
            p.soot_yield = 0.022

        elif goal == "oscillating flame filament":
            p.style = "whip"
            p.injection_power_W = self.rng.uniform(500, 700)
            p.fuel_injection_rate = self.rng.uniform(0.42, 0.66)
            p.oxygen_injection_rate = self.rng.uniform(0.26, 0.42)
            p.momentum = self.rng.uniform(0.75, 1.10)
            p.duration_steps = self.config.steps_per_round
            p.direction_deg = 0.0
            p.cone_half_angle_deg = self.rng.uniform(4, 8)
            p.turbulence = self.rng.uniform(0.05, 0.12)
            p.reaction_strength = self.rng.uniform(2.1, 3.0)
            p.convective_loss_coeff = self.rng.uniform(0.46, 0.68)
            p.soot_yield = 0.025

        elif goal == "distributed ember field":
            p.style = "sparks"
            p.injection_power_W = self.rng.uniform(230, 360)
            p.fuel_injection_rate = self.rng.uniform(0.08, 0.18)
            p.oxygen_injection_rate = self.rng.uniform(0.10, 0.22)
            p.momentum = self.rng.uniform(1.0, 1.8)
            p.duration_steps = self.rng.randint(85, 130)
            p.direction_deg = self.rng.uniform(-5, 5)
            p.cone_half_angle_deg = self.rng.uniform(35, 55)
            p.turbulence = self.rng.uniform(0.16, 0.30)
            p.reaction_strength = self.rng.uniform(1.0, 1.8)
            p.convective_loss_coeff = self.rng.uniform(0.72, 1.05)
            p.soot_yield = 0.020

        elif goal == "oxygen-rich blue precision jet":
            p.style = "blue"
            p.injection_power_W = self.rng.uniform(680, 860)
            p.fuel_injection_rate = self.rng.uniform(0.42, 0.66)
            p.oxygen_injection_rate = self.rng.uniform(0.55, 0.85)
            p.momentum = self.rng.uniform(1.35, 1.85)
            p.duration_steps = self.rng.randint(90, 135)
            p.direction_deg = self.rng.uniform(-1.5, 1.5)
            p.cone_half_angle_deg = self.rng.uniform(1.0, 3.0)
            p.turbulence = self.rng.uniform(0.005, 0.025)
            p.reaction_strength = self.rng.uniform(3.0, 4.2)
            p.convective_loss_coeff = self.rng.uniform(0.34, 0.52)
            p.soot_yield = self.rng.uniform(0.004, 0.010)

        elif goal == "obstacle-coupled ignition jet":
            p.style = "obstacle"
            p.injection_power_W = self.rng.uniform(560, 760)
            p.fuel_injection_rate = self.rng.uniform(0.42, 0.66)
            p.oxygen_injection_rate = self.rng.uniform(0.34, 0.58)
            p.momentum = self.rng.uniform(0.85, 1.25)
            p.duration_steps = self.rng.randint(115, 165)
            p.direction_deg = self.rng.uniform(-3, 3)
            p.cone_half_angle_deg = self.rng.uniform(5, 10)
            p.turbulence = self.rng.uniform(0.06, 0.13)
            p.reaction_strength = self.rng.uniform(2.2, 3.2)
            p.convective_loss_coeff = self.rng.uniform(0.42, 0.62)
            p.soot_yield = self.rng.uniform(0.010, 0.022)

        # Reactive tuning.
        if previous is not None:
            if previous.target_peak_temperature_K < 480 and p.style not in ("shield", "sparks"):
                p.momentum *= 1.06
                p.injection_power_W *= 1.06
                p.direction_deg *= 0.55
            if previous.target_peak_temperature_K > 1100:
                p.injection_power_W *= 0.93
                p.convective_loss_coeff *= 1.05
            if previous.smoke_index > 8.0:
                p.oxygen_injection_rate *= 1.12
                p.fuel_injection_rate *= 0.94
                p.soot_yield *= 0.90

        p.injection_power_W = clamp_float(p.injection_power_W, 50, 800)
        p.fuel_injection_rate = clamp_float(p.fuel_injection_rate, 0.01, 1.0)
        p.oxygen_injection_rate = clamp_float(p.oxygen_injection_rate, 0.01, 1.0)
        p.momentum = clamp_float(p.momentum, 0.0, 2.5)
        p.duration_steps = clamp_int(p.duration_steps, 1, self.config.steps_per_round)
        p.direction_deg = clamp_float(p.direction_deg, -25.0, 25.0)
        p.cone_half_angle_deg = clamp_float(p.cone_half_angle_deg, 0.0, 180.0)
        p.turbulence = clamp_float(p.turbulence, 0.0, 0.5)
        p.reaction_strength = clamp_float(p.reaction_strength, 0.1, 5.0)
        p.convective_loss_coeff = clamp_float(p.convective_loss_coeff, 0.1, 2.0)
        p.mixing_loss_coeff = clamp_float(p.mixing_loss_coeff, 0.0, 0.2)
        p.soot_yield = clamp_float(p.soot_yield, 0.0, 0.1)
        p.smoke_yield = clamp_float(p.smoke_yield, 0.0, 0.2)

        print(f"\nAI controller selected next scientific goal: {goal}")
        return p


# -----------------------------------------------------------------------------
# Round loop
# -----------------------------------------------------------------------------

def run_round(round_number: int, params: FireParams, config: SimConfig, rng: random.Random) -> RoundSummary:
    grid = make_grid(config, params)

    totals = {
        "energy_J": 0.0,
        "loss_J": 0.0,
        "fuel_burned": 0.0,
        "oxygen_used": 0.0,
        "peak_rate": 0.0,
        "rate_sum": 0.0,
        "rate_samples": 0.0,
        "hot_cell_time": 0.0,
        "current_rate": 0.0,
    }

    peak_T = config.ambient_temperature_K
    avg_T_integral = 0.0
    target_peak_T = config.ambient_temperature_K
    target_avg_T_final = config.ambient_temperature_K
    target_dose = 0.0

    print("\n" + "=" * 150)
    print(f"ROUND {round_number}: {params.style.upper()} SCIENTIFIC CONTROL MODE v3")
    print("=" * 150)
    print(
        f"power={params.injection_power_W:.1f} W, fuel_rate={params.fuel_injection_rate:.3f}/s, "
        f"O2_rate={params.oxygen_injection_rate:.3f}/s, momentum={params.momentum:.2f}, duration={params.duration_steps} steps"
    )
    print(
        f"reaction={params.reaction_strength:.2f}, ignition={params.ignition_temperature_K:.1f} K, "
        f"extinction={params.extinction_temperature_K:.1f} K, direction={params.direction_deg:.1f} deg, "
        f"cone={params.cone_half_angle_deg:.1f} deg, turbulence={params.turbulence:.3f}"
    )
    print(
        f"loss: conv={params.convective_loss_coeff:.3f}, rad={params.radiative_loss_coeff:.2e}, "
        f"mix={params.mixing_loss_coeff:.3f}, soot_yield={params.soot_yield:.3f}"
    )
    print("-" * 150)
    print(
        "time     | peak/avg/hot cells       | reactants      | combustion/energy balance          "
        "| reach/target exposure          | byproducts"
    )
    print("-" * 150)

    for step in range(config.steps_per_round + 1):
        inject_control(grid, params, config, step, rng)
        diag = update_physics(grid, params, config, rng)

        totals["energy_J"] += diag["energy_J"]
        totals["loss_J"] += diag["loss_J"]
        totals["fuel_burned"] += diag["fuel_burned"]
        totals["oxygen_used"] += diag["oxygen_used"]
        totals["peak_rate"] = max(totals["peak_rate"], diag["peak_rate"])
        totals["rate_sum"] += diag["peak_rate"]
        totals["rate_samples"] += 1.0
        totals["hot_cell_time"] += diag["hot_cells"] * config.dt_s
        totals["current_rate"] = diag["peak_rate"]

        current_peak = peak_temperature(grid)
        current_avg = average_temperature(grid)
        peak_T = max(peak_T, current_peak)
        avg_T_integral += current_avg * config.dt_s

        tm = target_metrics(grid, params, config)
        target_peak_T = max(target_peak_T, tm["peak_temperature_K"])
        target_avg_T_final = tm["average_temperature_K"]
        target_dose += max(0.0, tm["average_temperature_K"] - config.ambient_temperature_K) * config.dt_s

        if step % config.print_every_steps == 0 or step == config.steps_per_round:
            print_frame(step, config, params, grid, totals, target_dose)

    total_time = (config.steps_per_round + 1) * config.dt_s
    avg_T = avg_T_integral / max(config.dt_s, total_time)
    final_avg_T = average_temperature(grid)
    reach = thermal_reach(grid, params, config.hot_threshold_K)
    smoke = total_field(grid, "smoke")
    soot = total_field(grid, "soot")
    efficiency = totals["energy_J"] / max(1e-6, totals["energy_J"] + totals["loss_J"])

    # Localized stability: rewards persistent heat, but penalizes global blow-up.
    hot_cells_final = count_cells_above(grid, config.hot_threshold_K)
    domain_cells = config.width * config.height
    localization_penalty = max(0.0, hot_cells_final / domain_cells - 0.20) * 130.0
    stability = (
        totals["hot_cell_time"] * 0.35
        + efficiency * 32.0
        + min(reach, 40.0) * 0.60
        - smoke * 0.10
        - soot * 0.25
        - localization_penalty
    )

    # Precision rewards target heating with fewer hot cells and lower smoke.
    precision = (
        max(0.0, target_peak_T - config.ambient_temperature_K) * 0.08
        + target_dose * 0.020
        - smoke * 0.15
        - hot_cells_final * 0.05
    )

    summary = RoundSummary(
        round_number=round_number,
        params=params,
        peak_temperature_K=peak_T,
        average_temperature_K=avg_T,
        final_average_temperature_K=final_avg_T,
        total_energy_released_J=totals["energy_J"],
        total_heat_lost_J=totals["loss_J"],
        fuel_burned=totals["fuel_burned"],
        oxygen_consumed=totals["oxygen_used"],
        peak_combustion_rate=totals["peak_rate"],
        average_combustion_rate=totals["rate_sum"] / max(1.0, totals["rate_samples"]),
        target_peak_temperature_K=target_peak_T,
        target_average_temperature_K=target_avg_T_final,
        target_heat_dose_Ks=target_dose,
        thermal_reach_cells=reach,
        hot_cell_time=totals["hot_cell_time"],
        smoke_index=smoke,
        soot_index=soot,
        efficiency=efficiency,
        stability_score=stability,
        precision_score=precision,
        result="",
    )
    summary.result = classify_result(summary, config)

    print("-" * 150)
    print_round_summary(summary)
    return summary


def print_final_summary(summaries: List[RoundSummary]) -> None:
    print("\n" + "=" * 176)
    print("FINAL SCIENTIFIC FIRE BENDING v3 COMPARISON")
    print("=" * 176)
    header = (
        f"{'Round':>5} | {'Mode':>10} | {'Peak K':>8} | {'Avg K':>8} | {'Target K':>8} | "
        f"{'Dose':>8} | {'Erel J':>9} | {'Loss J':>9} | {'Fuel':>7} | {'O2':>7} | "
        f"{'Rate':>7} | {'Reach':>6} | {'Eff':>6} | {'Stab':>7} | {'Prec':>7} | Result"
    )
    print(header)
    print("-" * len(header))

    for s in summaries:
        print(
            f"{s.round_number:5d} | {s.params.style:>10} | "
            f"{s.peak_temperature_K:8.1f} | {s.average_temperature_K:8.1f} | "
            f"{s.target_peak_temperature_K:8.1f} | {s.target_heat_dose_Ks:8.1f} | "
            f"{s.total_energy_released_J:9.1f} | {s.total_heat_lost_J:9.1f} | "
            f"{s.fuel_burned:7.3f} | {s.oxygen_consumed:7.3f} | "
            f"{s.peak_combustion_rate:7.3f} | {s.thermal_reach_cells:6.2f} | "
            f"{s.efficiency:6.3f} | {s.stability_score:7.2f} | {s.precision_score:7.2f} | {s.result}"
        )

    best_stability = max(summaries, key=lambda s: s.stability_score)
    hottest = max(summaries, key=lambda s: s.peak_temperature_K)
    best_target = max(summaries, key=lambda s: s.target_heat_dose_Ks)
    cleanest = min(summaries, key=lambda s: s.smoke_index + s.soot_index)
    most_efficient = max(summaries, key=lambda s: s.efficiency)
    best_precision = max(summaries, key=lambda s: s.precision_score)

    print("\nBest scientific observations:")
    print(f"  Best stability/control: Round {best_stability.round_number}, {best_stability.params.style}, score {best_stability.stability_score:.2f}")
    print(f"  Hottest controlled mode: Round {hottest.round_number}, {hottest.params.style}, peak {hottest.peak_temperature_K:.1f} K")
    print(f"  Strongest target exposure: Round {best_target.round_number}, {best_target.params.style}, dose {best_target.target_heat_dose_Ks:.1f} K*s")
    print(f"  Cleanest burn: Round {cleanest.round_number}, {cleanest.params.style}, smoke+soot {cleanest.smoke_index + cleanest.soot_index:.2f}")
    print(f"  Highest heat efficiency: Round {most_efficient.round_number}, {most_efficient.params.style}, efficiency {most_efficient.efficiency:.3f}")
    print(f"  Best precision score: Round {best_precision.round_number}, {best_precision.params.style}, precision {best_precision.precision_score:.2f}")


def main() -> None:
    config = SimConfig()
    rng = random.Random(config.seed)
    controller = ScientificFireBendingAIController(rng, config)

    params = FireParams(
        bender_x=8,
        bender_y=config.height // 2,
        target_x=config.width - 10,
        target_y=config.height // 2,
    )

    previous_summary: RoundSummary | None = None
    summaries: List[RoundSummary] = []

    print("Scientific Fire Bending AI Rounds Simulation v3")
    print("No graphics. No external packages.")
    print("Model: calibrated 2D reactive thermal flow with local heat diffusion, advection, combustion, losses, smoke, soot, and target heat dose.")
    print("v3 tuning: calibrated pilot ignition creates mixed outcomes without full-grid thermal saturation.")

    for round_number in range(1, config.rounds + 1):
        params = controller.choose_next(previous_summary, params, round_number)
        summary = run_round(round_number, params, config, rng)
        summaries.append(summary)
        previous_summary = summary

    print_final_summary(summaries)


if __name__ == "__main__":
    main()
