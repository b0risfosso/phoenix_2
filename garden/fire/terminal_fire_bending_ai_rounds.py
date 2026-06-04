#!/usr/bin/env python3
"""
Terminal Fire Bending AI Rounds Simulation
------------------------------------------

A terminal-only Python simulation of stylized fire bending.

Run:
    python terminal_fire_bending_ai_rounds.py

No external packages are required.

The simulation uses a small grid world. Fire is represented by heat, flame,
sparks, smoke, fuel, oxygen, and direction. A fire bender emits controlled
bursts, streams, waves, shields, whips, and blue-fire beams. A rule-based AI
controller changes bending style and environment parameters between rounds.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import cos, sin, radians, sqrt
import random
import time
from typing import Dict, List, Tuple


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

@dataclass
class SimConfig:
    rounds: int = 8
    width: int = 64
    height: int = 22
    steps_per_round: int = 90
    print_every_steps: int = 6
    sleep_between_frames: float = 0.0
    seed: int = 19


@dataclass
class FireParams:
    style: str = "jab"                 # jab, stream, wave, shield, whip, sparks, blue, obstacle
    intensity: float = 9.0             # heat injected per emission cell
    duration_steps: int = 28
    direction_deg: float = 0.0
    spread: float = 0.25
    turbulence: float = 0.12
    spark_rate: float = 0.08
    cooling_rate: float = 0.18
    smoke_threshold: float = 2.2
    ignition_threshold: float = 5.5
    oxygen: float = 1.0
    fuel: float = 1.0
    bender_x: int = 8
    bender_y: int = 11
    target_x: int = 54
    target_y: int = 11


@dataclass
class FireCell:
    heat: float = 0.0
    flame: float = 0.0
    smoke: float = 0.0
    fuel: float = 0.0
    vx: float = 0.0
    vy: float = 0.0


@dataclass
class RoundSummary:
    round_number: int
    params: FireParams
    max_heat: float
    total_heat: float
    flame_cells: int
    smoke_cells: int
    sparks_created: int
    target_hits: int
    fuel_burned: float
    distance_reached: float
    control_score: float
    note: str


Grid = List[List[FireCell]]


def clamp_float(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def clamp_int(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def make_grid(config: SimConfig, params: FireParams) -> Grid:
    grid = [[FireCell() for _ in range(config.width)] for _ in range(config.height)]

    # Environmental fuel patches: dry brush/wood-like cells.
    if params.style == "obstacle":
        for y in range(6, config.height - 5):
            x = config.width // 2
            if y not in (9, 10, 11, 12):
                grid[y][x].fuel = 4.5
        for x in range(42, 58):
            grid[15][x].fuel = 3.5
    else:
        for x in range(38, 58):
            grid[16][x].fuel = 2.0
        for x in range(44, 52):
            grid[9][x].fuel = 1.4

    # A target with fuel-like burnable marker.
    tx = clamp_int(params.target_x, 0, config.width - 1)
    ty = clamp_int(params.target_y, 0, config.height - 1)
    grid[ty][tx].fuel = max(grid[ty][tx].fuel, 6.0)
    return grid


def emit_fire(grid: Grid, params: FireParams, step: int, rng: random.Random) -> int:
    """Inject controlled heat/flame into the grid. Returns sparks created."""
    height = len(grid)
    width = len(grid[0])
    bx = clamp_int(params.bender_x, 1, width - 2)
    by = clamp_int(params.bender_y, 1, height - 2)

    if step > params.duration_steps and params.style not in ("shield", "whip"):
        return 0

    angle = radians(params.direction_deg)
    dx = cos(angle)
    dy = sin(angle)
    sparks_created = 0

    def add_cell(x: int, y: int, heat: float, flame: float, vx: float, vy: float) -> None:
        nonlocal sparks_created
        if 0 <= x < width and 0 <= y < height:
            c = grid[y][x]
            c.heat += heat * params.oxygen
            c.flame = max(c.flame, flame * params.fuel * params.oxygen)
            c.vx += vx
            c.vy += vy
            if rng.random() < params.spark_rate:
                c.heat += heat * 0.25
                sparks_created += 1

    if params.style == "jab":
        # Short direct punch.
        if step <= params.duration_steps:
            reach = 3 + step // 2
            for r in range(1, min(reach, 14)):
                x = int(round(bx + dx * r))
                y = int(round(by + dy * r))
                add_cell(x, y, params.intensity / (1 + 0.07 * r), 1.2, dx * 1.4, dy * 1.4)

    elif params.style == "stream":
        # Continuous flame hose.
        if step <= params.duration_steps:
            for r in range(1, 18):
                sway = rng.uniform(-params.spread, params.spread) * r
                x = int(round(bx + dx * r - dy * sway))
                y = int(round(by + dy * r + dx * sway))
                add_cell(x, y, params.intensity / (1 + 0.04 * r), 1.5, dx * 1.2, dy * 1.2)

    elif params.style == "wave":
        # Expanding fan.
        if step <= params.duration_steps:
            radius = 3 + step // 2
            for offset_deg in range(-35, 36, 10):
                a = radians(params.direction_deg + offset_deg)
                wx, wy = cos(a), sin(a)
                for r in range(max(1, radius - 3), radius + 1):
                    x = int(round(bx + wx * r))
                    y = int(round(by + wy * r))
                    add_cell(x, y, params.intensity * 0.75 / (1 + 0.05 * r), 1.0, wx, wy)

    elif params.style == "shield":
        # Ring of fire around the bender.
        radius = 3 + int(1.5 * sin(step * 0.35))
        for offset_deg in range(0, 360, 15):
            a = radians(offset_deg)
            x = int(round(bx + cos(a) * radius))
            y = int(round(by + sin(a) * radius))
            add_cell(x, y, params.intensity * 0.65, 1.1, cos(a) * 0.3, sin(a) * 0.3)

    elif params.style == "whip":
        # Curved flame path.
        length = 22
        phase = step * 0.22
        for r in range(1, length):
            curve = sin(r * 0.45 + phase) * 3.0
            x = int(round(bx + r * 1.0))
            y = int(round(by + curve))
            add_cell(x, y, params.intensity / (1 + 0.06 * r), 1.2, 1.0, cos(r * 0.45 + phase))

    elif params.style == "sparks":
        # Many small ember projectiles.
        if step <= params.duration_steps:
            for _ in range(8):
                a = radians(params.direction_deg + rng.uniform(-45, 45))
                r = rng.randint(1, 10 + step // 4)
                x = int(round(bx + cos(a) * r))
                y = int(round(by + sin(a) * r))
                add_cell(x, y, params.intensity * rng.uniform(0.35, 0.85), 0.6, cos(a) * 1.8, sin(a) * 1.8)
                sparks_created += 1

    elif params.style == "blue":
        # Narrow, hotter precision beam.
        if step <= params.duration_steps:
            for r in range(1, 28):
                x = int(round(bx + dx * r))
                y = int(round(by + dy * r))
                add_cell(x, y, params.intensity * 1.7 / (1 + 0.025 * r), 2.0, dx * 1.8, dy * 1.8)

    elif params.style == "obstacle":
        # Controlled stream through/around obstacles.
        if step <= params.duration_steps:
            bend = sin(step * 0.18) * 0.8
            for r in range(1, 25):
                local_y = by + dy * r + sin(r * 0.28 + step * 0.1) * bend * r * 0.08
                x = int(round(bx + dx * r))
                y = int(round(local_y))
                add_cell(x, y, params.intensity / (1 + 0.04 * r), 1.3, dx * 1.1, dy * 1.1)

    return sparks_created


def update_grid(grid: Grid, params: FireParams, rng: random.Random) -> Tuple[int, float]:
    """Move heat/flame, burn fuel, cool cells, and produce smoke."""
    height = len(grid)
    width = len(grid[0])
    next_grid = [[FireCell(fuel=grid[y][x].fuel) for x in range(width)] for y in range(height)]
    target_hits = 0
    fuel_burned = 0.0

    for y in range(height):
        for x in range(width):
            c = grid[y][x]
            heat = c.heat
            flame = c.flame
            smoke = c.smoke

            # Burn local fuel if hot enough.
            if c.fuel > 0 and heat > params.ignition_threshold:
                burn = min(c.fuel, 0.18 * params.oxygen * max(0.2, flame + heat * 0.05))
                c.fuel -= burn
                heat += burn * 2.5
                flame += burn * 0.8
                smoke += burn * 0.35
                fuel_burned += burn

            # Target hit detection.
            if abs(x - params.target_x) <= 1 and abs(y - params.target_y) <= 1 and heat > params.ignition_threshold:
                target_hits += 1

            # Directional advection plus hot air rising.
            vx = c.vx + rng.uniform(-params.turbulence, params.turbulence)
            vy = c.vy - 0.18 + rng.uniform(-params.turbulence, params.turbulence)

            nx = clamp_int(int(round(x + vx)), 0, width - 1)
            ny = clamp_int(int(round(y + vy)), 0, height - 1)

            # Keep some heat in place and move some forward.
            moved_heat = heat * 0.58
            local_heat = heat * 0.22
            spread_heat = heat * 0.20

            next_grid[ny][nx].heat += moved_heat
            next_grid[ny][nx].flame += flame * 0.62
            next_grid[ny][nx].smoke += smoke * 0.55
            next_grid[ny][nx].vx += vx * 0.45
            next_grid[ny][nx].vy += vy * 0.45

            next_grid[y][x].heat += local_heat
            next_grid[y][x].flame += flame * 0.18
            next_grid[y][x].smoke += smoke * 0.25

            # Neighbor diffusion.
            for dy, dx in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                sx = x + dx
                sy = y + dy
                if 0 <= sx < width and 0 <= sy < height:
                    next_grid[sy][sx].heat += spread_heat * 0.25
                    next_grid[sy][sx].flame += flame * 0.035
                    next_grid[sy][sx].smoke += smoke * 0.03

    # Cooling and smoke conversion.
    for y in range(height):
        for x in range(width):
            c = next_grid[y][x]
            if c.heat > params.smoke_threshold and c.flame < 0.25:
                c.smoke += 0.04 * c.heat
            c.heat = max(0.0, c.heat - params.cooling_rate)
            c.flame *= max(0.0, 0.82 - params.cooling_rate * 0.03)
            c.smoke *= 0.94
            c.vx *= 0.65
            c.vy *= 0.65
            c.fuel = max(0.0, c.fuel)

    # Copy back into existing grid object.
    for y in range(height):
        for x in range(width):
            grid[y][x] = next_grid[y][x]

    return target_hits, fuel_burned


def cell_char(c: FireCell, x: int, y: int, params: FireParams) -> str:
    if x == params.bender_x and y == params.bender_y:
        return "@"
    if abs(x - params.target_x) <= 0 and abs(y - params.target_y) <= 0:
        if c.heat > 5.0:
            return "X"
        return "T"

    if c.flame > 2.0 or c.heat > 12.0:
        return "#"
    if c.flame > 1.0 or c.heat > 8.0:
        return "*"
    if c.heat > 5.0:
        return "+"
    if c.heat > 2.0:
        return "-"
    if c.smoke > 1.4:
        return "S"
    if c.smoke > 0.35:
        return ","
    if c.fuel > 3.0:
        return "F"
    if c.fuel > 0.5:
        return "f"
    return "."


def render_grid(round_number: int, step: int, grid: Grid, params: FireParams, stats: Dict[str, float]) -> None:
    print("\n" + "=" * 86)
    print(
        f"ROUND {round_number} | step={step:03d} | style={params.style} | "
        f"intensity={params.intensity:.1f} oxygen={params.oxygen:.2f} fuel={params.fuel:.2f}"
    )
    print(
        f"sparks={int(stats['sparks'])} target_hits={int(stats['target_hits'])} "
        f"fuel_burned={stats['fuel_burned']:.2f} total_heat={total_heat(grid):.1f} max_heat={max_heat(grid):.1f}"
    )
    print("-" * 86)
    for y in range(len(grid)):
        print("".join(cell_char(grid[y][x], x, y, params) for x in range(len(grid[0]))))
    print("legend: @ bender | T target | X target hit | # core | * flame | + hot | - warm | S smoke | F/f fuel")


def total_heat(grid: Grid) -> float:
    return sum(cell.heat for row in grid for cell in row)


def max_heat(grid: Grid) -> float:
    return max(cell.heat for row in grid for cell in row)


def count_flame_cells(grid: Grid) -> int:
    return sum(1 for row in grid for cell in row if cell.flame > 0.35 or cell.heat > 4.0)


def count_smoke_cells(grid: Grid) -> int:
    return sum(1 for row in grid for cell in row if cell.smoke > 0.35)


def distance_reached(grid: Grid, params: FireParams) -> float:
    best = 0.0
    for y, row in enumerate(grid):
        for x, c in enumerate(row):
            if c.heat > 3.0 or c.flame > 0.3:
                d = sqrt((x - params.bender_x) ** 2 + (y - params.bender_y) ** 2)
                best = max(best, d)
    return best


def summarize_round(round_number: int, params: FireParams, grid: Grid, stats: Dict[str, float]) -> RoundSummary:
    mh = max_heat(grid)
    th = total_heat(grid)
    flames = count_flame_cells(grid)
    smoke = count_smoke_cells(grid)
    reach = distance_reached(grid, params)

    control_score = stats["target_hits"] * 12.0 + reach * 0.6 + flames * 0.05 - smoke * 0.04
    if params.style == "shield":
        control_score += flames * 0.12
    if params.style == "blue":
        control_score += stats["target_hits"] * 8.0

    if stats["target_hits"] > 8:
        note = "target engaged"
    elif smoke > flames * 2 and smoke > 20:
        note = "smoke dominated"
    elif reach > 35:
        note = "long reach"
    elif flames > 140:
        note = "wide flame field"
    elif mh > 25:
        note = "hot focused core"
    else:
        note = "controlled burn"

    return RoundSummary(
        round_number=round_number,
        params=params,
        max_heat=mh,
        total_heat=th,
        flame_cells=flames,
        smoke_cells=smoke,
        sparks_created=int(stats["sparks"]),
        target_hits=int(stats["target_hits"]),
        fuel_burned=stats["fuel_burned"],
        distance_reached=reach,
        control_score=control_score,
        note=note,
    )


def print_round_summary(summary: RoundSummary) -> None:
    print("\nRound result:")
    print(f"  style: {summary.params.style}")
    print(f"  max heat: {summary.max_heat:.2f}")
    print(f"  total heat: {summary.total_heat:.2f}")
    print(f"  flame cells: {summary.flame_cells}")
    print(f"  smoke cells: {summary.smoke_cells}")
    print(f"  sparks created: {summary.sparks_created}")
    print(f"  target hits: {summary.target_hits}")
    print(f"  fuel burned: {summary.fuel_burned:.2f}")
    print(f"  distance reached: {summary.distance_reached:.2f} cells")
    print(f"  control score: {summary.control_score:.2f}")
    print(f"  note: {summary.note}")


class FireBendingAIController:
    """Rule-based controller that chooses a bending style and parameters."""

    def __init__(self, rng: random.Random, config: SimConfig):
        self.rng = rng
        self.config = config
        self.goal_cycle = [
            "straight fire jab",
            "continuous fire stream",
            "wide fire wave",
            "defensive fire shield",
            "curved fire whip",
            "spark storm",
            "blue flame precision",
            "obstacle ignition",
        ]

    def choose_next(self, previous: RoundSummary | None, current: FireParams, round_number: int) -> FireParams:
        goal = self.goal_cycle[(round_number - 1) % len(self.goal_cycle)]
        p = replace(current)

        p.bender_x = 8
        p.bender_y = self.config.height // 2
        p.target_x = self.config.width - 10
        p.target_y = self.config.height // 2

        if goal == "straight fire jab":
            p.style = "jab"
            p.intensity = self.rng.uniform(7.5, 10.5)
            p.duration_steps = self.rng.randint(14, 24)
            p.direction_deg = self.rng.uniform(-6, 6)
            p.spread = self.rng.uniform(0.08, 0.18)
            p.turbulence = self.rng.uniform(0.04, 0.10)
            p.spark_rate = self.rng.uniform(0.03, 0.08)
            p.cooling_rate = self.rng.uniform(0.14, 0.22)
            p.oxygen = self.rng.uniform(0.85, 1.05)
            p.fuel = self.rng.uniform(0.85, 1.10)

        elif goal == "continuous fire stream":
            p.style = "stream"
            p.intensity = self.rng.uniform(8.0, 12.0)
            p.duration_steps = self.rng.randint(36, 54)
            p.direction_deg = self.rng.uniform(-8, 8)
            p.spread = self.rng.uniform(0.18, 0.36)
            p.turbulence = self.rng.uniform(0.08, 0.16)
            p.spark_rate = self.rng.uniform(0.06, 0.12)
            p.cooling_rate = self.rng.uniform(0.13, 0.20)
            p.oxygen = self.rng.uniform(0.90, 1.15)
            p.fuel = self.rng.uniform(0.90, 1.15)

        elif goal == "wide fire wave":
            p.style = "wave"
            p.intensity = self.rng.uniform(7.0, 11.0)
            p.duration_steps = self.rng.randint(20, 35)
            p.direction_deg = self.rng.uniform(-5, 5)
            p.spread = self.rng.uniform(0.30, 0.55)
            p.turbulence = self.rng.uniform(0.12, 0.25)
            p.spark_rate = self.rng.uniform(0.06, 0.14)
            p.cooling_rate = self.rng.uniform(0.15, 0.24)
            p.oxygen = self.rng.uniform(0.85, 1.10)
            p.fuel = self.rng.uniform(0.90, 1.20)

        elif goal == "defensive fire shield":
            p.style = "shield"
            p.intensity = self.rng.uniform(5.5, 8.5)
            p.duration_steps = self.config.steps_per_round
            p.direction_deg = 0.0
            p.spread = self.rng.uniform(0.20, 0.35)
            p.turbulence = self.rng.uniform(0.06, 0.14)
            p.spark_rate = self.rng.uniform(0.03, 0.10)
            p.cooling_rate = self.rng.uniform(0.16, 0.24)
            p.oxygen = self.rng.uniform(0.80, 1.05)
            p.fuel = self.rng.uniform(0.75, 1.05)

        elif goal == "curved fire whip":
            p.style = "whip"
            p.intensity = self.rng.uniform(7.5, 10.5)
            p.duration_steps = self.config.steps_per_round
            p.direction_deg = 0.0
            p.spread = self.rng.uniform(0.10, 0.25)
            p.turbulence = self.rng.uniform(0.08, 0.16)
            p.spark_rate = self.rng.uniform(0.04, 0.12)
            p.cooling_rate = self.rng.uniform(0.14, 0.22)
            p.oxygen = self.rng.uniform(0.85, 1.10)
            p.fuel = self.rng.uniform(0.85, 1.10)

        elif goal == "spark storm":
            p.style = "sparks"
            p.intensity = self.rng.uniform(5.5, 8.0)
            p.duration_steps = self.rng.randint(38, 62)
            p.direction_deg = self.rng.uniform(-4, 4)
            p.spread = self.rng.uniform(0.45, 0.75)
            p.turbulence = self.rng.uniform(0.18, 0.34)
            p.spark_rate = self.rng.uniform(0.22, 0.40)
            p.cooling_rate = self.rng.uniform(0.18, 0.28)
            p.oxygen = self.rng.uniform(0.95, 1.20)
            p.fuel = self.rng.uniform(0.80, 1.05)

        elif goal == "blue flame precision":
            p.style = "blue"
            p.intensity = self.rng.uniform(10.0, 14.5)
            p.duration_steps = self.rng.randint(24, 42)
            p.direction_deg = self.rng.uniform(-3, 3)
            p.spread = self.rng.uniform(0.02, 0.08)
            p.turbulence = self.rng.uniform(0.02, 0.06)
            p.spark_rate = self.rng.uniform(0.02, 0.06)
            p.cooling_rate = self.rng.uniform(0.11, 0.18)
            p.oxygen = self.rng.uniform(1.05, 1.25)
            p.fuel = self.rng.uniform(0.95, 1.15)

        elif goal == "obstacle ignition":
            p.style = "obstacle"
            p.intensity = self.rng.uniform(8.5, 12.5)
            p.duration_steps = self.rng.randint(38, 60)
            p.direction_deg = self.rng.uniform(-5, 5)
            p.spread = self.rng.uniform(0.12, 0.28)
            p.turbulence = self.rng.uniform(0.08, 0.18)
            p.spark_rate = self.rng.uniform(0.05, 0.12)
            p.cooling_rate = self.rng.uniform(0.12, 0.20)
            p.oxygen = self.rng.uniform(0.95, 1.20)
            p.fuel = self.rng.uniform(1.00, 1.25)

        # Reactive tuning.
        if previous is not None:
            if previous.target_hits < 2 and goal not in ("defensive fire shield", "spark storm"):
                p.direction_deg *= 0.5
                p.intensity += 1.0
                p.cooling_rate *= 0.9
            if previous.smoke_cells > previous.flame_cells * 2 and previous.smoke_cells > 20:
                p.oxygen += 0.10
                p.cooling_rate += 0.03
                p.fuel *= 0.95
            if previous.max_heat > 35:
                p.intensity *= 0.90
                p.cooling_rate += 0.02

        p.intensity = clamp_float(p.intensity, 2.0, 18.0)
        p.duration_steps = clamp_int(p.duration_steps, 6, self.config.steps_per_round)
        p.direction_deg = clamp_float(p.direction_deg, -45.0, 45.0)
        p.spread = clamp_float(p.spread, 0.0, 1.0)
        p.turbulence = clamp_float(p.turbulence, 0.0, 0.6)
        p.spark_rate = clamp_float(p.spark_rate, 0.0, 0.6)
        p.cooling_rate = clamp_float(p.cooling_rate, 0.02, 0.7)
        p.oxygen = clamp_float(p.oxygen, 0.25, 1.6)
        p.fuel = clamp_float(p.fuel, 0.25, 1.6)

        print(f"\nAI controller selected next goal: {goal}")
        return p


def run_round(round_number: int, params: FireParams, config: SimConfig, rng: random.Random) -> RoundSummary:
    grid = make_grid(config, params)
    stats = {
        "sparks": 0.0,
        "target_hits": 0.0,
        "fuel_burned": 0.0,
    }

    print("\n" + "#" * 86)
    print(f"START ROUND {round_number}: {params.style.upper()}")
    print("#" * 86)

    for step in range(config.steps_per_round + 1):
        stats["sparks"] += emit_fire(grid, params, step, rng)
        hits, burned = update_grid(grid, params, rng)
        stats["target_hits"] += hits
        stats["fuel_burned"] += burned

        if step % config.print_every_steps == 0 or step == config.steps_per_round:
            render_grid(round_number, step, grid, params, stats)
            if config.sleep_between_frames > 0:
                time.sleep(config.sleep_between_frames)

    summary = summarize_round(round_number, params, grid, stats)
    print_round_summary(summary)
    return summary


def print_final_summary(summaries: List[RoundSummary]) -> None:
    print("\n" + "=" * 118)
    print("FINAL FIRE BENDING ROUND COMPARISON")
    print("=" * 118)
    header = (
        f"{'Round':>5} | {'Style':>9} | {'MaxHeat':>8} | {'TotHeat':>8} | "
        f"{'Flames':>6} | {'Smoke':>6} | {'Sparks':>6} | {'Hits':>5} | "
        f"{'Fuel':>7} | {'Reach':>7} | {'Score':>8} | Note"
    )
    print(header)
    print("-" * len(header))
    for s in summaries:
        print(
            f"{s.round_number:5d} | {s.params.style:>9} | "
            f"{s.max_heat:8.2f} | {s.total_heat:8.2f} | "
            f"{s.flame_cells:6d} | {s.smoke_cells:6d} | "
            f"{s.sparks_created:6d} | {s.target_hits:5d} | "
            f"{s.fuel_burned:7.2f} | {s.distance_reached:7.2f} | "
            f"{s.control_score:8.2f} | {s.note}"
        )

    best_score = max(summaries, key=lambda s: s.control_score)
    best_heat = max(summaries, key=lambda s: s.max_heat)
    best_hits = max(summaries, key=lambda s: s.target_hits)

    print("\nBest control score:")
    print(f"  Round {best_score.round_number}: {best_score.params.style}, score {best_score.control_score:.2f}")
    print("Hottest focused fire:")
    print(f"  Round {best_heat.round_number}: {best_heat.params.style}, max heat {best_heat.max_heat:.2f}")
    print("Most target engagement:")
    print(f"  Round {best_hits.round_number}: {best_hits.params.style}, target hits {best_hits.target_hits}")


def main() -> None:
    config = SimConfig()
    rng = random.Random(config.seed)
    controller = FireBendingAIController(rng, config)

    params = FireParams(
        bender_x=8,
        bender_y=config.height // 2,
        target_x=config.width - 10,
        target_y=config.height // 2,
    )
    previous_summary: RoundSummary | None = None
    summaries: List[RoundSummary] = []

    print("Terminal Fire Bending AI Rounds Simulation")
    print("No graphics. No external packages.")
    print("Model: grid-based heat, flame, sparks, smoke, fuel, oxygen, turbulence, and target engagement.")

    for round_number in range(1, config.rounds + 1):
        params = controller.choose_next(previous_summary, params, round_number)
        summary = run_round(round_number, params, config, rng)
        summaries.append(summary)
        previous_summary = summary

    print_final_summary(summaries)


if __name__ == "__main__":
    main()
