#!/usr/bin/env python3
"""
Terminal Water Flow AI Rounds Simulation
----------------------------------------

A terminal-only Python simulation of simple water behavior.

Run:
    python terminal_water_flow_ai_rounds.py

No external packages are required.

The simulation uses a small grid world. Water is represented as integer
amounts inside cells. Each round, water falls under gravity, spreads sideways,
splashes, evaporates, gets absorbed by soil, and may overflow out of the basin.
A simple AI controller changes conditions between rounds.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import random
import time
from typing import List, Tuple, Dict


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

@dataclass
class SimConfig:
    rounds: int = 8
    width: int = 56
    height: int = 18
    steps_per_round: int = 120
    print_every_steps: int = 8
    sleep_between_frames: float = 0.0
    seed: int = 11


@dataclass
class WaterParams:
    source_rate: int = 7              # water units added per step
    source_x: int = 28
    gravity_flow: int = 5             # max units that fall downward per update
    lateral_flow: int = 2             # max units spreading sideways per update
    splash_chance: float = 0.06       # chance fast falling water throws droplets upward
    evaporation_rate: float = 0.002   # chance surface water loses 1 unit
    absorption_rate: float = 0.035    # chance soil absorbs nearby water
    basin_leakiness: float = 0.010    # chance bottom cracks remove water
    obstacle_mode: str = "basin"      # basin, slope, maze, sponge, overflow
    max_cell_water: int = 9


@dataclass
class RoundSummary:
    round_number: int
    params: WaterParams
    added_water: int
    remaining_water: int
    evaporated: int
    absorbed: int
    leaked: int
    overflowed: int
    max_depth: int
    average_depth: float
    average_speed_estimate: float
    note: str


# Cell types:
#   "." empty air
#   "#" wall/ground
#   "s" soil/sponge
# Water amount is stored separately as 0..max_cell_water.


def clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def make_world(config: SimConfig, params: WaterParams) -> Tuple[List[List[str]], List[List[int]]]:
    terrain = [["." for _ in range(config.width)] for _ in range(config.height)]
    water = [[0 for _ in range(config.width)] for _ in range(config.height)]

    bottom = config.height - 1

    # Ground floor.
    for x in range(config.width):
        terrain[bottom][x] = "#"

    if params.obstacle_mode == "basin":
        for y in range(8, config.height):
            terrain[y][6] = "#"
            terrain[y][config.width - 7] = "#"
        for x in range(6, config.width - 6):
            terrain[bottom][x] = "#"

    elif params.obstacle_mode == "slope":
        for i, x in enumerate(range(8, config.width - 6)):
            y = min(bottom, 7 + i // 5)
            terrain[y][x] = "#"
        for y in range(10, config.height):
            terrain[y][4] = "#"

    elif params.obstacle_mode == "maze":
        for x in range(4, config.width - 4):
            if x % 11 != 0:
                terrain[6][x] = "#"
        for x in range(8, config.width - 8):
            if x % 13 != 0:
                terrain[10][x] = "#"
        for x in range(4, config.width - 4):
            if x % 17 != 0:
                terrain[14][x] = "#"
        for y in range(8, config.height):
            terrain[y][5] = "#"
            terrain[y][config.width - 6] = "#"

    elif params.obstacle_mode == "sponge":
        for x in range(5, config.width - 5):
            terrain[bottom][x] = "#"
        for y in range(10, bottom):
            for x in range(18, 39):
                if (x + y) % 2 == 0:
                    terrain[y][x] = "s"

    elif params.obstacle_mode == "overflow":
        for y in range(9, config.height):
            terrain[y][8] = "#"
            terrain[y][config.width - 9] = "#"
        # Lower left wall so overflow can escape.
        terrain[9][8] = "."
        terrain[10][8] = "."

    return terrain, water


def cell_char(terrain_cell: str, water_amount: int) -> str:
    if water_amount <= 0:
        if terrain_cell == "#":
            return "#"
        if terrain_cell == "s":
            return "%"
        return "."

    if water_amount <= 2:
        return "~"
    if water_amount <= 5:
        return "="
    if water_amount <= 8:
        return "W"
    return "M"


def render_world(
    round_number: int,
    step: int,
    terrain: List[List[str]],
    water: List[List[int]],
    params: WaterParams,
    stats: Dict[str, int],
) -> None:
    print("\n" + "=" * 78)
    print(
        f"ROUND {round_number} | step={step:03d} | mode={params.obstacle_mode} | "
        f"source={params.source_rate}/step | gravity={params.gravity_flow} | lateral={params.lateral_flow}"
    )
    print(
        f"added={stats['added']} remaining={total_water(water)} "
        f"evap={stats['evaporated']} absorbed={stats['absorbed']} "
        f"leaked={stats['leaked']} overflow={stats['overflowed']}"
    )
    print("-" * 78)

    source_x = clamp(params.source_x, 0, len(water[0]) - 1)
    source_line = [" " for _ in range(len(water[0]))]
    source_line[source_x] = "v"
    print("".join(source_line))

    for y in range(len(water)):
        print("".join(cell_char(terrain[y][x], water[y][x]) for x in range(len(water[0]))))

    print("legend: . air | # wall | % sponge/soil | ~ low water | = medium | W deep | M packed")


def total_water(water: List[List[int]]) -> int:
    return sum(sum(row) for row in water)


def add_source(water: List[List[int]], terrain: List[List[str]], params: WaterParams, stats: Dict[str, int]) -> None:
    x = clamp(params.source_x, 0, len(water[0]) - 1)
    amount = params.source_rate
    y = 0
    if terrain[y][x] == "#":
        return
    capacity = max(0, params.max_cell_water - water[y][x])
    placed = min(capacity, amount)
    water[y][x] += placed
    stats["added"] += placed

    overflow = amount - placed
    if overflow > 0:
        stats["overflowed"] += overflow


def try_move(water: List[List[int]], terrain: List[List[str]], x1: int, y1: int, x2: int, y2: int, amount: int, max_cell: int) -> int:
    height = len(water)
    width = len(water[0])
    if x2 < 0 or x2 >= width or y2 < 0 or y2 >= height:
        return 0
    if terrain[y2][x2] == "#":
        return 0
    if water[y1][x1] <= 0:
        return 0

    capacity = max_cell - water[y2][x2]
    moved = min(amount, water[y1][x1], capacity)
    if moved <= 0:
        return 0

    water[y1][x1] -= moved
    water[y2][x2] += moved
    return moved


def update_water(
    terrain: List[List[str]],
    water: List[List[int]],
    params: WaterParams,
    rng: random.Random,
    stats: Dict[str, int],
) -> int:
    height = len(water)
    width = len(water[0])
    movement = 0

    # Falling pass: bottom-up prevents a single unit from falling through
    # the entire world in one step.
    for y in range(height - 2, -1, -1):
        for x in range(width):
            if water[y][x] <= 0:
                continue

            below_blocked = terrain[y + 1][x] == "#"
            if not below_blocked:
                moved = try_move(water, terrain, x, y, x, y + 1, params.gravity_flow, params.max_cell_water)
                movement += moved

                # Splash after downward collision/slowdown.
                if moved > 0 and y + 2 < height and terrain[y + 2][x] == "#" and rng.random() < params.splash_chance:
                    for dx in (-1, 1):
                        sx = x + dx
                        sy = max(0, y - 1)
                        if 0 <= sx < width and terrain[sy][sx] != "#":
                            splash = try_move(water, terrain, x, y + 1, sx, sy, 1, params.max_cell_water)
                            movement += splash
            else:
                # Spread when blocked below.
                dirs = [-1, 1]
                rng.shuffle(dirs)
                for dx in dirs:
                    nx = x + dx
                    if 0 <= nx < width:
                        moved = try_move(water, terrain, x, y, nx, y, params.lateral_flow, params.max_cell_water)
                        movement += moved

    # Equalization pass: water spreads from deeper cells to shallower neighbors.
    for y in range(height - 2, 0, -1):
        for x in range(width):
            if water[y][x] <= 1:
                continue
            dirs = [-1, 1]
            rng.shuffle(dirs)
            for dx in dirs:
                nx = x + dx
                if 0 <= nx < width and terrain[y][nx] != "#":
                    difference = water[y][x] - water[y][nx]
                    if difference > 1:
                        moved = try_move(water, terrain, x, y, nx, y, min(params.lateral_flow, difference // 2), params.max_cell_water)
                        movement += moved

    # Absorption by sponge/soil cells.
    for y in range(height):
        for x in range(width):
            if terrain[y][x] != "s":
                continue
            for dy, dx in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)):
                ny = y + dy
                nx = x + dx
                if 0 <= ny < height and 0 <= nx < width and water[ny][nx] > 0:
                    if rng.random() < params.absorption_rate:
                        water[ny][nx] -= 1
                        stats["absorbed"] += 1

    # Evaporation from exposed water cells.
    for y in range(height):
        for x in range(width):
            if water[y][x] > 0:
                exposed = y == 0 or water[y - 1][x] == 0
                if exposed and rng.random() < params.evaporation_rate:
                    water[y][x] -= 1
                    stats["evaporated"] += 1

    # Leaking from cracked bottom cells.
    bottom = height - 2
    for x in range(width):
        if water[bottom][x] > 0 and rng.random() < params.basin_leakiness:
            water[bottom][x] -= 1
            stats["leaked"] += 1

    # Overflow off left/right edges when water sits at borders.
    for y in range(height):
        for x in (0, width - 1):
            if water[y][x] > 0:
                stats["overflowed"] += water[y][x]
                water[y][x] = 0

    return movement


def summarize_round(round_number: int, params: WaterParams, water: List[List[int]], stats: Dict[str, int], movement_total: int, config: SimConfig) -> RoundSummary:
    remaining = total_water(water)
    wet_cells = [amount for row in water for amount in row if amount > 0]
    max_depth = max(wet_cells) if wet_cells else 0
    average_depth = sum(wet_cells) / len(wet_cells) if wet_cells else 0.0
    average_speed_estimate = movement_total / max(1, config.steps_per_round)

    if remaining == 0:
        note = "dry system"
    elif stats["overflowed"] > stats["added"] * 0.15:
        note = "overflow dominated"
    elif stats["absorbed"] > stats["added"] * 0.15:
        note = "absorption dominated"
    elif max_depth >= params.max_cell_water - 1:
        note = "deep pooling"
    elif average_speed_estimate > 60:
        note = "fast turbulent flow"
    else:
        note = "stable flowing water"

    return RoundSummary(
        round_number=round_number,
        params=params,
        added_water=stats["added"],
        remaining_water=remaining,
        evaporated=stats["evaporated"],
        absorbed=stats["absorbed"],
        leaked=stats["leaked"],
        overflowed=stats["overflowed"],
        max_depth=max_depth,
        average_depth=average_depth,
        average_speed_estimate=average_speed_estimate,
        note=note,
    )


def print_round_summary(summary: RoundSummary) -> None:
    print("\nRound result:")
    print(f"  added water: {summary.added_water}")
    print(f"  remaining water: {summary.remaining_water}")
    print(f"  evaporated: {summary.evaporated}")
    print(f"  absorbed: {summary.absorbed}")
    print(f"  leaked: {summary.leaked}")
    print(f"  overflowed: {summary.overflowed}")
    print(f"  max depth: {summary.max_depth}")
    print(f"  average wet-cell depth: {summary.average_depth:.2f}")
    print(f"  average movement estimate: {summary.average_speed_estimate:.2f} units/step")
    print(f"  note: {summary.note}")


class WaterAIController:
    """Rule-based controller that chooses the next water scenario."""

    def __init__(self, rng: random.Random, config: SimConfig):
        self.rng = rng
        self.config = config
        self.goal_cycle = [
            "gentle basin fill",
            "fast river slope",
            "maze flow",
            "sponge absorption",
            "overflow stress test",
            "misty evaporation test",
        ]

    def choose_next(self, previous: RoundSummary | None, current: WaterParams, round_number: int) -> WaterParams:
        goal = self.goal_cycle[(round_number - 1) % len(self.goal_cycle)]
        p = replace(current)

        if goal == "gentle basin fill":
            p.obstacle_mode = "basin"
            p.source_rate = self.rng.randint(4, 7)
            p.gravity_flow = self.rng.randint(3, 5)
            p.lateral_flow = self.rng.randint(1, 2)
            p.splash_chance = self.rng.uniform(0.01, 0.04)
            p.evaporation_rate = self.rng.uniform(0.000, 0.003)
            p.absorption_rate = self.rng.uniform(0.015, 0.035)
            p.basin_leakiness = self.rng.uniform(0.000, 0.010)

        elif goal == "fast river slope":
            p.obstacle_mode = "slope"
            p.source_rate = self.rng.randint(7, 11)
            p.gravity_flow = self.rng.randint(5, 8)
            p.lateral_flow = self.rng.randint(2, 4)
            p.splash_chance = self.rng.uniform(0.05, 0.12)
            p.evaporation_rate = self.rng.uniform(0.000, 0.002)
            p.absorption_rate = self.rng.uniform(0.010, 0.030)
            p.basin_leakiness = self.rng.uniform(0.010, 0.030)

        elif goal == "maze flow":
            p.obstacle_mode = "maze"
            p.source_rate = self.rng.randint(5, 9)
            p.gravity_flow = self.rng.randint(4, 7)
            p.lateral_flow = self.rng.randint(2, 3)
            p.splash_chance = self.rng.uniform(0.02, 0.07)
            p.evaporation_rate = self.rng.uniform(0.000, 0.003)
            p.absorption_rate = self.rng.uniform(0.015, 0.040)
            p.basin_leakiness = self.rng.uniform(0.005, 0.020)

        elif goal == "sponge absorption":
            p.obstacle_mode = "sponge"
            p.source_rate = self.rng.randint(5, 8)
            p.gravity_flow = self.rng.randint(3, 6)
            p.lateral_flow = self.rng.randint(1, 3)
            p.splash_chance = self.rng.uniform(0.01, 0.05)
            p.evaporation_rate = self.rng.uniform(0.000, 0.002)
            p.absorption_rate = self.rng.uniform(0.060, 0.120)
            p.basin_leakiness = self.rng.uniform(0.000, 0.010)

        elif goal == "overflow stress test":
            p.obstacle_mode = "overflow"
            p.source_rate = self.rng.randint(10, 15)
            p.gravity_flow = self.rng.randint(5, 9)
            p.lateral_flow = self.rng.randint(3, 5)
            p.splash_chance = self.rng.uniform(0.06, 0.16)
            p.evaporation_rate = self.rng.uniform(0.000, 0.002)
            p.absorption_rate = self.rng.uniform(0.005, 0.020)
            p.basin_leakiness = self.rng.uniform(0.000, 0.020)

        elif goal == "misty evaporation test":
            p.obstacle_mode = "basin"
            p.source_rate = self.rng.randint(3, 6)
            p.gravity_flow = self.rng.randint(2, 5)
            p.lateral_flow = self.rng.randint(1, 2)
            p.splash_chance = self.rng.uniform(0.02, 0.08)
            p.evaporation_rate = self.rng.uniform(0.010, 0.030)
            p.absorption_rate = self.rng.uniform(0.020, 0.050)
            p.basin_leakiness = self.rng.uniform(0.000, 0.010)

        # Reactive tuning.
        if previous is not None:
            if previous.overflowed > previous.added_water * 0.25:
                p.source_rate = max(2, p.source_rate - 2)
                p.lateral_flow = max(1, p.lateral_flow - 1)
            if previous.remaining_water < previous.added_water * 0.10:
                p.source_rate += 2
                p.evaporation_rate *= 0.5
            if previous.max_depth >= p.max_cell_water - 1:
                p.lateral_flow += 1
                p.basin_leakiness += 0.005

        p.source_x = clamp(self.config.width // 2 + self.rng.randint(-8, 8), 1, self.config.width - 2)
        p.source_rate = clamp(p.source_rate, 1, 18)
        p.gravity_flow = clamp(p.gravity_flow, 1, 12)
        p.lateral_flow = clamp(p.lateral_flow, 1, 8)
        p.splash_chance = max(0.0, min(0.4, p.splash_chance))
        p.evaporation_rate = max(0.0, min(0.08, p.evaporation_rate))
        p.absorption_rate = max(0.0, min(0.3, p.absorption_rate))
        p.basin_leakiness = max(0.0, min(0.1, p.basin_leakiness))

        print(f"\nAI controller selected next goal: {goal}")
        return p


def run_round(round_number: int, params: WaterParams, config: SimConfig, rng: random.Random) -> RoundSummary:
    terrain, water = make_world(config, params)
    stats = {
        "added": 0,
        "evaporated": 0,
        "absorbed": 0,
        "leaked": 0,
        "overflowed": 0,
    }
    movement_total = 0

    print("\n" + "#" * 78)
    print(f"START ROUND {round_number}: {params.obstacle_mode.upper()}")
    print("#" * 78)

    for step in range(config.steps_per_round + 1):
        add_source(water, terrain, params, stats)
        movement_total += update_water(terrain, water, params, rng, stats)

        if step % config.print_every_steps == 0 or step == config.steps_per_round:
            render_world(round_number, step, terrain, water, params, stats)
            if config.sleep_between_frames > 0:
                time.sleep(config.sleep_between_frames)

    summary = summarize_round(round_number, params, water, stats, movement_total, config)
    print_round_summary(summary)
    return summary


def print_final_summary(summaries: List[RoundSummary]) -> None:
    print("\n" + "=" * 104)
    print("FINAL WATER ROUND COMPARISON")
    print("=" * 104)
    header = (
        f"{'Round':>5} | {'Mode':>9} | {'Added':>7} | {'Remain':>7} | "
        f"{'Evap':>5} | {'Absorb':>6} | {'Leak':>5} | {'Overflow':>8} | "
        f"{'MaxD':>4} | {'AvgD':>6} | {'Move':>8} | Note"
    )
    print(header)
    print("-" * len(header))
    for s in summaries:
        print(
            f"{s.round_number:5d} | {s.params.obstacle_mode:>9} | "
            f"{s.added_water:7d} | {s.remaining_water:7d} | "
            f"{s.evaporated:5d} | {s.absorbed:6d} | {s.leaked:5d} | "
            f"{s.overflowed:8d} | {s.max_depth:4d} | "
            f"{s.average_depth:6.2f} | {s.average_speed_estimate:8.2f} | {s.note}"
        )

    best_storage = max(summaries, key=lambda s: s.remaining_water)
    best_flow = max(summaries, key=lambda s: s.average_speed_estimate)
    print("\nBest water storage:")
    print(f"  Round {best_storage.round_number}: {best_storage.remaining_water} units remaining in {best_storage.params.obstacle_mode} mode")
    print("Best active flow:")
    print(f"  Round {best_flow.round_number}: movement estimate {best_flow.average_speed_estimate:.2f} units/step in {best_flow.params.obstacle_mode} mode")


def main() -> None:
    config = SimConfig()
    rng = random.Random(config.seed)
    controller = WaterAIController(rng, config)

    params = WaterParams(source_x=config.width // 2)
    previous_summary: RoundSummary | None = None
    summaries: List[RoundSummary] = []

    print("Terminal Water Flow AI Rounds Simulation")
    print("No graphics. No external packages.")
    print("Model: grid-based water falling, spreading, pooling, splashing, evaporating, absorbing, leaking, and overflowing.")

    for round_number in range(1, config.rounds + 1):
        params = controller.choose_next(previous_summary, params, round_number)
        summary = run_round(round_number, params, config, rng)
        summaries.append(summary)
        previous_summary = summary

    print_final_summary(summaries)


if __name__ == "__main__":
    main()
