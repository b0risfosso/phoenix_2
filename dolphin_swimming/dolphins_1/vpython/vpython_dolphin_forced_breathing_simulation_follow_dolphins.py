"""
VPython Dolphin Pod Simulation Based on Terminal Output
------------------------------------------------------
Visualizes the terminal-only dolphin pod hydrodynamics simulation with:
- 8 autonomous rounds matching the terminal output behaviors
- six dolphins, including one calf
- prey cloud, predator, current arrow, surface and depth reference grid
- forced dive-rise breathing cycle in Round 2
- AI controller values and round summary HUD

Controls:
    SPACE  pause/play
    n      next round
    b      previous round
    r      reset current round
    f      toggle dolphin-follow camera
    c      cycle followed dolphin
    p      follow whole pod
    h      hide/show help

Run:
    python vpython_dolphin_forced_breathing_simulation.py

Notes:
    This script uses ring(...) instead of torus(...), for compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Dict, List, Optional

from vpython import (
    vector, mag, norm, dot, cross, rate,
    sphere, ellipsoid, cone, cylinder, box, arrow, curve, ring,
    label, color, scene, compound, local_light
)


# -----------------------------------------------------------------------------
# Configuration data derived from the forced-breathing terminal output summary
# -----------------------------------------------------------------------------

@dataclass
class RoundConfig:
    name: str
    behavior: str
    note: str
    current: vector
    target_depth: float
    thrust_scale: float
    spiral: float
    predator_pressure: float
    prey_density: float
    desired_spacing: float
    speed_cap: float
    turn_control: float
    summary_speed: float
    summary_spread: float
    expected_captures: int
    initial_prey_distance: float
    final_prey_distance: float
    surface_events: int = 0
    breath_cycles: int = 0


ROUND_CONFIGS: List[RoundConfig] = [
    RoundConfig(
        name="Low-Drag Cruise",
        behavior="cruise",
        note="Efficient alignment, low drag, and steady depth control.",
        current=vector(0.7, -0.2, 0.08),
        target_depth=20.7,
        thrust_scale=1.00,
        spiral=0.00,
        predator_pressure=0.00,
        prey_density=0.65,
        desired_spacing=8.0,
        speed_cap=4.0,
        turn_control=0.85,
        summary_speed=2.41,
        summary_spread=6.32,
        expected_captures=2,
        initial_prey_distance=59.1,
        final_prey_distance=12.8,
    ),
    RoundConfig(
        name="Forced Dive-Rise Breathing Cycle",
        behavior="forced_breathing",
        note="Scheduled dive, rise, surface-hold, and recovery descent.",
        current=vector(0.4, -0.3, 0.02),
        target_depth=26.0,
        thrust_scale=1.03,
        spiral=0.00,
        predator_pressure=0.00,
        prey_density=0.62,
        desired_spacing=8.0,
        speed_cap=4.2,
        turn_control=0.90,
        summary_speed=3.19,
        summary_spread=10.38,
        expected_captures=4,
        initial_prey_distance=49.6,
        final_prey_distance=12.7,
        surface_events=27,
        breath_cycles=6,
    ),
    RoundConfig(
        name="Active Prey Pursuit",
        behavior="prey_pursuit",
        note="Direct chase with stronger steering toward the prey cloud.",
        current=vector(0.5, 0.0, 0.0),
        target_depth=16.9,
        thrust_scale=1.22,
        spiral=0.00,
        predator_pressure=0.00,
        prey_density=0.56,
        desired_spacing=7.5,
        speed_cap=4.6,
        turn_control=1.20,
        summary_speed=2.65,
        summary_spread=9.50,
        expected_captures=9,
        initial_prey_distance=37.9,
        final_prey_distance=33.8,
    ),
    RoundConfig(
        name="Spiral Compression Hunt",
        behavior="spiral_hunt",
        note="Inward force plus tangential orbiting compresses prey.",
        current=vector(0.5, 0.5, 0.0),
        target_depth=22.0,
        thrust_scale=1.10,
        spiral=1.35,
        predator_pressure=0.00,
        prey_density=0.77,
        desired_spacing=7.1,
        speed_cap=4.4,
        turn_control=1.10,
        summary_speed=3.43,
        summary_spread=5.14,
        expected_captures=7,
        initial_prey_distance=35.8,
        final_prey_distance=58.8,
    ),
    RoundConfig(
        name="Current-Assisted Surfing",
        behavior="current_surfing",
        note="Current alignment and glide intervals reduce direct thrust.",
        current=vector(0.6, -0.2, 0.0),
        target_depth=31.2,
        thrust_scale=0.79,
        spiral=0.00,
        predator_pressure=0.00,
        prey_density=0.61,
        desired_spacing=6.6,
        speed_cap=4.2,
        turn_control=0.95,
        summary_speed=3.88,
        summary_spread=3.74,
        expected_captures=3,
        initial_prey_distance=46.0,
        final_prey_distance=61.3,
    ),
    RoundConfig(
        name="Predator Evasion",
        behavior="predator_avoidance",
        note="Threat pressure causes burst speed and defensive grouping.",
        current=vector(0.6, -0.3, -0.05),
        target_depth=16.7,
        thrust_scale=1.31,
        spiral=0.00,
        predator_pressure=0.73,
        prey_density=0.63,
        desired_spacing=6.2,
        speed_cap=5.4,
        turn_control=1.15,
        summary_speed=4.90,
        summary_spread=7.99,
        expected_captures=0,
        initial_prey_distance=40.1,
        final_prey_distance=100.6,
    ),
    RoundConfig(
        name="Calf Escort Formation",
        behavior="calf_protection",
        note="Adults form a moving shell around the slower calf.",
        current=vector(0.9, 0.5, -0.05),
        target_depth=25.8,
        thrust_scale=0.84,
        spiral=0.00,
        predator_pressure=0.00,
        prey_density=0.46,
        desired_spacing=5.9,
        speed_cap=4.3,
        turn_control=1.00,
        summary_speed=4.08,
        summary_spread=7.43,
        expected_captures=8,
        initial_prey_distance=40.8,
        final_prey_distance=70.5,
    ),
    RoundConfig(
        name="Mixed Adaptive Foraging",
        behavior="mixed_adaptive",
        note="AI blends hunting, formation, current use, and mild threat awareness.",
        current=vector(0.9, 0.4, 0.02),
        target_depth=22.6,
        thrust_scale=0.83,
        spiral=0.59,
        predator_pressure=0.30,
        prey_density=0.75,
        desired_spacing=5.5,
        speed_cap=4.6,
        turn_control=1.20,
        summary_speed=4.20,
        summary_spread=4.20,
        expected_captures=10,
        initial_prey_distance=35.7,
        final_prey_distance=77.8,
    ),
]


# -----------------------------------------------------------------------------
# Visual constants
# -----------------------------------------------------------------------------

random.seed(7)
DT = 0.045
STEPS_PER_ROUND = 860
WORLD_SCALE = 1.0
SURFACE_Z = 0.0
BOTTOM_Z = -42.0
PREY_CAPTURE_RADIUS = 5.2

DOLPHIN_COLORS = [
    vector(0.46, 0.68, 0.88),
    vector(0.41, 0.62, 0.82),
    vector(0.50, 0.72, 0.90),
    vector(0.36, 0.58, 0.78),
    vector(0.52, 0.75, 0.92),
    vector(0.72, 0.86, 0.96),
]


# -----------------------------------------------------------------------------
# Helper math
# -----------------------------------------------------------------------------

def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def safe_norm(v: vector, fallback: Optional[vector] = None) -> vector:
    if mag(v) < 1e-9:
        return fallback if fallback is not None else vector(1, 0, 0)
    return norm(v)


def limit_vector(v: vector, max_mag: float) -> vector:
    m = mag(v)
    if m > max_mag and m > 1e-9:
        return v * (max_mag / m)
    return v


def depth_to_z(depth: float) -> float:
    return -abs(depth)


def z_to_depth(z: float) -> float:
    return max(0.0, -z)


def horizontal(v: vector) -> vector:
    return vector(v.x, v.y, 0)


# -----------------------------------------------------------------------------
# Scene setup
# -----------------------------------------------------------------------------

scene.title = "VPython Dolphin Pod Hydrodynamics: Forced Breathing Cycles"
scene.width = 1280
scene.height = 760
scene.background = vector(0.88, 0.96, 1.0)
scene.forward = vector(-0.80, -0.40, -0.35)
scene.up = vector(0, 0, 1)
scene.center = vector(22, 0, -15)
scene.range = 72
scene.userspin = True
scene.userzoom = True

local_light(pos=vector(-30, -40, 40), color=color.white)

# ocean volume and reference objects
surface = box(
    pos=vector(40, 0, SURFACE_Z),
    size=vector(160, 90, 0.08),
    color=vector(0.52, 0.78, 0.96),
    opacity=0.25,
)
bottom = box(
    pos=vector(40, 0, BOTTOM_Z),
    size=vector(160, 90, 0.08),
    color=vector(0.35, 0.62, 0.72),
    opacity=0.18,
)
center_line = curve(color=vector(0.45, 0.70, 0.85), radius=0.03)
for x in range(-40, 121, 10):
    center_line.append(vector(x, -42, SURFACE_Z))
    center_line.append(vector(x, 42, SURFACE_Z))
    center_line.append(vector(x, -42, SURFACE_Z))

# depth rings use ring, not torus
for depth, rad in [(10, 18), (20, 26), (30, 34)]:
    ring(
        pos=vector(40, 0, depth_to_z(depth)),
        axis=vector(0, 0, 1),
        radius=rad,
        thickness=0.04,
        color=vector(0.60, 0.78, 0.92),
        opacity=0.32,
    )

surface_label = label(
    pos=vector(-38, -42, 1.6),
    text="surface z=0 | depth increases downward",
    color=vector(0.12, 0.28, 0.40),
    box=False,
    height=13,
)


# -----------------------------------------------------------------------------
# Visual entity classes
# -----------------------------------------------------------------------------

class DolphinVisual:
    def __init__(self, idx: int, role: str, pos: vector):
        self.idx = idx
        self.role = role
        self.pos = pos
        self.vel = vector(1.2 + 0.2 * idx, 0.05 * idx, 0.0)
        self.energy = 100.0
        self.oxygen = 100.0
        self.captures = 0
        self.prey_transit_timer = 0.0
        self.prey_loop_sign = 1.0 if idx % 2 == 0 else -1.0
        self.prey_passes = 0
        self.phase = "neutral"
        self.phase_time = 0.0
        self.breath_cycles = 0
        self.surface_events = 0
        self.max_speed_multiplier = 0.76 if role == "CALF" else 1.0
        base_color = DOLPHIN_COLORS[idx - 1]

        # Body points along +x by default; compound axis follows velocity.
        body = ellipsoid(
            pos=vector(0, 0, 0),
            length=4.5 if role != "CALF" else 3.4,
            height=1.15 if role != "CALF" else 0.86,
            width=1.25 if role != "CALF" else 0.95,
            color=base_color,
            shininess=0.65,
        )
        nose = cone(
            pos=vector(2.1 if role != "CALF" else 1.65, 0, 0),
            axis=vector(0.85, 0, 0),
            radius=0.42 if role != "CALF" else 0.32,
            color=base_color,
        )
        tail_stem = cylinder(
            pos=vector(-2.15 if role != "CALF" else -1.62, 0, 0),
            axis=vector(-0.95 if role != "CALF" else -0.70, 0, 0),
            radius=0.22 if role != "CALF" else 0.17,
            color=base_color,
        )
        tail_top = cone(
            pos=vector(-3.05 if role != "CALF" else -2.28, 0, 0.05),
            axis=vector(-0.15, 0.75, 0.35),
            radius=0.42 if role != "CALF" else 0.30,
            color=base_color * 0.85,
        )
        tail_bottom = cone(
            pos=vector(-3.05 if role != "CALF" else -2.28, 0, -0.05),
            axis=vector(-0.15, -0.75, -0.35),
            radius=0.42 if role != "CALF" else 0.30,
            color=base_color * 0.85,
        )
        dorsal = cone(
            pos=vector(-0.15, 0, 0.45 if role != "CALF" else 0.35),
            axis=vector(-0.25, 0, 0.95 if role != "CALF" else 0.70),
            radius=0.34 if role != "CALF" else 0.24,
            color=base_color * 0.75,
        )
        left_fin = cone(
            pos=vector(0.35, 0.48 if role != "CALF" else 0.36, -0.12),
            axis=vector(-0.35, 0.75, -0.30),
            radius=0.25 if role != "CALF" else 0.18,
            color=base_color * 0.72,
        )
        right_fin = cone(
            pos=vector(0.35, -0.48 if role != "CALF" else -0.36, -0.12),
            axis=vector(-0.35, -0.75, -0.30),
            radius=0.25 if role != "CALF" else 0.18,
            color=base_color * 0.72,
        )
        self.obj = compound([body, nose, tail_stem, tail_top, tail_bottom, dorsal, left_fin, right_fin])
        self.obj.pos = self.pos
        self.obj.axis = vector(1, 0, 0)

        self.trail = curve(color=base_color * 0.75, radius=0.035)
        self.vel_arrow = arrow(pos=self.pos, axis=self.vel, shaftwidth=0.11, color=base_color * 0.8, opacity=0.55)
        # Dolphin name/status labels removed for a cleaner visual scene.
        self.name_label = None
        self.oxygen_bar = cylinder(pos=self.pos + vector(-1.2, 0, 1.5), axis=vector(1.8, 0, 0), radius=0.05, color=vector(0.2, 0.6, 1.0))

    def set_state(self, pos: vector, vel: vector):
        self.pos = pos
        self.vel = vel
        self.obj.pos = pos
        self.obj.axis = safe_norm(vel, self.obj.axis)
        # VPython curve objects vary by version. Some expose .npoints, but not .points.
        # Avoid AttributeError on newer VPython builds while still trimming long trails.
        try:
            if getattr(self.trail, "npoints", 0) > 220:
                self.trail.pop(0)
        except Exception:
            pass
        self.trail.append(pos=pos)
        self.vel_arrow.pos = pos
        self.vel_arrow.axis = safe_norm(vel) * min(4.0, mag(vel))
        self.oxygen_bar.pos = pos + vector(-0.9, 0, 1.55)
        self.oxygen_bar.axis = vector(1.8 * clamp(self.oxygen / 100.0, 0.05, 1.0), 0, 0)
        self.oxygen_bar.color = vector(0.15 + 0.7 * (1 - self.oxygen / 100.0), 0.55, 1.0)

    def reset_trail(self):
        self.trail.clear()


class PreyCloud:
    def __init__(self):
        self.center = vector(55, -8, -20)
        self.vel = vector(0.2, -0.05, 0.0)
        self.density = 0.7
        self.particles: List[sphere] = []
        self.ring = ring(pos=self.center, axis=vector(0, 0, 1), radius=7.0, thickness=0.08, color=vector(1.0, 0.72, 0.20), opacity=0.45)
        for _ in range(36):
            p = sphere(radius=random.uniform(0.10, 0.25), color=vector(1.0, random.uniform(0.65, 0.90), 0.12), emissive=True)
            self.particles.append(p)
        self.label = label(pos=self.center + vector(0, 0, 5), text="prey cloud", box=False, height=12, color=vector(0.55, 0.36, 0.00))
        self.refresh_particles()

    def set_center(self, center: vector, density: float):
        self.center = center
        self.density = density
        self.ring.pos = center
        self.ring.radius = 5.5 + 5.5 * density
        self.label.pos = center + vector(0, 0, 5)
        self.refresh_particles()

    def update(self, dt: float, round_cfg: RoundConfig, t: float):
        swirl = vector(0.15 * math.sin(t * 0.45), 0.12 * math.cos(t * 0.37), 0.03 * math.sin(t * 0.52))
        self.center += (self.vel + swirl + 0.04 * round_cfg.current) * dt
        self.center.z = clamp(self.center.z, BOTTOM_Z + 5, -4)
        self.ring.pos = self.center
        self.ring.axis = vector(0, 0, 1)
        self.label.pos = self.center + vector(0, 0, 5)
        self.refresh_particles(t)

    def refresh_particles(self, t: float = 0.0):
        radius = 4.0 + 5.5 * self.density
        for i, p in enumerate(self.particles):
            angle = 2 * math.pi * (i / len(self.particles)) + 0.35 * math.sin(t + i)
            rr = radius * (0.25 + 0.75 * ((i * 37) % 100) / 100.0)
            zoff = 1.8 * math.sin(angle * 1.7 + i)
            p.pos = self.center + vector(rr * math.cos(angle), rr * math.sin(angle), zoff)
            p.visible = i < int(12 + 24 * self.density)

    def reduce_density(self, amount: float):
        self.density = clamp(self.density - amount, 0.12, 1.0)
        self.ring.radius = 5.5 + 5.5 * self.density


class PredatorVisual:
    def __init__(self):
        body = ellipsoid(pos=vector(0, 0, 0), length=6.5, height=1.3, width=1.1, color=vector(0.30, 0.34, 0.38))
        nose = cone(pos=vector(3.0, 0, 0), axis=vector(1.1, 0, 0), radius=0.55, color=vector(0.24, 0.27, 0.30))
        tail = cone(pos=vector(-3.1, 0, 0), axis=vector(-1.2, 0.6, 0.2), radius=0.55, color=vector(0.24, 0.27, 0.30))
        fin = cone(pos=vector(0.0, 0, 0.55), axis=vector(-0.1, 0, 1.2), radius=0.45, color=vector(0.18, 0.20, 0.23))
        self.obj = compound([body, nose, tail, fin])
        self.obj.visible = False
        self.pos = vector(-50, 10, -14)
        self.vel = vector(0.6, -0.1, 0)
        self.threat_ring = ring(pos=self.pos, axis=vector(0, 0, 1), radius=11, thickness=0.08, color=vector(0.9, 0.2, 0.1), opacity=0.45)
        self.threat_ring.visible = False
        self.label = label(pos=self.pos + vector(0, 0, 4), text="predator", box=False, height=12, color=vector(0.55, 0.05, 0.04))
        self.label.visible = False

    def set_active(self, active: bool):
        self.obj.visible = active
        self.threat_ring.visible = active
        self.label.visible = active

    def reset_for_round(self, cfg: RoundConfig, pod_center: vector):
        active = cfg.predator_pressure > 0
        self.set_active(active)
        if active:
            self.pos = pod_center + vector(-58, 22, random.uniform(-5, 5))
            self.vel = safe_norm(vector(1, -0.18, 0.02)) * (0.6 + 0.9 * cfg.predator_pressure)
            self.update_visual()

    def update(self, dt: float, pod_center: vector, cfg: RoundConfig):
        if cfg.predator_pressure <= 0:
            self.set_active(False)
            return
        self.set_active(True)
        direction = safe_norm(pod_center - self.pos, self.vel)
        self.vel = limit_vector(self.vel * 0.97 + direction * (0.04 + 0.04 * cfg.predator_pressure), 2.4)
        self.pos += self.vel * dt
        self.pos.z = clamp(self.pos.z, BOTTOM_Z + 8, -4)
        self.update_visual()

    def update_visual(self):
        self.obj.pos = self.pos
        self.obj.axis = safe_norm(self.vel)
        self.threat_ring.pos = self.pos
        self.label.pos = self.pos + vector(0, 0, 4)


# -----------------------------------------------------------------------------
# Simulation state
# -----------------------------------------------------------------------------

class DolphinSimulation:
    def __init__(self):
        self.round_index = 0
        self.step = 0
        self.t = 0.0
        self.paused = False
        self.follow_camera = True
        self.follow_mode = "dolphin"
        self.follow_index = 0
        self.show_help = True
        self.ai_spacing = 8.0
        self.ai_hunt = 1.0
        self.ai_oxygen = 1.0
        self.ai_energy = 1.0
        self.ai_predator = 1.0
        self.round_captures = 0
        self.round_surface_events = 0
        self.round_cycles = 0
        self.round_forced_dive_transitions = 0
        self.round_forced_rise_transitions = 0
        self.round_surface_hold_transitions = 0

        roles = ["LEAD", "ADULT", "ADULT", "ADULT", "ADULT", "CALF"]
        starts = [
            vector(6.5, -0.4, -13.1),
            vector(-5.0, -1.9, -8.9),
            vector(2.5, 1.2, -9.6),
            vector(-5.9, -0.4, -12.4),
            vector(0.1, -2.1, -18.0),
            vector(-2.2, -2.7, -10.9),
        ]
        self.dolphins = [DolphinVisual(i + 1, roles[i], starts[i]) for i in range(6)]
        self.prey = PreyCloud()
        self.predator = PredatorVisual()

        self.current_arrow = arrow(
            pos=vector(-34, 36, -2),
            axis=vector(8, 0, 0),
            shaftwidth=0.35,
            color=vector(0.08, 0.46, 0.78),
        )
        self.current_label = label(pos=self.current_arrow.pos + vector(3, 0, 3), text="current", box=False, height=12, color=vector(0.06, 0.30, 0.50))

        self.hud = label(
            pos=vector(-45, -44, 18),
            text="",
            height=12,
            box=True,
            color=vector(0.05, 0.13, 0.22),
            background=vector(0.94, 0.98, 1.0),
            border=8,
            line=False,
            opacity=0.88,
        )
        self.help_label = label(
            pos=vector(50, -44, 18),
            text="",
            height=11,
            box=True,
            color=vector(0.06, 0.12, 0.18),
            background=vector(0.95, 0.98, 1.0),
            border=6,
            opacity=0.82,
        )
        self.reset_round(0, full_reset=True)

    @property
    def cfg(self) -> RoundConfig:
        return ROUND_CONFIGS[self.round_index]

    def pod_center(self) -> vector:
        acc = vector(0, 0, 0)
        for d in self.dolphins:
            acc += d.pos
        return acc / len(self.dolphins)

    def pod_spread(self) -> float:
        c = self.pod_center()
        return sum(mag(d.pos - c) for d in self.dolphins) / len(self.dolphins)

    def average_speed(self) -> float:
        return sum(mag(d.vel) for d in self.dolphins) / len(self.dolphins)

    def min_oxygen(self) -> float:
        return min(d.oxygen for d in self.dolphins)

    def avg_energy(self) -> float:
        return sum(d.energy for d in self.dolphins) / len(self.dolphins)

    def reset_round(self, index: Optional[int] = None, full_reset: bool = False):
        if index is not None:
            self.round_index = index % len(ROUND_CONFIGS)
        cfg = self.cfg
        self.step = 0
        self.t = 0.0
        self.round_captures = 0
        self.round_surface_events = 0
        self.round_cycles = 0
        self.round_forced_dive_transitions = 0
        self.round_forced_rise_transitions = 0
        self.round_surface_hold_transitions = 0

        if full_reset:
            base_positions = [
                vector(6.5, -0.4, -13.1),
                vector(-5.0, -1.9, -8.9),
                vector(2.5, 1.2, -9.6),
                vector(-5.9, -0.4, -12.4),
                vector(0.1, -2.1, -18.0),
                vector(-2.2, -2.7, -10.9),
            ]
            for d, p in zip(self.dolphins, base_positions):
                d.pos = p
                d.vel = vector(1.3, 0.05 * d.idx, 0.0)
                d.energy = 100.0
                d.oxygen = 100.0
                d.captures = 0
                d.prey_transit_timer = 0.0
                d.prey_loop_sign = 1.0 if d.idx % 2 == 0 else -1.0
                d.prey_passes = 0
                d.phase = "neutral"
                d.phase_time = 0.0
                d.breath_cycles = 0
                d.surface_events = 0
                d.reset_trail()
                d.set_state(d.pos, d.vel)
        else:
            # Damp inherited motion between rounds to mirror v2/v3 terminal improvements.
            c = self.pod_center()
            for i, d in enumerate(self.dolphins):
                offset = vector((i - 2.5) * 2.2, ((i % 2) - 0.5) * 3.5, random.uniform(-1.0, 1.0))
                target_z = depth_to_z(cfg.target_depth * random.uniform(0.65, 0.95))
                d.pos = vector(c.x * 0.35 + offset.x, c.y * 0.35 + offset.y, target_z)
                d.vel *= 0.30
                d.oxygen = 100.0
                d.energy = max(58.0, d.energy + 10.0)
                d.prey_transit_timer = 0.0
                d.prey_loop_sign = 1.0 if d.idx % 2 == 0 else -1.0
                d.prey_passes = 0
                d.phase = "neutral"
                d.phase_time = 0.0
                d.reset_trail()
                d.set_state(d.pos, d.vel)

        pod_c = self.pod_center()
        prey_direction = safe_norm(vector(1.0, random.uniform(-0.4, 0.4), random.uniform(-0.10, 0.10)))
        distance = cfg.initial_prey_distance
        prey_center = pod_c + prey_direction * distance
        prey_center.z = depth_to_z(cfg.target_depth + random.uniform(-5, 5))
        self.prey.vel = vector(random.uniform(-0.15, 0.25), random.uniform(-0.15, 0.25), random.uniform(-0.02, 0.02))
        self.prey.set_center(prey_center, cfg.prey_density)
        self.predator.reset_for_round(cfg, pod_c)

        self.current_arrow.axis = cfg.current * 10.0
        if mag(self.current_arrow.axis) < 1e-6:
            self.current_arrow.axis = vector(2, 0, 0)
        self.current_label.text = f"current=({cfg.current.x:.1f},{cfg.current.y:.1f},{cfg.current.z:.1f})"
        self.update_camera()
        self.update_hud()

    def next_round(self):
        self.update_ai_after_round()
        self.reset_round(self.round_index + 1)

    def previous_round(self):
        self.reset_round(self.round_index - 1)

    def update_ai_after_round(self):
        cfg = self.cfg
        if self.round_captures < max(1, cfg.expected_captures // 2):
            self.ai_hunt = clamp(self.ai_hunt * 1.06, 0.75, 1.8)
        else:
            self.ai_hunt = clamp(self.ai_hunt * 0.98, 0.75, 1.8)
        if self.pod_spread() > cfg.desired_spacing * 1.4:
            self.ai_spacing = clamp(self.ai_spacing * 0.94, 4.5, 10.0)
        if self.avg_energy() < 72:
            self.ai_energy = clamp(self.ai_energy * 1.08, 0.80, 1.60)
        elif self.avg_energy() > 92:
            self.ai_energy = clamp(self.ai_energy * 0.98, 0.80, 1.60)
        if self.min_oxygen() > 88:
            self.ai_oxygen = clamp(self.ai_oxygen * 0.97, 0.60, 1.30)
        else:
            self.ai_oxygen = clamp(self.ai_oxygen * 1.10, 0.60, 1.30)
        if cfg.predator_pressure > 0:
            self.ai_predator = clamp(self.ai_predator * 1.02, 0.90, 1.30)

    def forced_breathing_target_depth(self, d: DolphinVisual, dt: float) -> float:
        """Return target depth for scheduled breathing cycle."""
        d.phase_time += dt
        # Offset each dolphin so transitions are visible across the pod.
        if d.phase == "neutral":
            d.phase = "forced_dive"
            d.phase_time = (d.idx - 1) * 0.18
            self.round_forced_dive_transitions += 1

        if d.phase == "forced_dive":
            if d.phase_time > 6.2 + 0.25 * d.idx:
                d.phase = "forced_rise"
                d.phase_time = 0.0
                self.round_forced_rise_transitions += 1
            return 26.0 + 5.0 * math.sin(0.35 * self.t + d.idx)

        if d.phase == "forced_rise":
            if d.phase_time > 4.6:
                d.phase = "surface_hold"
                d.phase_time = 0.0
                self.round_surface_hold_transitions += 1
            return 1.5

        if d.phase == "surface_hold":
            if d.phase_time > 2.9:
                d.phase = "recovery_descent"
                d.phase_time = 0.0
                d.breath_cycles += 1
                self.round_cycles += 1
            return 0.8

        if d.phase == "recovery_descent":
            if d.phase_time > 3.5:
                d.phase = "forced_dive"
                d.phase_time = 0.0
                self.round_forced_dive_transitions += 1
            return 12.0 + 4.0 * math.sin(0.2 * self.t + d.idx)

        return self.cfg.target_depth

    def compute_forces(self, d: DolphinVisual, neighbors: List[DolphinVisual]) -> vector:
        cfg = self.cfg
        pod_c = self.pod_center()
        to_prey = self.prey.center - d.pos
        prey_dir = safe_norm(to_prey, safe_norm(d.vel))
        prey_dist = mag(to_prey)
        forward_dir = safe_norm(d.vel, prey_dir)
        radial_out = safe_norm(d.pos - self.prey.center, forward_dir)
        loop_side = safe_norm(cross(vector(0, 0, 1), forward_dir), vector(0, 1, 0)) * d.prey_loop_sign
        in_prey_zone = prey_dist < PREY_CAPTURE_RADIUS + 4.0 * self.prey.density
        near_prey_zone = prey_dist < (PREY_CAPTURE_RADIUS + 4.0 * self.prey.density) * 1.85
        if in_prey_zone and d.prey_transit_timer <= 0.0:
            d.prey_transit_timer = 6.8
            d.prey_loop_sign *= -1.0
            d.prey_passes += 1
        flythrough_mode = d.prey_transit_timer > 0.0
        active_target_dir = prey_dir
        if flythrough_mode:
            if d.prey_transit_timer > 3.2:
                active_target_dir = safe_norm(forward_dir * 0.95 + radial_out * 1.35 + loop_side * 0.22, forward_dir)
            else:
                active_target_dir = safe_norm(forward_dir * 0.45 + radial_out * 0.35 + loop_side * 1.20 + prey_dir * 0.12, forward_dir)
        depth_target = cfg.target_depth

        if cfg.behavior == "forced_breathing":
            depth_target = self.forced_breathing_target_depth(d, DT)
        elif cfg.behavior == "calf_protection" and d.role != "CALF":
            calf = self.dolphins[-1]
            to_calf = calf.pos - d.pos
            if mag(to_calf) > cfg.desired_spacing * 1.4:
                active_target_dir = safe_norm(active_target_dir * 0.35 + safe_norm(to_calf) * 0.65)
        elif cfg.behavior == "predator_avoidance" and cfg.predator_pressure > 0:
            away = safe_norm(d.pos - self.predator.pos, safe_norm(d.vel))
            active_target_dir = safe_norm(active_target_dir * 0.15 + away * 0.85)

        # Prey force differs by behavior.
        prey_weight = 0.35
        if cfg.behavior == "prey_pursuit":
            prey_weight = 1.35 * self.ai_hunt
        elif cfg.behavior == "spiral_hunt":
            prey_weight = 1.05 * self.ai_hunt
        elif cfg.behavior == "mixed_adaptive":
            prey_weight = 1.15 * self.ai_hunt
        elif cfg.behavior == "current_surfing":
            prey_weight = 0.55
        elif cfg.behavior == "forced_breathing":
            prey_weight = 0.55
        elif cfg.behavior == "predator_avoidance":
            prey_weight = 0.10
        elif cfg.behavior == "calf_protection":
            prey_weight = 0.42

        # Stronger steering if moving away from the active target.
        moving_away = dot(safe_norm(d.vel), active_target_dir) < -0.15
        if moving_away and cfg.behavior in ("prey_pursuit", "spiral_hunt", "mixed_adaptive", "forced_breathing"):
            prey_weight *= 1.55

        force = vector(0, 0, 0)
        force += active_target_dir * prey_weight * cfg.turn_control

        # Spiral force: outside the prey circle, orbit and compress. Inside/near it,
        # convert the spiral to an outward fly-through arc so dolphins do not loiter.
        if cfg.spiral > 0:
            radial_in = safe_norm(horizontal(self.prey.center - d.pos), vector(1, 0, 0))
            tangent = safe_norm(cross(vector(0, 0, 1), radial_in), vector(0, 1, 0))
            if flythrough_mode or near_prey_zone:
                force += radial_out * (0.85 * cfg.spiral) + tangent * (0.55 * cfg.spiral)
            else:
                force += radial_in * (0.55 * cfg.spiral) + tangent * (0.80 * cfg.spiral)

        # Current-assisted surfing: align with current but keep steering correction.
        if cfg.behavior == "current_surfing":
            force += safe_norm(cfg.current, vector(1, 0, 0)) * 0.65

        # Cohesion and separation.
        for other in neighbors:
            delta = other.pos - d.pos
            dist = mag(delta)
            if dist < 1e-6:
                continue
            desired = cfg.desired_spacing * (0.75 if cfg.behavior in ("spiral_hunt", "current_surfing", "mixed_adaptive") else 1.0)
            if dist > desired:
                force += safe_norm(delta) * min(0.55, 0.035 * (dist - desired))
            if dist < desired * 0.62:
                force -= safe_norm(delta) * min(0.95, 0.10 * (desired * 0.62 - dist))

        # Pod center cohesion.
        force += safe_norm(pod_c - d.pos, safe_norm(d.vel)) * 0.10

        # Vertical target depth control.
        target_z = depth_to_z(depth_target)
        vertical_error = target_z - d.pos.z
        force += vector(0, 0, clamp(vertical_error * 0.055, -1.2, 1.2))

        # Surface oxygen override remains available outside the forced cycle.
        if d.oxygen < 38 and cfg.behavior != "forced_breathing":
            force += vector(0, 0, 1.7)
            d.phase = "oxygen_surface"

        # Predator repulsion.
        if cfg.predator_pressure > 0:
            away = d.pos - self.predator.pos
            dist = mag(away)
            if dist < 38:
                force += safe_norm(away, safe_norm(d.vel)) * (cfg.predator_pressure * (38 - dist) / 9.0)

        # Drag and thrust scaling.
        speed = mag(d.vel)
        drag = -safe_norm(d.vel, vector(1, 0, 0)) * (0.035 * speed * speed)
        thrust = safe_norm(force, safe_norm(d.vel)) * (0.70 * cfg.thrust_scale / self.ai_energy)
        return force + thrust + cfg.current * 0.18 + drag

    def update_dolphin_resources(self, d: DolphinVisual, speed: float):
        cfg = self.cfg
        oxygen_cost = 0.011 + 0.010 * speed
        energy_cost = 0.006 + 0.012 * speed * speed
        if cfg.behavior == "forced_breathing":
            if d.phase in ("forced_dive", "forced_rise"):
                oxygen_cost *= 1.45
                energy_cost *= 1.15
            if d.phase == "surface_hold" or d.pos.z > -1.3:
                if d.phase == "surface_hold":
                    self.round_surface_events += 1
                    d.surface_events += 1
                d.oxygen += 0.85
                energy_cost *= 0.55
        else:
            if d.pos.z > -1.2:
                d.oxygen += 0.45
                self.round_surface_events += 1
                d.surface_events += 1
        if d.prey_transit_timer > 0.0:
            d.prey_transit_timer = max(0.0, d.prey_transit_timer - DT)
        d.oxygen = clamp(d.oxygen - oxygen_cost, 0, 100)
        d.energy = clamp(d.energy - energy_cost, 0, 100)

    def capture_logic(self, d: DolphinVisual):
        cfg = self.cfg
        dist = mag(d.pos - self.prey.center)
        if dist < PREY_CAPTURE_RADIUS + 4.0 * self.prey.density:
            # Captures happen during pass-through contact. The force model then steers
            # the dolphin out and around for another approach instead of circling here.
            base_prob = 0.008
            if cfg.behavior == "prey_pursuit":
                base_prob = 0.035
            elif cfg.behavior == "spiral_hunt":
                base_prob = 0.030
            elif cfg.behavior == "mixed_adaptive":
                base_prob = 0.038
            elif cfg.behavior == "calf_protection":
                base_prob = 0.026
            elif cfg.behavior == "current_surfing":
                base_prob = 0.014
            elif cfg.behavior == "forced_breathing":
                base_prob = 0.018
            elif cfg.behavior == "cruise":
                base_prob = 0.010
            if self.round_captures < max(0, cfg.expected_captures + 2) and random.random() < base_prob:
                d.captures += 1
                self.round_captures += 1
                d.energy = clamp(d.energy + 2.4, 0, 100)
                self.prey.reduce_density(0.035)
                sphere(pos=d.pos + vector(0, 0, 1.2), radius=0.35, color=vector(1.0, 0.85, 0.1), emissive=True)

    def followed_dolphin(self) -> DolphinVisual:
        return self.dolphins[self.follow_index % len(self.dolphins)]

    def update_camera(self):
        if not self.follow_camera:
            return
        if self.follow_mode == "pod":
            target_center = self.pod_center()
            scene.center = scene.center * 0.84 + (target_center + vector(8, 0, -2)) * 0.16
            desired_forward = safe_norm(vector(-0.80, -0.40, -0.35))
            scene.forward = scene.forward * 0.88 + desired_forward * 0.12
            return

        target = self.followed_dolphin()
        forward = safe_norm(target.vel, vector(1, 0, 0))
        # Chase camera: behind and above the selected dolphin, looking ahead.
        chase_offset = -forward * 14 + vector(0, 0, 5.5)
        focus_point = target.pos + forward * 6 + vector(0, 0, 1.2)
        desired_center = target.pos + forward * 3 + vector(0, 0, 0.8)
        scene.center = scene.center * 0.78 + desired_center * 0.22
        try:
            scene.camera.pos = scene.camera.pos * 0.74 + (target.pos + chase_offset) * 0.26
            scene.camera.axis = focus_point - scene.camera.pos
        except Exception:
            # Fallback for environments with limited camera support.
            scene.forward = scene.forward * 0.80 + safe_norm(scene.center - (target.pos + chase_offset), vector(-1, 0, -0.2)) * 0.20

    def step_simulation(self):
        cfg = self.cfg
        self.t += DT
        self.step += 1
        pod_c = self.pod_center()
        self.prey.update(DT, cfg, self.t)
        self.predator.update(DT, pod_c, cfg)

        for d in self.dolphins:
            neighbors = [o for o in self.dolphins if o is not d]
            acc = self.compute_forces(d, neighbors)
            d.vel += acc * DT
            cap = cfg.speed_cap * d.max_speed_multiplier
            d.vel = limit_vector(d.vel, cap)
            d.pos += d.vel * DT
            d.pos.z = clamp(d.pos.z, BOTTOM_Z + 3, SURFACE_Z - 0.08)
            speed = mag(d.vel)
            self.update_dolphin_resources(d, speed)
            self.capture_logic(d)
            d.set_state(d.pos, d.vel)

        self.update_camera()

        self.update_hud()
        if self.step >= STEPS_PER_ROUND:
            self.next_round()

    def behavior_text(self) -> str:
        behavior_map: Dict[str, str] = {
            "cruise": "steady cruise with alignment and separation control",
            "forced_breathing": "forced dive-rise-surface-hold breathing cycle",
            "prey_pursuit": "active prey pursuit",
            "spiral_hunt": "fly-through spiral: pass through prey, exit, arc around, re-enter",
            "current_surfing": "current-assisted gliding",
            "predator_avoidance": "tight evasive formation",
            "calf_protection": "adults escorting calf near pod center",
            "mixed_adaptive": "blended hunting, current use, and formation control",
        }
        return behavior_map.get(self.cfg.behavior, self.cfg.behavior)

    def update_hud(self):
        cfg = self.cfg
        prey_distance = mag(self.pod_center() - self.prey.center)
        predator_line = ""
        if cfg.predator_pressure > 0:
            predator_line = f"\npredator pressure={cfg.predator_pressure:.2f} | predator distance={mag(self.predator.pos - self.pod_center()):.1f}m"
        transit_count = sum(1 for d in self.dolphins if d.prey_transit_timer > 0.0)
        flythrough_line = f"\nprey fly-through: {transit_count} dolphin(s) exiting/looping outside circle" if transit_count else ""
        breathing_line = ""
        if cfg.behavior == "forced_breathing":
            breathing_line = (
                f"\nforced breathing: surface events={self.round_surface_events} | "
                f"cycles={self.round_cycles} | dive={self.round_forced_dive_transitions} | "
                f"rise={self.round_forced_rise_transitions} | hold={self.round_surface_hold_transitions}"
            )
        follow_line = "off"
        if self.follow_camera:
            if self.follow_mode == "pod":
                follow_line = "pod"
            else:
                target = self.followed_dolphin()
                follow_line = f"dolphin {target.idx} [{target.role}]"
        self.hud.text = (
            f"ROUND {self.round_index + 1}/8: {cfg.name}\n"
            f"behavior: {self.behavior_text()}\n"
            f"note: {cfg.note}\n"
            f"t={self.t:5.1f}s step={self.step:03d} | pod speed={self.average_speed():4.2f} m/s | spread={self.pod_spread():4.2f}m\n"
            f"min O2={self.min_oxygen():4.1f}% | avg energy={self.avg_energy():4.1f}% | captures={self.round_captures}\n"
            f"prey distance={prey_distance:5.1f}m | prey density={self.prey.density:4.2f} | target depth={cfg.target_depth:4.1f}m\n"
            f"AI spacing={self.ai_spacing:3.1f} | hunt={self.ai_hunt:3.2f} | oxygen={self.ai_oxygen:3.2f} | energy={self.ai_energy:3.2f} | camera={follow_line}{predator_line}{flythrough_line}{breathing_line}"
        )
        if self.show_help:
            self.help_label.visible = True
            self.help_label.text = (
                "Terminal-output behavior map\n"
                "1 cruise: efficient travel + incidental prey\n"
                "2 forced breathing: dive → rise → hold → descend\n"
                "3 pursuit: direct prey chasing\n"
                "4 spiral: circular prey compression\n"
                "5 current: gliding with flow\n"
                "6 predator: evasive burst\n"
                "7 calf: protective escort\n"
                "8 mixed: adaptive foraging\n\n"
                "Controls: SPACE pause | n next | b back | r reset | f follow on/off | c next dolphin | p follow pod | h help"
            )
        else:
            self.help_label.visible = False


def handle_key(evt):
    key = evt.key
    if key == " ":
        sim.paused = not sim.paused
    elif key == "n":
        sim.next_round()
    elif key == "b":
        sim.previous_round()
    elif key == "r":
        sim.reset_round(sim.round_index)
    elif key == "f":
        sim.follow_camera = not sim.follow_camera
        if sim.follow_camera and sim.follow_mode not in ("dolphin", "pod"):
            sim.follow_mode = "dolphin"
        sim.update_hud()
    elif key == "c":
        sim.follow_mode = "dolphin"
        sim.follow_camera = True
        sim.follow_index = (sim.follow_index + 1) % len(sim.dolphins)
        sim.update_hud()
    elif key == "p":
        sim.follow_mode = "pod"
        sim.follow_camera = True
        sim.update_hud()
    elif key == "h":
        sim.show_help = not sim.show_help
        sim.update_hud()


sim = DolphinSimulation()
scene.bind("keydown", handle_key)

while True:
    rate(60)
    if not sim.paused:
        sim.step_simulation()
