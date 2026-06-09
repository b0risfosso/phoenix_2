#!/usr/bin/env python3
"""
Terminal-only scientific/mathematical dolphin pod hydrodynamics simulation.

This is the non-visual counterpart of the forced-breathing VPython pod scene.
It simulates six dolphins, prey transit/fly-through behavior, eight round modes,
oxygen and energy budgets, current forcing, depth regulation, predator pressure,
calf escorting, and forced dive-rise-surface-hold breathing cycles.

Run:
    python terminal_dolphin_forced_breathing_pod_scientific.py
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import List, Optional


@dataclass
class Vec3:
    x: float
    y: float
    z: float

    def __add__(self, other): return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)
    def __sub__(self, other): return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)
    def __mul__(self, k: float): return Vec3(self.x * k, self.y * k, self.z * k)
    __rmul__ = __mul__
    def __truediv__(self, k: float): return Vec3(self.x / k, self.y / k, self.z / k)
    def mag(self) -> float: return math.sqrt(self.x*self.x + self.y*self.y + self.z*self.z)
    def norm(self, fallback=None):
        m = self.mag()
        if m < 1e-9:
            return fallback if fallback is not None else Vec3(1, 0, 0)
        return self / m
    def dot(self, other) -> float: return self.x*other.x + self.y*other.y + self.z*other.z
    def cross(self, other):
        return Vec3(self.y*other.z - self.z*other.y, self.z*other.x - self.x*other.z, self.x*other.y - self.y*other.x)


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def limit(v: Vec3, max_mag: float) -> Vec3:
    m = v.mag()
    return v * (max_mag / m) if m > max_mag and m > 1e-9 else v


def depth_to_z(depth: float) -> float:
    return -abs(depth)


def z_to_depth(z: float) -> float:
    return max(0.0, -z)


def horizontal(v: Vec3) -> Vec3:
    return Vec3(v.x, v.y, 0)


@dataclass
class RoundConfig:
    name: str
    behavior: str
    note: str
    current: Vec3
    target_depth: float
    thrust_scale: float
    spiral: float
    predator_pressure: float
    prey_density: float
    desired_spacing: float
    speed_cap: float
    turn_control: float
    expected_captures: int
    initial_prey_distance: float


ROUND_CONFIGS = [
    RoundConfig("Low-Drag Cruise", "cruise", "efficient alignment, low drag, and steady depth control", Vec3(0.7, -0.2, 0.08), 20.7, 1.00, 0.00, 0.00, 0.65, 8.0, 4.0, 0.85, 2, 59.1),
    RoundConfig("Forced Dive-Rise Breathing Cycle", "forced_breathing", "scheduled dive, rise, surface hold, and recovery descent", Vec3(0.4, -0.3, 0.02), 26.0, 1.03, 0.00, 0.00, 0.62, 8.0, 4.2, 0.90, 4, 49.6),
    RoundConfig("Active Prey Pursuit", "prey_pursuit", "direct chase with stronger steering toward prey", Vec3(0.5, 0.0, 0.0), 16.9, 1.22, 0.00, 0.00, 0.56, 7.5, 4.6, 1.20, 9, 37.9),
    RoundConfig("Spiral Compression Hunt", "spiral_hunt", "inward force plus tangential orbiting compresses prey", Vec3(0.5, 0.5, 0.0), 22.0, 1.10, 1.35, 0.00, 0.77, 7.1, 4.4, 1.10, 7, 35.8),
    RoundConfig("Current-Assisted Surfing", "current_surfing", "current alignment and glide intervals reduce thrust", Vec3(0.6, -0.2, 0.0), 31.2, 0.79, 0.00, 0.00, 0.61, 6.6, 4.2, 0.95, 3, 46.0),
    RoundConfig("Predator Evasion", "predator_avoidance", "threat pressure causes burst speed and defensive grouping", Vec3(0.6, -0.3, -0.05), 16.7, 1.31, 0.00, 0.73, 0.63, 6.2, 5.4, 1.15, 0, 40.1),
    RoundConfig("Calf Escort Formation", "calf_protection", "adults form a moving shell around the calf", Vec3(0.9, 0.5, -0.05), 25.8, 0.84, 0.00, 0.00, 0.46, 5.9, 4.3, 1.00, 8, 40.8),
    RoundConfig("Mixed Adaptive Foraging", "mixed_adaptive", "blended hunting, current use, and formation control", Vec3(0.9, 0.4, 0.02), 22.6, 0.83, 0.59, 0.30, 0.75, 5.5, 4.6, 1.20, 10, 35.7),
]


@dataclass
class Dolphin:
    idx: int
    role: str
    pos: Vec3
    vel: Vec3
    oxygen: float = 100.0
    energy: float = 100.0
    captures: int = 0
    prey_transit_timer: float = 0.0
    prey_loop_sign: float = 1.0
    phase: str = "neutral"
    phase_time: float = 0.0
    breath_cycles: int = 0

    @property
    def max_speed_multiplier(self):
        return 0.76 if self.role == "CALF" else 1.0


class PodSimulation:
    def __init__(self, seed: int = 7):
        random.seed(seed)
        self.dt = 0.045
        self.steps_per_round = 860
        self.capture_radius = 5.2
        self.bottom_z = -42.0
        self.surface_z = 0.0
        self.ai_hunt = 1.0
        self.ai_energy = 1.0
        roles = ["LEAD", "ADULT", "ADULT", "ADULT", "ADULT", "CALF"]
        starts = [Vec3(6.5, -0.4, -13.1), Vec3(-5.0, -1.9, -8.9), Vec3(2.5, 1.2, -9.6), Vec3(-5.9, -0.4, -12.4), Vec3(0.1, -2.1, -18.0), Vec3(-2.2, -2.7, -10.9)]
        self.dolphins = [Dolphin(i + 1, roles[i], starts[i], Vec3(1.3, 0.05 * (i+1), 0.0), prey_loop_sign=1.0 if i % 2 else -1.0) for i in range(6)]
        self.prey_center = Vec3(55, -8, -20)
        self.prey_density = 0.7
        self.predator_pos = Vec3(-50, 10, -14)
        self.predator_vel = Vec3(0.6, -0.1, 0.0)
        self.t = 0.0
        self.round_captures = 0
        self.surface_events = 0
        self.round_cycles = 0

    def pod_center(self):
        c = Vec3(0, 0, 0)
        for d in self.dolphins:
            c += d.pos
        return c / len(self.dolphins)

    def pod_spread(self):
        c = self.pod_center()
        return sum((d.pos - c).mag() for d in self.dolphins) / len(self.dolphins)

    def average_speed(self):
        return sum(d.vel.mag() for d in self.dolphins) / len(self.dolphins)

    def min_oxygen(self):
        return min(d.oxygen for d in self.dolphins)

    def avg_energy(self):
        return sum(d.energy for d in self.dolphins) / len(self.dolphins)

    def reset_round(self, cfg: RoundConfig):
        self.t = 0.0
        self.round_captures = 0
        self.surface_events = 0
        self.round_cycles = 0
        c = self.pod_center()
        for i, d in enumerate(self.dolphins):
            d.pos = Vec3(c.x * 0.35 + (i - 2.5) * 2.2, c.y * 0.35 + ((i % 2) - 0.5) * 3.5, depth_to_z(cfg.target_depth * random.uniform(0.65, 0.95)))
            d.vel = d.vel * 0.3
            d.oxygen = 100.0
            d.energy = max(58.0, d.energy + 10.0)
            d.prey_transit_timer = 0.0
            d.phase = "neutral"
            d.phase_time = 0.0
        direction = Vec3(1.0, random.uniform(-0.4, 0.4), random.uniform(-0.1, 0.1)).norm()
        self.prey_center = c + direction * cfg.initial_prey_distance
        self.prey_center.z = depth_to_z(cfg.target_depth + random.uniform(-5, 5))
        self.prey_density = cfg.prey_density
        self.predator_pos = c + Vec3(-58, 22, random.uniform(-5, 5))
        self.predator_vel = Vec3(1, -0.18, 0.02).norm() * (0.6 + 0.9 * cfg.predator_pressure)

    def forced_breathing_target_depth(self, d: Dolphin, cfg: RoundConfig):
        d.phase_time += self.dt
        if d.phase == "neutral":
            d.phase = "forced_dive"
            d.phase_time = (d.idx - 1) * 0.18
        if d.phase == "forced_dive":
            if d.phase_time > 6.2 + 0.25 * d.idx:
                d.phase, d.phase_time = "forced_rise", 0.0
            return 26.0 + 5.0 * math.sin(0.35 * self.t + d.idx)
        if d.phase == "forced_rise":
            if d.phase_time > 4.6:
                d.phase, d.phase_time = "surface_hold", 0.0
            return 1.5
        if d.phase == "surface_hold":
            if d.phase_time > 2.9:
                d.phase, d.phase_time = "recovery_descent", 0.0
                d.breath_cycles += 1
                self.round_cycles += 1
            return 0.8
        if d.phase == "recovery_descent":
            if d.phase_time > 3.5:
                d.phase, d.phase_time = "forced_dive", 0.0
            return 12.0 + 4.0 * math.sin(0.2 * self.t + d.idx)
        return cfg.target_depth

    def compute_force(self, d: Dolphin, cfg: RoundConfig) -> Vec3:
        pod_c = self.pod_center()
        to_prey = self.prey_center - d.pos
        prey_dir = to_prey.norm(d.vel.norm())
        prey_dist = to_prey.mag()
        forward = d.vel.norm(prey_dir)
        radial_out = (d.pos - self.prey_center).norm(forward)
        loop_side = Vec3(0, 0, 1).cross(forward).norm(Vec3(0, 1, 0)) * d.prey_loop_sign
        in_zone = prey_dist < self.capture_radius + 4.0 * self.prey_density
        near_zone = prey_dist < (self.capture_radius + 4.0 * self.prey_density) * 1.85
        if in_zone and d.prey_transit_timer <= 0.0:
            d.prey_transit_timer = 6.8
            d.prey_loop_sign *= -1.0
        flythrough = d.prey_transit_timer > 0.0
        active_dir = prey_dir
        if flythrough:
            if d.prey_transit_timer > 3.2:
                active_dir = (forward * 0.95 + radial_out * 1.35 + loop_side * 0.22).norm(forward)
            else:
                active_dir = (forward * 0.45 + radial_out * 0.35 + loop_side * 1.20 + prey_dir * 0.12).norm(forward)

        target_depth = cfg.target_depth
        if cfg.behavior == "forced_breathing":
            target_depth = self.forced_breathing_target_depth(d, cfg)
        elif cfg.behavior == "predator_avoidance" and cfg.predator_pressure > 0:
            active_dir = (active_dir * 0.15 + (d.pos - self.predator_pos).norm(forward) * 0.85).norm(forward)
        elif cfg.behavior == "calf_protection" and d.role != "CALF":
            calf = self.dolphins[-1]
            active_dir = (active_dir * 0.35 + (calf.pos - d.pos).norm(active_dir) * 0.65).norm(active_dir)

        prey_weight = {
            "prey_pursuit": 1.35,
            "spiral_hunt": 1.05,
            "mixed_adaptive": 1.15,
            "current_surfing": 0.55,
            "forced_breathing": 0.55,
            "predator_avoidance": 0.10,
            "calf_protection": 0.42,
            "cruise": 0.35,
        }.get(cfg.behavior, 0.35)
        force = active_dir * prey_weight * self.ai_hunt * cfg.turn_control

        if cfg.spiral > 0:
            radial_in = horizontal(self.prey_center - d.pos).norm(Vec3(1, 0, 0))
            tangent = Vec3(0, 0, 1).cross(radial_in).norm(Vec3(0, 1, 0))
            if flythrough or near_zone:
                force += radial_out * (0.85 * cfg.spiral) + tangent * (0.55 * cfg.spiral)
            else:
                force += radial_in * (0.55 * cfg.spiral) + tangent * (0.80 * cfg.spiral)

        if cfg.behavior == "current_surfing":
            force += cfg.current.norm(Vec3(1, 0, 0)) * 0.65

        for o in self.dolphins:
            if o is d: continue
            delta = o.pos - d.pos
            dist = delta.mag()
            if dist < 1e-6: continue
            desired = cfg.desired_spacing * (0.75 if cfg.behavior in ("spiral_hunt", "current_surfing", "mixed_adaptive") else 1.0)
            if dist > desired:
                force += delta.norm() * min(0.55, 0.035 * (dist - desired))
            if dist < desired * 0.62:
                force -= delta.norm() * min(0.95, 0.10 * (desired * 0.62 - dist))

        force += (pod_c - d.pos).norm(forward) * 0.10
        vertical_error = depth_to_z(target_depth) - d.pos.z
        force += Vec3(0, 0, clamp(vertical_error * 0.055, -1.2, 1.2))
        if d.oxygen < 38 and cfg.behavior != "forced_breathing":
            force += Vec3(0, 0, 1.7)
        if cfg.predator_pressure > 0:
            away = d.pos - self.predator_pos
            dist = away.mag()
            if dist < 38:
                force += away.norm(forward) * (cfg.predator_pressure * (38 - dist) / 9.0)
        drag = forward * (-0.035 * d.vel.mag() ** 2)
        thrust = force.norm(forward) * (0.70 * cfg.thrust_scale / self.ai_energy)
        return force + thrust + cfg.current * 0.18 + drag

    def update_resources_and_capture(self, d: Dolphin, cfg: RoundConfig):
        speed = d.vel.mag()
        oxygen_cost = 0.011 + 0.010 * speed
        energy_cost = 0.006 + 0.012 * speed * speed
        if cfg.behavior == "forced_breathing":
            if d.phase in ("forced_dive", "forced_rise"):
                oxygen_cost *= 1.45
                energy_cost *= 1.15
            if d.phase == "surface_hold" or d.pos.z > -1.3:
                self.surface_events += 1
                d.oxygen += 0.85
                energy_cost *= 0.55
        elif d.pos.z > -1.2:
            d.oxygen += 0.45
            self.surface_events += 1
        if d.prey_transit_timer > 0:
            d.prey_transit_timer = max(0.0, d.prey_transit_timer - self.dt)
        d.oxygen = clamp(d.oxygen - oxygen_cost, 0.0, 100.0)
        d.energy = clamp(d.energy - energy_cost, 0.0, 100.0)

        dist = (d.pos - self.prey_center).mag()
        if dist < self.capture_radius + 4.0 * self.prey_density:
            base = {"prey_pursuit": 0.035, "spiral_hunt": 0.030, "mixed_adaptive": 0.038, "calf_protection": 0.026, "current_surfing": 0.014, "forced_breathing": 0.018, "cruise": 0.010}.get(cfg.behavior, 0.008)
            if self.round_captures < cfg.expected_captures + 2 and random.random() < base:
                d.captures += 1
                self.round_captures += 1
                d.energy = clamp(d.energy + 2.4, 0, 100)
                self.prey_density = clamp(self.prey_density - 0.035, 0.12, 1.0)

    def step(self, cfg: RoundConfig):
        self.t += self.dt
        pc = self.pod_center()
        if cfg.predator_pressure > 0:
            direction = (pc - self.predator_pos).norm(self.predator_vel)
            self.predator_vel = limit(self.predator_vel * 0.97 + direction * (0.04 + 0.04 * cfg.predator_pressure), 2.4)
            self.predator_pos += self.predator_vel * self.dt
            self.predator_pos.z = clamp(self.predator_pos.z, self.bottom_z + 8, -4)
        self.prey_center += (Vec3(0.15*math.sin(self.t*0.45), 0.12*math.cos(self.t*0.37), 0.03*math.sin(self.t*0.52)) + cfg.current * 0.04) * self.dt
        self.prey_center.z = clamp(self.prey_center.z, self.bottom_z + 5, -4)
        for d in self.dolphins:
            acc = self.compute_force(d, cfg)
            d.vel = limit(d.vel + acc * self.dt, cfg.speed_cap * d.max_speed_multiplier)
            d.pos += d.vel * self.dt
            d.pos.z = clamp(d.pos.z, self.bottom_z + 3, self.surface_z - 0.08)
            self.update_resources_and_capture(d, cfg)

    def run_round(self, index: int, cfg: RoundConfig):
        self.reset_round(cfg)
        print(f"\nROUND {index + 1}/8: {cfg.name}")
        print(f"behavior={cfg.behavior} | {cfg.note}")
        print("time  speed spread minO2 energy preyDist preyDensity captures surface cycles predatorDist flythrough")
        for step_i in range(self.steps_per_round):
            self.step(cfg)
            if step_i % 120 == 0 or step_i == self.steps_per_round - 1:
                pred_dist = (self.predator_pos - self.pod_center()).mag() if cfg.predator_pressure > 0 else 0.0
                fly = sum(1 for d in self.dolphins if d.prey_transit_timer > 0)
                print(f"{self.t:5.1f} {self.average_speed():5.2f} {self.pod_spread():6.2f} {self.min_oxygen():5.1f} {self.avg_energy():6.1f} {(self.pod_center()-self.prey_center).mag():8.1f} {self.prey_density:10.2f} {self.round_captures:8d} {self.surface_events:7d} {self.round_cycles:6d} {pred_dist:11.1f} {fly:10d}")
        print(f"SUMMARY: captures={self.round_captures}, final_spread={self.pod_spread():.2f} m, mean_speed={self.average_speed():.2f} m/s, min_oxygen={self.min_oxygen():.1f}%, mean_energy={self.avg_energy():.1f}%")

    def run(self):
        print("Dolphin Pod Terminal Scientific Simulation")
        print("Model terms: force = steering + cohesion/separation/alignment + depth control + current + drag + optional predator repulsion")
        print("Prey-zone behavior: dolphins pass through, exit, arc outside, then re-enter; no persistent prey-circle loitering.")
        for i, cfg in enumerate(ROUND_CONFIGS):
            self.run_round(i, cfg)
            if self.round_captures < max(1, cfg.expected_captures // 2):
                self.ai_hunt = clamp(self.ai_hunt * 1.06, 0.75, 1.8)
            else:
                self.ai_hunt = clamp(self.ai_hunt * 0.98, 0.75, 1.8)
            if self.avg_energy() < 72:
                self.ai_energy = clamp(self.ai_energy * 1.08, 0.80, 1.60)


if __name__ == "__main__":
    PodSimulation().run()
