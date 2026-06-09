#!/usr/bin/env python3
"""
Terminal-only scientific/mathematical dolphin life and ocean migration model.

This is the non-visual counterpart of the VPython ocean migration map. It models
three pods crossing an XYZ ocean corridor, switching through four behavior rounds,
random environmental regimes, breathing, energy, prey gains, predator mortality,
aging, reproduction, deaths, and ocean-wrap respawns.

Run:
    python terminal_dolphin_life_ocean_migration_scientific.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
from typing import List, Optional


@dataclass
class Vec3:
    x: float
    y: float
    z: float
    def __add__(self, o): return Vec3(self.x + o.x, self.y + o.y, self.z + o.z)
    def __sub__(self, o): return Vec3(self.x - o.x, self.y - o.y, self.z - o.z)
    def __mul__(self, k: float): return Vec3(self.x*k, self.y*k, self.z*k)
    __rmul__ = __mul__
    def __truediv__(self, k: float): return Vec3(self.x/k, self.y/k, self.z/k)
    def mag(self): return math.sqrt(self.x*self.x + self.y*self.y + self.z*self.z)
    def norm(self):
        m = self.mag()
        return Vec3(0, 0, 0) if m < 1e-9 else self / m


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def rand_vec(a, b):
    return Vec3(random.uniform(a, b), random.uniform(a, b), random.uniform(a, b))


@dataclass
class Environment:
    name: str
    current: Vec3
    prey_center: Vec3
    prey_radius: float
    predator_center: Vec3
    predator_radius: float
    reef_y: float
    reef_half_width: float
    preferred_depth: float
    visibility: float
    food_density: float
    turbulence: float
    temperature: float
    calf_safety: float

    @staticmethod
    def random_environment(index: int):
        names = ["Warm Shelf Current", "Cold Bluewater Channel", "Reef-Edge Feeding Ground", "Storm-Mixed Open Ocean", "Predator Shadow Passage", "Clear Pelagic Corridor", "Deep Nutrient Upwelling"]
        return Environment(
            name=random.choice(names) + f" #{index}",
            current=Vec3(random.uniform(0.18, 0.95), random.uniform(-0.24, 0.24), random.uniform(-0.04, 0.06)),
            prey_center=Vec3(random.uniform(-220, 620), random.uniform(-150, 150), random.uniform(-75, -15)),
            prey_radius=random.uniform(85, 180),
            predator_center=Vec3(random.uniform(-60, 760), random.uniform(-200, 200), random.uniform(-100, -18)),
            predator_radius=random.uniform(120, 240),
            reef_y=random.uniform(-210, 210),
            reef_half_width=random.uniform(45, 95),
            preferred_depth=random.uniform(-85, -25),
            visibility=random.uniform(0.35, 1.0),
            food_density=random.uniform(0.35, 1.0),
            turbulence=random.uniform(0.02, 0.18),
            temperature=random.uniform(14.0, 27.0),
            calf_safety=random.uniform(0.35, 1.0),
        )


@dataclass
class Dolphin:
    ident: int
    pod_id: int
    pos: Vec3
    vel: Vec3
    age: float
    energy: float
    oxygen: float
    sex: str
    sociality: float
    boldness: float
    is_leader: bool = False
    calf_until: float = 2.0
    alive: bool = True
    breath_phase: float = field(default_factory=lambda: random.random() * math.tau)
    mother_id: Optional[int] = None

    @property
    def is_calf(self): return self.age < self.calf_until
    @property
    def age_class(self):
        if self.age < 2: return "calf"
        if self.age < 8: return "juvenile"
        if self.age < 35: return "adult"
        return "elder"


@dataclass
class Pod:
    pod_id: int
    dolphins: List[Dolphin]
    lane_y: float
    wraps: int = 0
    births: int = 0
    deaths: int = 0

    def living(self): return [d for d in self.dolphins if d.alive]
    def centroid(self):
        living = self.living()
        if not living: return Vec3(0, 0, -40)
        c = Vec3(0, 0, 0)
        for d in living: c += d.pos
        return c / len(living)
    def mean_speed(self):
        living = self.living()
        return 0.0 if not living else sum(d.vel.mag() for d in living) / len(living)
    def leader(self):
        living = self.living()
        leaders = [d for d in living if d.is_leader]
        if leaders: return max(leaders, key=lambda d: d.energy + d.boldness * 20)
        adults = [d for d in living if d.age_class == "adult"]
        if adults:
            chosen = max(adults, key=lambda d: d.energy + d.boldness * 10)
            chosen.is_leader = True
            return chosen
        return living[0] if living else None


class MigrationLifeSimulation:
    def __init__(self, seed: int = 21):
        random.seed(seed)
        self.dt = 0.42
        self.step_index = 0
        self.round_duration = 70
        self.ocean_start_x = -520.0
        self.ocean_end_x = 980.0
        self.surface_z = 0.0
        self.max_depth_z = -160.0
        self.next_ident = 0
        self.environment_index = 1
        self.environment = Environment.random_environment(self.environment_index)
        self.pods = self.make_initial_pods()

    def new_ident(self):
        self.next_ident += 1
        return self.next_ident

    def current_round(self):
        return [1, 2, 3, 4][(self.step_index // self.round_duration) % 4]

    def round_name(self, rid):
        return {1: "Current-Riding Drift", 2: "Prey Corridor Crossing", 3: "Predator-Avoidance Spread", 4: "Regrouping Social Spiral"}[rid]

    def make_initial_pods(self):
        pods = []
        for pod_id, lane in enumerate([-110.0, 0.0, 115.0]):
            dolphins = []
            for i in range(random.randint(7, 11)):
                age = random.choice([random.uniform(0.4, 1.8), random.uniform(3, 7), random.uniform(8, 30), random.uniform(31, 43)])
                dolphins.append(Dolphin(
                    ident=self.new_ident(), pod_id=pod_id,
                    pos=Vec3(self.ocean_start_x + random.uniform(0, 80), lane + random.uniform(-26, 26), random.uniform(-80, -18)),
                    vel=Vec3(random.uniform(1.0, 2.2), random.uniform(-0.15, 0.15), random.uniform(-0.04, 0.04)),
                    age=age, energy=random.uniform(58, 96), oxygen=random.uniform(62, 100),
                    sex=random.choice(["F", "M"]), sociality=random.uniform(0.45, 1.0), boldness=random.uniform(0.2, 1.0), is_leader=(i == 0)
                ))
            pods.append(Pod(pod_id, dolphins, lane))
        return pods

    def behavior_force(self, d: Dolphin, pod: Pod, rid: int):
        env = self.environment
        centroid = pod.centroid()
        leader = pod.leader()
        desired_depth = env.preferred_depth
        if rid == 1:
            return Vec3(1, 0, 0) * 0.75 + env.current * 1.15 + Vec3(0, (pod.lane_y - d.pos.y) * 0.004, (desired_depth - d.pos.z) * 0.010)
        if rid == 2:
            prey_vec = env.prey_center - d.pos
            return Vec3(1.25, prey_vec.y * 0.004, prey_vec.z * 0.005) + prey_vec.norm() * (0.42 * env.food_density)
        if rid == 3:
            away = (d.pos - env.predator_center).norm()
            risk_distance = (d.pos - env.predator_center).mag()
            risk_gain = max(0.0, (env.predator_radius - risk_distance) / max(1.0, env.predator_radius))
            side = 1.0 if d.ident % 2 == 0 else -1.0
            return Vec3(1.20 + 0.4 * d.boldness, side * (0.26 + 1.2 * risk_gain), away.z * (0.8 + risk_gain))
        theta = 0.035 * self.step_index + d.ident * 0.75
        spiral = Vec3(0.6, math.sin(theta) * 0.34, math.cos(theta) * 0.20)
        leader_pull = Vec3(0, 0, 0)
        if leader and leader.ident != d.ident:
            leader_pull = (leader.pos - d.pos).norm() * (0.45 * d.sociality)
        center_pull = (centroid - d.pos).norm() * (0.65 * d.sociality)
        return spiral + leader_pull + center_pull

    def social_force(self, d: Dolphin, pod: Pod):
        cohesion = Vec3(0, 0, 0); separation = Vec3(0, 0, 0); alignment = Vec3(0, 0, 0); calf_guard = Vec3(0, 0, 0)
        neighbors = 0
        for other in pod.living():
            if other.ident == d.ident: continue
            diff = other.pos - d.pos
            dist = diff.mag()
            if dist < 1e-6: continue
            if dist < 130:
                cohesion += diff
                alignment += other.vel
                neighbors += 1
            if dist < 22:
                separation += (d.pos - other.pos).norm() * ((22 - dist) * 0.05)
            if other.is_calf and not d.is_calf and dist < 95:
                calf_guard += diff.norm() * (0.22 * self.environment.calf_safety)
        if neighbors:
            cohesion = cohesion.norm() * (0.30 * d.sociality)
            alignment = alignment.norm() * (0.22 * d.sociality)
        return cohesion + separation + alignment + calf_guard

    def breathing_force(self, d: Dolphin):
        d.breath_phase += self.dt * (0.040 + 0.015 * random.random())
        periodic_need = 0.5 + 0.5 * math.sin(d.breath_phase)
        oxygen_need = max(0.0, (45.0 - d.oxygen) / 45.0)
        if periodic_need > 0.88 or oxygen_need > 0.25:
            return Vec3(0, 0, 1.2 + 2.4 * oxygen_need)
        if d.oxygen > 72 and d.pos.z > self.environment.preferred_depth:
            return Vec3(0, 0, -0.20)
        return Vec3(0, 0, 0)

    def update_life(self, d: Dolphin, pod: Pod):
        env = self.environment
        d.age += self.dt / 5200.0
        speed = d.vel.mag()
        d.energy -= self.dt * (0.015 + 0.012 * speed + 0.010 * env.turbulence)
        d.oxygen -= self.dt * (0.12 + 0.030 * speed)
        if d.pos.z > -4:
            d.oxygen = min(100.0, d.oxygen + 7.0 * self.dt)
            d.energy = min(100.0, d.energy + 0.22 * self.dt)
        if (d.pos - env.prey_center).mag() < env.prey_radius:
            d.energy = min(100.0, d.energy + env.food_density * 0.45 * self.dt)
        predator_risk = (d.pos - env.predator_center).mag() < env.predator_radius
        mortality = 0.00002 + max(0.0, d.age - 38.0) * 0.00002
        if d.energy < 8: mortality += 0.004
        if d.oxygen <= 0: mortality += 0.02
        if predator_risk: mortality += 0.0009 * (1.4 - env.visibility)
        if d.is_calf: mortality *= 1.45
        if random.random() < mortality:
            d.alive = False
            pod.deaths += 1

    def reproduction(self, pod: Pod):
        living = pod.living()
        adult_f = [d for d in living if d.sex == "F" and 8 < d.age < 32 and d.energy > 72]
        adult_m = [d for d in living if d.sex == "M" and 8 < d.age < 35 and d.energy > 65]
        if not adult_f or not adult_m or len(living) > 20:
            return
        p_birth = 0.0022 * self.environment.food_density * self.environment.calf_safety
        if random.random() < p_birth:
            mother = random.choice(adult_f)
            calf = Dolphin(self.new_ident(), pod.pod_id, mother.pos + Vec3(random.uniform(-8, 8), random.uniform(-6, 6), random.uniform(-2, 2)), mother.vel, 0.0, 62.0, 88.0, random.choice(["F", "M"]), random.uniform(0.75, 1.0), random.uniform(0.10, 0.55), mother_id=mother.ident)
            mother.energy -= 12.0
            pod.dolphins.append(calf)
            pod.births += 1

    def wrap_pod_if_needed(self, pod: Pod):
        c = pod.centroid()
        if c.x <= self.ocean_end_x:
            return False
        dx = self.ocean_start_x - c.x + random.uniform(10, 45)
        dy = random.uniform(-25, 25)
        for d in pod.living():
            d.pos = Vec3(d.pos.x + dx, clamp(d.pos.y + dy, -260, 260), clamp(d.pos.z + random.uniform(-8, 8), self.max_depth_z, self.surface_z))
            d.vel = Vec3(max(0.8, d.vel.x), d.vel.y * 0.4, d.vel.z * 0.4)
        pod.wraps += 1
        self.environment_index += 1
        self.environment = Environment.random_environment(self.environment_index)
        print(f"ENVIRONMENT WRAP: pod {pod.pod_id} reached x>{self.ocean_end_x:.0f}; new environment = {self.environment.name}")
        return True

    def step(self):
        rid = self.current_round()
        for pod in self.pods:
            for d in list(pod.living()):
                force = self.behavior_force(d, pod, rid) + self.social_force(d, pod) + self.breathing_force(d)
                force += Vec3(random.uniform(-1, 1) * self.environment.turbulence, random.uniform(-1, 1) * self.environment.turbulence, random.uniform(-1, 1) * self.environment.turbulence * 0.45)
                d.vel = d.vel * 0.88 + force * self.dt
                max_speed = 4.6 if rid == 3 else 3.8
                if d.vel.mag() > max_speed:
                    d.vel = d.vel.norm() * max_speed
                d.pos += d.vel * self.dt * 4.5
                d.pos = Vec3(d.pos.x, clamp(d.pos.y, -290, 290), clamp(d.pos.z, self.max_depth_z, self.surface_z))
                self.update_life(d, pod)
            pod.dolphins = [d for d in pod.dolphins if d.alive]
            self.reproduction(pod)
            self.wrap_pod_if_needed(pod)
        self.step_index += 1

    def status_line(self):
        rid = self.current_round()
        total = sum(len(p.living()) for p in self.pods)
        calves = sum(1 for p in self.pods for d in p.living() if d.is_calf)
        surfacing = sum(1 for p in self.pods for d in p.living() if d.pos.z > -5)
        pod_summary = " | ".join(f"pod{p.pod_id}:n={len(p.living())},wrap={p.wraps},birth={p.births},death={p.deaths},speed={p.mean_speed():.1f},x={p.centroid().x:.0f}" for p in self.pods)
        return (f"step={self.step_index:04d} R{rid} {self.round_name(rid):<26} env={self.environment.name:<30} total={total:2d} calves={calves:2d} surfacing={surfacing:2d} "
                f"food={self.environment.food_density:.2f} vis={self.environment.visibility:.2f} temp={self.environment.temperature:4.1f}C | {pod_summary}")

    def run(self, steps: int = 840, print_every: int = 20):
        print("Dolphin Life Ocean Migration Terminal Scientific Simulation")
        print("Axes: x=migration distance, y=ocean lane, z=depth. Rounds: current drift, prey crossing, predator spread, social regrouping.")
        print("Life terms: oxygen, energy, aging, mortality, calf safety, reproduction, prey feeding, and environment wraps.")
        print("-" * 180)
        for _ in range(steps):
            if self.step_index % print_every == 0:
                print(self.status_line())
            self.step()
        print("-" * 180)
        print(self.status_line())
        print("FINAL POD SUMMARY")
        for p in self.pods:
            living = p.living()
            avg_energy = sum(d.energy for d in living)/len(living) if living else 0.0
            avg_oxygen = sum(d.oxygen for d in living)/len(living) if living else 0.0
            print(f"pod {p.pod_id}: living={len(living)}, wraps={p.wraps}, births={p.births}, deaths={p.deaths}, mean_speed={p.mean_speed():.2f}, mean_energy={avg_energy:.1f}, mean_oxygen={avg_oxygen:.1f}")


if __name__ == "__main__":
    MigrationLifeSimulation().run()
