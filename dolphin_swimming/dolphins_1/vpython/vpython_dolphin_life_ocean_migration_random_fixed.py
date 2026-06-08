#!/usr/bin/env python3
"""
VPython visual simulation: dolphin pods migrating through changing ocean environments.

Features
- Full XYZ ocean map: x migration distance, y ocean lane, z depth.
- Rounds loop through R1-R4, with different social/migration behaviors.
- When pods reach the ocean end, they respawn at the start and the ocean randomly
  transitions into a new environment / biome.
- Dolphins periodically rise to the surface for air, then dive.
- Dolphins age, reproduce, die, and show social pod behaviors.
- Camera follows all pods or selected pod, with keyboard zoom.

Controls
space  pause/play
r      reset
1      follow all pods
2      follow pod 0
3      follow pod 1
4      follow pod 2
+ z    zoom in
- x    zoom out
n      next round
p      previous round
"""

from __future__ import annotations

import math
import random as pyrandom
from dataclasses import dataclass, field
from typing import List, Optional

from vpython import *  # VPython intentionally uses a visual global API.


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def safe_norm(v):
    if mag(v) < 1e-9:
        return vector(0, 0, 0)
    return norm(v)


def random_vec(a, b):
    return vector(pyrandom.uniform(a, b), pyrandom.uniform(a, b), pyrandom.uniform(a, b))


@dataclass
class Environment:
    name: str
    current: object
    prey_center: object
    prey_radius: float
    predator_center: object
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
        names = [
            "Warm Shelf Current",
            "Cold Bluewater Channel",
            "Reef-Edge Feeding Ground",
            "Storm-Mixed Open Ocean",
            "Predator Shadow Passage",
            "Clear Pelagic Corridor",
            "Deep Nutrient Upwelling",
        ]
        return Environment(
            name=pyrandom.choice(names) + f" #{index}",
            current=vector(pyrandom.uniform(0.18, 0.95), pyrandom.uniform(-0.24, 0.24), pyrandom.uniform(-0.04, 0.06)),
            prey_center=vector(pyrandom.uniform(-220, 620), pyrandom.uniform(-150, 150), pyrandom.uniform(-75, -15)),
            prey_radius=pyrandom.uniform(85, 180),
            predator_center=vector(pyrandom.uniform(-60, 760), pyrandom.uniform(-200, 200), pyrandom.uniform(-100, -18)),
            predator_radius=pyrandom.uniform(120, 240),
            reef_y=pyrandom.uniform(-210, 210),
            reef_half_width=pyrandom.uniform(45, 95),
            preferred_depth=pyrandom.uniform(-85, -25),
            visibility=pyrandom.uniform(0.35, 1.0),
            food_density=pyrandom.uniform(0.35, 1.0),
            turbulence=pyrandom.uniform(0.02, 0.18),
            temperature=pyrandom.uniform(14.0, 27.0),
            calf_safety=pyrandom.uniform(0.35, 1.0),
        )


@dataclass
class Dolphin:
    ident: int
    pod_id: int
    pos: object
    vel: object
    age: float
    energy: float
    oxygen: float
    sex: str
    sociality: float
    boldness: float
    is_leader: bool = False
    calf_until: float = 2.0
    alive: bool = True
    breath_phase: float = field(default_factory=lambda: pyrandom.random() * math.tau)
    mother_id: Optional[int] = None
    body: Optional[object] = None
    nose: Optional[object] = None
    fin: Optional[object] = None
    dorsal: Optional[object] = None
    tail: Optional[object] = None
    trail: Optional[object] = None
    label: Optional[object] = None

    @property
    def is_calf(self):
        return self.age < self.calf_until

    @property
    def age_class(self):
        if self.age < 2:
            return "calf"
        if self.age < 8:
            return "juvenile"
        if self.age < 35:
            return "adult"
        return "elder"


@dataclass
class Pod:
    pod_id: int
    dolphins: List[Dolphin]
    lane_y: float
    wraps: int = 0

    def living(self):
        return [d for d in self.dolphins if d.alive]

    def centroid(self):
        living = self.living()
        if not living:
            return vector(0, 0, -40)
        c = vector(0, 0, 0)
        for d in living:
            c += d.pos
        return c / len(living)

    def mean_speed(self):
        living = self.living()
        if not living:
            return 0.0
        return sum(mag(d.vel) for d in living) / len(living)

    def leader(self):
        living = self.living()
        leaders = [d for d in living if d.is_leader]
        if leaders:
            return max(leaders, key=lambda d: d.energy + d.boldness * 20)
        adults = [d for d in living if d.age_class == "adult"]
        if adults:
            chosen = max(adults, key=lambda d: d.energy + d.boldness * 10)
            chosen.is_leader = True
            return chosen
        return living[0] if living else None


class DolphinLifeOceanVisual:
    def __init__(self):
        pyrandom.seed(21)
        self.dt = 0.42
        self.step_index = 0
        self.round_duration = 70
        self.round_ids = [1, 2, 3, 4]
        self.forced_round_index = None
        self.ocean_start_x = -520.0
        self.ocean_end_x = 980.0
        self.surface_z = 0.0
        self.max_depth_z = -160.0
        self.next_ident = 0
        self.environment_index = 1
        self.environment = Environment.random_environment(self.environment_index)
        self.paused = False
        self.follow_mode = "all"
        self.zoom = 520.0
        self.min_zoom = 95.0
        self.max_zoom = 1150.0
        self.last_environment_name = self.environment.name

        self.scene = canvas(
            title="Dolphin Life Ocean Migration Map | Rounds 1-4 | Environment changes after each ocean wrap",
            width=1180,
            height=760,
            background=vector(0.78, 0.90, 1.0),
            center=vector(180, 0, -55),
            range=self.zoom,
        )
        self.scene.forward = vector(-0.58, -0.34, -0.74)
        self.scene.up = vector(0, 0, 1)
        self.scene.bind("keydown", self.on_keydown)

        self.status = wtext(text="")
        self.pods = self.make_initial_pods()
        self.map_objects = []
        self.build_static_map()
        self.build_environment_objects()

    def new_ident(self):
        self.next_ident += 1
        return self.next_ident

    def current_round(self):
        if self.forced_round_index is not None:
            return self.round_ids[self.forced_round_index % len(self.round_ids)]
        return self.round_ids[(self.step_index // self.round_duration) % len(self.round_ids)]

    def round_name(self, rid):
        return {
            1: "Current-Riding Drift",
            2: "Prey Corridor Crossing",
            3: "Predator-Avoidance Spread",
            4: "Regrouping Social Spiral",
        }[rid]

    def pod_color(self, pod_id):
        return [vector(0.05, 0.38, 0.78), vector(0.12, 0.54, 0.50), vector(0.44, 0.30, 0.72)][pod_id % 3]

    def make_initial_pods(self):
        pods = []
        for pod_id, lane in enumerate([-110.0, 0.0, 115.0]):
            dolphins = []
            for i in range(pyrandom.randint(7, 11)):
                age = pyrandom.choice([pyrandom.uniform(0.4, 1.8), pyrandom.uniform(3, 7), pyrandom.uniform(8, 30), pyrandom.uniform(31, 43)])
                d = Dolphin(
                    ident=self.new_ident(),
                    pod_id=pod_id,
                    pos=vector(self.ocean_start_x + pyrandom.uniform(0, 80), lane + pyrandom.uniform(-26, 26), pyrandom.uniform(-80, -18)),
                    vel=vector(pyrandom.uniform(1.0, 2.2), pyrandom.uniform(-0.15, 0.15), pyrandom.uniform(-0.04, 0.04)),
                    age=age,
                    energy=pyrandom.uniform(58, 96),
                    oxygen=pyrandom.uniform(62, 100),
                    sex=pyrandom.choice(["F", "M"]),
                    sociality=pyrandom.uniform(0.45, 1.0),
                    boldness=pyrandom.uniform(0.2, 1.0),
                    is_leader=(i == 0),
                )
                self.create_dolphin_visual(d)
                dolphins.append(d)
            pods.append(Pod(pod_id=pod_id, dolphins=dolphins, lane_y=lane))
        return pods

    def create_dolphin_visual(self, d):
        base = self.pod_color(d.pod_id)
        scale = 5.2 if not d.is_calf else 3.4
        if d.age_class == "elder":
            body_color = base * 0.72 + vector(0.18, 0.18, 0.18)
        elif d.is_calf:
            body_color = base * 0.72 + vector(0.22, 0.38, 0.48)
        else:
            body_color = base
        d.body = ellipsoid(pos=d.pos, length=scale * 3.0, height=scale * 0.78, width=scale * 0.82, color=body_color, shininess=0.65)
        d.nose = cone(pos=d.pos + vector(scale * 1.8, 0, 0), axis=vector(scale * 0.9, 0, 0), radius=scale * 0.27, color=body_color)
        d.fin = cone(pos=d.pos + vector(-scale * 1.65, 0, 0), axis=vector(-scale * 0.9, 0, 0), radius=scale * 0.45, color=body_color * 0.9)
        d.dorsal = pyramid(pos=d.pos + vector(-scale * 0.25, 0, scale * 0.45), size=vector(scale * 0.70, scale * 0.25, scale * 0.70), color=body_color * 0.75)
        d.tail = box(pos=d.pos + vector(-scale * 2.05, 0, 0), size=vector(scale * 0.20, scale * 1.0, scale * 0.30), color=body_color * 0.82)
        d.trail = curve(color=body_color * 0.7, radius=0.65 if not d.is_calf else 0.35)
        if d.is_leader:
            d.label = label(pos=d.pos + vector(0, 0, 11), text="leader", height=9, box=False, opacity=0, color=vector(0.1, 0.1, 0.1))

    def delete_dolphin_visual(self, d):
        for obj_name in ["body", "nose", "fin", "dorsal", "tail", "trail", "label"]:
            obj = getattr(d, obj_name, None)
            if obj is not None:
                obj.visible = False
                setattr(d, obj_name, None)

    def build_static_map(self):
        # Water volume and coordinate references.
        self.water = box(pos=vector((self.ocean_start_x + self.ocean_end_x) / 2, 0, self.max_depth_z / 2),
                         size=vector(self.ocean_end_x - self.ocean_start_x, 620, abs(self.max_depth_z)),
                         color=vector(0.18, 0.55, 0.82), opacity=0.16)
        self.surface = box(pos=vector((self.ocean_start_x + self.ocean_end_x) / 2, 0, 0.7),
                           size=vector(self.ocean_end_x - self.ocean_start_x, 620, 1.2),
                           color=vector(0.42, 0.78, 0.95), opacity=0.25)
        self.start_gate = box(pos=vector(self.ocean_start_x, 0, -70), size=vector(4, 610, 140), color=vector(0.1, 0.8, 0.35), opacity=0.27)
        self.end_gate = box(pos=vector(self.ocean_end_x, 0, -70), size=vector(4, 610, 140), color=vector(0.9, 0.25, 0.18), opacity=0.27)
        for y in [-250, -125, 0, 125, 250]:
            curve(pos=[vector(self.ocean_start_x, y, -2), vector(self.ocean_end_x, y, -2)], color=vector(0.1, 0.25, 0.45), radius=0.45)
        for z in [-40, -80, -120, -160]:
            curve(pos=[vector(self.ocean_start_x, -305, z), vector(self.ocean_end_x, -305, z)], color=vector(0.2, 0.35, 0.45), radius=0.35)
        label(pos=vector(self.ocean_start_x, -325, 9), text="START / RESPAWN", height=12, box=False, color=vector(0.05, 0.28, 0.12))
        label(pos=vector(self.ocean_end_x, -325, 9), text="OCEAN END / WRAP", height=12, box=False, color=vector(0.45, 0.05, 0.03))

    def clear_environment_objects(self):
        for obj in self.map_objects:
            obj.visible = False
        self.map_objects = []

    def add_env_obj(self, obj):
        self.map_objects.append(obj)
        return obj

    def build_environment_objects(self):
        self.clear_environment_objects()
        env = self.environment
        self.add_env_obj(sphere(pos=env.prey_center, radius=env.prey_radius, color=vector(0.05, 0.85, 0.38), opacity=0.10))
        self.add_env_obj(label(pos=env.prey_center + vector(0, 0, 28), text="prey corridor", height=10, box=False, color=vector(0.03, 0.38, 0.12)))
        self.add_env_obj(sphere(pos=env.predator_center, radius=env.predator_radius, color=vector(0.95, 0.20, 0.12), opacity=0.075))
        self.add_env_obj(label(pos=env.predator_center + vector(0, 0, 35), text="predator-risk sector", height=10, box=False, color=vector(0.42, 0.05, 0.04)))
        self.add_env_obj(box(pos=vector((self.ocean_start_x + self.ocean_end_x) / 2, env.reef_y, -132),
                             size=vector(self.ocean_end_x - self.ocean_start_x, env.reef_half_width * 2, 10),
                             color=vector(0.78, 0.62, 0.36), opacity=0.18))
        self.add_env_obj(label(pos=vector(self.ocean_start_x + 170, env.reef_y, -114), text="reef shelf", height=10, box=False, color=vector(0.38, 0.25, 0.08)))
        # Current arrows.
        for x in [-350, -80, 190, 460, 730]:
            arr = arrow(pos=vector(x, -275, -35), axis=env.current * 42, shaftwidth=2.8, color=vector(0.05, 0.35, 0.95), opacity=0.55)
            self.add_env_obj(arr)
        self.add_env_obj(label(pos=vector(120, -282, -12), text="current band", height=10, box=False, color=vector(0.05, 0.18, 0.45)))
        self.last_environment_name = env.name

    def behavior_force(self, d, pod, rid):
        env = self.environment
        centroid = pod.centroid()
        leader = pod.leader()
        migration = vector(1, 0, 0)
        desired_depth = env.preferred_depth
        if rid == 1:
            return migration * 0.75 + env.current * 1.15 + vector(0, (pod.lane_y - d.pos.y) * 0.004, (desired_depth - d.pos.z) * 0.010)
        if rid == 2:
            prey_vec = env.prey_center - d.pos
            return vector(1.25, prey_vec.y * 0.004, prey_vec.z * 0.005) + safe_norm(prey_vec) * (0.42 * env.food_density)
        if rid == 3:
            away = safe_norm(d.pos - env.predator_center)
            risk_distance = mag(d.pos - env.predator_center)
            risk_gain = max(0.0, (env.predator_radius - risk_distance) / max(1.0, env.predator_radius))
            spread_side = 1.0 if d.ident % 2 == 0 else -1.0
            return vector(1.20 + 0.4 * d.boldness, spread_side * (0.26 + 1.2 * risk_gain), away.z * (0.8 + risk_gain))
        theta = 0.035 * self.step_index + d.ident * 0.75
        spiral = vector(0.6, math.sin(theta) * 0.34, math.cos(theta) * 0.20)
        leader_pull = vector(0, 0, 0)
        if leader and leader.ident != d.ident:
            leader_pull = safe_norm(leader.pos - d.pos) * (0.45 * d.sociality)
        center_pull = safe_norm(centroid - d.pos) * (0.65 * d.sociality)
        return spiral + leader_pull + center_pull

    def social_force(self, d, pod):
        cohesion = vector(0, 0, 0)
        separation = vector(0, 0, 0)
        alignment = vector(0, 0, 0)
        calf_guard = vector(0, 0, 0)
        neighbors = 0
        for other in pod.living():
            if other.ident == d.ident:
                continue
            diff = other.pos - d.pos
            dist = mag(diff)
            if dist < 1e-6:
                continue
            if dist < 130:
                cohesion += diff
                alignment += other.vel
                neighbors += 1
            if dist < 22:
                separation += safe_norm(d.pos - other.pos) * ((22 - dist) * 0.05)
            if other.is_calf and not d.is_calf and dist < 95:
                calf_guard += safe_norm(diff) * (0.22 * self.environment.calf_safety)
        if neighbors:
            cohesion = safe_norm(cohesion) * (0.30 * d.sociality)
            alignment = safe_norm(alignment) * (0.22 * d.sociality)
        return cohesion + separation + alignment + calf_guard

    def breathing_force(self, d):
        d.breath_phase += self.dt * (0.040 + 0.015 * pyrandom.random())
        periodic_need = 0.5 + 0.5 * math.sin(d.breath_phase)
        oxygen_need = max(0.0, (45.0 - d.oxygen) / 45.0)
        if periodic_need > 0.88 or oxygen_need > 0.25:
            return vector(0, 0, 1.2 + 2.4 * oxygen_need)
        if d.oxygen > 72 and d.pos.z > self.environment.preferred_depth:
            return vector(0, 0, -0.20)
        return vector(0, 0, 0)

    def update_life(self, d):
        d.age += self.dt / 5200.0
        speed = mag(d.vel)
        d.energy -= self.dt * (0.015 + 0.012 * speed + 0.010 * self.environment.turbulence)
        d.oxygen -= self.dt * (0.12 + 0.030 * speed)
        if d.pos.z > -4:
            d.oxygen = min(100.0, d.oxygen + 7.0 * self.dt)
            d.energy = min(100.0, d.energy + 0.22 * self.dt)
        if mag(d.pos - self.environment.prey_center) < self.environment.prey_radius:
            d.energy = min(100.0, d.energy + self.environment.food_density * 0.45 * self.dt)
        predator_risk = mag(d.pos - self.environment.predator_center) < self.environment.predator_radius
        mortality = 0.00002 + max(0.0, d.age - 38.0) * 0.00002
        if d.energy < 8:
            mortality += 0.004
        if d.oxygen <= 0:
            mortality += 0.02
        if predator_risk:
            mortality += 0.0009 * (1.4 - self.environment.visibility)
        if d.is_calf:
            mortality *= 1.45
        if pyrandom.random() < mortality:
            d.alive = False
            self.delete_dolphin_visual(d)

    def reproduction(self, pod):
        living = pod.living()
        adult_f = [d for d in living if d.sex == "F" and 8 < d.age < 32 and d.energy > 72]
        adult_m = [d for d in living if d.sex == "M" and 8 < d.age < 35 and d.energy > 65]
        if not adult_f or not adult_m or len(living) > 20:
            return
        p_birth = 0.0022 * self.environment.food_density * self.environment.calf_safety
        if pyrandom.random() < p_birth:
            mother = pyrandom.choice(adult_f)
            calf = Dolphin(
                ident=self.new_ident(),
                pod_id=pod.pod_id,
                pos=mother.pos + vector(pyrandom.uniform(-8, 8), pyrandom.uniform(-6, 6), pyrandom.uniform(-2, 2)),
                vel=mother.vel,
                age=0.0,
                energy=62.0,
                oxygen=88.0,
                sex=pyrandom.choice(["F", "M"]),
                sociality=pyrandom.uniform(0.75, 1.0),
                boldness=pyrandom.uniform(0.10, 0.55),
                mother_id=mother.ident,
            )
            mother.energy -= 12.0
            self.create_dolphin_visual(calf)
            pod.dolphins.append(calf)

    def wrap_pod_if_needed(self, pod):
        c = pod.centroid()
        if c.x <= self.ocean_end_x:
            return False
        dx = self.ocean_start_x - c.x + pyrandom.uniform(10, 45)
        dy = pyrandom.uniform(-25, 25)
        for d in pod.living():
            d.pos = vector(d.pos.x + dx, clamp(d.pos.y + dy, -260, 260), clamp(d.pos.z + pyrandom.uniform(-8, 8), self.max_depth_z, self.surface_z))
            d.vel = vector(max(0.8, d.vel.x), d.vel.y * 0.4, d.vel.z * 0.4)
            if d.trail is not None:
                d.trail.clear()
        pod.wraps += 1
        self.environment_index += 1
        self.environment = Environment.random_environment(self.environment_index)
        self.build_environment_objects()
        return True

    def update_dolphin_visual(self, d):
        if d.body is None:
            return
        direction = safe_norm(d.vel)
        if mag(direction) < 1e-6:
            direction = vector(1, 0, 0)
        side = safe_norm(cross(vector(0, 0, 1), direction))
        if mag(side) < 1e-6:
            side = vector(0, 1, 0)
        up = safe_norm(cross(direction, side))
        size = 5.2 if not d.is_calf else 3.4
        bob = math.sin(self.step_index * 0.25 + d.ident) * 0.75
        basepos = d.pos + up * bob
        for obj in [d.body, d.nose, d.fin, d.dorsal, d.tail]:
            if obj is not None:
                obj.axis = direction
        d.body.pos = basepos
        d.body.axis = direction * size * 3.0
        d.nose.pos = basepos + direction * (size * 1.8)
        d.nose.axis = direction * (size * 0.9)
        d.fin.pos = basepos - direction * (size * 1.65)
        d.fin.axis = -direction * (size * 0.9)
        d.dorsal.pos = basepos - direction * (size * 0.25) + up * (size * 0.55)
        d.tail.pos = basepos - direction * (size * 2.05)
        d.tail.axis = side * (size * 1.0)
        if d.trail is not None:
            d.trail.append(pos=basepos)
            while getattr(d.trail, "npoints", 0) > 70:
                d.trail.pop(0)
        if d.label is not None:
            d.label.pos = basepos + vector(0, 0, 12)
            d.label.visible = d.is_leader

    def step(self):
        rid = self.current_round()
        for pod in self.pods:
            for d in list(pod.living()):
                force = self.behavior_force(d, pod, rid) + self.social_force(d, pod) + self.breathing_force(d)
                force += vector(pyrandom.uniform(-1, 1) * self.environment.turbulence,
                                pyrandom.uniform(-1, 1) * self.environment.turbulence,
                                pyrandom.uniform(-1, 1) * self.environment.turbulence * 0.45)
                d.vel = d.vel * 0.88 + force * self.dt
                max_speed = 4.6 if rid == 3 else 3.8
                if mag(d.vel) > max_speed:
                    d.vel = norm(d.vel) * max_speed
                d.pos += d.vel * self.dt * 4.5
                d.pos = vector(d.pos.x, clamp(d.pos.y, -290, 290), clamp(d.pos.z, self.max_depth_z, self.surface_z))
                self.update_life(d)
                self.update_dolphin_visual(d)
            pod.dolphins = [d for d in pod.dolphins if d.alive]
            self.reproduction(pod)
            self.wrap_pod_if_needed(pod)
        self.step_index += 1

    def selected_center(self):
        if self.follow_mode == "all":
            living = []
            for pod in self.pods:
                living.extend(pod.living())
            if not living:
                return vector(180, 0, -60)
            c = vector(0, 0, 0)
            for d in living:
                c += d.pos
            return c / len(living)
        pod_id = int(self.follow_mode)
        if 0 <= pod_id < len(self.pods):
            return self.pods[pod_id].centroid()
        return vector(180, 0, -60)

    def update_camera_and_status(self):
        target = self.selected_center()
        self.scene.center = self.scene.center * 0.92 + target * 0.08
        self.scene.range = self.zoom
        rid = self.current_round()
        total = sum(len(p.living()) for p in self.pods)
        calves = sum(1 for p in self.pods for d in p.living() if d.is_calf)
        surfacing = sum(1 for p in self.pods for d in p.living() if d.pos.z > -5)
        pod_summary = " | ".join(
            f"pod {p.pod_id}: n={len(p.living())} wraps={p.wraps} speed={p.mean_speed():.1f}" for p in self.pods
        )
        self.status.text = (
            f"\nR{rid} {self.round_name(rid)} | env: {self.environment.name} | total dolphins={total} calves={calves} surfacing={surfacing} "
            f"| food={self.environment.food_density:.2f} visibility={self.environment.visibility:.2f} temp={self.environment.temperature:.1f}C "
            f"| camera={self.follow_mode} zoom={self.zoom:.0f}\n{pod_summary}\n"
            "Controls: space pause | r reset | 1 all | 2/3/4 pod | +/- or z/x zoom | n/p round\n"
        )

    def on_keydown(self, evt):
        key = evt.key
        if key == " ":
            self.paused = not self.paused
        elif key == "r":
            self.reset()
        elif key == "1":
            self.follow_mode = "all"
        elif key == "2":
            self.follow_mode = "0"
        elif key == "3":
            self.follow_mode = "1"
        elif key == "4":
            self.follow_mode = "2"
        elif key in ["+", "=", "z"]:
            self.zoom = max(self.min_zoom, self.zoom * 0.88)
        elif key in ["-", "_", "x"]:
            self.zoom = min(self.max_zoom, self.zoom * 1.14)
        elif key == "n":
            if self.forced_round_index is None:
                self.forced_round_index = self.round_ids.index(self.current_round())
            self.forced_round_index = (self.forced_round_index + 1) % len(self.round_ids)
        elif key == "p":
            if self.forced_round_index is None:
                self.forced_round_index = self.round_ids.index(self.current_round())
            self.forced_round_index = (self.forced_round_index - 1) % len(self.round_ids)

    def reset(self):
        for pod in self.pods:
            for d in pod.dolphins:
                self.delete_dolphin_visual(d)
        self.step_index = 0
        self.next_ident = 0
        self.environment_index = 1
        self.environment = Environment.random_environment(self.environment_index)
        self.build_environment_objects()
        self.pods = self.make_initial_pods()
        self.forced_round_index = None

    def run(self):
        while True:
            rate(34)
            if not self.paused:
                self.step()
            self.update_camera_and_status()


if __name__ == "__main__":
    DolphinLifeOceanVisual().run()
