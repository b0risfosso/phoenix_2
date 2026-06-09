"""
VPython City Traffic Network Simulation
Urban planning, congestion, signals, and routing

Run:
    python vpython_city_traffic_network_urban_planning_congestion_signals_routing.py

Controls:
    Space : pause / resume
    r     : reset simulation
    + / = : add cars
    -     : remove cars
    s     : toggle signal cycling
    a     : toggle adaptive signal timing
    d     : toggle dynamic routing
    c     : toggle congestion heat overlay
    f     : fast / normal mode
    v     : toggle cinematic camera
    1     : orbit whole city
    2     : follow a car close
    3     : overhead planning view
    4     : low street-level tracking view
    n     : follow next car

Notes:
    Uses simple VPython primitives only. No torus objects are used.
"""

from vpython import *
import random
import math
from collections import deque, defaultdict

# -----------------------------
# Scene setup
# -----------------------------
scene = canvas(
    title="City Traffic Network — Urban Planning, Congestion, Signals, Routing",
    width=1280,
    height=820,
    background=vector(0.86, 0.91, 0.96),
)
scene.camera.pos = vector(0, 58, 46)
scene.camera.axis = vector(0, -58, -46)
scene.forward = vector(0, -0.78, -0.63)
scene.up = vector(0, 1, 0)
scene.autoscale = False
scene.range = 38

# -----------------------------
# Simulation parameters
# -----------------------------
GRID_N = 5
SPACING = 12
ROAD_W = 2.4
LANE_OFFSET = 0.75
BLOCK_SIZE = SPACING - ROAD_W
BASE_SPEED = 3.2
CAR_COUNT_START = 42
MAX_CARS = 120
MIN_CARS = 3
DT = 0.04

paused = False
signal_enabled = True
adaptive_signals = True
dynamic_routing = True
show_congestion = True
fast_mode = False
cinematic_camera = True
camera_mode = "orbit"
camera_timer = 0.0
camera_mode_duration = 9.0
camera_target_index = 0
camera_orbit_angle = 0.0

random.seed(12)

# -----------------------------
# Utility functions
# -----------------------------
def node_pos(i, j):
    center = (GRID_N - 1) / 2
    return vector((i - center) * SPACING, 0.05, (j - center) * SPACING)


def manhattan_path(start, goal, congestion=None, dynamic=True):
    """Return a list of grid nodes from start to goal.

    When dynamic routing is enabled, each step chooses the lower-cost next move.
    Congested outgoing road segments become more expensive.
    """
    if start == goal:
        return [start]

    path = [start]
    current = start
    safety = 0

    while current != goal and safety < 30:
        safety += 1
        ci, cj = current
        gi, gj = goal
        candidates = []

        if ci < gi:
            candidates.append((ci + 1, cj))
        elif ci > gi:
            candidates.append((ci - 1, cj))
        if cj < gj:
            candidates.append((ci, cj + 1))
        elif cj > gj:
            candidates.append((ci, cj - 1))

        if not candidates:
            break

        if dynamic and congestion is not None and len(candidates) > 1:
            def segment_cost(nxt):
                seg = tuple(sorted([current, nxt]))
                return 1.0 + 3.5 * congestion.get(seg, 0.0) + random.random() * 0.2
            nxt = min(candidates, key=segment_cost)
        else:
            nxt = random.choice(candidates)

        path.append(nxt)
        current = nxt

    return path


def road_segment_key(a, b):
    return tuple(sorted([a, b]))


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def lerp(a, b, t):
    return a * (1 - t) + b * t


def make_text(pos, text, height=0.75, color_value=color.black, align="center"):
    return label(
        pos=pos,
        text=text,
        height=height,
        color=color_value,
        box=False,
        opacity=0,
        align=align,
    )

# -----------------------------
# City geometry
# -----------------------------
ground = box(pos=vector(0, -0.08, 0), size=vector(70, 0.12, 70), color=vector(0.72, 0.82, 0.72))

# Blocks / buildings
buildings = []
for i in range(GRID_N - 1):
    for j in range(GRID_N - 1):
        p1 = node_pos(i, j)
        p2 = node_pos(i + 1, j + 1)
        center = (p1 + p2) / 2
        h = random.uniform(1.0, 5.5)
        b = box(
            pos=vector(center.x, h / 2, center.z),
            size=vector(BLOCK_SIZE * 0.72, h, BLOCK_SIZE * 0.72),
            color=vector(random.uniform(0.45, 0.72), random.uniform(0.48, 0.70), random.uniform(0.55, 0.80)),
            opacity=0.48,
        )
        buildings.append(b)

# Roads
roads = []
for j in range(GRID_N):
    z = node_pos(0, j).z
    roads.append(box(pos=vector(0, 0, z), size=vector((GRID_N - 1) * SPACING + ROAD_W, 0.08, ROAD_W), color=vector(0.18, 0.19, 0.20)))
for i in range(GRID_N):
    x = node_pos(i, 0).x
    roads.append(box(pos=vector(x, 0.01, 0), size=vector(ROAD_W, 0.08, (GRID_N - 1) * SPACING + ROAD_W), color=vector(0.18, 0.19, 0.20)))

# Lane markings
for j in range(GRID_N):
    z = node_pos(0, j).z
    for k in range(-2, 3):
        x = k * SPACING
        box(pos=vector(x, 0.08, z), size=vector(2.4, 0.025, 0.08), color=vector(0.95, 0.90, 0.52))
for i in range(GRID_N):
    x = node_pos(i, 0).x
    for k in range(-2, 3):
        z = k * SPACING
        box(pos=vector(x, 0.08, z), size=vector(0.08, 0.025, 2.4), color=vector(0.95, 0.90, 0.52))

# Intersections and signal markers
nodes = [(i, j) for i in range(GRID_N) for j in range(GRID_N)]
intersection_markers = {}
signal_poles = {}
signal_lights = {}
queue_labels = {}

for n in nodes:
    p = node_pos(*n)
    intersection_markers[n] = box(pos=vector(p.x, 0.09, p.z), size=vector(2.7, 0.03, 2.7), color=vector(0.25, 0.25, 0.25), opacity=0.55)
    signal_poles[n] = cylinder(pos=vector(p.x + 1.8, 0.1, p.z + 1.8), axis=vector(0, 2.5, 0), radius=0.055, color=vector(0.08, 0.08, 0.08))
    signal_lights[n] = sphere(pos=vector(p.x + 1.8, 2.8, p.z + 1.8), radius=0.32, color=color.green, emissive=True)
    if 0 < n[0] < GRID_N - 1 and 0 < n[1] < GRID_N - 1:
        queue_labels[n] = make_text(vector(p.x, 3.4, p.z), "", height=0.55, color_value=vector(0.1, 0.1, 0.1))

# Congestion overlay on road segments
segments = []
congestion_boxes = {}
for i in range(GRID_N):
    for j in range(GRID_N):
        if i < GRID_N - 1:
            a, b = (i, j), (i + 1, j)
            pa, pb = node_pos(*a), node_pos(*b)
            mid = (pa + pb) / 2
            key = road_segment_key(a, b)
            congestion_boxes[key] = box(pos=vector(mid.x, 0.14, mid.z), size=vector(SPACING - 2.3, 0.025, ROAD_W * 0.75), color=color.green, opacity=0.0)
            segments.append(key)
        if j < GRID_N - 1:
            a, b = (i, j), (i, j + 1)
            pa, pb = node_pos(*a), node_pos(*b)
            mid = (pa + pb) / 2
            key = road_segment_key(a, b)
            congestion_boxes[key] = box(pos=vector(mid.x, 0.14, mid.z), size=vector(ROAD_W * 0.75, 0.025, SPACING - 2.3), color=color.green, opacity=0.0)
            segments.append(key)

# Legend / dashboard
status = label(
    pos=vector(-33, 13, -33),
    text="",
    height=0.75,
    color=color.black,
    box=True,
    background=vector(0.98, 0.98, 0.94),
    opacity=0.75,
    align="left",
)

controls = label(
    pos=vector(18, 13, -33),
    text="Space pause | r reset | +/- cars | s signals | a adaptive | d routing | c heat | f speed | v camera | 1-4 views | n car",
    height=0.65,
    color=color.black,
    box=True,
    background=vector(0.98, 0.98, 0.94),
    opacity=0.72,
)

phase_banner = label(
    pos=vector(0, 10.5, 31),
    text="",
    height=0.95,
    color=color.black,
    box=False,
    opacity=0,
)

# -----------------------------
# Signals
# -----------------------------
class SignalSystem:
    def __init__(self):
        self.timer = 0.0
        self.phase = 0  # 0: east-west green, 1: north-south green
        self.base_duration = 5.0
        self.duration = self.base_duration
        self.switch_count = 0

    def reset(self):
        self.timer = 0.0
        self.phase = 0
        self.duration = self.base_duration
        self.switch_count = 0

    def update(self, dt, queues):
        if not signal_enabled:
            for n in nodes:
                signal_lights[n].color = color.green
            return

        self.timer += dt

        if adaptive_signals:
            ew_q = sum(queues.get((n, "EW"), 0) for n in nodes)
            ns_q = sum(queues.get((n, "NS"), 0) for n in nodes)
            dominant = ew_q if self.phase == 0 else ns_q
            opposing = ns_q if self.phase == 0 else ew_q
            self.duration = clamp(3.0 + 0.12 * dominant - 0.05 * opposing, 3.0, 8.5)
        else:
            self.duration = self.base_duration

        if self.timer >= self.duration:
            self.phase = 1 - self.phase
            self.timer = 0.0
            self.switch_count += 1

        for n in nodes:
            # Center signal color represents the current permitted direction.
            signal_lights[n].color = color.green if self.phase == 0 else vector(0.1, 0.55, 1.0)

    def can_pass(self, prev_node, current_node, next_node):
        if not signal_enabled:
            return True
        if prev_node is None or next_node is None:
            return True
        # Determine outgoing direction at current intersection.
        ci, cj = current_node
        ni, nj = next_node
        if ni != ci:
            direction = "EW"
        else:
            direction = "NS"
        return (self.phase == 0 and direction == "EW") or (self.phase == 1 and direction == "NS")

signals = SignalSystem()

# -----------------------------
# Cars
# -----------------------------
car_palette = [
    vector(0.95, 0.22, 0.18),
    vector(0.10, 0.42, 0.95),
    vector(0.98, 0.70, 0.20),
    vector(0.15, 0.70, 0.35),
    vector(0.70, 0.30, 0.95),
    vector(0.95, 0.45, 0.20),
]

class Car:
    next_id = 0

    def __init__(self):
        self.id = Car.next_id
        Car.next_id += 1
        self.body = box(pos=vector(0, 0.45, 0), size=vector(1.25, 0.42, 0.72), color=random.choice(car_palette))
        self.roof = box(pos=vector(0, 0.78, 0), size=vector(0.65, 0.28, 0.50), color=vector(0.88, 0.92, 0.96), opacity=0.75)
        self.route_line = curve(color=self.body.color, radius=0.035, opacity=0.35)
        self.wait_time = 0.0
        self.total_distance = 0.0
        self.completed_trips = 0
        self.speed = BASE_SPEED * random.uniform(0.85, 1.15)
        self.replan_cooldown = 0.0
        self.assign_trip({})

    def choose_edge_node(self):
        side = random.choice([0, 1, 2, 3])
        if side == 0:
            return (0, random.randrange(GRID_N))
        if side == 1:
            return (GRID_N - 1, random.randrange(GRID_N))
        if side == 2:
            return (random.randrange(GRID_N), 0)
        return (random.randrange(GRID_N), GRID_N - 1)

    def assign_trip(self, congestion):
        start = self.choose_edge_node()
        goal = self.choose_edge_node()
        while goal == start:
            goal = self.choose_edge_node()
        self.origin = start
        self.destination = goal
        self.path = manhattan_path(start, goal, congestion, dynamic_routing)
        self.path_index = 0
        self.t = 0.0
        self.prev_node = None
        self.current_node = self.path[0]
        self.next_node = self.path[1] if len(self.path) > 1 else None
        self.place_on_segment()
        self.update_route_line()

    def update_route_line(self):
        self.route_line.clear()
        for n in self.path[self.path_index:]:
            p = node_pos(*n)
            self.route_line.append(vector(p.x, 0.22, p.z))

    def place_on_segment(self):
        if self.next_node is None:
            p = node_pos(*self.current_node)
            self.body.pos = vector(p.x, 0.45, p.z)
            self.roof.pos = self.body.pos + vector(0, 0.33, 0)
            return
        a = node_pos(*self.current_node)
        b = node_pos(*self.next_node)
        direction = norm(b - a)
        # Lane offset gives opposite sides of the road depending on direction.
        side = vector(-direction.z, 0, direction.x) * LANE_OFFSET
        p = a + (b - a) * self.t + side
        self.body.pos = vector(p.x, 0.45, p.z)
        self.roof.pos = self.body.pos + vector(0, 0.33, 0)
        if abs(direction.x) > abs(direction.z):
            self.body.size = vector(1.25, 0.42, 0.72)
            self.roof.size = vector(0.65, 0.28, 0.50)
        else:
            self.body.size = vector(0.72, 0.42, 1.25)
            self.roof.size = vector(0.50, 0.28, 0.65)

    def distance_to_car_ahead(self, other):
        if self.current_node != other.current_node or self.next_node != other.next_node:
            return 99
        if other.t <= self.t:
            return 99
        return (other.t - self.t) * SPACING

    def update(self, dt, cars, congestion):
        if self.next_node is None:
            self.completed_trips += 1
            self.assign_trip(congestion)
            return

        self.replan_cooldown = max(0.0, self.replan_cooldown - dt)

        # Dynamic routing: approaching an intersection, replan the remaining trip.
        if dynamic_routing and self.t > 0.82 and self.replan_cooldown <= 0:
            remaining = manhattan_path(self.next_node, self.destination, congestion, True)
            self.path = self.path[:self.path_index + 1] + remaining
            self.update_route_line()
            self.replan_cooldown = 1.2

        near_intersection = self.t > 0.83
        pass_allowed = signals.can_pass(self.prev_node, self.next_node, self.path[self.path_index + 2] if self.path_index + 2 < len(self.path) else None)

        # Stop before the intersection if red for the intended next movement.
        red_stop = signal_enabled and near_intersection and not pass_allowed

        # Car-following behavior.
        gap = min(self.distance_to_car_ahead(other) for other in cars if other is not self) if len(cars) > 1 else 99
        blocked_by_car = gap < 1.55

        if red_stop or blocked_by_car:
            self.wait_time += dt
            # Inch forward slightly until stop line.
            if self.t < 0.90 and not blocked_by_car:
                self.t += (self.speed * 0.22 * dt) / SPACING
            self.place_on_segment()
            return

        local_congestion = congestion.get(road_segment_key(self.current_node, self.next_node), 0.0)
        speed_factor = clamp(1.0 - 0.62 * local_congestion, 0.25, 1.0)
        self.t += (self.speed * speed_factor * dt) / SPACING
        self.total_distance += self.speed * speed_factor * dt

        if self.t >= 1.0:
            self.prev_node = self.current_node
            self.current_node = self.next_node
            self.path_index += 1
            self.t = 0.0
            if self.path_index >= len(self.path) - 1:
                self.next_node = None
                self.completed_trips += 1
                self.assign_trip(congestion)
                return
            self.next_node = self.path[self.path_index + 1]
            self.update_route_line()

        self.place_on_segment()

    def destroy(self):
        self.body.visible = False
        self.roof.visible = False
        self.route_line.visible = False

cars = []
congestion = defaultdict(float)

# -----------------------------
# Metrics and overlays
# -----------------------------
def compute_congestion(cars):
    counts = defaultdict(int)
    for car in cars:
        if car.next_node is not None:
            counts[road_segment_key(car.current_node, car.next_node)] += 1
    result = defaultdict(float)
    for seg in segments:
        # Four cars on one segment is heavy traffic in this stylized scale.
        result[seg] = clamp(counts.get(seg, 0) / 4.0, 0.0, 1.0)
    return result


def compute_queues(cars):
    queues = defaultdict(int)
    for car in cars:
        if car.next_node is None:
            continue
        if car.t > 0.78:
            ci, cj = car.current_node
            ni, nj = car.next_node
            direction = "EW" if ni != ci else "NS"
            queues[(car.next_node, direction)] += 1
    return queues


def update_congestion_overlay(congestion):
    for seg, overlay in congestion_boxes.items():
        level = congestion.get(seg, 0.0)
        if not show_congestion:
            overlay.opacity = 0.0
        else:
            overlay.opacity = 0.13 + 0.48 * level if level > 0.03 else 0.02
            # Green -> yellow -> red without relying on color maps.
            overlay.color = vector(clamp(level * 1.7, 0.0, 1.0), clamp(1.0 - level * 0.75, 0.05, 1.0), 0.08)


def update_queue_labels(queues):
    for n, lab in queue_labels.items():
        ew = queues.get((n, "EW"), 0)
        ns = queues.get((n, "NS"), 0)
        total = ew + ns
        lab.text = f"Q {total}" if total > 0 else ""
        lab.color = vector(0.75, 0.1, 0.05) if total >= 4 else vector(0.1, 0.1, 0.1)


def city_phase(avg_cong):
    if avg_cong < 0.18:
        return "Free-flowing traffic"
    if avg_cong < 0.38:
        return "Moderate congestion"
    if avg_cong < 0.62:
        return "Rush-hour pressure"
    return "Gridlock risk"


def update_dashboard(t, congestion, queues):
    avg_cong = sum(congestion.values()) / max(1, len(congestion))
    max_cong = max(congestion.values()) if congestion else 0
    total_wait = sum(car.wait_time for car in cars)
    trips = sum(car.completed_trips for car in cars)
    phase = city_phase(avg_cong)
    signal_mode = "adaptive" if adaptive_signals else "fixed"
    signal_state = "EW green" if signals.phase == 0 else "NS green"

    status.text = (
        f"CITY TRAFFIC NETWORK\n"
        f"Cars: {len(cars)}   Trips completed: {trips}\n"
        f"Average congestion: {avg_cong:0.2f}   Peak road: {max_cong:0.2f}\n"
        f"Total waiting: {total_wait:0.1f}s\n"
        f"Signals: {'on' if signal_enabled else 'off'} / {signal_mode} / {signal_state}\n"
        f"Routing: {'dynamic congestion-aware' if dynamic_routing else 'static random shortest path'}\n"
        f"Heat overlay: {'on' if show_congestion else 'off'}   Speed: {'fast' if fast_mode else 'normal'}\n"
        f"Camera: {'cinematic ' + camera_mode if cinematic_camera else 'manual'}"
    )
    phase_banner.text = phase
    if avg_cong >= 0.62:
        phase_banner.color = vector(0.8, 0.05, 0.03)
    elif avg_cong >= 0.38:
        phase_banner.color = vector(0.75, 0.45, 0.02)
    else:
        phase_banner.color = vector(0.05, 0.35, 0.12)


# -----------------------------
# Cinematic camera
# -----------------------------
def smooth_camera(desired_pos, desired_target, blend=0.045):
    """Move the camera gradually so the view glides instead of snapping."""
    scene.camera.pos = scene.camera.pos * (1 - blend) + desired_pos * blend
    scene.camera.axis = (desired_target - scene.camera.pos)
    scene.forward = norm(scene.camera.axis)


def camera_focus_point():
    if not cars:
        return vector(0, 0, 0)
    idx = camera_target_index % len(cars)
    return cars[idx].body.pos


def switch_camera_mode(mode):
    global camera_mode, camera_timer
    camera_mode = mode
    camera_timer = 0.0


def follow_next_car():
    global camera_target_index
    if cars:
        camera_target_index = (camera_target_index + 1) % len(cars)


def update_cinematic_camera(dt, sim_time):
    """Cycle through city-scale and car-scale camera behaviors.

    orbit:     rotating overview around the full network.
    follow:    zoomed follow camera attached to a selected car.
    overhead:  urban-planning map view from above.
    street:    low tracking view that watches traffic move through intersections.
    """
    global camera_timer, camera_orbit_angle, camera_mode, camera_target_index

    if not cinematic_camera:
        return

    if not cars:
        smooth_camera(vector(0, 55, 45), vector(0, 0, 0), 0.035)
        return

    camera_timer += dt
    camera_orbit_angle += dt * 0.18

    # Automatic shot list. The camera loops through wide, close, top-down, and street views.
    if camera_timer >= camera_mode_duration:
        modes = ["orbit", "follow", "overhead", "street"]
        camera_mode = modes[(modes.index(camera_mode) + 1) % len(modes)] if camera_mode in modes else "orbit"
        camera_timer = 0.0
        if camera_mode in ["follow", "street"]:
            camera_target_index = random.randrange(len(cars))

    focus = camera_focus_point()

    if camera_mode == "orbit":
        radius = 55 + 8 * math.sin(sim_time * 0.20)
        height = 35 + 8 * math.sin(sim_time * 0.13 + 1.1)
        desired_pos = vector(radius * math.cos(camera_orbit_angle), height, radius * math.sin(camera_orbit_angle))
        desired_target = vector(0, 1.5, 0)
        smooth_camera(desired_pos, desired_target, 0.028)
        scene.range = scene.range * 0.97 + 39 * 0.03

    elif camera_mode == "follow":
        idx = camera_target_index % len(cars)
        car = cars[idx]
        focus = car.body.pos
        if car.next_node is not None:
            a = node_pos(*car.current_node)
            b = node_pos(*car.next_node)
            travel = norm(b - a)
        else:
            travel = vector(1, 0, 0)
        desired_pos = focus - travel * 7.5 + vector(0, 4.2, 0) + vector(-travel.z, 0, travel.x) * 2.3
        desired_target = focus + travel * 4 + vector(0, 0.8, 0)
        smooth_camera(desired_pos, desired_target, 0.080)
        scene.range = scene.range * 0.94 + 8.5 * 0.06

    elif camera_mode == "overhead":
        # Slight drift prevents the planning view from feeling static.
        drift = vector(7 * math.sin(sim_time * 0.08), 0, 7 * math.cos(sim_time * 0.07))
        desired_pos = vector(0, 73, 0.1) + drift * 0.15
        desired_target = vector(0, 0, 0)
        smooth_camera(desired_pos, desired_target, 0.035)
        scene.up = vector(0, 0, -1)
        scene.range = scene.range * 0.96 + 42 * 0.04

    elif camera_mode == "street":
        idx = camera_target_index % len(cars)
        car = cars[idx]
        focus = car.body.pos
        if car.next_node is not None:
            a = node_pos(*car.current_node)
            b = node_pos(*car.next_node)
            travel = norm(b - a)
        else:
            travel = vector(1, 0, 0)
        side = vector(-travel.z, 0, travel.x)
        desired_pos = focus - travel * 11 + side * 5 + vector(0, 2.0, 0)
        desired_target = focus + travel * 8 + vector(0, 1.2, 0)
        smooth_camera(desired_pos, desired_target, 0.060)
        scene.range = scene.range * 0.93 + 12.0 * 0.07

    # Restore a normal upright view after overhead mode leaves; overhead resets it each frame.
    if camera_mode != "overhead":
        scene.up = scene.up * 0.90 + vector(0, 1, 0) * 0.10

# -----------------------------
# Simulation control
# -----------------------------
def reset_simulation():
    global cars, congestion
    for car in cars:
        car.destroy()
    cars = []
    Car.next_id = 0
    congestion = defaultdict(float)
    signals.reset()
    for _ in range(CAR_COUNT_START):
        cars.append(Car())


def add_car():
    if len(cars) < MAX_CARS:
        cars.append(Car())


def remove_car():
    if len(cars) > MIN_CARS:
        car = cars.pop()
        car.destroy()


def keydown(evt):
    global paused, signal_enabled, adaptive_signals, dynamic_routing, show_congestion, fast_mode, cinematic_camera, camera_mode, camera_timer
    key = evt.key
    if key == " ":
        paused = not paused
    elif key in ["r", "R"]:
        reset_simulation()
    elif key in ["+", "="]:
        for _ in range(5):
            add_car()
    elif key in ["-"]:
        for _ in range(5):
            remove_car()
    elif key in ["s", "S"]:
        signal_enabled = not signal_enabled
    elif key in ["a", "A"]:
        adaptive_signals = not adaptive_signals
    elif key in ["d", "D"]:
        dynamic_routing = not dynamic_routing
    elif key in ["c", "C"]:
        show_congestion = not show_congestion
    elif key in ["f", "F"]:
        fast_mode = not fast_mode
    elif key in ["v", "V"]:
        cinematic_camera = not cinematic_camera
    elif key == "1":
        cinematic_camera = True
        switch_camera_mode("orbit")
    elif key == "2":
        cinematic_camera = True
        switch_camera_mode("follow")
    elif key == "3":
        cinematic_camera = True
        switch_camera_mode("overhead")
    elif key == "4":
        cinematic_camera = True
        switch_camera_mode("street")
    elif key in ["n", "N"]:
        follow_next_car()

scene.bind("keydown", keydown)

# -----------------------------
# Main loop
# -----------------------------
reset_simulation()
sim_time = 0.0

while True:
    rate(90 if fast_mode else 60)
    if paused:
        update_dashboard(sim_time, congestion, compute_queues(cars))
        continue

    steps = 3 if fast_mode else 1
    for _ in range(steps):
        sim_time += DT
        congestion = compute_congestion(cars)
        queues = compute_queues(cars)
        signals.update(DT, queues)
        for car in cars:
            car.update(DT, cars, congestion)

    queues = compute_queues(cars)
    update_congestion_overlay(congestion)
    update_queue_labels(queues)
    update_dashboard(sim_time, congestion, queues)
    update_cinematic_camera(DT * steps, sim_time)
