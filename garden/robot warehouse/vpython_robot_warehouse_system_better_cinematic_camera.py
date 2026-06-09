"""
VPython Simulation: Robot Warehouse System
Engineering | Pathfinding | Task Scheduling | Collisions

Run:
    python vpython_robot_warehouse_system_pathfinding_scheduling_collisions.py

Controls:
    Space : pause / resume
    r     : reset simulation
    t     : add task
    + / = : add robot
    -     : remove robot
    a     : toggle automatic task generation
    c     : toggle collision avoidance
    p     : toggle path display
    v     : toggle cinematic camera
    1     : smooth orbit overview
    2     : predictive close follow robot
    3     : steady top-down planning view
    4     : low aisle dolly view
    n     : follow next robot

Notes:
    - Uses VPython primitives only; no torus is used.
    - Robots use grid-based A* pathfinding.
    - A scheduler assigns waiting tasks to idle robots.
    - Collision avoidance reserves next grid cells and causes robots to wait/reroute.
"""

from vpython import *
import random
import math
import heapq
from collections import deque, defaultdict

# -----------------------------
# Scene setup
# -----------------------------
scene = canvas(
    title="Robot Warehouse System — Pathfinding, Scheduling, Collision Avoidance",
    width=1280,
    height=760,
    background=vector(0.92, 0.95, 0.98),
    center=vector(0, 0, 0),
)
scene.forward = vector(-0.55, -0.72, -0.42)
scene.up = vector(0, 1, 0)
scene.range = 22

# -----------------------------
# Parameters
# -----------------------------
GRID_W = 18
GRID_H = 12
CELL = 2.0
HALF_W = GRID_W * CELL * 0.5
HALF_H = GRID_H * CELL * 0.5

MAX_TRAILS_PER_ROBOT = 28
BASE_ROBOT_SPEED = 3.8
TASK_AUTO_INTERVAL = 2.4
REPLAN_INTERVAL = 1.2

paused = False
auto_tasks = True
avoid_collisions = True
show_paths = True
cinematic_camera = True
camera_mode = 0  # 0 = automatic cinematic shot cycle
follow_index = 0

# Smooth cinematic camera state. VPython camera jumps feel harsh when camera.pos
# is assigned directly every frame, so desired camera positions are eased toward.
camera_pos_smooth = vector(28, 24, 24)
camera_center_smooth = vector(0, 0, 0)
camera_range_smooth = 22.0
camera_last_mode = None
manual_camera_timer = 0.0
sim_time = 0.0
last_auto_task = 0.0
next_task_id = 1

# -----------------------------
# Colors
# -----------------------------
COL_FLOOR = vector(0.78, 0.82, 0.86)
COL_GRID = vector(0.60, 0.65, 0.70)
COL_SHELF = vector(0.50, 0.34, 0.18)
COL_BIN = vector(0.20, 0.42, 0.75)
COL_PICK = vector(0.20, 0.70, 0.35)
COL_DROP = vector(0.92, 0.55, 0.12)
COL_ROBOT_IDLE = vector(0.25, 0.50, 0.95)
COL_ROBOT_BUSY = vector(0.95, 0.62, 0.18)
COL_ROBOT_WAIT = vector(0.95, 0.22, 0.18)
COL_PATH = vector(0.18, 0.45, 0.95)
COL_TASK_WAIT = vector(0.95, 0.82, 0.18)
COL_TASK_DONE = vector(0.30, 0.85, 0.45)

# -----------------------------
# Geometry helpers
# -----------------------------
def grid_to_world(cell):
    x, z = cell
    return vector((x - GRID_W / 2 + 0.5) * CELL, 0, (z - GRID_H / 2 + 0.5) * CELL)


def world_to_grid(pos):
    x = int(round(pos.x / CELL + GRID_W / 2 - 0.5))
    z = int(round(pos.z / CELL + GRID_H / 2 - 0.5))
    return (max(0, min(GRID_W - 1, x)), max(0, min(GRID_H - 1, z)))


def in_bounds(cell):
    x, z = cell
    return 0 <= x < GRID_W and 0 <= z < GRID_H


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

# -----------------------------
# Warehouse layout
# -----------------------------
blocked = set()
shelf_cells = []

# Shelf blocks in vertical aisle rows, leaving corridors.
for x in [3, 4, 7, 8, 11, 12, 15]:
    for z in range(2, 10):
        if z in [5, 6]:
            continue
        blocked.add((x, z))
        shelf_cells.append((x, z))

pickup_stations = [(0, 1), (0, 10), (17, 1)]
dropoff_stations = [(17, 10), (9, 0), (9, 11)]
charging_stations = [(0, 5), (17, 5)]

# Ensure stations remain open.
for s in pickup_stations + dropoff_stations + charging_stations:
    blocked.discard(s)

# -----------------------------
# Static visual objects
# -----------------------------
floor = box(pos=vector(0, -0.08, 0), size=vector(GRID_W * CELL + 1.5, 0.12, GRID_H * CELL + 1.5), color=COL_FLOOR)

# Grid lines.
for x in range(GRID_W + 1):
    wx = (x - GRID_W / 2) * CELL
    curve(pos=[vector(wx, 0.01, -HALF_H), vector(wx, 0.01, HALF_H)], color=COL_GRID, radius=0.01)
for z in range(GRID_H + 1):
    wz = (z - GRID_H / 2) * CELL
    curve(pos=[vector(-HALF_W, 0.01, wz), vector(HALF_W, 0.01, wz)], color=COL_GRID, radius=0.01)

# Shelves.
shelf_objects = []
for cell in shelf_cells:
    p = grid_to_world(cell)
    shelf = box(pos=p + vector(0, 0.65, 0), size=vector(1.45, 1.3, 1.45), color=COL_SHELF)
    top_bin = box(pos=p + vector(0, 1.38, 0), size=vector(1.18, 0.22, 1.18), color=COL_BIN)
    shelf_objects.append((shelf, top_bin))

# Stations.
station_objects = []
for i, cell in enumerate(pickup_stations):
    p = grid_to_world(cell)
    station_objects.append(box(pos=p + vector(0, 0.08, 0), size=vector(1.65, 0.16, 1.65), color=COL_PICK))
    label(pos=p + vector(0, 1.0, 0), text=f"PICK {i+1}", height=10, color=COL_PICK, box=False, opacity=0)
for i, cell in enumerate(dropoff_stations):
    p = grid_to_world(cell)
    station_objects.append(box(pos=p + vector(0, 0.08, 0), size=vector(1.65, 0.16, 1.65), color=COL_DROP))
    label(pos=p + vector(0, 1.0, 0), text=f"DROP {i+1}", height=10, color=COL_DROP, box=False, opacity=0)
for i, cell in enumerate(charging_stations):
    p = grid_to_world(cell)
    station_objects.append(cylinder(pos=p + vector(0, 0.03, 0), axis=vector(0, 0.14, 0), radius=0.72, color=vector(0.35, 0.35, 0.38)))
    label(pos=p + vector(0, 0.95, 0), text="CHARGE", height=9, color=vector(0.2, 0.2, 0.25), box=False, opacity=0)

# Side boards.
hud = label(pos=vector(-20.5, 8.2, -12.5), text="", height=11, align="left", box=True, opacity=0.72, color=vector(0.05, 0.07, 0.10), background=vector(0.96, 0.98, 1.0))
dispatch_label = label(pos=vector(19.5, 7.8, -12.5), text="", height=10, align="left", box=True, opacity=0.72, color=vector(0.05, 0.07, 0.10), background=vector(1.0, 0.98, 0.92))

# -----------------------------
# Pathfinding
# -----------------------------
def neighbors(cell):
    x, z = cell
    for dx, dz in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        nxt = (x + dx, z + dz)
        if in_bounds(nxt) and nxt not in blocked:
            yield nxt


def astar(start, goal, soft_cost=None):
    if start == goal:
        return [start]
    if goal in blocked:
        return []
    soft_cost = soft_cost or {}
    open_heap = []
    heapq.heappush(open_heap, (0, start))
    came_from = {}
    g_score = {start: 0}

    while open_heap:
        _, current = heapq.heappop(open_heap)
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        for nxt in neighbors(current):
            congestion_penalty = soft_cost.get(nxt, 0.0)
            tentative = g_score[current] + 1.0 + congestion_penalty
            if tentative < g_score.get(nxt, 999999):
                came_from[nxt] = current
                g_score[nxt] = tentative
                f = tentative + manhattan(nxt, goal)
                heapq.heappush(open_heap, (f, nxt))
    return []


def random_open_cell():
    while True:
        cell = (random.randrange(GRID_W), random.randrange(GRID_H))
        if cell not in blocked and cell not in pickup_stations and cell not in dropoff_stations:
            return cell

# -----------------------------
# Task and robot classes
# -----------------------------
class Task:
    def __init__(self, task_id, pickup, dropoff):
        self.id = task_id
        self.pickup = pickup
        self.dropoff = dropoff
        self.status = "waiting"  # waiting, assigned, picked, done
        self.assigned_robot = None
        self.age = 0.0
        p = grid_to_world(pickup)
        d = grid_to_world(dropoff)
        self.pick_marker = sphere(pos=p + vector(0, 0.55, 0), radius=0.25, color=COL_TASK_WAIT, emissive=True)
        self.drop_marker = pyramid(pos=d + vector(0, 0.55, 0), size=vector(0.55, 0.55, 0.55), color=COL_DROP)
        self.link = curve(pos=[p + vector(0, 0.35, 0), d + vector(0, 0.35, 0)], color=COL_TASK_WAIT, radius=0.025)

    def update_visual(self):
        pulse = 0.18 + 0.06 * math.sin(sim_time * 4 + self.id)
        if self.status == "waiting":
            self.pick_marker.color = COL_TASK_WAIT
            self.pick_marker.radius = 0.25 + pulse
            self.link.color = COL_TASK_WAIT
            self.link.visible = True
            self.drop_marker.visible = True
        elif self.status in ["assigned", "picked"]:
            self.pick_marker.color = vector(0.2, 0.55, 1.0)
            self.pick_marker.radius = 0.20
            self.link.color = vector(0.2, 0.55, 1.0)
            self.link.visible = True
            self.drop_marker.visible = True
        elif self.status == "done":
            self.pick_marker.visible = False
            self.drop_marker.visible = False
            self.link.visible = False

    def remove_visual(self):
        self.pick_marker.visible = False
        self.drop_marker.visible = False
        self.link.visible = False


class Robot:
    def __init__(self, robot_id, start_cell):
        self.id = robot_id
        self.cell = start_cell
        self.pos = grid_to_world(start_cell) + vector(0, 0.38, 0)
        self.target_cell = start_cell
        self.path = []
        self.path_index = 0
        self.task = None
        self.state = "idle"  # idle, to_pickup, to_dropoff, waiting, charging
        self.wait_timer = 0.0
        self.replan_timer = random.uniform(0.0, REPLAN_INTERVAL)
        self.speed = BASE_ROBOT_SPEED * random.uniform(0.90, 1.12)
        self.battery = random.uniform(0.65, 1.0)
        self.completed = 0
        self.collision_waits = 0
        self.body = box(pos=self.pos, size=vector(1.0, 0.48, 1.0), color=COL_ROBOT_IDLE)
        self.top = cylinder(pos=self.pos + vector(0, 0.26, 0), axis=vector(0, 0.22, 0), radius=0.38, color=vector(0.12, 0.16, 0.22))
        self.sensor = sphere(pos=self.pos + vector(0, 0.55, 0), radius=0.13, color=vector(0.1, 0.95, 1.0), emissive=True)
        self.payload = box(pos=self.pos + vector(0, 0.78, 0), size=vector(0.55, 0.32, 0.55), color=COL_TASK_WAIT, visible=False)
        self.path_curve = None  # Created lazily after a path has at least two points
        self.trail_dots = deque(maxlen=MAX_TRAILS_PER_ROBOT)

    def set_path(self, path):
        self.path = path or [self.cell]
        self.path_index = 0
        self.update_path_curve()

    def update_path_curve(self):
        pts = []
        for c in self.path[self.path_index:]:
            pts.append(grid_to_world(c) + vector(0, 0.08, 0))

        # Some VPython versions crash when curve(pos=[]) is used.
        # Create the curve only when there are enough points to draw.
        if len(pts) < 2:
            if self.path_curve:
                self.path_curve.clear()
                self.path_curve.visible = False
            return

        if self.path_curve is None:
            self.path_curve = curve(color=COL_PATH, radius=0.035, visible=show_paths)
        else:
            self.path_curve.clear()

        self.path_curve.append(pts)
        if self.path_curve:
            self.path_curve.visible = show_paths

    def assign_task(self, task, congestion_cost):
        self.task = task
        task.assigned_robot = self
        task.status = "assigned"
        self.state = "to_pickup"
        self.target_cell = task.pickup
        self.set_path(astar(self.cell, task.pickup, congestion_cost))

    def current_goal(self):
        if self.state == "to_pickup" and self.task:
            return self.task.pickup
        if self.state == "to_dropoff" and self.task:
            return self.task.dropoff
        if self.state == "charging":
            return min(charging_stations, key=lambda c: manhattan(self.cell, c))
        return self.cell

    def maybe_replan(self, dt, congestion_cost):
        self.replan_timer -= dt
        if self.replan_timer <= 0:
            self.replan_timer = REPLAN_INTERVAL + random.uniform(-0.35, 0.35)
            goal = self.current_goal()
            if goal != self.cell:
                new_path = astar(self.cell, goal, congestion_cost)
                if new_path:
                    self.set_path(new_path)

    def next_cell(self):
        if not self.path or self.path_index >= len(self.path) - 1:
            return self.cell
        return self.path[self.path_index + 1]

    def can_enter_next(self, reserved_next, occupied_now):
        nxt = self.next_cell()
        if nxt == self.cell:
            return True
        if not avoid_collisions:
            return True
        if nxt in reserved_next:
            return False
        # Avoid driving into an occupied cell unless that robot is leaving.
        if nxt in occupied_now:
            return False
        return True

    def update(self, dt, reserved_next, occupied_now):
        self.battery = max(0.0, self.battery - dt * (0.0025 if self.state != "idle" else 0.0012))

        if self.wait_timer > 0:
            self.wait_timer -= dt
            self.body.color = COL_ROBOT_WAIT
            self.collision_waits += 1
            self.update_visual()
            return

        if self.state == "idle" and self.battery < 0.18:
            self.state = "charging"
            self.target_cell = min(charging_stations, key=lambda c: manhattan(self.cell, c))
            self.set_path(astar(self.cell, self.target_cell))

        if self.state == "charging" and self.cell in charging_stations:
            self.battery = min(1.0, self.battery + dt * 0.18)
            if self.battery >= 0.85:
                self.state = "idle"
            self.update_visual()
            return

        if not self.path or self.path_index >= len(self.path) - 1:
            self.arrive_at_goal()
            self.update_visual()
            return

        nxt = self.next_cell()
        if not self.can_enter_next(reserved_next, occupied_now):
            self.wait_timer = random.uniform(0.12, 0.35)
            self.body.color = COL_ROBOT_WAIT
            self.collision_waits += 1
            self.update_visual()
            return

        reserved_next.add(nxt)
        target_pos = grid_to_world(nxt) + vector(0, 0.38, 0)
        delta = target_pos - self.pos
        dist = mag(delta)
        step = self.speed * dt
        if dist <= step:
            self.pos = target_pos
            self.cell = nxt
            self.path_index += 1
            self.update_path_curve()
            self.add_trail_dot()
        else:
            self.pos += norm(delta) * step

        self.update_visual()

    def arrive_at_goal(self):
        if self.state == "to_pickup" and self.task and self.cell == self.task.pickup:
            self.task.status = "picked"
            self.state = "to_dropoff"
            self.payload.visible = True
            self.target_cell = self.task.dropoff
            self.set_path(astar(self.cell, self.task.dropoff))
        elif self.state == "to_dropoff" and self.task and self.cell == self.task.dropoff:
            self.task.status = "done"
            self.completed += 1
            self.payload.visible = False
            self.task = None
            self.state = "idle"
            self.set_path([self.cell])
        elif self.state == "charging" and self.cell in charging_stations:
            pass
        else:
            if self.state not in ["idle", "charging"]:
                self.state = "idle"

    def add_trail_dot(self):
        dot = sphere(pos=self.pos - vector(0, 0.27, 0), radius=0.12, color=vector(0.2, 0.45, 1.0), opacity=0.42)
        self.trail_dots.append({"obj": dot, "age": 0.0})

    def update_trails(self, dt):
        alive = deque(maxlen=MAX_TRAILS_PER_ROBOT)
        for item in list(self.trail_dots):
            item["age"] += dt
            age = item["age"]
            dot = item["obj"]
            fade = max(0.0, 1.0 - age / 3.2)
            dot.opacity = 0.42 * fade
            dot.radius = 0.12 * (0.55 + 0.45 * fade)
            dot.pos.y -= dt * 0.015
            if fade > 0.02:
                alive.append(item)
            else:
                dot.visible = False
        self.trail_dots = alive

    def update_visual(self):
        if self.state == "idle":
            self.body.color = COL_ROBOT_IDLE
        elif self.wait_timer > 0:
            self.body.color = COL_ROBOT_WAIT
        elif self.state == "charging":
            self.body.color = vector(0.45, 0.35, 0.95)
        else:
            self.body.color = COL_ROBOT_BUSY

        self.body.pos = self.pos
        self.top.pos = self.pos + vector(0, 0.26, 0)
        self.sensor.pos = self.pos + vector(0, 0.55 + 0.04 * math.sin(sim_time * 7 + self.id), 0)
        self.payload.pos = self.pos + vector(0, 0.78, 0)
        if self.path_curve:
            self.path_curve.visible = show_paths

    def remove(self):
        for obj in [self.body, self.top, self.sensor, self.payload]:
            obj.visible = False
        if self.path_curve:
            self.path_curve.visible = False
        for item in self.trail_dots:
            item["obj"].visible = False

# -----------------------------
# Dynamic state
# -----------------------------
robots = []
tasks = []
completed_tasks = 0
collision_warnings = []


def congestion_cost_map():
    cost = defaultdict(float)
    for r in robots:
        for c in r.path[r.path_index:r.path_index + 5]:
            cost[c] += 0.25
        cost[r.cell] += 0.75
    return cost


def create_task():
    global next_task_id
    pickup = random.choice(pickup_stations)
    # Sometimes pick shelf-adjacent open cells to simulate inventory retrieval.
    if random.random() < 0.65:
        candidates = []
        for s in shelf_cells:
            for n in neighbors(s):
                candidates.append(n)
        if candidates:
            pickup = random.choice(candidates)
    dropoff = random.choice(dropoff_stations)
    task = Task(next_task_id, pickup, dropoff)
    next_task_id += 1
    tasks.append(task)
    return task


def add_robot():
    robot_id = len(robots) + 1
    start = random.choice(charging_stations + pickup_stations)
    # Try to avoid exact overlap at spawn.
    used = {r.cell for r in robots}
    if start in used:
        open_starts = [c for c in charging_stations + pickup_stations + dropoff_stations if c not in used]
        if open_starts:
            start = random.choice(open_starts)
        else:
            start = random_open_cell()
    robots.append(Robot(robot_id, start))


def remove_robot():
    if robots:
        r = robots.pop()
        if r.task and r.task.status != "done":
            r.task.status = "waiting"
            r.task.assigned_robot = None
        r.remove()


def assign_tasks():
    waiting = [t for t in tasks if t.status == "waiting"]
    idle = [r for r in robots if r.state == "idle" and r.battery > 0.20]
    if not waiting or not idle:
        return
    cost = congestion_cost_map()
    for task in waiting:
        if not idle:
            break
        # Choose robot by distance plus battery penalty.
        robot = min(idle, key=lambda r: manhattan(r.cell, task.pickup) + (1.0 - r.battery) * 5.0)
        robot.assign_task(task, cost)
        idle.remove(robot)


def cleanup_done_tasks():
    global completed_tasks
    keep = []
    for t in tasks:
        if t.status == "done":
            t.age += 1
            t.update_visual()
            completed_tasks += 1
            t.remove_visual()
        else:
            keep.append(t)
    tasks[:] = keep


def reset_simulation():
    global tasks, completed_tasks, next_task_id, sim_time, last_auto_task, follow_index
    for t in tasks:
        t.remove_visual()
    for r in robots:
        r.remove()
    tasks = []
    robots.clear()
    completed_tasks = 0
    next_task_id = 1
    sim_time = 0.0
    last_auto_task = 0.0
    follow_index = 0
    for _ in range(6):
        add_robot()
    for _ in range(8):
        create_task()

# -----------------------------
# Metrics and visualization
# -----------------------------
metric_bars = []
for i in range(4):
    base = vector(-20.5, 2.8 - i * 0.75, -12.5)
    label(pos=base + vector(0, 0.18, 0), text=["Queue", "Busy", "Battery", "Waits"][i], height=8, align="left", box=False, opacity=0, color=vector(0.08, 0.08, 0.08))
    bg = box(pos=base + vector(3.2, 0, 0), size=vector(4.0, 0.18, 0.18), color=vector(0.78, 0.78, 0.78))
    fg = box(pos=base + vector(1.25, 0.01, 0), size=vector(0.1, 0.22, 0.22), color=vector(0.2, 0.55, 0.95))
    metric_bars.append((bg, fg, base))


def set_bar(index, value, max_value, col):
    value = max(0.0, min(value, max_value))
    frac = value / max_value if max_value > 0 else 0
    bg, fg, base = metric_bars[index]
    width = max(0.04, 4.0 * frac)
    fg.size = vector(width, 0.22, 0.22)
    fg.pos = base + vector(1.2 + width * 0.5, 0.01, 0)
    fg.color = col


def update_hud():
    waiting = sum(1 for t in tasks if t.status == "waiting")
    assigned = sum(1 for t in tasks if t.status in ["assigned", "picked"])
    busy = sum(1 for r in robots if r.state not in ["idle"])
    avg_batt = sum(r.battery for r in robots) / len(robots) if robots else 0
    waits = sum(r.collision_waits for r in robots)
    hud.text = (
        "ROBOT WAREHOUSE SYSTEM\n"
        f"time: {sim_time:5.1f}s\n"
        f"robots: {len(robots)}   busy: {busy}\n"
        f"waiting tasks: {waiting}\n"
        f"active tasks: {assigned}\n"
        f"completed tasks: {completed_tasks}\n"
        f"avg battery: {avg_batt * 100:4.0f}%\n"
        f"collision waits: {waits}\n"
        f"auto tasks: {'ON' if auto_tasks else 'OFF'}\n"
        f"avoid collisions: {'ON' if avoid_collisions else 'OFF'}\n"
        f"paths: {'ON' if show_paths else 'OFF'}\n\n"
        "Controls: Space r t +/- a c p v 1-4 n"
    )
    set_bar(0, waiting, 16, vector(0.95, 0.75, 0.15))
    set_bar(1, busy, max(1, len(robots)), vector(0.95, 0.48, 0.12))
    set_bar(2, avg_batt, 1.0, vector(0.22, 0.70, 0.35))
    set_bar(3, min(waits / 30.0, 1.0), 1.0, vector(0.90, 0.20, 0.15))


def update_dispatch_board():
    lines = ["DISPATCH BOARD"]
    shown = 0
    for t in tasks[:10]:
        rid = t.assigned_robot.id if t.assigned_robot else "--"
        lines.append(f"T{t.id:02d} {t.status:8s} P{t.pickup}->D{t.dropoff} R{rid}")
        shown += 1
    if len(tasks) > shown:
        lines.append(f"... {len(tasks) - shown} more tasks")
    if robots:
        r = robots[follow_index % len(robots)]
        lines.append("")
        lines.append(f"FOLLOW R{r.id}: {r.state}")
        lines.append(f"cell: {r.cell}  battery: {r.battery*100:.0f}%")
        lines.append(f"completed: {r.completed}  waits: {r.collision_waits}")
    dispatch_label.text = "\n".join(lines)


def smooth_vec(current, desired, gain, dt):
    # Frame-rate independent exponential easing.
    alpha = 1.0 - math.exp(-gain * max(0.0, dt))
    return current + (desired - current) * alpha


def smooth_float(current, desired, gain, dt):
    alpha = 1.0 - math.exp(-gain * max(0.0, dt))
    return current + (desired - current) * alpha


def best_follow_robot():
    if not robots:
        return None
    busy = [r for r in robots if r.state in ["to_pickup", "to_dropoff", "waiting"] or r.wait_timer > 0]
    pool = busy if busy else robots
    return pool[follow_index % len(pool)]


def robot_motion_direction(r):
    if r and r.path and r.path_index < len(r.path) - 1:
        nxt = grid_to_world(r.path[r.path_index + 1]) + vector(0, 0.38, 0)
        delta = nxt - r.pos
        if mag(delta) > 0.05:
            return norm(delta)
    # Fallback diagonal keeps the follow camera readable when the robot is stopped.
    return norm(vector(1, 0, 1))


def cinematic_shot_id():
    # Slow automatic shot cycle. Mode 0 means auto; numbered modes remain manual.
    return int(sim_time // 9.5) % 5 + 1


def update_camera(dt):
    if not cinematic_camera:
        return

    global camera_pos_smooth, camera_center_smooth, camera_range_smooth, camera_last_mode

    shot = cinematic_shot_id() if camera_mode == 0 else camera_mode

    # Soft cut when switching shots: initialize the smoother near the new shot so
    # the transition is intentional, not a long accidental drift across the scene.
    switching = camera_last_mode != shot
    camera_last_mode = shot

    desired_center = vector(0, 0, 0)
    desired_pos = vector(30, 22, 28)
    desired_range = 20.0

    # 1. Slow orbit around the full warehouse. Good overview without losing context.
    if shot == 1:
        angle = sim_time * 0.10
        radius = 35.0
        desired_center = vector(0, 0.2, 0)
        desired_pos = vector(math.cos(angle) * radius, 25.0, math.sin(angle) * radius)
        desired_range = 22.0

    # 2. Predictive close follow. Looks ahead along the robot path instead of
    # staring exactly at the robot body.
    elif shot == 2:
        r = best_follow_robot()
        if r:
            direction = robot_motion_direction(r)
            side = norm(vector(-direction.z, 0, direction.x))
            lookahead = direction * 3.2
            desired_center = r.pos + lookahead + vector(0, 0.75, 0)
            desired_pos = r.pos - direction * 7.0 + side * 2.8 + vector(0, 4.2, 0)
            desired_range = 7.0

    # 3. Top-down planning view. This is steady and does not rotate, so paths,
    # congestion, and task routing are easy to read.
    elif shot == 3:
        desired_center = vector(0, 0, 0)
        desired_pos = vector(0.01, 38.0, 0.01)
        desired_range = 18.5

    # 4. Low aisle dolly view. Camera glides along an aisle while looking across
    # shelves, useful for seeing near-collisions and robot spacing.
    elif shot == 4:
        lane_z = -HALF_H + 4.0 + (math.sin(sim_time * 0.18) + 1.0) * (HALF_H - 4.0)
        x_slide = math.sin(sim_time * 0.11) * 7.0
        desired_center = vector(x_slide, 0.75, lane_z + 3.8)
        desired_pos = vector(x_slide - 20.0, 4.8, lane_z - 4.2)
        desired_range = 11.5

    # 5. Dispatch board / queue perspective: pulled back enough to include the
    # side board and the busy warehouse. Automatic cycle uses this extra shot.
    elif shot == 5:
        desired_center = vector(6.0, 1.2, 0.0)
        desired_pos = vector(29.0, 14.0, 21.0)
        desired_range = 20.0

    if switching:
        camera_pos_smooth = camera_pos_smooth * 0.35 + desired_pos * 0.65
        camera_center_smooth = camera_center_smooth * 0.35 + desired_center * 0.65
        camera_range_smooth = camera_range_smooth * 0.35 + desired_range * 0.65

    gain = 1.7 if shot in [1, 3, 5] else 2.7
    camera_pos_smooth = smooth_vec(camera_pos_smooth, desired_pos, gain, dt)
    camera_center_smooth = smooth_vec(camera_center_smooth, desired_center, gain, dt)
    camera_range_smooth = smooth_float(camera_range_smooth, desired_range, gain, dt)

    scene.camera.pos = camera_pos_smooth
    scene.center = camera_center_smooth
    scene.camera.axis = camera_center_smooth - camera_pos_smooth
    scene.up = vector(0, 1, 0)
    scene.range = camera_range_smooth

# -----------------------------
# Keyboard controls
# -----------------------------
def on_keydown(evt):
    global paused, auto_tasks, avoid_collisions, show_paths, cinematic_camera, camera_mode, follow_index
    key = evt.key
    if key == " ":
        paused = not paused
    elif key == "r":
        reset_simulation()
    elif key == "t":
        create_task()
    elif key in ["+", "="]:
        add_robot()
    elif key == "-":
        remove_robot()
    elif key == "a":
        auto_tasks = not auto_tasks
    elif key == "c":
        avoid_collisions = not avoid_collisions
    elif key == "p":
        show_paths = not show_paths
        for r in robots:
            if r.path_curve:
                r.path_curve.visible = show_paths
    elif key == "v":
        cinematic_camera = not cinematic_camera
    elif key in ["1", "2", "3", "4"]:
        camera_mode = int(key)
        cinematic_camera = True
    elif key == "n":
        if robots:
            follow_index = (follow_index + 1) % len(robots)
            camera_mode = 2
            cinematic_camera = True

scene.bind("keydown", on_keydown)

# -----------------------------
# Main loop
# -----------------------------
reset_simulation()

while True:
    rate(60)
    dt = 1.0 / 60.0
    if paused:
        update_hud()
        update_dispatch_board()
        continue

    sim_time += dt

    if auto_tasks and sim_time - last_auto_task > TASK_AUTO_INTERVAL:
        last_auto_task = sim_time
        if len(tasks) < 22:
            # Add bursts sometimes to show scheduling pressure.
            for _ in range(1 + (1 if random.random() < 0.25 else 0)):
                create_task()

    assign_tasks()
    cost = congestion_cost_map()

    # Replanning with congestion cost.
    for r in robots:
        if avoid_collisions and r.state not in ["idle"]:
            r.maybe_replan(dt, cost)

    occupied_now = {r.cell for r in robots}
    reserved_next = set()
    # Update order rotates so the same robot does not always get priority.
    ordered = robots[:]
    random.shuffle(ordered)
    for r in ordered:
        r.update(dt, reserved_next, occupied_now)
        r.update_trails(dt)

    for t in tasks:
        t.age += dt
        t.update_visual()
    cleanup_done_tasks()

    update_hud()
    update_dispatch_board()
    update_camera(dt)
