"""
VPython Immune System Response Simulation
Biology + infection spread + immune-cell targeting

Run:
    python vpython_immune_system_response_fading_trails.py

Controls:
    Space : pause / resume
    r     : reset simulation
    i     : add infection burst
    +/=   : add immune cells
    -     : remove immune cells
    c     : toggle chemokine field
    t     : toggle immune targeting
    a     : toggle auto-looping infection rounds
    v     : toggle cinematic camera
    n     : follow next immune cell

Notes:
    This is a conceptual biology visualization, not a medical model.
"""

from vpython import *
import random
import math

# -----------------------------
# Scene setup
# -----------------------------
scene = canvas(
    title="Immune System Response — Infection Spread, Targeting, and Clearance",
    width=1280,
    height=760,
    background=vector(0.92, 0.96, 1.0),
    center=vector(0, 0, 0),
)
scene.forward = vector(-0.65, -0.55, -0.52)
scene.range = 46
scene.caption = """
Immune system response simulation\n
Legend:\n
  Red/orange spheres   = pathogen particles / infected tissue zones\n  Blue-white spheres   = immune cells\n  Purple particles     = chemokine signal released by infection\n  Green flashes        = cleared infection events\n  Tissue tiles         = healthy, stressed, infected, or recovering cells\n
Controls:\n  Space pause/resume | r reset | i infection burst | +/- immune count\n  c chemokine field | t targeting | a auto-loop | v camera | n next followed cell\n"""

# -----------------------------
# Constants and state
# -----------------------------
GRID_N = 17
TILE_SIZE = 3.0
WORLD_HALF = (GRID_N - 1) * TILE_SIZE * 0.5
MAX_PATHOGENS = 150
MAX_SIGNALS = 180
MAX_FLASHES = 70
MAX_TRAIL_DOTS_PER_CELL = 42
TRAIL_LIFETIME = 3.6
TRAIL_DROP_INTERVAL = 0.12

HEALTHY = 0
STRESSED = 1
INFECTED = 2
RECOVERING = 3

paused = False
targeting_enabled = True
chemokines_enabled = True
auto_loop = True
cinematic_camera = True
follow_index = 0
round_number = 1
sim_time = 0.0
round_timer = 0.0
infection_seed_timer = 0.0
pressure_like_infection_load = 0.0

random.seed(17)

# Visual containers
terrain = []
tissue = []
pathogens = []
immune_cells = []
signals = []
flashes = []
labels = []

# -----------------------------
# Utility functions
# -----------------------------
def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def mag2(v):
    return v.x * v.x + v.y * v.y + v.z * v.z


def rand_vec2(scale=1.0):
    angle = random.random() * 2 * math.pi
    return vector(math.cos(angle) * scale, 0, math.sin(angle) * scale)


def bounded_position(pos, margin=2.0):
    pos.x = clamp(pos.x, -WORLD_HALF - margin, WORLD_HALF + margin)
    pos.z = clamp(pos.z, -WORLD_HALF - margin, WORLD_HALF + margin)
    return pos


def color_for_state(state, infection, recovery):
    if state == HEALTHY:
        return vector(0.64, 0.90, 0.70)
    if state == STRESSED:
        return vector(0.95, 0.82, 0.43)
    if state == INFECTED:
        return vector(1.0, clamp(0.34 + 0.22 * (1 - infection), 0.12, 0.45), 0.18)
    if state == RECOVERING:
        return vector(0.45, 0.78, 0.92 + 0.05 * math.sin(recovery))
    return color.white


def nearest_tile(pos):
    best = None
    best_d = 1e9
    for cell in tissue:
        d = mag2(cell["pos"] - pos)
        if d < best_d:
            best_d = d
            best = cell
    return best


def infection_center():
    total = vector(0, 0, 0)
    count = 0
    for cell in tissue:
        if cell["state"] == INFECTED:
            total += cell["pos"]
            count += 1
    if count > 0:
        return total / count
    if pathogens:
        total = vector(0, 0, 0)
        for p in pathogens:
            total += p["pos"]
        return total / len(pathogens)
    return vector(0, 0, 0)


def infection_load():
    load = len(pathogens) * 0.45
    for cell in tissue:
        if cell["state"] == INFECTED:
            load += 1.0 + cell["infection"]
        elif cell["state"] == STRESSED:
            load += 0.25
    return load


def clear_scene_objects():
    for group in [terrain, labels]:
        for obj in group:
            obj.visible = False
        group.clear()
    for group in [pathogens, immune_cells, signals, flashes]:
        for item in group:
            item["obj"].visible = False
            if "trail" in item and item["trail"]:
                # Trail is now stored as fading dots, not a permanent curve.
                for dot in item["trail"]:
                    dot["obj"].visible = False
                item["trail"].clear()
        group.clear()
    for cell in tissue:
        cell["obj"].visible = False
    tissue.clear()

# -----------------------------
# Build scene
# -----------------------------
def make_static_geography():
    # Petri-dish/tissue plane
    base = cylinder(
        pos=vector(0, -0.24, 0),
        axis=vector(0, 0.16, 0),
        radius=WORLD_HALF + 7,
        color=vector(0.78, 0.88, 0.94),
        opacity=0.35,
    )
    rim = ring(
        pos=vector(0, -0.13, 0),
        axis=vector(0, 1, 0),
        radius=WORLD_HALF + 7,
        thickness=0.25,
        color=vector(0.55, 0.66, 0.76),
        opacity=0.7,
    )
    terrain.extend([base, rim])

    # Blood vessel / immune entry lane
    vessel = cylinder(
        pos=vector(-WORLD_HALF - 7.5, 0.35, -WORLD_HALF - 4),
        axis=vector(0, 0, WORLD_HALF * 2 + 8),
        radius=1.05,
        color=vector(0.85, 0.16, 0.20),
        opacity=0.42,
    )
    vessel_label = label(
        pos=vector(-WORLD_HALF - 8, 3.2, 0),
        text="immune entry vessel",
        height=12,
        box=False,
        color=vector(0.35, 0.05, 0.08),
    )
    terrain.append(vessel)
    labels.append(vessel_label)


def make_tissue_grid():
    for ix in range(GRID_N):
        for iz in range(GRID_N):
            x = (ix - (GRID_N - 1) / 2) * TILE_SIZE
            z = (iz - (GRID_N - 1) / 2) * TILE_SIZE
            pos = vector(x, 0, z)
            tile = box(
                pos=pos,
                size=vector(TILE_SIZE * 0.88, 0.20, TILE_SIZE * 0.88),
                color=color_for_state(HEALTHY, 0, 0),
                opacity=0.74,
            )
            tissue.append(
                {
                    "obj": tile,
                    "pos": pos,
                    "ix": ix,
                    "iz": iz,
                    "state": HEALTHY,
                    "infection": 0.0,
                    "chemokine": 0.0,
                    "recovery": 0.0,
                    "stress": random.random() * 0.05,
                }
            )


def spawn_pathogen(pos=None, burst=False):
    if len(pathogens) >= MAX_PATHOGENS:
        return
    if pos is None:
        pos = vector(random.uniform(-8, 8), 1.2, random.uniform(-8, 8))
    radius = random.uniform(0.26, 0.48) * (1.35 if burst else 1.0)
    obj = sphere(
        pos=pos + vector(random.uniform(-1.0, 1.0), random.uniform(0.35, 1.2), random.uniform(-1.0, 1.0)),
        radius=radius,
        color=vector(1.0, random.uniform(0.16, 0.34), 0.05),
        emissive=True,
        opacity=0.88,
    )
    pathogens.append(
        {
            "obj": obj,
            "pos": obj.pos,
            "vel": rand_vec2(random.uniform(0.025, 0.09)),
            "age": 0.0,
            "infectivity": random.uniform(0.55, 1.15),
            "split_timer": random.uniform(2.5, 6.0),
        }
    )


def spawn_signal(pos, strength=1.0):
    if not chemokines_enabled or len(signals) >= MAX_SIGNALS:
        return
    obj = sphere(
        pos=pos + vector(random.uniform(-0.35, 0.35), random.uniform(0.18, 1.5), random.uniform(-0.35, 0.35)),
        radius=0.11 + 0.07 * strength,
        color=vector(0.70, 0.22, 1.0),
        opacity=0.45,
        emissive=True,
    )
    signals.append(
        {
            "obj": obj,
            "pos": obj.pos,
            "vel": rand_vec2(random.uniform(0.01, 0.035)) + vector(0, random.uniform(0.008, 0.025), 0),
            "age": 0.0,
            "life": random.uniform(3.0, 7.0),
        }
    )


def spawn_flash(pos, kind="clear"):
    if len(flashes) >= MAX_FLASHES:
        old = flashes.pop(0)
        old["obj"].visible = False
    col = vector(0.20, 1.0, 0.32) if kind == "clear" else vector(1.0, 0.8, 0.1)
    obj = sphere(pos=pos + vector(0, 1.1, 0), radius=0.18, color=col, opacity=0.78, emissive=True)
    flashes.append({"obj": obj, "age": 0.0, "life": 1.2, "base": obj.radius})


def spawn_immune_cell(pos=None):
    if pos is None:
        pos = vector(-WORLD_HALF - 7.5, 1.1, random.uniform(-WORLD_HALF, WORLD_HALF))
    body = sphere(pos=pos, radius=random.uniform(0.55, 0.78), color=vector(0.65, 0.88, 1.0), emissive=True)
    ring_marker = ring(pos=pos, axis=vector(0, 1, 0), radius=body.radius * 1.45, thickness=0.035, color=vector(0.25, 0.55, 1.0), opacity=0.9)
    immune_cells.append(
        {
            "obj": body,
            "ring": ring_marker,
            "trail": [],
            "trail_timer": random.random() * TRAIL_DROP_INTERVAL,
            "pos": pos,
            "vel": vector(random.uniform(0.02, 0.08), 0, random.uniform(-0.05, 0.05)),
            "mode": "patrol",
            "target": None,
            "cooldown": random.uniform(0.0, 1.0),
            "energy": random.uniform(0.82, 1.0),
            "wander_phase": random.random() * 100,
        }
    )


def seed_initial_infection():
    for _ in range(12):
        spawn_pathogen(vector(random.uniform(-6, 6), 1.0, random.uniform(-6, 6)), burst=True)
    for _ in range(5):
        cell = nearest_tile(vector(random.uniform(-5, 5), 0, random.uniform(-5, 5)))
        if cell:
            cell["state"] = INFECTED
            cell["infection"] = random.uniform(0.35, 0.8)
            cell["chemokine"] = random.uniform(0.55, 1.0)


def reset_simulation():
    global sim_time, round_timer, infection_seed_timer, pressure_like_infection_load, round_number, follow_index
    clear_scene_objects()
    sim_time = 0.0
    round_timer = 0.0
    infection_seed_timer = 0.0
    pressure_like_infection_load = 0.0
    follow_index = 0
    make_static_geography()
    make_tissue_grid()
    for _ in range(22):
        spawn_immune_cell()
    seed_initial_infection()

# -----------------------------
# Simulation update logic
# -----------------------------
def update_tissue(dt):
    for cell in tissue:
        if cell["state"] == INFECTED:
            cell["infection"] += dt * random.uniform(0.025, 0.07)
            cell["chemokine"] = clamp(cell["chemokine"] + dt * 0.16, 0, 1.5)
            if random.random() < 0.038:
                spawn_signal(cell["pos"], clamp(cell["chemokine"], 0.4, 1.4))
            if cell["infection"] > 1.0:
                cell["infection"] = 1.0
                # local spread to neighbors
                if random.random() < 0.025:
                    nx = cell["ix"] + random.choice([-1, 0, 1])
                    nz = cell["iz"] + random.choice([-1, 0, 1])
                    for nb in tissue:
                        if nb["ix"] == nx and nb["iz"] == nz and nb["state"] in (HEALTHY, STRESSED):
                            nb["state"] = STRESSED
                            nb["stress"] = 0.65
                            break
        elif cell["state"] == STRESSED:
            cell["stress"] += dt * 0.045
            cell["chemokine"] = clamp(cell["chemokine"] + dt * 0.08, 0, 0.7)
            if cell["stress"] > random.uniform(0.72, 1.25):
                cell["state"] = INFECTED
                cell["infection"] = 0.22
        elif cell["state"] == RECOVERING:
            cell["recovery"] += dt
            cell["infection"] = max(0, cell["infection"] - dt * 0.18)
            cell["chemokine"] = max(0, cell["chemokine"] - dt * 0.14)
            if cell["recovery"] > 7.0:
                cell["state"] = HEALTHY
                cell["recovery"] = 0.0
                cell["stress"] = 0.0
        else:
            cell["chemokine"] = max(0, cell["chemokine"] - dt * 0.035)

        cell["obj"].color = color_for_state(cell["state"], cell["infection"], cell["recovery"])
        cell["obj"].height = 0.20 + 0.22 * cell["infection"] + 0.07 * cell["chemokine"]
        cell["obj"].pos.y = cell["obj"].height * 0.5 - 0.05
        cell["obj"].opacity = 0.64 + 0.2 * min(1.0, cell["chemokine"])


def update_pathogens(dt):
    global infection_seed_timer
    remove = []
    for p in pathogens:
        p["age"] += dt
        p["split_timer"] -= dt
        center = infection_center()
        away_or_spread = norm(p["pos"] - center) if mag(p["pos"] - center) > 0.1 else rand_vec2(1)
        p["vel"] += away_or_spread * 0.006 * dt + rand_vec2(0.014)
        p["vel"] *= 0.985
        if mag(p["vel"]) > 0.16:
            p["vel"] = norm(p["vel"]) * 0.16
        p["pos"] += p["vel"]
        p["pos"] = bounded_position(p["pos"])
        p["pos"].y = 1.0 + 0.35 * math.sin(sim_time * 3.0 + p["age"])
        p["obj"].pos = p["pos"]
        p["obj"].radius = p["obj"].radius * 0.995 + (0.33 + 0.08 * math.sin(sim_time * 5 + p["age"])) * 0.005

        tile = nearest_tile(p["pos"])
        if tile and mag(tile["pos"] - p["pos"]) < TILE_SIZE * 1.15:
            if tile["state"] == HEALTHY and random.random() < 0.018 * p["infectivity"]:
                tile["state"] = STRESSED
                tile["stress"] = 0.42
            elif tile["state"] == STRESSED and random.random() < 0.022 * p["infectivity"]:
                tile["state"] = INFECTED
                tile["infection"] = 0.20

        if p["split_timer"] <= 0 and len(pathogens) < MAX_PATHOGENS and random.random() < 0.55:
            spawn_pathogen(p["pos"] + rand_vec2(0.8))
            p["split_timer"] = random.uniform(5.0, 9.0)

        if p["age"] > 95:
            remove.append(p)

    for p in remove:
        if p in pathogens:
            p["obj"].visible = False
            pathogens.remove(p)


def best_target_for(cell):
    best = None
    best_score = -1e9
    pos = cell["pos"]

    for p in pathogens:
        d = mag(p["pos"] - pos)
        score = 30.0 / (1.0 + d) + 4.0
        if score > best_score:
            best_score = score
            best = ("pathogen", p)

    for tile in tissue:
        if tile["state"] == INFECTED:
            d = mag(tile["pos"] - pos)
            score = 22.0 / (1.0 + d) + 12.0 * tile["chemokine"] + 4 * tile["infection"]
            if score > best_score:
                best_score = score
                best = ("tile", tile)
        elif tile["state"] == STRESSED and tile["chemokine"] > 0.25:
            d = mag(tile["pos"] - pos)
            score = 10.0 / (1.0 + d) + 4.0 * tile["chemokine"]
            if score > best_score:
                best_score = score
                best = ("tile", tile)
    return best


def update_immune_cells(dt):
    for idx, cell in enumerate(immune_cells):
        body = cell["obj"]
        ring_obj = cell["ring"]
        cell["cooldown"] = max(0, cell["cooldown"] - dt)
        cell["wander_phase"] += dt

        target = best_target_for(cell) if targeting_enabled else None
        if target:
            cell["mode"] = "targeting"
            cell["target"] = target
            target_pos = target[1]["pos"] if target[0] == "tile" else target[1]["pos"]
            desired = target_pos - cell["pos"]
            desired.y = 0
            if mag(desired) > 0.01:
                desired = norm(desired)
            chem_boost = 1.0 + 0.65 * min(1.0, infection_load() / 80.0)
            cell["vel"] += desired * 0.04 * chem_boost
        else:
            cell["mode"] = "patrol"
            patrol_wave = vector(
                math.sin(cell["wander_phase"] * 1.6 + idx) * 0.018,
                0,
                math.cos(cell["wander_phase"] * 1.3 + idx * 0.7) * 0.018,
            )
            # drift from blood vessel into tissue
            if cell["pos"].x < -WORLD_HALF + 3:
                patrol_wave += vector(0.035, 0, 0)
            cell["vel"] += patrol_wave + rand_vec2(0.009)

        speed_limit = 0.19 if cell["mode"] == "targeting" else 0.115
        if mag(cell["vel"]) > speed_limit:
            cell["vel"] = norm(cell["vel"]) * speed_limit
        cell["vel"] *= 0.97
        cell["pos"] += cell["vel"]
        cell["pos"] = bounded_position(cell["pos"], 8.0)
        cell["pos"].y = 1.05 + 0.12 * math.sin(sim_time * 4 + idx)

        # Interactions: engulf pathogen or heal infected tissue
        if cell["cooldown"] <= 0:
            for p in list(pathogens):
                if mag(p["pos"] - cell["pos"]) < body.radius + p["obj"].radius + 0.35:
                    spawn_flash(p["pos"], "clear")
                    p["obj"].visible = False
                    pathogens.remove(p)
                    cell["cooldown"] = random.uniform(0.35, 0.8)
                    cell["energy"] = max(0.35, cell["energy"] - 0.035)
                    break

            tile = nearest_tile(cell["pos"])
            if tile and mag(tile["pos"] - cell["pos"]) < TILE_SIZE * 0.95:
                if tile["state"] == INFECTED:
                    tile["infection"] -= dt * 0.9
                    tile["chemokine"] = max(0, tile["chemokine"] - dt * 0.4)
                    if random.random() < 0.05:
                        spawn_flash(tile["pos"], "clear")
                    if tile["infection"] <= 0.08:
                        tile["state"] = RECOVERING
                        tile["recovery"] = 0.0
                        tile["infection"] = 0.05
                elif tile["state"] == STRESSED:
                    tile["stress"] -= dt * 0.32
                    if tile["stress"] <= 0.1:
                        tile["state"] = RECOVERING
                        tile["recovery"] = 0.0

        body.pos = cell["pos"]
        ring_obj.pos = cell["pos"]
        ring_obj.axis = vector(0, 1, 0)
        ring_obj.radius = body.radius * (1.35 + 0.08 * math.sin(sim_time * 8 + idx))
        body.color = vector(0.55, 0.82, 1.0) if cell["mode"] == "patrol" else vector(0.90, 0.96, 1.0)
        ring_obj.color = vector(0.18, 0.48, 1.0) if cell["mode"] == "patrol" else vector(0.05, 0.85, 1.0)
        add_immune_trail_dot(cell, idx, dt)


def add_immune_trail_dot(cell, idx, dt):
    """Leave short-lived immune-cell trail dots that fade and then disappear."""
    cell["trail_timer"] -= dt
    if cell["trail_timer"] > 0:
        return

    cell["trail_timer"] = TRAIL_DROP_INTERVAL

    # Keep each cell trail bounded so old dots cannot accumulate forever.
    while len(cell["trail"]) >= MAX_TRAIL_DOTS_PER_CELL:
        old = cell["trail"].pop(0)
        old["obj"].visible = False

    mode_color = vector(0.28, 0.52, 1.0) if cell["mode"] == "patrol" else vector(0.05, 0.90, 1.0)
    dot = sphere(
        pos=cell["pos"] - cell["vel"] * 2.0 + vector(0, -0.12, 0),
        radius=0.10 if cell["mode"] == "patrol" else 0.13,
        color=mode_color,
        opacity=0.42,
        emissive=True,
    )
    cell["trail"].append(
        {
            "obj": dot,
            "age": 0.0,
            "life": TRAIL_LIFETIME * random.uniform(0.75, 1.15),
            "base_radius": dot.radius,
        }
    )


def update_immune_trails(dt):
    for cell in immune_cells:
        for dot in list(cell["trail"]):
            dot["age"] += dt
            fade = 1.0 - dot["age"] / dot["life"]
            if fade <= 0:
                dot["obj"].visible = False
                cell["trail"].remove(dot)
                continue
            dot["obj"].opacity = 0.42 * fade
            dot["obj"].radius = dot["base_radius"] * (0.45 + 0.55 * fade)
            dot["obj"].pos.y = max(0.35, dot["obj"].pos.y - dt * 0.018)


def update_signals_and_flashes(dt):
    for s in list(signals):
        s["age"] += dt
        s["pos"] += s["vel"]
        s["obj"].pos = s["pos"]
        s["obj"].opacity = max(0, 0.55 * (1 - s["age"] / s["life"]))
        s["obj"].radius *= 1.002
        if s["age"] >= s["life"]:
            s["obj"].visible = False
            signals.remove(s)

    for f in list(flashes):
        f["age"] += dt
        pulse = f["age"] / f["life"]
        f["obj"].radius = f["base"] + 1.25 * pulse
        f["obj"].opacity = max(0, 0.75 * (1 - pulse))
        if f["age"] >= f["life"]:
            f["obj"].visible = False
            flashes.remove(f)


def auto_round_logic(dt):
    global round_timer, round_number, infection_seed_timer
    round_timer += dt
    infection_seed_timer += dt
    load = infection_load()

    # Add occasional infection bursts early or if immune system cleared everything.
    if auto_loop:
        if infection_seed_timer > 16 and load < 18:
            burst_center = vector(random.uniform(-10, 10), 1.0, random.uniform(-10, 10))
            for _ in range(random.randint(8, 16)):
                spawn_pathogen(burst_center + rand_vec2(random.uniform(0.2, 3.0)), burst=True)
            cell = nearest_tile(burst_center)
            if cell:
                cell["state"] = INFECTED
                cell["infection"] = 0.35
                cell["chemokine"] = 0.8
            infection_seed_timer = 0.0

        # Reset into a new round after major clearance and enough time has passed.
        if round_timer > 75 and load < 8:
            round_number += 1
            reset_simulation()


def update_hud():
    infected_tiles = sum(1 for c in tissue if c["state"] == INFECTED)
    stressed_tiles = sum(1 for c in tissue if c["state"] == STRESSED)
    recovering_tiles = sum(1 for c in tissue if c["state"] == RECOVERING)
    load = infection_load()
    scene.title = (
        "Immune System Response — "
        f"Round {round_number} | Load {load:05.1f} | Pathogens {len(pathogens)} | "
        f"Infected {infected_tiles} | Stressed {stressed_tiles} | Recovering {recovering_tiles} | "
        f"Immune cells {len(immune_cells)}"
    )


def update_camera(dt):
    if not cinematic_camera:
        return
    if not immune_cells:
        return
    cycle = (sim_time % 42.0)
    center = infection_center()
    if cycle < 13:
        # Orbit whole tissue field
        angle = sim_time * 0.18
        scene.center = vector(0, 0.5, 0) * 0.96 + center * 0.04
        scene.range = 45
        scene.forward = norm(vector(math.cos(angle) * -0.85, -0.52, math.sin(angle) * -0.85))
    elif cycle < 24:
        # Follow an immune cell approaching infection
        target_cell = immune_cells[follow_index % len(immune_cells)]
        scene.center = target_cell["pos"] * 0.75 + center * 0.25
        scene.range = 17
        direction = norm(target_cell["pos"] - center + vector(0.1, -5.0, 0.1))
        scene.forward = direction
    elif cycle < 33:
        # Overhead planning view
        scene.center = vector(0, 0, 0)
        scene.range = 39
        scene.forward = vector(0, -1, -0.03)
    else:
        # Low tissue-level tracking view
        scene.center = center * 0.8 + vector(0, 1.2, 0)
        scene.range = 25
        angle = sim_time * 0.25
        scene.forward = norm(vector(-0.65 * math.cos(angle), -0.20, -0.65 * math.sin(angle)))

# -----------------------------
# Keyboard controls
# -----------------------------
def keydown(evt):
    global paused, targeting_enabled, chemokines_enabled, auto_loop, cinematic_camera, follow_index, round_number
    key = evt.key
    if key == " ":
        paused = not paused
    elif key == "r":
        round_number += 1
        reset_simulation()
    elif key == "i":
        burst_center = vector(random.uniform(-12, 12), 1.0, random.uniform(-12, 12))
        for _ in range(20):
            spawn_pathogen(burst_center + rand_vec2(random.uniform(0.0, 3.4)), burst=True)
        cell = nearest_tile(burst_center)
        if cell:
            cell["state"] = INFECTED
            cell["infection"] = 0.45
            cell["chemokine"] = 1.1
    elif key in ["+", "="]:
        for _ in range(4):
            spawn_immune_cell()
    elif key == "-":
        for _ in range(min(4, len(immune_cells))):
            c = immune_cells.pop()
            c["obj"].visible = False
            c["ring"].visible = False
            for dot in c["trail"]:
                dot["obj"].visible = False
            c["trail"].clear()
    elif key == "c":
        chemokines_enabled = not chemokines_enabled
    elif key == "t":
        targeting_enabled = not targeting_enabled
    elif key == "a":
        auto_loop = not auto_loop
    elif key == "v":
        cinematic_camera = not cinematic_camera
    elif key == "n":
        follow_index = (follow_index + 1) % max(1, len(immune_cells))

scene.bind("keydown", keydown)

# -----------------------------
# Main loop
# -----------------------------
reset_simulation()

while True:
    rate(60)
    if paused:
        update_camera(1 / 60)
        continue

    dt = 1 / 60
    sim_time += dt
    update_tissue(dt)
    update_pathogens(dt)
    update_immune_cells(dt)
    update_immune_trails(dt)
    update_signals_and_flashes(dt)
    auto_round_logic(dt)
    update_camera(dt)
    if int(sim_time * 5) % 5 == 0:
        update_hud()
