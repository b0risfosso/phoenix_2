"""
VPython Simulation: Music Rhythm Generator — sound, pattern, tempo, variation

A visual rhythm generator showing tempo, beat pulses, instrument lanes, pattern steps,
variation, syncopation, and an AI rhythm mutator.

Run:
    python vpython_music_rhythm_generator_sound_pattern_tempo_variation.py

Controls:
    Space : pause / resume
    r     : reset patterns
    +/=   : increase tempo
    -     : decrease tempo
    v     : force variation
    a     : toggle AI rhythm mutator
    m     : toggle metronome pulse
    c     : toggle cinematic camera
    1     : whole sequencer view
    2     : close beat grid view
    3     : orbit stage view
    4     : side tempo-wave view

Notes:
    This is a visual/simulation rhythm generator. It does not require audio output.
    Beat color/intensity, pulse size, and wave motion represent sound events.
"""

from vpython import *
import random
import math

# -----------------------------
# Scene setup
# -----------------------------
scene = canvas(
    title="Music Rhythm Generator — sound, pattern, tempo, variation",
    width=1280,
    height=760,
    background=vector(0.86, 0.91, 0.96),
    center=vector(0, 0, 0),
    forward=vector(-0.35, -0.35, -1),
)
scene.range = 17
scene.autoscale = False

# -----------------------------
# Constants and state
# -----------------------------
STEP_COUNT = 16
LANE_COUNT = 5
LANE_NAMES = ["KICK", "SNARE", "HIHAT", "BASS", "MELODY"]
LANE_COLORS = [
    vector(1.00, 0.28, 0.18),
    vector(0.25, 0.50, 1.00),
    vector(1.00, 0.82, 0.18),
    vector(0.40, 0.95, 0.45),
    vector(0.80, 0.35, 1.00),
]

GRID_X0 = -8.5
GRID_Y0 = 4.4
STEP_SPACING = 1.12
LANE_SPACING = 1.55
CELL_SIZE = vector(0.82, 0.18, 0.82)

MIN_TEMPO = 55
MAX_TEMPO = 190
DEFAULT_TEMPO = 105

paused = False
auto_ai = True
metronome_enabled = True
cinematic_camera = True
camera_mode = 0
camera_timer = 0.0

bpm = DEFAULT_TEMPO
current_step = 0
beat_phase = 0.0
measure_count = 0
variation_energy = 0.0
swing = 0.08
syncopation = 0.22
complexity = 0.48
humanize = 0.015
master_energy = 0.0

# Visual containers
cells = []
cell_glows = []
lane_labels = []
step_labels = []
active_bar = None
pulse_objects = []
wave_objects = []
particle_objects = []

# Patterns: each cell stores velocity 0..1
patterns = []
base_patterns = []

# -----------------------------
# Helpers
# -----------------------------
def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def mix(a, b, t):
    return a * (1 - t) + b * t


def cell_pos(lane, step):
    x = GRID_X0 + step * STEP_SPACING
    y = GRID_Y0 - lane * LANE_SPACING
    z = 0
    return vector(x, y, z)


def seconds_per_step():
    # 16th-note grid: one quarter note = 60/bpm; step = quarter/4
    return (60.0 / bpm) / 4.0


def make_initial_patterns():
    global patterns, base_patterns
    # Groove: kick anchors, snare backbeat, hats steady, bass offbeats, melody sparse.
    base_patterns = [
        [0.95, 0.0, 0.0, 0.25, 0.78, 0.0, 0.18, 0.0, 0.95, 0.0, 0.0, 0.25, 0.74, 0.0, 0.22, 0.0],
        [0.0, 0.0, 0.12, 0.0, 1.0, 0.0, 0.0, 0.0, 0.08, 0.0, 0.18, 0.0, 1.0, 0.0, 0.0, 0.0],
        [0.42, 0.22, 0.55, 0.24, 0.48, 0.22, 0.58, 0.24, 0.46, 0.22, 0.55, 0.28, 0.50, 0.22, 0.60, 0.30],
        [0.55, 0.0, 0.20, 0.0, 0.0, 0.42, 0.0, 0.0, 0.52, 0.0, 0.32, 0.0, 0.0, 0.48, 0.0, 0.0],
        [0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 0.45, 0.0, 0.0, 0.35, 0.0, 0.0, 0.58, 0.0, 0.0, 0.0],
    ]
    patterns = [[v for v in lane] for lane in base_patterns]


def vary_patterns(amount=0.25):
    """Mutate rhythm with controlled variation."""
    global variation_energy, complexity, syncopation, swing
    variation_energy = min(1.0, variation_energy + amount)

    for lane in range(LANE_COUNT):
        for step in range(STEP_COUNT):
            anchor = step in (0, 4, 8, 12)
            offbeat = step % 2 == 1
            chance = amount * (0.15 + complexity * 0.45)
            if lane == 0 and anchor:
                chance *= 0.35
            if lane == 1 and step in (4, 12):
                chance *= 0.25
            if offbeat:
                chance *= 1.0 + syncopation

            if random.random() < chance:
                if patterns[lane][step] > 0.05:
                    patterns[lane][step] = clamp(patterns[lane][step] + random.uniform(-0.45, 0.35), 0.0, 1.0)
                    if random.random() < 0.18:
                        patterns[lane][step] = 0.0
                else:
                    patterns[lane][step] = clamp(random.uniform(0.18, 0.95) * (0.55 + complexity), 0.0, 1.0)

    # Preserve some musical skeleton.
    patterns[0][0] = max(patterns[0][0], 0.88)
    patterns[1][4] = max(patterns[1][4], 0.82)
    patterns[1][12] = max(patterns[1][12], 0.82)
    swing = clamp(swing + random.uniform(-0.02, 0.025), 0.0, 0.18)


def reset_patterns():
    global bpm, current_step, beat_phase, measure_count, variation_energy
    global swing, syncopation, complexity, master_energy
    bpm = DEFAULT_TEMPO
    current_step = 0
    beat_phase = 0.0
    measure_count = 0
    variation_energy = 0.0
    swing = 0.08
    syncopation = 0.22
    complexity = 0.48
    master_energy = 0.0
    make_initial_patterns()
    rebuild_grid_visuals()

# -----------------------------
# Geometry creation
# -----------------------------
def clear_visual_list(items):
    for obj in items:
        try:
            obj.visible = False
        except Exception:
            pass
    items.clear()


def make_stage():
    # Floor
    box(pos=vector(0, -3.4, -0.15), size=vector(23, 0.08, 10.5), color=vector(0.74, 0.80, 0.86), opacity=0.72)
    # Beat grid backing board
    box(pos=vector(0.0, 0.9, -0.18), size=vector(19.4, 8.7, 0.12), color=vector(0.93, 0.96, 0.98), opacity=0.92)
    # Tempo wave rail
    cylinder(pos=vector(-9.5, -5.05, 0.0), axis=vector(19, 0, 0), radius=0.035, color=vector(0.36, 0.41, 0.46))
    # Speaker towers
    for x in (-11.2, 11.2):
        box(pos=vector(x, 1.0, 0.2), size=vector(0.55, 7.8, 0.6), color=vector(0.18, 0.20, 0.23))
        for yy in [3.7, 2.0, 0.3, -1.4]:
            cylinder(pos=vector(x, yy, 0.52), axis=vector(0, 0, 0.12), radius=0.23, color=vector(0.55, 0.60, 0.65))
            ring(pos=vector(x, yy, 0.59), axis=vector(0, 0, 1), radius=0.26, thickness=0.025, color=vector(0.95, 0.95, 0.95), opacity=0.7)


def rebuild_grid_visuals():
    global cells, cell_glows, lane_labels, step_labels, active_bar
    clear_visual_list(cells)
    clear_visual_list(cell_glows)
    clear_visual_list(lane_labels)
    clear_visual_list(step_labels)
    if active_bar is not None:
        try:
            active_bar.visible = False
        except Exception:
            pass

    for lane in range(LANE_COUNT):
        lp = cell_pos(lane, -1)
        lane_labels.append(label(
            pos=vector(lp.x - 0.75, lp.y, 0.45),
            text=LANE_NAMES[lane],
            height=12,
            color=LANE_COLORS[lane],
            box=False,
            opacity=0,
        ))
        row_cells = []
        row_glows = []
        for step in range(STEP_COUNT):
            p = cell_pos(lane, step)
            velocity = patterns[lane][step]
            base_col = mix(vector(0.72, 0.76, 0.80), LANE_COLORS[lane], velocity)
            cell = box(pos=p, size=CELL_SIZE, color=base_col, opacity=0.92)
            glow = sphere(pos=vector(p.x, p.y + 0.12, p.z + 0.08), radius=0.06 + velocity * 0.12,
                          color=LANE_COLORS[lane], opacity=0.12 + velocity * 0.35, emissive=True)
            row_cells.append(cell)
            row_glows.append(glow)
        cells.append(row_cells)
        cell_glows.append(row_glows)

    for step in range(STEP_COUNT):
        p = cell_pos(LANE_COUNT - 1, step)
        step_labels.append(label(
            pos=vector(p.x, p.y - 0.85, 0.45),
            text=str(step + 1),
            height=10,
            color=vector(0.25, 0.28, 0.31),
            box=False,
            opacity=0,
        ))

    active_bar = box(pos=vector(cell_pos(0, 0).x, 1.35, -0.04), size=vector(0.95, 8.15, 0.05),
                     color=vector(1.0, 1.0, 1.0), opacity=0.35)


def create_hud():
    hud = []
    hud.append(label(pos=vector(-10.7, 6.05, 0.5), text="MUSIC RHYTHM GENERATOR", height=18,
                     color=vector(0.12, 0.14, 0.16), box=False, opacity=0))
    hud.append(label(pos=vector(-10.7, 5.55, 0.5), text="", height=12,
                     color=vector(0.16, 0.18, 0.20), box=False, opacity=0))
    hud.append(label(pos=vector(5.6, 6.05, 0.5), text="", height=12,
                     color=vector(0.16, 0.18, 0.20), box=False, opacity=0))
    hud.append(label(pos=vector(5.6, 5.55, 0.5), text="", height=11,
                     color=vector(0.16, 0.18, 0.20), box=False, opacity=0))
    return hud

# -----------------------------
# Pulse objects
# -----------------------------
class BeatPulse:
    def __init__(self, lane, step, velocity):
        self.lane = lane
        self.step = step
        self.velocity = velocity
        p = cell_pos(lane, step)
        self.age = 0.0
        self.life = 0.55 + velocity * 0.35
        self.body = sphere(pos=vector(p.x, p.y, 0.55), radius=0.16 + 0.18 * velocity,
                           color=LANE_COLORS[lane], opacity=0.75, emissive=True)
        self.ring = ring(pos=vector(p.x, p.y, 0.56), axis=vector(0, 0, 1), radius=0.20 + velocity * 0.12,
                         thickness=0.025, color=LANE_COLORS[lane], opacity=0.6)
        self.vel = vector(random.uniform(-0.04, 0.04), random.uniform(0.05, 0.22), random.uniform(0.3, 0.9)) * (0.7 + velocity)

    def update(self, dt):
        self.age += dt
        t = self.age / self.life
        self.body.pos += self.vel * dt
        self.body.radius = (0.16 + self.velocity * 0.18) * (1.0 + 0.6 * t)
        self.body.opacity = max(0, 0.75 * (1 - t))
        self.ring.radius = (0.20 + self.velocity * 0.12) * (1 + 2.5 * t)
        self.ring.opacity = max(0, 0.60 * (1 - t))
        if t >= 1:
            self.body.visible = False
            self.ring.visible = False
            return True
        return False


class WaveParticle:
    def __init__(self, lane, step, velocity):
        p = cell_pos(lane, step)
        self.age = 0.0
        self.life = random.uniform(0.8, 1.4)
        self.body = sphere(pos=vector(p.x, -5.05, random.uniform(-0.3, 0.3)),
                           radius=0.05 + 0.06 * velocity,
                           color=LANE_COLORS[lane], opacity=0.6, emissive=True)
        self.speed = random.uniform(1.6, 3.2) * (0.5 + velocity)
        self.amp = random.uniform(0.15, 0.45) * (0.5 + velocity)
        self.phase = random.uniform(0, math.tau)
        self.start_x = self.body.pos.x

    def update(self, dt):
        self.age += dt
        self.body.pos.x += self.speed * dt
        self.body.pos.y = -5.05 + math.sin(self.age * 10 + self.phase) * self.amp
        t = self.age / self.life
        self.body.opacity = max(0, 0.6 * (1 - t))
        if self.body.pos.x > 10.2 or t >= 1:
            self.body.visible = False
            return True
        return False

# -----------------------------
# Simulation mechanics
# -----------------------------
def trigger_step(step):
    global master_energy
    step_energy = 0.0
    for lane in range(LANE_COUNT):
        velocity = patterns[lane][step]
        # Humanized probability for softer notes.
        if velocity > 0.05 and random.random() < clamp(velocity + 0.10, 0.0, 1.0):
            pulse_objects.append(BeatPulse(lane, step, velocity))
            wave_objects.append(WaveParticle(lane, step, velocity))
            step_energy += velocity
            # Brighten hit cell.
            cells[lane][step].color = mix(cells[lane][step].color, vector(1, 1, 1), 0.38)
            cells[lane][step].size.y = 0.34 + velocity * 0.22
            cell_glows[lane][step].opacity = 0.55 + velocity * 0.35
            cell_glows[lane][step].radius = 0.16 + velocity * 0.16
    master_energy = clamp(master_energy + step_energy / 6.5, 0.0, 1.0)


def update_cells(dt):
    for lane in range(LANE_COUNT):
        for step in range(STEP_COUNT):
            velocity = patterns[lane][step]
            target_col = mix(vector(0.72, 0.76, 0.80), LANE_COLORS[lane], velocity)
            cells[lane][step].color = mix(cells[lane][step].color, target_col, min(1, dt * 6))
            target_y = 0.18
            cells[lane][step].size.y += (target_y - cells[lane][step].size.y) * min(1, dt * 7)
            cell_glows[lane][step].opacity += ((0.10 + velocity * 0.28) - cell_glows[lane][step].opacity) * min(1, dt * 5)
            cell_glows[lane][step].radius += ((0.06 + velocity * 0.12) - cell_glows[lane][step].radius) * min(1, dt * 5)

    active_bar.pos.x += (cell_pos(0, current_step).x - active_bar.pos.x) * min(1, dt * 13)
    active_bar.opacity = 0.22 + 0.14 * math.sin(beat_phase * math.tau)


def update_pulses(dt):
    for collection in (pulse_objects, wave_objects):
        dead = []
        for obj in collection:
            if obj.update(dt):
                dead.append(obj)
        for obj in dead:
            collection.remove(obj)


def ai_controller():
    global bpm, complexity, syncopation, swing
    # Adjust groove pressure over time.
    avg_density = sum(1 for lane in patterns for v in lane if v > 0.08) / (LANE_COUNT * STEP_COUNT)
    if avg_density < 0.30:
        complexity = clamp(complexity + 0.03, 0.1, 1.0)
        vary_patterns(0.16)
    elif avg_density > 0.68:
        complexity = clamp(complexity - 0.02, 0.1, 1.0)
        # remove a few non-anchor notes
        for _ in range(4):
            lane = random.randrange(LANE_COUNT)
            step = random.randrange(STEP_COUNT)
            if not (lane == 0 and step in (0, 8)) and not (lane == 1 and step in (4, 12)):
                patterns[lane][step] *= random.uniform(0.0, 0.45)

    if random.random() < 0.55:
        syncopation = clamp(syncopation + random.uniform(-0.05, 0.06), 0.0, 0.65)
    if random.random() < 0.35:
        bpm = int(clamp(bpm + random.choice([-4, -2, 2, 3, 5]), MIN_TEMPO, MAX_TEMPO))

    # Every AI cycle creates a little mutation.
    vary_patterns(random.uniform(0.08, 0.18))


def update_hud():
    density = sum(1 for lane in patterns for v in lane if v > 0.08) / (LANE_COUNT * STEP_COUNT)
    hud_labels[1].text = f"Tempo: {bpm} BPM   Step: {current_step + 1}/16   Measure: {measure_count}"
    hud_labels[2].text = f"Density: {density:.2f}   Complexity: {complexity:.2f}   Syncopation: {syncopation:.2f}   Swing: {swing:.2f}"
    hud_labels[3].text = (
        f"AI: {'ON' if auto_ai else 'OFF'}   Metronome: {'ON' if metronome_enabled else 'OFF'}   "
        f"Camera: {'ON' if cinematic_camera else 'OFF'}\n"
        "Space pause | +/- tempo | v variation | a AI | m metronome | c camera | 1-4 views | r reset"
    )

# -----------------------------
# Camera
# -----------------------------
def set_camera_mode(mode):
    global camera_mode, camera_timer
    camera_mode = mode
    camera_timer = 0.0


def update_camera(dt):
    global camera_timer, camera_mode
    camera_timer += dt
    if cinematic_camera and camera_timer > 8.0:
        camera_mode = (camera_mode + 1) % 4
        camera_timer = 0.0

    t = scene.time if hasattr(scene, 'time') else 0
    tt = camera_timer
    if camera_mode == 0:  # whole sequencer
        target_center = vector(0, 0.5, 0.0)
        target_forward = vector(-0.20, -0.28, -1)
        target_range = 15.5
    elif camera_mode == 1:  # close grid
        target_center = vector(0, 0.4, 0.1)
        target_forward = vector(-0.12, -0.20, -1)
        target_range = 9.8
    elif camera_mode == 2:  # orbit stage
        angle = tt * 0.25
        target_center = vector(0, 0.0, 0.2)
        target_forward = vector(-0.45 * math.sin(angle), -0.24, -1)
        target_range = 16.2
    else:  # side tempo wave
        target_center = vector(0, -2.2, 0.1)
        target_forward = vector(-0.65, -0.18, -1)
        target_range = 12.5

    scene.center += (target_center - scene.center) * min(1, dt * 1.2)
    scene.forward += (target_forward - scene.forward) * min(1, dt * 1.2)
    scene.range += (target_range - scene.range) * min(1, dt * 1.0)

# -----------------------------
# Keyboard
# -----------------------------
def on_keydown(evt):
    global paused, bpm, auto_ai, metronome_enabled, cinematic_camera
    key = evt.key
    if key == ' ':
        paused = not paused
    elif key == 'r':
        reset_patterns()
    elif key in ['+', '=']:
        bpm = int(clamp(bpm + 5, MIN_TEMPO, MAX_TEMPO))
    elif key == '-':
        bpm = int(clamp(bpm - 5, MIN_TEMPO, MAX_TEMPO))
    elif key == 'v':
        vary_patterns(0.35)
    elif key == 'a':
        auto_ai = not auto_ai
    elif key == 'm':
        metronome_enabled = not metronome_enabled
    elif key == 'c':
        cinematic_camera = not cinematic_camera
    elif key in ['1', '2', '3', '4']:
        set_camera_mode(int(key) - 1)

scene.bind('keydown', on_keydown)

# -----------------------------
# Build initial world
# -----------------------------
make_initial_patterns()
make_stage()
rebuild_grid_visuals()
hud_labels = create_hud()

# Metronome object
metronome = sphere(pos=vector(-10.2, 5.25, 0.55), radius=0.16, color=vector(1, 1, 1), opacity=0.65, emissive=True)
metronome_ring = ring(pos=metronome.pos, axis=vector(0, 0, 1), radius=0.25, thickness=0.025, color=vector(1, 1, 1), opacity=0.4)

# -----------------------------
# Main loop
# -----------------------------
dt = 1 / 60
step_clock = 0.0
ai_clock = 0.0

while True:
    rate(60)
    if paused:
        update_camera(dt)
        update_hud()
        continue

    # Advance timing with swing and humanize.
    base_step_duration = seconds_per_step()
    swung = 1.0
    if current_step % 2 == 1:
        swung += swing
    else:
        swung -= swing * 0.45
    step_duration = max(0.035, base_step_duration * swung + random.uniform(-humanize, humanize) * base_step_duration)

    step_clock += dt
    beat_phase = step_clock / step_duration

    if step_clock >= step_duration:
        step_clock = 0.0
        current_step = (current_step + 1) % STEP_COUNT
        if current_step == 0:
            measure_count += 1
            if auto_ai and measure_count % 2 == 0:
                ai_controller()
            elif random.random() < 0.18:
                vary_patterns(0.12)
        trigger_step(current_step)

    if metronome_enabled:
        pulse = 0.5 + 0.5 * math.sin(beat_phase * math.tau)
        metronome.radius = 0.14 + 0.11 * pulse
        metronome.opacity = 0.38 + 0.42 * pulse
        metronome_ring.radius = 0.24 + 0.22 * pulse
        metronome_ring.opacity = 0.18 + 0.32 * (1 - pulse)
    else:
        metronome.opacity = 0.08
        metronome_ring.opacity = 0.05

    variation_energy *= 0.995
    master_energy *= 0.965

    update_cells(dt)
    update_pulses(dt)
    update_camera(dt)
    update_hud()
