"""
Autonomous Suspended Bird Puppet Build + Play/Test Simulation
-------------------------------------------------------------

This VPython script animates the full sequence from the provided human instructions:

1. Materials appear on the table
2. Body sheet is traced and cut into a bird body
3. Wings slide into place
4. Tail attaches
5. Balance weights are tested
6. Support frame rises
7. Suspension strings attach
8. Puppet lifts from the table
9. 3D coordinate axes and octants are built
10. Bird automatically enters play/test rounds:
    - direct octant chain
    - axis-crossing survey
    - large looping orbit
    - diagonal opposite cuts
    - spiral octant expansion
    - chaotic octant hunt
    - figure-eight across axes
    - full-coverage mixed wander

Run:
    python autonomous_suspended_bird_puppet_build_simulation.py

Controls:
    SPACE  pause/play
    A      toggle autonomous progression
    N      jump to next phase
    B      jump to previous phase
    R      restart from materials
    F      faster
    S      slower
    H      show/hide help
"""

from vpython import *
import math
import random

# --------------------------------------------------------------------------------------
# Scene
# --------------------------------------------------------------------------------------

scene.title = "Autonomous Suspended Bird Puppet Build + Play/Test"
scene.width = 1180
scene.height = 760
scene.background = vector(0.88, 0.94, 1.0)
scene.center = vector(0, 1.9, 0)
scene.forward = vector(-0.85, -0.38, -0.75)
scene.range = 9.5
scene.autoscale = False

# --------------------------------------------------------------------------------------
# Timing
# --------------------------------------------------------------------------------------

DT = 0.02
TIME_SCALE = 1.0
paused = False
autonomous = True
show_help = True

# Phase durations in seconds.
PHASES = [
    ("Materials appear", 7.0),
    ("Trace and cut body", 7.0),
    ("Attach wings", 7.0),
    ("Attach tail", 5.5),
    ("Balance puppet", 6.0),
    ("Build support frame", 6.0),
    ("Attach suspension strings", 6.0),
    ("Lift puppet", 5.0),
    ("Build 3D octant space", 7.0),
    ("Play/test: direct octant chain", 10.0),
    ("Play/test: axis-crossing survey", 10.0),
    ("Play/test: large looping orbit", 10.0),
    ("Play/test: diagonal opposite cuts", 10.0),
    ("Play/test: spiral octant expansion", 10.0),
    ("Play/test: chaotic octant hunt", 10.0),
    ("Play/test: figure-eight across axes", 10.0),
    ("Play/test: full-coverage mixed wander", 12.0),
]

phase_index = 0
phase_time = 0.0
global_time = 0.0

# --------------------------------------------------------------------------------------
# Colors
# --------------------------------------------------------------------------------------

COLORS = {
    "body": vector(0.95, 0.72, 0.32),
    "wing": vector(0.35, 0.62, 0.95),
    "tail": vector(0.95, 0.42, 0.38),
    "string": vector(0.10, 0.10, 0.10),
    "frame": vector(0.42, 0.30, 0.18),
    "table": vector(0.70, 0.53, 0.36),
    "paper": vector(1.00, 0.98, 0.86),
    "metal": vector(0.65, 0.68, 0.72),
    "x": vector(0.95, 0.25, 0.25),
    "y": vector(0.20, 0.70, 0.25),
    "z": vector(0.25, 0.35, 0.95),
    "origin": vector(1.00, 0.85, 0.15),
    "ghost": vector(0.75, 0.78, 0.82),
}

# --------------------------------------------------------------------------------------
# Persistent object registries
# --------------------------------------------------------------------------------------

objects = []
labels = []
bird_parts = {}
strings = []
trail_segments = []
status_labels = []
octant_markers = []
target_marker = None
target_ring = None

# Assembly objects
table_top = None
body_sheet = None
body_outline = None
body_flat = None
beak_flat = None
eye_flat = None
left_wing_sheet = None
right_wing_sheet = None
tail_sheet = None
weight_objs = []
tool_objs = []
frame_objs = []
axis_objs = []
grid_objs = []

# Metrics
last_pos = vector(0, 2.3, 0)
prev_signs = None
visited_octants = set()
axis_crossings = {"x": 0, "y": 0, "z": 0}
near_origin_passes = 0
loop_proxy = 0

# --------------------------------------------------------------------------------------
# Utility
# --------------------------------------------------------------------------------------

def reg(obj):
    objects.append(obj)
    return obj

def reg_label(obj):
    labels.append(obj)
    objects.append(obj)
    return obj

def clamp(x, a, b):
    return max(a, min(b, x))

def smooth(x):
    x = clamp(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)

def lerp(a, b, u):
    return a * (1 - u) + b * u

def make_label(pos, text, height=13, color=vector(0.05, 0.06, 0.08), box=False, opacity=0.0):
    return reg_label(label(pos=pos, text=text, height=height, color=color, box=box, opacity=opacity, billboard=True))

def hide_all():
    for obj in objects:
        obj.visible = False
    for seg in trail_segments:
        seg.visible = False

def show_list(items, visible=True):
    for item in items:
        item.visible = visible

def set_opacity_safe(obj, value):
    try:
        obj.opacity = value
    except Exception:
        pass

def phase_name():
    return PHASES[phase_index][0]

def phase_duration():
    return PHASES[phase_index][1]

def help_text():
    if not show_help:
        return ""
    return "\nSPACE pause/play | A auto | N next | B previous | R restart | F faster | S slower | H help"

def clear_status():
    global status_labels
    for lab in status_labels:
        lab.visible = False
    status_labels = []

def status(text):
    clear_status()
    status_labels.append(label(
        pos=vector(-7.4, 6.0, 0),
        text=text,
        height=12,
        color=vector(0.04, 0.05, 0.07),
        box=True,
        opacity=0.18,
        billboard=True,
        border=8,
    ))

def title_text(extra=""):
    status(
        f"Phase {phase_index + 1}/{len(PHASES)}: {phase_name()}\n"
        f"Autonomous progression: {'ON' if autonomous else 'OFF'} | speed x{TIME_SCALE:.1f}"
        f"{help_text()}"
        + (f"\n{extra}" if extra else "")
    )

def reset_metrics():
    global last_pos, prev_signs, visited_octants, axis_crossings, near_origin_passes, loop_proxy
    last_pos = vector(0, 2.3, 0)
    prev_signs = None
    visited_octants = set()
    axis_crossings = {"x": 0, "y": 0, "z": 0}
    near_origin_passes = 0
    loop_proxy = 0

def sign(v):
    return 1 if v >= 0 else -1

def octant_name(pos):
    xs = "+" if pos.x >= 0 else "-"
    ys = "+" if pos.z >= 0 else "-"
    zs = "+" if pos.y >= 2.25 else "-"
    key = xs + ys + zs
    names = {
        "+++": "O1",
        "-++": "O2",
        "--+": "O3",
        "+-+": "O4",
        "++-": "O5",
        "-+-": "O6",
        "---": "O7",
        "+--": "O8",
    }
    return names[key], key

def target_for_octant(name):
    coords = {
        "O1": vector(4.0, 4.3, 4.0),
        "O2": vector(-4.0, 4.3, 4.0),
        "O3": vector(-4.0, 4.3, -4.0),
        "O4": vector(4.0, 4.3, -4.0),
        "O5": vector(4.0, 1.15, 4.0),
        "O6": vector(-4.0, 1.15, 4.0),
        "O7": vector(-4.0, 1.15, -4.0),
        "O8": vector(4.0, 1.15, -4.0),
        "CENTER": vector(0, 2.25, 0),
    }
    return coords[name]

def move_sequence(seq, t, segment):
    i = int(t / segment) % len(seq)
    j = (i + 1) % len(seq)
    u = smooth((t % segment) / segment)
    return lerp(target_for_octant(seq[i]), target_for_octant(seq[j]), u)

# --------------------------------------------------------------------------------------
# Build static object skeleton
# --------------------------------------------------------------------------------------

def create_base_objects():
    global table_top
    table_top = reg(box(pos=vector(0, -0.12, 0), size=vector(9.4, 0.18, 5.4), color=COLORS["table"]))
    for x in [-4.2, 4.2]:
        for z in [-2.3, 2.3]:
            reg(cylinder(pos=vector(x, -1.25, z), axis=vector(0, 1.15, 0), radius=0.06, color=COLORS["frame"]))

def create_material_objects():
    global body_sheet, left_wing_sheet, right_wing_sheet, tail_sheet, weight_objs, tool_objs
    body_sheet = reg(box(pos=vector(-7.0, 0.04, 0.45), size=vector(2.3, 0.04, 1.55), color=COLORS["paper"]))
    left_wing_sheet = reg(box(pos=vector(-7.0, 0.07, 0.95), size=vector(2.2, 0.035, 0.85), color=COLORS["wing"]))
    right_wing_sheet = reg(box(pos=vector(-7.0, 0.07, -0.95), size=vector(2.2, 0.035, 0.85), color=COLORS["wing"]))
    tail_sheet = reg(box(pos=vector(7.0, 0.07, 0.35), size=vector(1.1, 0.035, 0.8), color=COLORS["tail"]))

    # Tools: ruler, blade/scissor proxy, tape.
    tool_objs = [
        reg(box(pos=vector(7.0, 0.09, -1.6), size=vector(1.3, 0.07, 0.18), color=vector(0.9, 0.15, 0.12))),
        reg(cylinder(pos=vector(7.0, 0.10, -1.95), axis=vector(0.85, 0, 0), radius=0.04, color=COLORS["metal"])),
        reg(ring(pos=vector(7.0, 0.12, -1.20), axis=vector(0, 1, 0), radius=0.20, thickness=0.045, color=vector(0.8, 0.8, 0.9))),
    ]

    weight_objs = []
    for i in range(5):
        weight_objs.append(reg(sphere(pos=vector(7.0 + i * 0.22, 0.12, 1.55), radius=0.07, color=COLORS["metal"])))

def create_flat_body_objects():
    global body_outline, body_flat, beak_flat, eye_flat
    body_flat = reg(ellipsoid(pos=vector(-0.6, 0.15, 0), length=2.25, height=0.08, width=0.75, color=COLORS["body"], opacity=0.0))
    beak_flat = reg(cone(pos=vector(0.62, 0.18, 0), axis=vector(0.42, 0, 0), radius=0.11, color=vector(1.0, 0.55, 0.16), opacity=0.0))
    eye_flat = reg(sphere(pos=vector(0.38, 0.27, 0.16), radius=0.045, color=vector(0, 0, 0), opacity=0.0))

    pts = [
        vector(-1.9, 0.22, 0),
        vector(-1.35, 0.22, 0.38),
        vector(-0.35, 0.22, 0.48),
        vector(0.55, 0.22, 0.22),
        vector(1.0, 0.22, 0),
        vector(0.55, 0.22, -0.20),
        vector(-0.35, 0.22, -0.38),
        vector(-1.35, 0.22, -0.30),
        vector(-1.9, 0.22, 0),
    ]
    body_outline = reg(curve(pos=pts, color=COLORS["body"], radius=0.018))
    body_outline.visible = False

def create_bird_parts():
    global bird_parts
    bird_parts = {
        "body": reg(ellipsoid(pos=vector(0, 0.6, 0), length=1.75, height=0.42, width=0.34, color=COLORS["body"], opacity=0.0)),
        "head": reg(sphere(pos=vector(0.9, 0.68, 0), radius=0.22, color=COLORS["body"], opacity=0.0)),
        "beak": reg(cone(pos=vector(1.08, 0.68, 0), axis=vector(0.32, 0, 0), radius=0.09, color=vector(1.0, 0.56, 0.18), opacity=0.0)),
        "eye": reg(sphere(pos=vector(1.02, 0.78, 0.12), radius=0.035, color=vector(0, 0, 0), opacity=0.0)),
        "left_wing": reg(pyramid(pos=vector(-0.15, 0.6, 0.28), size=vector(0.12, 1.2, 0.95), color=COLORS["wing"], opacity=0.0)),
        "right_wing": reg(pyramid(pos=vector(-0.15, 0.6, -0.28), size=vector(0.12, 1.2, 0.95), color=COLORS["wing"], opacity=0.0)),
        "tail": reg(pyramid(pos=vector(-0.9, 0.6, 0), size=vector(0.8, 0.32, 0.8), color=COLORS["tail"], opacity=0.0)),
    }

def rotate_vec(v, yaw=0, pitch=0, bank=0):
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cb, sb = math.cos(bank), math.sin(bank)

    x1 = cy * v.x + sy * v.z
    y1 = v.y
    z1 = -sy * v.x + cy * v.z

    x2 = cp * x1 - sp * y1
    y2 = sp * x1 + cp * y1
    z2 = z1

    x3 = x2
    y3 = cb * y2 - sb * z2
    z3 = sb * y2 + cb * z2
    return vector(x3, y3, z3)

def set_bird_pose(pos, yaw=0, pitch=0, bank=0, wing_phase=0):
    flap = 0.18 * math.sin(wing_phase)
    offsets = {
        "body": vector(0, 0, 0),
        "head": vector(0.92, 0.08, 0),
        "beak": vector(1.10, 0.08, 0),
        "eye": vector(1.02, 0.17, 0.12),
        "left_wing": vector(-0.15, flap, 0.29),
        "right_wing": vector(-0.15, -flap, -0.29),
        "tail": vector(-0.9, -0.02, 0),
    }
    for name, off in offsets.items():
        bird_parts[name].pos = pos + rotate_vec(off, yaw, pitch, bank)

    forward = rotate_vec(vector(1, 0, 0), yaw, pitch, bank)
    bird_parts["body"].axis = forward * 1.75
    bird_parts["beak"].axis = forward * 0.32
    bird_parts["tail"].axis = rotate_vec(vector(-0.85, 0, 0), yaw, pitch, bank)
    bird_parts["left_wing"].axis = rotate_vec(vector(-0.25, 0.12 + flap, 0.95), yaw, pitch, bank)
    bird_parts["right_wing"].axis = rotate_vec(vector(-0.25, -0.12 - flap, -0.95), yaw, pitch, bank)

def set_bird_opacity(opacity):
    for part in bird_parts.values():
        set_opacity_safe(part, opacity)

def create_frame_objects():
    global frame_objs
    top_y = 5.0
    frame_objs = []
    for x in [-4.5, 4.5]:
        for z in [-3.1, 3.1]:
            frame_objs.append(reg(cylinder(pos=vector(x, 0, z), axis=vector(0, 0.01, 0), radius=0.055, color=COLORS["frame"], opacity=0.0)))
    frame_objs += [
        reg(cylinder(pos=vector(-4.5, top_y, -3.1), axis=vector(0.01, 0, 0), radius=0.055, color=COLORS["frame"], opacity=0.0)),
        reg(cylinder(pos=vector(-4.5, top_y, 3.1), axis=vector(0.01, 0, 0), radius=0.055, color=COLORS["frame"], opacity=0.0)),
        reg(cylinder(pos=vector(-4.5, top_y, -3.1), axis=vector(0, 0, 0.01), radius=0.055, color=COLORS["frame"], opacity=0.0)),
        reg(cylinder(pos=vector(4.5, top_y, -3.1), axis=vector(0, 0, 0.01), radius=0.055, color=COLORS["frame"], opacity=0.0)),
        reg(sphere(pos=vector(0, top_y, 0), radius=0.12, color=COLORS["metal"], opacity=0.0)),
    ]

def create_strings():
    global strings
    strings = []
    for _ in range(3):
        strings.append(reg(cylinder(pos=vector(0, 5.0, 0), axis=vector(0, -0.01, 0), radius=0.012, color=COLORS["string"], opacity=0.0)))

def update_strings(bird_pos, opacity=1.0):
    top = vector(0, 5.0, 0)
    offsets = [vector(0, 0.18, 0), vector(-0.2, 0.05, 0.33), vector(-0.2, 0.05, -0.33)]
    for s, off in zip(strings, offsets):
        s.pos = top
        s.axis = bird_pos + off - top
        set_opacity_safe(s, opacity)

def create_axes_objects():
    global axis_objs, grid_objs, octant_markers
    axis_objs = []
    grid_objs = []
    octant_markers = []

    for i in range(-6, 7):
        grid_objs.append(reg(curve(pos=[vector(-6, 0.01, i), vector(6, 0.01, i)], color=COLORS["ghost"], radius=0.006)))
        grid_objs.append(reg(curve(pos=[vector(i, 0.01, -6), vector(i, 0.01, 6)], color=COLORS["ghost"], radius=0.006)))

    axis_objs += [
        reg(cylinder(pos=vector(-6, 0.03, 0), axis=vector(12, 0, 0), radius=0.025, color=COLORS["x"])),
        reg(cone(pos=vector(6, 0.03, 0), axis=vector(0.45, 0, 0), radius=0.11, color=COLORS["x"])),
        reg(cylinder(pos=vector(0, 0.04, -6), axis=vector(0, 0, 12), radius=0.025, color=COLORS["y"])),
        reg(cone(pos=vector(0, 0.04, 6), axis=vector(0, 0, 0.45), radius=0.11, color=COLORS["y"])),
        reg(cylinder(pos=vector(0, 0.0, 0), axis=vector(0, 5.3, 0), radius=0.025, color=COLORS["z"])),
        reg(cone(pos=vector(0, 5.3, 0), axis=vector(0, 0.45, 0), radius=0.11, color=COLORS["z"])),
        reg(sphere(pos=vector(0, 0.08, 0), radius=0.14, color=COLORS["origin"])),
    ]

    labels_data = [
        ("+X", vector(6.75, 0.35, 0), COLORS["x"]),
        ("+Y", vector(0, 0.35, 6.75), COLORS["y"]),
        ("+Z", vector(0.35, 5.9, 0), COLORS["z"]),
        ("ORIGIN", vector(0.6, 0.42, 0), vector(0.15, 0.12, 0.02)),
        ("O1 +++", vector(3.8, 4.2, 3.8), vector(0.05, 0.06, 0.08)),
        ("O2 -++", vector(-3.8, 4.2, 3.8), vector(0.05, 0.06, 0.08)),
        ("O3 --+", vector(-3.8, 4.2, -3.8), vector(0.05, 0.06, 0.08)),
        ("O4 +-+", vector(3.8, 4.2, -3.8), vector(0.05, 0.06, 0.08)),
        ("O5 ++-", vector(3.8, 0.75, 3.8), vector(0.05, 0.06, 0.08)),
        ("O6 -+-", vector(-3.8, 0.75, 3.8), vector(0.05, 0.06, 0.08)),
        ("O7 ---", vector(-3.8, 0.75, -3.8), vector(0.05, 0.06, 0.08)),
        ("O8 +--", vector(3.8, 0.75, -3.8), vector(0.05, 0.06, 0.08)),
    ]
    for txt, pos, col in labels_data:
        axis_objs.append(make_label(pos, txt, height=11, color=col))
    for name in ["O1", "O2", "O3", "O4", "O5", "O6", "O7", "O8"]:
        marker = reg(sphere(pos=target_for_octant(name), radius=0.10, color=vector(1, 0.85, 0.15), opacity=0.0))
        octant_markers.append(marker)

def create_target_objects():
    global target_marker, target_ring
    target_marker = reg(sphere(pos=vector(0, 2.25, 0), radius=0.16, color=vector(1, 0.2, 0.1), emissive=True, opacity=0.0))
    target_ring = reg(ring(pos=vector(0, 2.25, 0), axis=vector(0, 1, 0), radius=0.42, thickness=0.025, color=vector(1, 0.35, 0.1), opacity=0.0))

def initialize_world():
    create_base_objects()
    create_material_objects()
    create_flat_body_objects()
    create_bird_parts()
    create_frame_objects()
    create_strings()
    create_axes_objects()
    create_target_objects()

initialize_world()

# --------------------------------------------------------------------------------------
# Visibility / phase state
# --------------------------------------------------------------------------------------

def prepare_phase_visibility():
    # Hide all, then phase update shows relevant items. This keeps old stages from cluttering.
    hide_all()
    table_top.visible = True
    # table legs are first 5 objects? easiest: show all objects with table/frame low positions during build phases.
    for obj in objects:
        if isinstance(obj, cylinder) and obj.pos.y < 0 and obj.radius == 0.06:
            obj.visible = True

def show_materials():
    for obj in [body_sheet, left_wing_sheet, right_wing_sheet, tail_sheet] + tool_objs + weight_objs:
        obj.visible = True

def show_flat_body(opacity=1.0):
    for obj in [body_flat, beak_flat, eye_flat]:
        obj.visible = True
        set_opacity_safe(obj, opacity)

def show_bird(opacity=1.0):
    for part in bird_parts.values():
        part.visible = True
        set_opacity_safe(part, opacity)

def show_frame(opacity=1.0, build_u=1.0):
    top_y = 5.0
    # posts first four
    posts = frame_objs[:4]
    beams = frame_objs[4:8]
    hub = frame_objs[8]
    for post in posts:
        post.visible = True
        set_opacity_safe(post, opacity)
        post.axis = vector(0, top_y * build_u, 0)
    for i, beam in enumerate(beams):
        beam.visible = build_u > 0.65
        set_opacity_safe(beam, opacity)
    beam_u = smooth((build_u - 0.65) / 0.35)
    beams[0].axis = vector(9.0 * beam_u, 0, 0)
    beams[1].axis = vector(9.0 * beam_u, 0, 0)
    beams[2].axis = vector(0, 0, 6.2 * beam_u)
    beams[3].axis = vector(0, 0, 6.2 * beam_u)
    hub.visible = build_u > 0.85
    set_opacity_safe(hub, opacity)

def show_axes(opacity=1.0, build_u=1.0):
    for obj in grid_objs:
        obj.visible = build_u > 0.10
        set_opacity_safe(obj, opacity * smooth((build_u - 0.10) / 0.20))
    for obj in axis_objs:
        obj.visible = build_u > 0.25
        set_opacity_safe(obj, opacity * smooth((build_u - 0.25) / 0.35))
    for m in octant_markers:
        m.visible = build_u > 0.55
        set_opacity_safe(m, opacity * smooth((build_u - 0.55) / 0.25))

def show_strings(opacity=1.0, bird_pos=vector(0, 2.25, 0)):
    for s in strings:
        s.visible = True
    update_strings(bird_pos, opacity)

# --------------------------------------------------------------------------------------
# Phase animations
# --------------------------------------------------------------------------------------

def animate_materials(t, u):
    prepare_phase_visibility()
    # Materials slide in from sides and settle on the table.
    body_sheet.pos = lerp(vector(-7, 0.04, 0.45), vector(-2.7, 0.04, 0.55), smooth(u * 1.6))
    left_wing_sheet.pos = lerp(vector(-7, 0.07, 0.95), vector(0.15, 0.07, 0.95), smooth((u - 0.12) * 1.7))
    right_wing_sheet.pos = lerp(vector(-7, 0.07, -0.95), vector(0.15, 0.07, -0.95), smooth((u - 0.12) * 1.7))
    tail_sheet.pos = lerp(vector(7, 0.07, 0.35), vector(2.7, 0.07, 0.45), smooth((u - 0.25) * 1.8))

    for i, obj in enumerate(tool_objs):
        obj.pos.x = lerp(7.0, -0.8 + i * 0.7, smooth((u - 0.35) * 1.7))
    for i, w in enumerate(weight_objs):
        w.pos = lerp(vector(7.0 + i * 0.22, 0.12, 1.55), vector(2.0 + i * 0.22, 0.12, -1.65), smooth((u - 0.45) * 1.8))

    show_materials()
    title_text("Materials slide onto the work table before assembly begins.")

def animate_cut_body(t, u):
    prepare_phase_visibility()
    show_materials()

    # Body sheet moves to center; outline draws by increasing visible curve points.
    body_sheet.pos = lerp(vector(-2.7, 0.04, 0.55), vector(-0.6, 0.04, 0.0), smooth(u * 1.4))
    left_wing_sheet.pos = vector(2.7, 0.07, 1.15)
    right_wing_sheet.pos = vector(2.7, 0.07, -1.15)
    tail_sheet.pos = vector(3.6, 0.07, 0.0)

    body_outline.visible = u > 0.18
    if body_outline.visible:
        # Pulse outline radius/color as if tracing.
        body_outline.radius = 0.014 + 0.012 * abs(math.sin(t * 6))
        body_outline.color = vector(0.9, 0.45 + 0.25 * math.sin(t * 4), 0.15)

    cut_u = smooth((u - 0.45) / 0.45)
    show_flat_body(opacity=cut_u)
    set_opacity_safe(body_sheet, 1.0 - 0.55 * cut_u)

    # Tool sweeps across body as cutter.
    tool_objs[0].visible = True
    tool_objs[0].pos = vector(lerp(-2.1, 1.1, smooth((u - 0.18) / 0.65)), 0.35, 0.7 * math.sin(u * math.pi * 4))
    tool_objs[0].size = vector(0.55, 0.08, 0.16)

    title_text("The body is traced, cut out, and separated from the sheet.")

def animate_wings(t, u):
    prepare_phase_visibility()
    show_flat_body(1.0)
    show_bird(opacity=0.25)

    # Body rises from flat cutout into the 3D puppet body.
    lift_u = smooth(u * 1.4)
    set_bird_pose(vector(-0.4, lerp(0.2, 0.75, lift_u), 0), yaw=0, pitch=0.06 * math.sin(t * 2), bank=0, wing_phase=t * 2)
    set_bird_opacity(0.25 + 0.75 * lift_u)

    # Wing sheets move onto the bird as attached wings.
    left_start = vector(3.0, 0.08, 1.35)
    right_start = vector(3.0, 0.08, -1.35)
    left_end = vector(-0.15, 0.75, 0.30)
    right_end = vector(-0.15, 0.75, -0.30)

    left_wing_sheet.visible = True
    right_wing_sheet.visible = True
    left_wing_sheet.pos = lerp(left_start, left_end, smooth((u - 0.18) / 0.55))
    right_wing_sheet.pos = lerp(right_start, right_end, smooth((u - 0.28) / 0.55))
    left_wing_sheet.rotate(angle=0.01, axis=vector(0, 1, 0))
    right_wing_sheet.rotate(angle=-0.01, axis=vector(0, 1, 0))

    wing_u = smooth((u - 0.45) / 0.45)
    set_opacity_safe(bird_parts["left_wing"], wing_u)
    set_opacity_safe(bird_parts["right_wing"], wing_u)
    set_opacity_safe(left_wing_sheet, 1.0 - wing_u)
    set_opacity_safe(right_wing_sheet, 1.0 - wing_u)

    title_text("Wing pieces slide into place and become the left and right wings.")

def animate_tail(t, u):
    prepare_phase_visibility()
    show_bird(1.0)
    set_bird_pose(vector(-0.2, 0.75, 0), pitch=0.03 * math.sin(t * 3), bank=0.05 * math.sin(t * 2), wing_phase=t * 3)

    tail_sheet.visible = True
    tail_sheet.pos = lerp(vector(3.5, 0.08, 0), vector(-1.05, 0.72, 0), smooth(u))
    tail_sheet.rotate(angle=0.012, axis=vector(0, 1, 0))

    tail_u = smooth((u - 0.45) / 0.45)
    set_opacity_safe(bird_parts["tail"], tail_u)
    set_opacity_safe(tail_sheet, 1.0 - tail_u)

    title_text("The tail attaches to the back so the puppet can pitch and stabilize.")

def animate_balance(t, u):
    prepare_phase_visibility()
    show_bird(1.0)

    # Bird rocks around as weights are tested.
    bank = 0.35 * math.sin(t * 3.2) * (1 - u)
    pitch = 0.25 * math.cos(t * 2.7) * (1 - u) + 0.08
    set_bird_pose(vector(0, 0.9 + 0.08 * math.sin(t * 2), 0), pitch=pitch, bank=bank, wing_phase=t * 3)

    # Weights move toward nose/tail then settle.
    for i, w in enumerate(weight_objs):
        w.visible = True
        target = vector(0.85 + i * 0.05, 0.68, -0.52 + i * 0.08)
        start = vector(2.0 + i * 0.22, 0.12, -1.65)
        w.pos = lerp(start, target, smooth((u - 0.15 - i * 0.05) / 0.55))

    # Balance line
    reg(curve(pos=[vector(-2.2, 0.9, 0), vector(2.2, 0.9, 0)], color=vector(0.15, 0.15, 0.15), radius=0.01))

    title_text("Small weights are tested until the bird hangs almost level with a slight forward glide angle.")

def animate_frame(t, u):
    prepare_phase_visibility()
    show_bird(1.0)
    set_bird_pose(vector(0, 0.9, 0), pitch=0.06, bank=0.04 * math.sin(t * 2), wing_phase=t * 3)
    show_frame(opacity=1.0, build_u=u)
    title_text("A simple overhead frame rises around the puppet.")

def animate_strings(t, u):
    prepare_phase_visibility()
    show_bird(1.0)
    show_frame(1.0, 1.0)

    bird_pos = vector(0, 0.95 + 0.05 * math.sin(t * 2), 0)
    set_bird_pose(bird_pos, pitch=0.05, bank=0.05 * math.sin(t * 2.5), wing_phase=t * 3)

    string_u = smooth(u)
    # Strings lengthen from top toward the bird.
    top = vector(0, 5.0, 0)
    offsets = [vector(0, 0.18, 0), vector(-0.2, 0.05, 0.33), vector(-0.2, 0.05, -0.33)]
    for i, s in enumerate(strings):
        s.visible = True
        set_opacity_safe(s, string_u)
        end = bird_pos + offsets[i]
        partial_end = lerp(top, end, string_u)
        s.pos = top
        s.axis = partial_end - top

    title_text("Three suspension strings attach: one main string and two steering strings.")

def animate_lift(t, u):
    prepare_phase_visibility()
    show_bird(1.0)
    show_frame(1.0, 1.0)

    bird_pos = vector(0, lerp(0.95, 2.35, smooth(u)), 0)
    set_bird_pose(bird_pos, pitch=0.08 + 0.04 * math.sin(t * 3), bank=0.10 * math.sin(t * 2), wing_phase=t * 5)
    show_strings(1.0, bird_pos)

    title_text("The assembled puppet lifts from the table and begins to hang freely.")

def animate_octants(t, u):
    prepare_phase_visibility()
    show_bird(1.0)
    show_frame(1.0, 1.0)

    bird_pos = vector(0, 2.35 + 0.05 * math.sin(t * 2), 0)
    set_bird_pose(bird_pos, pitch=0.07, bank=0.08 * math.sin(t * 2), wing_phase=t * 5)
    show_strings(1.0, bird_pos)
    show_axes(opacity=1.0, build_u=u)

    title_text("The origin, axes, and eight octant target regions appear around the puppet.")

def play_name_for_phase(idx):
    return PHASES[idx][0].replace("Play/test: ", "")

def flight_position_for_phase(idx, t):
    # idx 9..16
    if idx == 9:
        return move_sequence(["O1", "O2", "O3", "O4", "O5", "O6", "O7", "O8"], t, 1.15)
    if idx == 10:
        return move_sequence(["O1", "CENTER", "O7", "CENTER", "O2", "CENTER", "O8", "CENTER", "O3", "CENTER", "O5", "CENTER"], t, 0.75)
    if idx == 11:
        r = 3.6 + 0.2 * math.sin(t * 0.5)
        return vector(r * math.cos(t * 0.9), 2.35 + 1.45 * math.sin(t * 0.65), r * math.sin(t * 0.9))
    if idx == 12:
        return move_sequence(["O1", "O7", "O2", "O8", "O3", "O5", "O4", "O6", "O1"], t, 1.0) + vector(0.18 * math.sin(t * 6), 0.12 * math.sin(t * 4), 0.18 * math.cos(t * 5))
    if idx == 13:
        r = min(4.5, 0.3 + 0.35 * t)
        return vector(r * math.cos(t * 1.4), 2.35 + 1.45 * math.sin(t * 0.75), r * math.sin(t * 1.4))
    if idx == 14:
        return vector(
            3.7 * math.sin(1.45 * t + 0.8 * math.sin(0.9 * t)),
            2.35 + 1.55 * math.sin(1.1 * t + 0.7 * math.sin(2.1 * t)),
            3.8 * math.sin(1.85 * t + 0.9 * math.cos(1.2 * t)),
        )
    if idx == 15:
        a = 3.4
        x = a * math.sin(t * 1.05)
        z = a * math.sin(t * 2.1) / 2.0
        y = 2.35 + 1.55 * math.sin(t * 1.05 + math.pi / 3)
        rot = 0.55 * math.sin(t * 0.32)
        return vector(x * math.cos(rot) - z * math.sin(rot), y, x * math.sin(rot) + z * math.cos(rot))
    if idx == 16:
        phase = int((t // 3.0) % 4)
        lt = t % 3.0
        if phase == 0:
            return move_sequence(["O1", "O7", "O2", "O8"], lt, 0.75)
        if phase == 1:
            r = 1.0 + 0.9 * lt
            return vector(r * math.cos(lt * 2.8), 2.35 + 1.35 * math.sin(lt * 1.9), r * math.sin(lt * 2.8))
        if phase == 2:
            return move_sequence(["O3", "O5", "O4", "O6"], lt, 0.75)
        return vector(3.2 * math.sin(lt * 3.1), 2.35 + 1.55 * math.sin(lt * 2.0), 3.2 * math.sin(lt * 4.0 + 1.0))
    return vector(0, 2.35, 0)

def target_for_play_phase(idx, t):
    if idx == 9:
        return target_for_octant(["O1", "O2", "O3", "O4", "O5", "O6", "O7", "O8"][int(t / 1.15) % 8])
    if idx == 10:
        seq = ["O1", "CENTER", "O7", "CENTER", "O2", "CENTER", "O8", "CENTER", "O3", "CENTER", "O5", "CENTER"]
        return target_for_octant(seq[int(t / 0.75) % len(seq)])
    if idx == 12:
        seq = ["O1", "O7", "O2", "O8", "O3", "O5", "O4", "O6", "O1"]
        return target_for_octant(seq[int(t / 1.0) % len(seq)])
    if idx == 14:
        seq = ["O6", "O4", "O3", "O5", "O2", "O8", "O7", "O1"]
        return target_for_octant(seq[int(t / 1.0) % len(seq)])
    if idx == 16:
        seq = ["O1", "O7", "O2", "O8", "O3", "O5", "O4", "O6"]
        return target_for_octant(seq[int(t / 1.2) % len(seq)])
    return flight_position_for_phase(idx, t + 0.8)

def update_metrics(pos, idx, t):
    global last_pos, prev_signs, near_origin_passes, loop_proxy
    octn, signs = octant_name(pos)
    visited_octants.add(octn)

    cur_signs = (sign(pos.x), sign(pos.z), sign(pos.y - 2.25))
    if prev_signs is not None:
        if cur_signs[0] != prev_signs[0]:
            axis_crossings["x"] += 1
        if cur_signs[1] != prev_signs[1]:
            axis_crossings["y"] += 1
        if cur_signs[2] != prev_signs[2]:
            axis_crossings["z"] += 1
    prev_signs = cur_signs

    if mag(vector(pos.x, 0, pos.z)) < 0.55 and mag(vector(last_pos.x, 0, last_pos.z)) >= 0.55:
        near_origin_passes += 1

    if idx in [11, 12, 15, 16]:
        loop_proxy = int(t / 3.8)

    last_pos = pos
    return octn, signs

def add_trail(pos):
    if len(trail_segments) > 260:
        old = trail_segments.pop(0)
        old.visible = False
    if len(trail_segments) == 0:
        trail_segments.append(reg(curve(pos=[pos, pos + vector(0.001, 0, 0)], color=vector(0.10, 0.16, 0.24), radius=0.018)))
    else:
        last_curve = trail_segments[-1]
        try:
            prev = last_curve.pos[-1]
        except Exception:
            prev = pos
        trail_segments.append(reg(curve(pos=[prev, pos], color=vector(0.10, 0.16, 0.24), radius=0.018)))

def animate_play(t, u):
    prepare_phase_visibility()
    show_frame(1.0, 1.0)
    show_axes(1.0, 1.0)
    show_bird(1.0)

    pos = flight_position_for_phase(phase_index, t)
    pos.y = clamp(pos.y, 0.65, 4.7)

    vel = pos - last_pos
    if mag(vel) > 0.001:
        yaw = math.atan2(vel.x, vel.z) - math.pi / 2
        pitch = 0.14 * math.sin(t * 2.0) + 0.04 * vel.y
        bank = -0.32 * math.sin(t * 1.4) + 0.10 * vel.x
    else:
        yaw = 0
        pitch = 0
        bank = 0

    set_bird_pose(pos, yaw=yaw, pitch=pitch, bank=bank, wing_phase=t * 8)
    show_strings(1.0, pos)
    add_trail(pos)

    if target_marker and target_ring:
        target = target_for_play_phase(phase_index, t)
        target_marker.visible = True
        target_ring.visible = True
        set_opacity_safe(target_marker, 1.0)
        set_opacity_safe(target_ring, 1.0)
        target_marker.pos = target
        target_ring.pos = target

    octn, signs = update_metrics(pos, phase_index, t)
    speed_proxy = mag(vel) / max(DT, 0.001)
    title_text(
        f"Behavior: {play_name_for_phase(phase_index)}\n"
        f"Current octant: {octn} ({signs}) | visited: {len(visited_octants)}/8 {', '.join(sorted(visited_octants))}\n"
        f"Axis crossings: x={axis_crossings['x']} y={axis_crossings['y']} z={axis_crossings['z']} total={sum(axis_crossings.values())}\n"
        f"Near-origin passes: {near_origin_passes} | loop proxy: {loop_proxy} | speed proxy: {speed_proxy:4.1f}"
    )

# --------------------------------------------------------------------------------------
# Phase dispatcher
# --------------------------------------------------------------------------------------

def jump_to_phase(idx):
    global phase_index, phase_time
    phase_index = idx % len(PHASES)
    phase_time = 0.0
    reset_metrics()
    # Hide old trails on phase changes.
    for seg in trail_segments:
        seg.visible = False
    trail_segments.clear()

def update_phase():
    u = phase_time / max(phase_duration(), 0.001)
    t = phase_time

    if phase_index == 0:
        animate_materials(t, u)
    elif phase_index == 1:
        animate_cut_body(t, u)
    elif phase_index == 2:
        animate_wings(t, u)
    elif phase_index == 3:
        animate_tail(t, u)
    elif phase_index == 4:
        animate_balance(t, u)
    elif phase_index == 5:
        animate_frame(t, u)
    elif phase_index == 6:
        animate_strings(t, u)
    elif phase_index == 7:
        animate_lift(t, u)
    elif phase_index == 8:
        animate_octants(t, u)
    else:
        animate_play(t, u)

# --------------------------------------------------------------------------------------
# Keyboard
# --------------------------------------------------------------------------------------

def on_key(evt):
    global paused, autonomous, TIME_SCALE, show_help
    k = evt.key.lower()
    if k == " ":
        paused = not paused
    elif k == "a":
        autonomous = not autonomous
    elif k == "n":
        jump_to_phase(phase_index + 1)
    elif k == "b":
        jump_to_phase(phase_index - 1)
    elif k == "r":
        jump_to_phase(0)
    elif k == "f":
        TIME_SCALE = min(4.0, TIME_SCALE + 0.25)
    elif k == "s":
        TIME_SCALE = max(0.25, TIME_SCALE - 0.25)
    elif k == "h":
        show_help = not show_help

scene.bind("keydown", on_key)

# --------------------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------------------

# Start with all objects hidden until first update makes them appear.
hide_all()

while True:
    rate(50)

    if not paused:
        phase_time += DT * TIME_SCALE
        global_time += DT * TIME_SCALE

    update_phase()

    if autonomous and phase_time >= phase_duration():
        jump_to_phase(phase_index + 1)
