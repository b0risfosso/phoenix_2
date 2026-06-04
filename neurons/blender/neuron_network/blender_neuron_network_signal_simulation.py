#!/usr/bin/env python3
"""
Blender Neuron Network Signal Simulation
----------------------------------------

Run inside Blender:
    blender --python blender_neuron_network_signal_simulation.py

Or:
    Blender > Scripting tab > Open this file > Run Script

This script builds a 3D animated neuron network:
- multiple neurons with soma, dendrites, axons, and synaptic terminals
- excitatory neurons send warm/orange signal pulses
- inhibitory neurons send cool/blue suppressive pulses
- active synapses brighten during transmission
- network rounds show sensory input, propagation, inhibition, synchronization,
  learning-strengthening, and dream/replay mode
- floating labels describe the current network phase
- a small activity panel shows firing-rate bars and a voltage trace

No external Python packages are required.
"""

import bpy
import math
from mathutils import Vector


# ---------------------------------------------------------------------------
# Clean scene
# ---------------------------------------------------------------------------

bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_mat(name, color, emission=False, strength=1.0, alpha=1.0):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    bsdf = nodes.get("Principled BSDF")

    if bsdf:
        bsdf.inputs["Base Color"].default_value = color
        bsdf.inputs["Alpha"].default_value = alpha
        if emission:
            try:
                bsdf.inputs["Emission Color"].default_value = color
                bsdf.inputs["Emission Strength"].default_value = strength
            except Exception:
                pass

    mat.blend_method = "BLEND" if alpha < 1.0 else "OPAQUE"
    mat.use_screen_refraction = alpha < 1.0
    return mat


def set_emission_strength(mat, frame, strength):
    nodes = mat.node_tree.nodes
    bsdf = nodes.get("Principled BSDF")
    if bsdf and "Emission Strength" in bsdf.inputs:
        bsdf.inputs["Emission Strength"].default_value = strength
        bsdf.inputs["Emission Strength"].keyframe_insert("default_value", frame=frame)


def add_uv_sphere(name, loc, radius, mat, segments=24, rings=12):
    bpy.ops.mesh.primitive_uv_sphere_add(
        segments=segments,
        ring_count=rings,
        radius=radius,
        location=loc,
    )
    obj = bpy.context.object
    obj.name = name
    obj.data.materials.append(mat)
    return obj


def add_cylinder_between(name, start, end, radius, mat, vertices=18):
    start = Vector(start)
    end = Vector(end)
    mid = (start + end) / 2.0
    direction = end - start
    length = direction.length

    bpy.ops.mesh.primitive_cylinder_add(
        vertices=vertices,
        radius=radius,
        depth=length,
        location=mid,
    )
    obj = bpy.context.object
    obj.name = name

    quat = direction.to_track_quat("Z", "Y")
    obj.rotation_euler = quat.to_euler()
    obj.data.materials.append(mat)
    return obj


def add_text(name, body, loc, size, mat, rot_x=68):
    bpy.ops.object.text_add(location=loc, rotation=(math.radians(rot_x), 0, 0))
    obj = bpy.context.object
    obj.name = name
    obj.data.body = body
    obj.data.align_x = "CENTER"
    obj.data.align_y = "CENTER"
    obj.data.size = size
    obj.data.materials.append(mat)
    return obj


def keyframe_visibility(obj, frame, visible):
    obj.hide_viewport = not visible
    obj.hide_render = not visible
    obj.keyframe_insert("hide_viewport", frame=frame)
    obj.keyframe_insert("hide_render", frame=frame)


def keyframe_location(obj, frame, loc):
    obj.location = loc
    obj.keyframe_insert("location", frame=frame)


def keyframe_scale(obj, frame, scale):
    obj.scale = scale
    obj.keyframe_insert("scale", frame=frame)


def animate_pulse(name, start, end, mat, frame_start, frame_end, radius=0.08):
    pulse = add_uv_sphere(name, start, radius, mat, segments=16, rings=8)
    keyframe_visibility(pulse, 1, False)
    keyframe_visibility(pulse, frame_start, True)
    keyframe_location(pulse, frame_start, start)
    keyframe_location(pulse, frame_end, end)
    keyframe_visibility(pulse, frame_end + 4, False)
    return pulse


def flash_object(obj, start_frame, peak_frame, end_frame, small=(1,1,1), large=(1.7,1.7,1.7)):
    keyframe_scale(obj, start_frame, small)
    keyframe_scale(obj, peak_frame, large)
    keyframe_scale(obj, end_frame, small)


def draw_polyline(name, points, radius, mat):
    objs = []
    for i in range(len(points) - 1):
        objs.append(add_cylinder_between(f"{name} segment {i+1}", points[i], points[i+1], radius, mat))
    return objs


# ---------------------------------------------------------------------------
# Materials
# ---------------------------------------------------------------------------

mat_floor = make_mat("matte light floor", (0.80, 0.84, 0.88, 1.0))
mat_text = make_mat("dark text", (0.02, 0.02, 0.03, 1.0))

mat_exc_soma = make_mat("excitatory soma transparent", (1.0, 0.55, 0.16, 0.34), alpha=0.34)
mat_inh_soma = make_mat("inhibitory soma transparent", (0.20, 0.52, 1.0, 0.34), alpha=0.34)
mat_soma_glow_exc = make_mat("excitatory activation glow", (1.0, 0.48, 0.05, 0.42), emission=True, strength=0.5, alpha=0.42)
mat_soma_glow_inh = make_mat("inhibitory activation glow", (0.08, 0.45, 1.0, 0.42), emission=True, strength=0.5, alpha=0.42)

mat_dendrite = make_mat("dendrites dark teal", (0.16, 0.38, 0.48, 1.0))
mat_axon_exc = make_mat("excitatory axon", (1.0, 0.72, 0.26, 1.0))
mat_axon_inh = make_mat("inhibitory axon", (0.25, 0.64, 1.0, 1.0))
mat_myelin = make_mat("myelin pale sheath", (0.92, 0.88, 0.72, 1.0))

mat_exc_signal = make_mat("excitatory signal pulse", (1.0, 0.72, 0.10, 1.0), emission=True, strength=4.0)
mat_inh_signal = make_mat("inhibitory signal pulse", (0.10, 0.55, 1.0, 1.0), emission=True, strength=4.0)
mat_input_signal = make_mat("sensory input signal", (0.25, 1.0, 0.42, 1.0), emission=True, strength=3.0)
mat_dream_signal = make_mat("dream replay signal", (0.88, 0.25, 1.0, 1.0), emission=True, strength=3.5)

mat_synapse_dim = make_mat("dim synapse line", (0.45, 0.45, 0.48, 0.32), alpha=0.32)
mat_synapse_exc = make_mat("active excitatory synapse", (1.0, 0.55, 0.05, 0.58), emission=True, strength=0.2, alpha=0.58)
mat_synapse_inh = make_mat("active inhibitory synapse", (0.10, 0.50, 1.0, 0.58), emission=True, strength=0.2, alpha=0.58)
mat_learning = make_mat("learning strengthened synapse", (1.0, 0.95, 0.18, 0.70), emission=True, strength=0.8, alpha=0.70)

mat_bar = make_mat("activity bars", (0.18, 0.75, 0.95, 1.0), emission=True, strength=0.8)
mat_trace = make_mat("voltage trace red", (1.0, 0.10, 0.08, 1.0), emission=True, strength=0.8)


# ---------------------------------------------------------------------------
# Scene base
# ---------------------------------------------------------------------------

bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, -0.08))
floor = bpy.context.object
floor.name = "light floor"
floor.dimensions = (13, 8, 0.08)
floor.data.materials.append(mat_floor)

add_text("title", "Neuron Network Signal Simulation", (0, -3.55, 3.55), 0.28, mat_text)


# ---------------------------------------------------------------------------
# Network definition
# ---------------------------------------------------------------------------

neurons = {
    "E0": {"type": "E", "pos": Vector((-3.7,  1.8, 1.3))},
    "E1": {"type": "E", "pos": Vector((-3.5, -1.6, 1.3))},
    "E2": {"type": "E", "pos": Vector((-0.7,  1.2, 1.6))},
    "E3": {"type": "E", "pos": Vector(( 1.5, -1.1, 1.35))},
    "I4": {"type": "I", "pos": Vector((-0.1, -1.9, 1.45))},
    "I5": {"type": "I", "pos": Vector(( 2.8,  1.2, 1.45))},
    "OUT": {"type": "E", "pos": Vector((4.7, 0.0, 1.35))},
}

connections = [
    ("E0", "E2", "E"),
    ("E1", "E2", "E"),
    ("E1", "I4", "E"),
    ("E2", "E3", "E"),
    ("E2", "I5", "E"),
    ("I4", "E3", "I"),
    ("I5", "OUT", "I"),
    ("E3", "OUT", "E"),
    ("E0", "I5", "E"),
    ("I4", "E2", "I"),
]


# ---------------------------------------------------------------------------
# Build neuron objects
# ---------------------------------------------------------------------------

for name, data in neurons.items():
    pos = data["pos"]
    is_inhibitory = data["type"] == "I"
    soma_mat = mat_inh_soma if is_inhibitory else mat_exc_soma
    glow_mat = mat_soma_glow_inh if is_inhibitory else mat_soma_glow_exc

    soma = add_uv_sphere(f"{name} soma", pos, 0.36, soma_mat)
    glow = add_uv_sphere(f"{name} activation glow", pos, 0.30, glow_mat)
    glow.scale = (0.2, 0.2, 0.2)

    data["soma"] = soma
    data["glow"] = glow

    # Small dendrite star around each soma.
    for j, angle in enumerate([0, 60, 125, 205, 290], start=1):
        a = math.radians(angle)
        start = pos + Vector((0.08 * math.cos(a), 0.08 * math.sin(a), 0))
        end = pos + Vector((0.62 * math.cos(a), 0.50 * math.sin(a), 0.22 * math.sin(a * 2)))
        add_cylinder_between(f"{name} dendrite {j}", start, end, 0.018, mat_dendrite, vertices=12)

    # Mini axon tail, mostly toward network output.
    direction = Vector((1, 0, 0))
    if name == "OUT":
        direction = Vector((0.7, 0.0, 0.15))
    axon_start = pos + direction * 0.33
    axon_end = pos + direction * 0.75 + Vector((0, 0, -0.07))
    add_cylinder_between(f"{name} local axon", axon_start, axon_end, 0.025, mat_axon_inh if is_inhibitory else mat_axon_exc, vertices=12)

    # Label
    kind = "inhibitory" if is_inhibitory else "excitatory"
    add_text(f"{name} label", f"{name}\n{kind}", (pos.x, pos.y, pos.z + 0.72), 0.12, mat_text)


# ---------------------------------------------------------------------------
# Build synaptic connections
# ---------------------------------------------------------------------------

synapse_objs = []
connection_paths = {}

for idx, (src, dst, kind) in enumerate(connections, start=1):
    start = neurons[src]["pos"]
    end = neurons[dst]["pos"]

    # Slight curve using midpoint offset.
    mid = (start + end) / 2 + Vector((0, 0, 0.35 + 0.08 * ((idx % 3) - 1)))
    points = [start, mid, end]
    line_mat = mat_synapse_exc if kind == "E" else mat_synapse_inh
    segments = draw_polyline(f"synapse {src}->{dst}", points, 0.012, mat_synapse_dim)
    synapse_objs.extend(segments)
    connection_paths[(src, dst)] = points

    # Terminal knob at target
    terminal_mat = mat_axon_exc if kind == "E" else mat_axon_inh
    term = add_uv_sphere(f"synaptic terminal {src}->{dst}", end - (end-start).normalized()*0.35, 0.055, terminal_mat, segments=12, rings=6)

    # Brighter overlay connection for active phases.
    active_segments = draw_polyline(f"active synapse {src}->{dst}", points, 0.018, line_mat)
    for obj in active_segments:
        keyframe_visibility(obj, 1, False)
    connection_paths[(src, dst, "active")] = active_segments


# ---------------------------------------------------------------------------
# Phase labels
# ---------------------------------------------------------------------------

phase_labels = [
    (1,   44,  "Round 1: Sensory input reaches E0 and E1"),
    (45,  86,  "Round 2: Excitatory propagation through E2"),
    (87,  128, "Round 3: Inhibitory neurons suppress part of the signal"),
    (129, 170, "Round 4: Synchronized firing drives the output neuron"),
    (171, 214, "Round 5: Learning strengthens repeated active synapses"),
    (215, 270, "Round 6: Dream/replay mode reactivates stored pathways"),
]
for f0, f1, body in phase_labels:
    label = add_text(f"phase label {f0}", body, (0, 3.25, 3.15), 0.18, mat_text)
    keyframe_visibility(label, 1, False)
    keyframe_visibility(label, f0, True)
    keyframe_visibility(label, f1, False)


# ---------------------------------------------------------------------------
# Animate activation glows
# ---------------------------------------------------------------------------

activation_schedule = {
    "E0": [(12, 28), (132, 150), (220, 238)],
    "E1": [(18, 36), (136, 154), (230, 248)],
    "E2": [(52, 72), (142, 162), (238, 256)],
    "I4": [(88, 112), (246, 264)],
    "E3": [(112, 132), (152, 172), (252, 270)],
    "I5": [(96, 122), (258, 270)],
    "OUT": [(158, 184), (264, 270)],
}

for name, windows in activation_schedule.items():
    glow = neurons[name]["glow"]
    keyframe_scale(glow, 1, (0.2, 0.2, 0.2))
    for f0, f1 in windows:
        keyframe_scale(glow, f0, (0.25, 0.25, 0.25))
        keyframe_scale(glow, (f0 + f1)//2, (1.25, 1.25, 1.25))
        keyframe_scale(glow, f1, (0.25, 0.25, 0.25))


# ---------------------------------------------------------------------------
# Animate signal pulses across connection paths
# ---------------------------------------------------------------------------

def animate_connection_pulse(src, dst, kind, frame_start, frame_end, mat=None, radius=0.07):
    points = connection_paths[(src, dst)]
    mat = mat or (mat_exc_signal if kind == "E" else mat_inh_signal)

    # Use two pulses along each curved connection segment.
    mid_frame = (frame_start + frame_end) // 2
    p1 = animate_pulse(f"pulse {src}->{dst} A {frame_start}", points[0], points[1], mat, frame_start, mid_frame, radius)
    p2 = animate_pulse(f"pulse {src}->{dst} B {frame_start}", points[1], points[2], mat, mid_frame, frame_end, radius)

    # Show active synapse overlay during the pulse.
    for obj in connection_paths[(src, dst, "active")]:
        keyframe_visibility(obj, 1, False)
        keyframe_visibility(obj, frame_start, True)
        keyframe_visibility(obj, frame_end + 5, False)

    return p1, p2


# Sensory inputs from outside.
animate_pulse("sensory pulse into E0", (-5.2, 2.2, 1.5), neurons["E0"]["pos"], mat_input_signal, 8, 28, 0.08)
animate_pulse("sensory pulse into E1", (-5.2, -2.0, 1.5), neurons["E1"]["pos"], mat_input_signal, 14, 36, 0.08)

# Round propagation
animate_connection_pulse("E0", "E2", "E", 42, 66)
animate_connection_pulse("E1", "E2", "E", 48, 72)
animate_connection_pulse("E1", "I4", "E", 74, 98)
animate_connection_pulse("E0", "I5", "E", 78, 104)

# Inhibition and mixed signal flow
animate_connection_pulse("I4", "E3", "I", 96, 122)
animate_connection_pulse("I4", "E2", "I", 104, 128)
animate_connection_pulse("E2", "E3", "E", 110, 138)
animate_connection_pulse("E2", "I5", "E", 116, 142)
animate_connection_pulse("I5", "OUT", "I", 126, 152)

# Synchronized firing to output
animate_connection_pulse("E0", "E2", "E", 132, 152)
animate_connection_pulse("E1", "E2", "E", 136, 156)
animate_connection_pulse("E2", "E3", "E", 148, 170)
animate_connection_pulse("E3", "OUT", "E", 162, 188)

# Learning phase: repeated active paths strengthen.
learning_connections = [("E0", "E2"), ("E1", "E2"), ("E2", "E3"), ("E3", "OUT")]
for i, (src, dst) in enumerate(learning_connections):
    points = connection_paths[(src, dst)]
    strong_segments = draw_polyline(f"strengthened learned path {src}->{dst}", points, 0.030, mat_learning)
    for seg in strong_segments:
        keyframe_visibility(seg, 1, False)
        keyframe_visibility(seg, 174 + i*4, True)
        keyframe_visibility(seg, 270, True)

animate_connection_pulse("E0", "E2", "E", 178, 198, mat_learning, radius=0.075)
animate_connection_pulse("E1", "E2", "E", 184, 204, mat_learning, radius=0.075)
animate_connection_pulse("E2", "E3", "E", 196, 218, mat_learning, radius=0.075)
animate_connection_pulse("E3", "OUT", "E", 210, 232, mat_learning, radius=0.075)

# Dream/replay mode: purple internal replay without outside input.
animate_connection_pulse("E0", "E2", "E", 222, 240, mat_dream_signal, radius=0.07)
animate_connection_pulse("E1", "E2", "E", 230, 248, mat_dream_signal, radius=0.07)
animate_connection_pulse("E2", "I5", "E", 242, 260, mat_dream_signal, radius=0.07)
animate_connection_pulse("I5", "OUT", "I", 252, 270, mat_dream_signal, radius=0.07)
animate_connection_pulse("E2", "E3", "E", 246, 266, mat_dream_signal, radius=0.07)
animate_connection_pulse("E3", "OUT", "E", 258, 270, mat_dream_signal, radius=0.07)


# ---------------------------------------------------------------------------
# Activity panel: firing-rate bars and voltage trace
# ---------------------------------------------------------------------------

panel_origin = Vector((-5.4, -3.05, 0.45))
add_text("activity panel title", "network activity panel", (panel_origin.x + 2.1, panel_origin.y - 0.15, panel_origin.z + 1.2), 0.12, mat_text)

bar_names = ["E0", "E1", "E2", "I4", "E3", "I5", "OUT"]
for i, name in enumerate(bar_names):
    x = panel_origin.x + i * 0.62
    bpy.ops.mesh.primitive_cube_add(size=1, location=(x, panel_origin.y, panel_origin.z))
    bar = bpy.context.object
    bar.name = f"{name} firing rate bar"
    bar.dimensions = (0.18, 0.18, 0.10)
    bar.data.materials.append(mat_bar)

    # Animate bar height by scaling z and moving upward.
    keyframe_scale(bar, 1, (0.18, 0.18, 0.10))
    for f0, f1 in activation_schedule.get(name, []):
        mid = (f0 + f1)//2
        keyframe_scale(bar, f0, (0.18, 0.18, 0.15))
        keyframe_scale(bar, mid, (0.18, 0.18, 1.2))
        keyframe_scale(bar, f1, (0.18, 0.18, 0.20))

    add_text(f"{name} bar label", name, (x, panel_origin.y - 0.25, panel_origin.z - 0.05), 0.08, mat_text)

# Voltage trace for OUT neuron.
trace_points_raw = [
    (0, -70), (40, -68), (80, -63), (120, -58), (150, -54),
    (166, 25), (178, -75), (210, -70), (240, -58), (268, 10)
]
trace_points = []
trace_origin = Vector((-5.25, -3.20, 1.95))
for frame, mv in trace_points_raw:
    trace_points.append(Vector((trace_origin.x + frame*0.020, trace_origin.y, trace_origin.z + (mv + 80)*0.012)))
draw_polyline("OUT voltage trace", trace_points, 0.010, mat_trace)
add_text("out voltage trace label", "OUT voltage trace", (-2.7, -3.35, 2.55), 0.10, mat_text)


# ---------------------------------------------------------------------------
# Lights and camera
# ---------------------------------------------------------------------------

bpy.ops.object.light_add(type="AREA", location=(0, -4.5, 7))
area = bpy.context.object
area.name = "large soft area light"
area.data.energy = 750
area.data.size = 6

bpy.ops.object.light_add(type="POINT", location=(-1.5, 1.5, 4.0))
point = bpy.context.object
point.name = "signal glow light"
point.data.energy = 130
point.data.color = (1.0, 0.72, 0.35)

bpy.ops.object.camera_add(location=(2.6, -8.2, 5.0), rotation=(math.radians(62), 0, math.radians(24)))
camera = bpy.context.object
camera.name = "main camera"
camera.data.lens = 30
bpy.context.scene.camera = camera


# ---------------------------------------------------------------------------
# Timeline/render settings
# ---------------------------------------------------------------------------

scene = bpy.context.scene
scene.frame_start = 1
scene.frame_end = 270
scene.frame_set(1)
scene.render.fps = 24
scene.render.resolution_x = 1500
scene.render.resolution_y = 950

# Prefer Eevee for bloom/glow speed.
try:
    engines = [item.identifier for item in scene.render.bl_rna.properties["engine"].enum_items]
    scene.render.engine = "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in engines else "BLENDER_EEVEE"
except Exception:
    pass

# Enable bloom when Blender version supports it.
if hasattr(scene, "eevee"):
    if hasattr(scene.eevee, "use_bloom"):
        scene.eevee.use_bloom = True


# ---------------------------------------------------------------------------
# Save .blend file if possible
# ---------------------------------------------------------------------------

try:
    bpy.ops.wm.save_as_mainfile(filepath="neuron_network_signal_simulation.blend")
except Exception:
    pass
