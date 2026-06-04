#!/usr/bin/env python3
"""
Blender Single Neuron Signal Simulation
---------------------------------------

Run inside Blender:
    blender --python blender_single_neuron_signal_simulation.py

Or:
    Blender > Scripting tab > Open this file > Run Script

This script builds a simple animated neuron scene:
- dendrites receive glowing input signals
- the soma accumulates activation
- the axon hillock flashes when threshold is reached
- an action-potential pulse travels down the axon
- synaptic terminals release neurotransmitter particles
- labels show voltage, threshold, and simulation phase

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
    """Create a material with optional emission and transparency."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes

    bsdf = nodes.get("Principled BSDF")
    if bsdf:
        if emission:
            try:
                bsdf.inputs["Base Color"].default_value = color
                bsdf.inputs["Emission Color"].default_value = color
                bsdf.inputs["Emission Strength"].default_value = strength
            except Exception:
                pass
        else:
            bsdf.inputs["Base Color"].default_value = color

        bsdf.inputs["Alpha"].default_value = alpha

    mat.blend_method = "BLEND" if alpha < 1.0 else "OPAQUE"
    mat.use_screen_refraction = alpha < 1.0
    return mat


def add_uv_sphere(name, loc, radius, mat, segments=32, rings=16):
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


def add_cylinder_between(name, start, end, radius, mat, vertices=24):
    """Create cylinder between two 3D points."""
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


def add_text(name, text, loc, size, mat):
    bpy.ops.object.text_add(location=loc, rotation=(math.radians(70), 0, 0))
    obj = bpy.context.object
    obj.name = name
    obj.data.body = text
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


def keyframe_scale(obj, frame, scale):
    obj.scale = scale
    obj.keyframe_insert("scale", frame=frame)


def keyframe_location(obj, frame, loc):
    obj.location = loc
    obj.keyframe_insert("location", frame=frame)


def set_emission_strength(mat, frame, strength):
    """Keyframe emission strength on material if available."""
    nodes = mat.node_tree.nodes
    bsdf = nodes.get("Principled BSDF")
    if bsdf and "Emission Strength" in bsdf.inputs:
        bsdf.inputs["Emission Strength"].default_value = strength
        bsdf.inputs["Emission Strength"].keyframe_insert("default_value", frame=frame)


# ---------------------------------------------------------------------------
# Materials
# ---------------------------------------------------------------------------

mat_soma = make_mat("transparent soma membrane", (0.35, 0.75, 1.0, 0.28), alpha=0.28)
mat_soma_glow = make_mat("soma activation glow", (0.20, 0.85, 1.0, 0.45), emission=True, strength=0.5, alpha=0.45)
mat_dendrite = make_mat("dendrite branches", (0.18, 0.42, 0.62, 1.0))
mat_axon = make_mat("axon cable", (0.95, 0.70, 0.35, 1.0))
mat_myelin = make_mat("myelin sheath", (0.92, 0.88, 0.72, 1.0))
mat_node = make_mat("nodes of Ranvier", (1.0, 0.52, 0.05, 1.0), emission=True, strength=0.8)
mat_input = make_mat("incoming dendrite signal", (0.35, 1.0, 0.45, 1.0), emission=True, strength=2.5)
mat_spike = make_mat("action potential pulse", (1.0, 0.78, 0.12, 1.0), emission=True, strength=4.0)
mat_syn = make_mat("neurotransmitter particles", (0.9, 0.25, 1.0, 1.0), emission=True, strength=2.2)
mat_text = make_mat("dark label text", (0.02, 0.02, 0.025, 1.0))
mat_threshold = make_mat("threshold red", (1.0, 0.10, 0.08, 1.0), emission=True, strength=1.2)
mat_floor = make_mat("matte light floor", (0.78, 0.82, 0.86, 1.0))


# ---------------------------------------------------------------------------
# Scene geometry
# ---------------------------------------------------------------------------

# Floor
bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, -0.08))
floor = bpy.context.object
floor.name = "light floor"
floor.dimensions = (12, 7, 0.08)
floor.data.materials.append(mat_floor)

# Soma and activation glow
soma = add_uv_sphere("soma / cell body", (0, 0, 1.35), 0.72, mat_soma)
soma_glow = add_uv_sphere("soma accumulated activation", (0, 0, 1.35), 0.45, mat_soma_glow)
soma_glow.scale = (0.2, 0.2, 0.2)

# Axon hillock
hillock = add_uv_sphere("axon hillock threshold zone", (0.72, 0, 1.28), 0.18, mat_threshold)

# Dendrites
dendrite_roots = [
    ((-0.55, 0.18, 1.55), (-2.1, 1.45, 2.15)),
    ((-0.60, -0.10, 1.40), (-2.35, -0.95, 1.55)),
    ((-0.35, 0.42, 1.65), (-1.65, 0.20, 2.85)),
    ((-0.42, -0.42, 1.25), (-1.75, -1.75, 0.82)),
]
dendrite_paths = []

for i, (start, end) in enumerate(dendrite_roots, start=1):
    dendrite_paths.append((Vector(end), Vector(start)))
    add_cylinder_between(f"dendrite main branch {i}", start, end, 0.035, mat_dendrite)

    # Add small secondary branches.
    end_v = Vector(end)
    for j, angle in enumerate((-0.55, 0.55), start=1):
        branch_end = end_v + Vector((0.45 * math.cos(angle+i), 0.45 * math.sin(angle+i), 0.35 * math.sin(angle*j)))
        add_cylinder_between(f"dendrite side branch {i}.{j}", end_v, branch_end, 0.022, mat_dendrite)

# Axon
axon_points = [
    Vector((0.78, 0, 1.28)),
    Vector((1.7, 0.05, 1.18)),
    Vector((2.65, -0.04, 1.25)),
    Vector((3.6, 0.03, 1.15)),
    Vector((4.55, 0.00, 1.22)),
]
for i in range(len(axon_points)-1):
    add_cylinder_between(f"axon segment {i+1}", axon_points[i], axon_points[i+1], 0.045, mat_axon)

# Myelin sheath rings/segments along axon
myelin_positions = [
    (1.25, 0.04, 1.22),
    (2.10, 0.00, 1.22),
    (3.05, 0.00, 1.20),
    (4.00, 0.02, 1.18),
]
for i, loc in enumerate(myelin_positions, start=1):
    bpy.ops.mesh.primitive_cylinder_add(vertices=32, radius=0.12, depth=0.28, location=loc, rotation=(0, math.radians(90), 0))
    obj = bpy.context.object
    obj.name = f"myelin sheath {i}"
    obj.data.materials.append(mat_myelin)

# Nodes of Ranvier
node_positions = [
    (0.92, 0.02, 1.26),
    (1.67, 0.03, 1.18),
    (2.55, -0.03, 1.24),
    (3.50, 0.02, 1.16),
    (4.40, 0.00, 1.21),
]
for i, loc in enumerate(node_positions, start=1):
    add_uv_sphere(f"node of Ranvier {i}", loc, 0.07, mat_node, segments=16, rings=8)

# Synaptic terminals
terminal_positions = [
    (4.95, 0.28, 1.38),
    (5.00, -0.02, 1.15),
    (4.92, -0.32, 0.95),
]
for i, loc in enumerate(terminal_positions, start=1):
    add_uv_sphere(f"synaptic terminal {i}", loc, 0.16, mat_axon, segments=16, rings=8)
    add_cylinder_between(f"terminal connector {i}", axon_points[-1], loc, 0.022, mat_axon)

# Receiving neuron target pad
receiver = add_uv_sphere("receiving neuron contact area", (5.75, 0.0, 1.12), 0.38, mat_soma)


# ---------------------------------------------------------------------------
# Animated signal pulses
# ---------------------------------------------------------------------------

# Incoming dendrite signal pulses move along dendrite paths into soma.
input_pulses = []
for i, (start, end) in enumerate(dendrite_paths, start=1):
    pulse = add_uv_sphere(f"dendrite input pulse {i}", start, 0.085, mat_input, segments=16, rings=8)
    input_pulses.append((pulse, start, end))

    frame_start = 10 + i * 10
    frame_end = frame_start + 42

    keyframe_visibility(pulse, 1, False)
    keyframe_visibility(pulse, frame_start, True)
    keyframe_location(pulse, frame_start, start)
    keyframe_location(pulse, frame_end, end)
    keyframe_visibility(pulse, frame_end + 4, False)

# Soma glow grows as inputs arrive.
for f, s in [(1, (0.2, 0.2, 0.2)), (35, (0.35, 0.35, 0.35)), (58, (0.62, 0.62, 0.62)), (80, (1.05, 1.05, 1.05)), (112, (0.45, 0.45, 0.45))]:
    keyframe_scale(soma_glow, f, s)

set_emission_strength(mat_soma_glow, 1, 0.2)
set_emission_strength(mat_soma_glow, 45, 0.9)
set_emission_strength(mat_soma_glow, 75, 2.8)
set_emission_strength(mat_soma_glow, 105, 0.6)

# Hillock threshold flash.
for f, s in [(1, (0.8, 0.8, 0.8)), (70, (1.0, 1.0, 1.0)), (84, (1.8, 1.8, 1.8)), (98, (1.0, 1.0, 1.0))]:
    keyframe_scale(hillock, f, s)

set_emission_strength(mat_threshold, 1, 0.4)
set_emission_strength(mat_threshold, 70, 0.8)
set_emission_strength(mat_threshold, 84, 5.0)
set_emission_strength(mat_threshold, 100, 0.8)

# Action potential pulse travels down axon.
spike_pulse = add_uv_sphere("action potential pulse traveling down axon", axon_points[0], 0.12, mat_spike)
keyframe_visibility(spike_pulse, 1, False)
keyframe_visibility(spike_pulse, 82, True)
for idx, point in enumerate(axon_points):
    keyframe_location(spike_pulse, 82 + idx * 14, point)
keyframe_visibility(spike_pulse, 150, False)

# Node flashes as spike passes.
for i, loc in enumerate(node_positions, start=1):
    flash = add_uv_sphere(f"node flash {i}", loc, 0.12, mat_spike, segments=16, rings=8)
    keyframe_visibility(flash, 1, False)
    start_frame = 84 + i * 10
    keyframe_visibility(flash, start_frame, True)
    keyframe_scale(flash, start_frame, (0.5, 0.5, 0.5))
    keyframe_scale(flash, start_frame + 4, (1.8, 1.8, 1.8))
    keyframe_scale(flash, start_frame + 10, (0.5, 0.5, 0.5))
    keyframe_visibility(flash, start_frame + 12, False)

# Neurotransmitter release at synapse.
for i in range(18):
    start = Vector(terminal_positions[i % len(terminal_positions)])
    offset = Vector((0.45 + 0.18 * (i % 3), -0.30 + 0.06 * i, -0.12 + 0.04 * (i % 7)))
    end = Vector((5.55, 0, 1.12)) + offset * 0.35
    nt = add_uv_sphere(f"neurotransmitter particle {i+1}", start, 0.04, mat_syn, segments=12, rings=6)
    keyframe_visibility(nt, 1, False)
    f0 = 142 + (i % 6) * 3
    keyframe_visibility(nt, f0, True)
    keyframe_location(nt, f0, start)
    keyframe_location(nt, f0 + 22, end)
    keyframe_visibility(nt, f0 + 28, False)

# Receiver glow pulse
receiver_glow = add_uv_sphere("receiving neuron response glow", receiver.location, 0.42, mat_input, segments=24, rings=12)
receiver_glow.name = "receiving neuron response glow"
for f, s in [(1, (0.2, 0.2, 0.2)), (150, (0.2, 0.2, 0.2)), (168, (1.2, 1.2, 1.2)), (190, (0.35, 0.35, 0.35))]:
    keyframe_scale(receiver_glow, f, s)


# ---------------------------------------------------------------------------
# Scientific labels
# ---------------------------------------------------------------------------

add_text(
    "title label",
    "Single Neuron Receiving Signals",
    (0, -2.9, 3.15),
    0.26,
    mat_text,
)

voltage_labels = [
    (1, "Voltage: -70 mV  | resting"),
    (42, "Voltage: -62 mV  | dendritic inputs accumulating"),
    (76, "Voltage: -54 mV  | threshold reached"),
    (96, "Voltage: +30 mV  | action potential"),
    (125, "Voltage: -75 mV  | reset / refractory"),
    (165, "Synapse: neurotransmitter release"),
]
for f, body in voltage_labels:
    label = add_text(f"voltage label frame {f}", body, (0, 2.75, 2.80), 0.18, mat_text)
    keyframe_visibility(label, 1, False)
    keyframe_visibility(label, f, True)
    keyframe_visibility(label, f + 28, False)

add_text("dendrite label", "dendrites receive input", (-2.45, 1.7, 2.55), 0.15, mat_text)
add_text("soma label", "soma integrates signals", (0, -1.10, 2.25), 0.15, mat_text)
add_text("axon label", "axon carries spike", (2.85, -0.78, 1.85), 0.15, mat_text)
add_text("synapse label", "synaptic output", (5.2, 0.9, 1.95), 0.15, mat_text)


# ---------------------------------------------------------------------------
# Lighting and camera
# ---------------------------------------------------------------------------

bpy.ops.object.light_add(type="AREA", location=(0, -4, 6))
area = bpy.context.object
area.name = "large softbox light"
area.data.energy = 600
area.data.size = 5

bpy.ops.object.light_add(type="POINT", location=(2, 1.5, 3.2))
point = bpy.context.object
point.name = "signal glow light"
point.data.energy = 100
point.data.color = (1.0, 0.78, 0.28)

bpy.ops.object.camera_add(location=(3.6, -7.0, 4.2), rotation=(math.radians(62), 0, math.radians(31)))
camera = bpy.context.object
bpy.context.scene.camera = camera
camera.name = "main camera"
camera.data.lens = 32


# ---------------------------------------------------------------------------
# Timeline / rendering settings
# ---------------------------------------------------------------------------

scene = bpy.context.scene
scene.frame_start = 1
scene.frame_end = 200
scene.frame_set(1)

scene.render.fps = 24
scene.render.engine = "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in [item.identifier for item in scene.render.bl_rna.properties["engine"].enum_items] else "BLENDER_EEVEE"

# Enable bloom when available.
if hasattr(scene, "eevee"):
    if hasattr(scene.eevee, "use_bloom"):
        scene.eevee.use_bloom = True

scene.render.resolution_x = 1400
scene.render.resolution_y = 900


# ---------------------------------------------------------------------------
# Add a simple voltage trace curve as a visual scientific panel
# ---------------------------------------------------------------------------

# Make a membrane-voltage style line using small cylinders.
trace_origin = Vector((-3.3, -2.25, 0.55))
trace_scale_x = 0.055
trace_scale_y = 0.020

voltage_points = [
    (0, -70), (20, -68), (40, -62), (60, -58), (75, -54),
    (86, 30), (96, -75), (125, -72), (155, -68), (180, -64)
]
trace_points = []
for frame, mv in voltage_points:
    x = trace_origin.x + frame * trace_scale_x
    z = trace_origin.z + (mv + 80) * trace_scale_y
    trace_points.append(Vector((x, trace_origin.y, z)))

for i in range(len(trace_points)-1):
    add_cylinder_between(f"voltage trace segment {i+1}", trace_points[i], trace_points[i+1], 0.012, mat_threshold, vertices=12)

add_text("voltage trace label", "membrane voltage trace", (-1.1, -2.35, 1.65), 0.12, mat_text)


# Save file next to script if running in normal Blender environment.
# Comment this out if you do not want an automatic .blend save.
try:
    bpy.ops.wm.save_as_mainfile(filepath="single_neuron_signal_simulation.blend")
except Exception:
    pass
