# waterproof_tape_unwind_rewind_aligned_blender.py
# Run in Blender with:
# blender --python waterproof_tape_unwind_rewind_aligned_blender.py

import bpy
import math
from mathutils import Vector

# ============================================================
# Scene setup
# ============================================================
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete()

scene = bpy.context.scene
scene.frame_start = 1
scene.frame_end = 220
scene.render.fps = 24
scene.world.color = (1.0, 1.0, 1.0)

try:
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 64
except Exception:
    scene.render.engine = "BLENDER_EEVEE"

try:
    scene.view_settings.view_transform = "Filmic"
    scene.view_settings.look = "Medium High Contrast"
except Exception:
    pass

# ============================================================
# Helpers
# ============================================================
def make_mat(name, color, roughness=0.4, metallic=0.0, alpha=1.0):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    mat.diffuse_color = color
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        if "Base Color" in bsdf.inputs:
            bsdf.inputs["Base Color"].default_value = color
        if "Roughness" in bsdf.inputs:
            bsdf.inputs["Roughness"].default_value = roughness
        if "Metallic" in bsdf.inputs:
            bsdf.inputs["Metallic"].default_value = metallic
        if "Alpha" in bsdf.inputs:
            bsdf.inputs["Alpha"].default_value = alpha
    if alpha < 1.0:
        mat.blend_method = "BLEND"
        mat.show_transparent_back = True
    return mat


def key_obj(obj, frame, loc=None, rot=None, scale=None, visible=None):
    if loc is not None:
        obj.location = loc
        obj.keyframe_insert(data_path="location", frame=frame)
    if rot is not None:
        obj.rotation_euler = rot
        obj.keyframe_insert(data_path="rotation_euler", frame=frame)
    if scale is not None:
        obj.scale = scale
        obj.keyframe_insert(data_path="scale", frame=frame)
    if visible is not None:
        obj.hide_viewport = not visible
        obj.hide_render = not visible
        obj.keyframe_insert(data_path="hide_viewport", frame=frame)
        obj.keyframe_insert(data_path="hide_render", frame=frame)


def set_interpolation(kind="BEZIER"):
    for obj in bpy.data.objects:
        ad = obj.animation_data
        if not ad or not ad.action:
            continue
        action = ad.action
        fcurves = getattr(action, "fcurves", None)
        if fcurves is None and hasattr(action, "layers"):
            found = []
            try:
                for layer in action.layers:
                    for strip in layer.strips:
                        if hasattr(strip, "channelbag") and strip.channelbag:
                            found.extend(strip.channelbag.fcurves)
                        elif hasattr(strip, "fcurves"):
                            found.extend(strip.fcurves)
            except Exception:
                found = []
            fcurves = found
        for fc in fcurves or []:
            for kp in fc.keyframe_points:
                kp.interpolation = kind


def add_cylinder(name, radius, depth, location, mat, vertices=128, rotation=(0, 0, 0)):
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=vertices,
        radius=radius,
        depth=depth,
        location=location,
        rotation=rotation,
    )
    obj = bpy.context.object
    obj.name = name
    obj.data.materials.append(mat)
    try:
        bpy.ops.object.shade_smooth()
    except Exception:
        pass
    return obj


def add_cube(name, location, dimensions, mat, rotation=(0, 0, 0)):
    bpy.ops.mesh.primitive_cube_add(location=location, rotation=rotation)
    obj = bpy.context.object
    obj.name = name
    obj.dimensions = dimensions
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.data.materials.append(mat)
    return obj


def add_text(name, body, location, size, mat, rotation=(math.radians(70), 0, 0)):
    bpy.ops.object.text_add(location=location, rotation=rotation)
    obj = bpy.context.object
    obj.name = name
    obj.data.body = body
    obj.data.align_x = "CENTER"
    obj.data.align_y = "CENTER"
    obj.data.size = size
    obj.data.extrude = 0.004
    obj.data.materials.append(mat)
    return obj

# ============================================================
# Materials
# ============================================================
tape_black = make_mat("matte black waterproof tape", (0.005, 0.006, 0.007, 1), roughness=0.78)
rubber_edge = make_mat("slightly glossy rubberized tape edge", (0.0, 0.018, 0.022, 1), roughness=0.42)
adhesive_blue = make_mat("blue waterproof adhesive underside", (0.02, 0.27, 0.95, 0.72), roughness=0.08, alpha=0.72)
cardboard = make_mat("brown cardboard inner core", (0.55, 0.34, 0.16, 1), roughness=0.72)
label_yellow = make_mat("yellow waterproof tape roll label", (1.0, 0.82, 0.05, 1), roughness=0.55)
ink = make_mat("black printed text", (0, 0, 0, 1), roughness=0.55)
indicator_orange = make_mat("orange rotation indicator marks", (1.0, 0.28, 0.02, 1), roughness=0.38)
indicator_white = make_mat("white rotation indicator highlight", (1.0, 1.0, 0.92, 1), roughness=0.32)
water = make_mat("small blue water beads", (0.28, 0.62, 1.0, 0.45), roughness=0.02, alpha=0.45)
ground_mat = make_mat("light studio ground", (0.86, 0.86, 0.84, 1), roughness=0.65)

# ============================================================
# Static model: waterproof tape roll
# ============================================================
# Roll axis runs along Y, so the roll face is vertical in the X/Z plane.
# The strip extends along +X from the right side of the roll, like a real roll unwinding.
roll_rot = (math.radians(90), 0, 0)
roll_center = (-1.05, 0.0, 0.0)

outer_roll = add_cylinder("rotating waterproof tape roll", 1.0, 0.54, roll_center, tape_black, rotation=roll_rot)
core = add_cylinder("visible cardboard core", 0.45, 0.58, roll_center, cardboard, rotation=roll_rot)
hole = add_cylinder("dark center opening", 0.31, 0.60, roll_center, make_mat("dark hollow core", (0.015, 0.01, 0.006, 1), roughness=0.9), rotation=roll_rot)
left_rim = add_cylinder("front roll rim", 1.03, 0.045, (roll_center[0], -0.30, 0), rubber_edge, rotation=roll_rot)
right_rim = add_cylinder("back roll rim", 1.03, 0.045, (roll_center[0], 0.30, 0), rubber_edge, rotation=roll_rot)
left_label = add_cylinder("front yellow label disk", 0.72, 0.012, (roll_center[0], -0.326, 0), label_yellow, rotation=roll_rot)
right_label = add_cylinder("back yellow label disk", 0.72, 0.012, (roll_center[0], 0.326, 0), label_yellow, rotation=roll_rot)
label_text = add_text("label text", "WATERPROOF\nTAPE", (roll_center[0], -0.34, 0), 0.18, ink, rotation=(math.radians(90), 0, 0))

# Thin circular layer marks on face of roll.
layer_marks = []
for i, radius in enumerate([0.92, 0.82, 0.72, 0.62, 0.52]):
    layer = add_cylinder(
        f"visible tape winding layer {i+1}",
        radius,
        0.008,
        (roll_center[0], -0.348 - i * 0.006, 0),
        rubber_edge,
        vertices=128,
        rotation=roll_rot,
    )
    layer_marks.append(layer)

# Rotation indicators on the front face of the roll.
# These asymmetrical spokes and the edge dot make the roll spin readable.
rotation_indicators = []
for idx, angle in enumerate([0, math.radians(72), math.radians(155), math.radians(250)]):
    radius_mid = 0.63
    x = roll_center[0] + math.cos(angle) * radius_mid
    z_pos = roll_center[2] + math.sin(angle) * radius_mid
    spoke = add_cube(
        f"orange rotating face spoke {idx+1}",
        (x, -0.365, z_pos),
        (0.58, 0.018, 0.045),
        indicator_orange,
        rotation=(0, -angle, 0),
    )
    rotation_indicators.append(spoke)

# A white dot near the outer edge gives a clear single reference point.
bpy.ops.mesh.primitive_uv_sphere_add(
    segments=24,
    ring_count=12,
    radius=0.07,
    location=(roll_center[0] + 0.78, -0.382, roll_center[2] + 0.22),
)
edge_dot = bpy.context.object
edge_dot.name = "white moving rotation indicator dot"
edge_dot.data.materials.append(indicator_white)
try:
    bpy.ops.object.shade_smooth()
except Exception:
    pass
rotation_indicators.append(edge_dot)

# A small arrow-like marker shows direction during unwind/rewind.
direction_tab = add_cube(
    "orange direction tab on roll face",
    (roll_center[0] + 0.18, -0.392, roll_center[2] + 0.83),
    (0.34, 0.02, 0.075),
    indicator_orange,
    rotation=(0, math.radians(-22), 0),
)
rotation_indicators.append(direction_tab)

roll_parts = [outer_roll, core, hole, left_rim, right_rim, left_label, right_label, label_text] + layer_marks + rotation_indicators


# Parent all circular roll parts to one empty so the roll spins around
# the true cylinder axis. Because the cylinder was rotated so its axis
# runs along global Y, the animation rotates this empty around Y.
# That places the roll beside the straight strip, which extends along +X.
roll_axis_empty = bpy.data.objects.new("centered roll rotation controller", None)
bpy.context.collection.objects.link(roll_axis_empty)
roll_axis_empty.empty_display_type = "ARROWS"
roll_axis_empty.empty_display_size = 0.7
roll_axis_empty.location = roll_center

# Parent each roll piece with local coordinates measured from roll_center.
# This removes the old offset-pivot behavior where the roll appeared to
# rotate around a point near the bottom/edge instead of its true center.
roll_center_vec = Vector(roll_center)
for part in roll_parts:
    world_loc = part.location.copy()
    world_rot = part.rotation_euler.copy()
    part.parent = roll_axis_empty
    part.matrix_parent_inverse.identity()
    part.location = world_loc - roll_center_vec
    part.rotation_euler = world_rot

# ============================================================
# Unwound tape built from short segments
# ============================================================
# Segments progressively appear to make the strip unwind; they disappear in reverse to rewind.
segments = []
adhesive_segments = []
num_segments = 28
segment_len = 0.135
segment_gap = 0.006
strip_width = 0.48
strip_thick = 0.035
start_x = -0.18
z = 0.04

for i in range(num_segments):
    x = start_x + i * (segment_len + segment_gap)
    top = add_cube(
        f"unwound black tape segment {i+1:02d}",
        (x, 0, z),
        (segment_len, strip_width, strip_thick),
        tape_black,
    )
    underside = add_cube(
        f"blue adhesive underside segment {i+1:02d}",
        (x, 0, z - 0.035),
        (segment_len * 0.96, strip_width * 0.88, 0.012),
        adhesive_blue,
    )
    segments.append(top)
    adhesive_segments.append(underside)

# Curved leader tab at the roll edge.
leader_tab = add_cube("loose tape end leader tab", (start_x - 0.12, 0, z + 0.02), (0.22, strip_width, strip_thick), tape_black, rotation=(0, math.radians(-10), 0))
leader_adh = add_cube("leader tab adhesive underside", (start_x - 0.12, 0, z - 0.02), (0.20, strip_width * 0.88, 0.012), adhesive_blue, rotation=(0, math.radians(-10), 0))

# Moving free end handle at the far end of the tape.
free_end = add_cube("moving free tape end", (start_x, 0, z + 0.015), (0.19, strip_width * 1.04, strip_thick * 1.18), rubber_edge)

# A faint path guide line under the tape, useful for seeing the length change.
path_shadow = add_cube("soft contact shadow under unwound tape", (1.75, 0, -0.005), (4.05, 0.54, 0.01), make_mat("soft tape shadow", (0.12, 0.12, 0.12, 0.18), roughness=0.9, alpha=0.18))

# Water beads on the top surface show that it is waterproof but remain secondary.
beads = []
for i in range(14):
    x = 0.35 + i * 0.22
    y = -0.16 + 0.08 * (i % 5)
    bpy.ops.mesh.primitive_uv_sphere_add(segments=20, ring_count=10, radius=0.035 + 0.006 * (i % 3), location=(x, y, z + 0.07))
    bead = bpy.context.object
    bead.name = f"water bead on waterproof tape {i+1}"
    bead.data.materials.append(water)
    try:
        bpy.ops.object.shade_smooth()
    except Exception:
        pass
    beads.append(bead)

# Stage label.
stage_text = add_text(
    "animation label",
    "Waterproof tape unwinds, then rewinds",
    (0.75, -2.05, 1.35),
    0.14,
    ink,
)

# Ground, lights, camera.
bpy.ops.mesh.primitive_plane_add(size=7.0, location=(0.8, 0, -1.06))
ground = bpy.context.object
ground.name = "light studio ground plane"
ground.data.materials.append(ground_mat)

bpy.ops.object.light_add(type="AREA", location=(0, -4.2, 5.0))
key = bpy.context.object
key.name = "large softbox light"
key.data.energy = 720
key.data.size = 5.2

bpy.ops.object.light_add(type="POINT", location=(3.5, 2.2, 3.0))
rim = bpy.context.object
rim.name = "small rim highlight"
rim.data.energy = 95

bpy.ops.object.camera_add(location=(4.2, -5.5, 2.7), rotation=(math.radians(62), 0, math.radians(39)))
cam = bpy.context.object
cam.name = "camera tape unwind rewind view"
cam.data.lens = 46
scene.camera = cam

# ============================================================
# Animation: unwind, hold, rewind
# ============================================================
# Frames:
# 1    compact roll, only leader visible
# 25   leader begins pulling outward
# 110  tape fully unwound
# 140  fully extended hold/waterproof display
# 205  rewound back into compact roll
# 220  final compact roll

# Keep roll geometry fixed in local coordinates relative to the centered
# controller; only the controller rotates.
for obj in roll_parts:
    key_obj(obj, 1, loc=obj.location.copy(), rot=obj.rotation_euler.copy(), scale=(1, 1, 1), visible=True)
    key_obj(obj, 220, loc=obj.location.copy(), rot=obj.rotation_euler.copy(), scale=(1, 1, 1), visible=True)

key_obj(roll_axis_empty, 1, loc=roll_center, rot=(0, 0, 0), scale=(1, 1, 1), visible=True)
key_obj(roll_axis_empty, 25, loc=roll_center, rot=(0, math.radians(-90), 0), scale=(1, 1, 1), visible=True)
key_obj(roll_axis_empty, 110, loc=roll_center, rot=(0, math.radians(900), 0), scale=(1, 1, 1), visible=True)
key_obj(roll_axis_empty, 140, loc=roll_center, rot=(0, math.radians(940), 0), scale=(1, 1, 1), visible=True)
key_obj(roll_axis_empty, 205, loc=roll_center, rot=(0, math.radians(-40), 0), scale=(1, 1, 1), visible=True)
key_obj(roll_axis_empty, 220, loc=roll_center, rot=(0, 0, 0), scale=(1, 1, 1), visible=True)

# Leader and free end move out and return.
key_obj(leader_tab, 1, loc=(start_x - 0.12, 0, z + 0.02), rot=(0, math.radians(-18), 0), scale=(0.75, 1, 1), visible=True)
key_obj(leader_adh, 1, loc=(start_x - 0.12, 0, z - 0.02), rot=(0, math.radians(-18), 0), scale=(0.75, 1, 1), visible=True)
key_obj(free_end, 1, loc=(start_x - 0.12, 0, z + 0.02), scale=(0.75, 1, 1), visible=True)

key_obj(leader_tab, 25, loc=(start_x + 0.1, 0, z + 0.02), rot=(0, 0, 0), scale=(1, 1, 1), visible=True)
key_obj(leader_adh, 25, loc=(start_x + 0.1, 0, z - 0.02), rot=(0, 0, 0), scale=(1, 1, 1), visible=True)
key_obj(free_end, 25, loc=(start_x + 0.1, 0, z + 0.02), scale=(1, 1, 1), visible=True)

far_x = start_x + (num_segments - 1) * (segment_len + segment_gap) + 0.20
key_obj(leader_tab, 110, loc=(start_x + 0.1, 0, z + 0.02), rot=(0, 0, 0), scale=(1, 1, 1), visible=True)
key_obj(leader_adh, 110, loc=(start_x + 0.1, 0, z - 0.02), rot=(0, 0, 0), scale=(1, 1, 1), visible=True)
key_obj(free_end, 110, loc=(far_x, 0, z + 0.02), scale=(1, 1, 1), visible=True)

key_obj(free_end, 140, loc=(far_x + 0.10, 0, z + 0.04), scale=(1.05, 1.03, 1), visible=True)

key_obj(free_end, 205, loc=(start_x - 0.11, 0, z + 0.02), scale=(0.75, 1, 1), visible=True)
key_obj(free_end, 220, loc=(start_x - 0.12, 0, z + 0.02), scale=(0.75, 1, 1), visible=True)
key_obj(leader_tab, 205, loc=(start_x - 0.12, 0, z + 0.02), rot=(0, math.radians(-18), 0), scale=(0.75, 1, 1), visible=True)
key_obj(leader_adh, 205, loc=(start_x - 0.12, 0, z - 0.02), rot=(0, math.radians(-18), 0), scale=(0.75, 1, 1), visible=True)
key_obj(leader_tab, 220, loc=(start_x - 0.12, 0, z + 0.02), rot=(0, math.radians(-18), 0), scale=(0.75, 1, 1), visible=True)
key_obj(leader_adh, 220, loc=(start_x - 0.12, 0, z - 0.02), rot=(0, math.radians(-18), 0), scale=(0.75, 1, 1), visible=True)

# Segment visibility: each segment appears during unwind and disappears during rewind.
for i, (top, underside) in enumerate(zip(segments, adhesive_segments)):
    appear = 25 + int((i + 1) * (85 / num_segments))
    disappear = 205 - int((i + 1) * (65 / num_segments))

    for obj in (top, underside):
        original_loc = obj.location.copy()
        key_obj(obj, 1, loc=original_loc, scale=(0.02, 1, 1), visible=False)
        key_obj(obj, max(1, appear - 2), loc=original_loc, scale=(0.02, 1, 1), visible=False)
        key_obj(obj, appear, loc=original_loc, scale=(1, 1, 1), visible=True)
        key_obj(obj, 140, loc=original_loc, scale=(1, 1, 1), visible=True)
        key_obj(obj, max(141, disappear - 2), loc=original_loc, scale=(1, 1, 1), visible=True)
        key_obj(obj, disappear, loc=original_loc, scale=(0.02, 1, 1), visible=False)
        key_obj(obj, 220, loc=original_loc, scale=(0.02, 1, 1), visible=False)

# Path shadow is visible only while tape is extended.
key_obj(path_shadow, 1, visible=False, scale=(0.01, 1, 1))
key_obj(path_shadow, 35, visible=True, scale=(0.05, 1, 1))
key_obj(path_shadow, 110, visible=True, scale=(1, 1, 1))
key_obj(path_shadow, 140, visible=True, scale=(1, 1, 1))
key_obj(path_shadow, 205, visible=False, scale=(0.01, 1, 1))
key_obj(path_shadow, 220, visible=False, scale=(0.01, 1, 1))

# Water beads appear on the fully unwound tape, then vanish as tape rewinds.
for i, bead in enumerate(beads):
    key_obj(bead, 1, visible=False, scale=(0.01, 0.01, 0.01))
    key_obj(bead, 98 + i % 6, visible=False, scale=(0.01, 0.01, 0.01))
    key_obj(bead, 118 + i % 6, visible=True, scale=(1, 1, 1))
    key_obj(bead, 140, loc=(bead.location.x, bead.location.y + 0.03 * math.sin(i), bead.location.z + 0.02), visible=True, scale=(1.05, 1.05, 1.05))
    key_obj(bead, 170, visible=False, scale=(0.01, 0.01, 0.01))
    key_obj(bead, 220, visible=False, scale=(0.01, 0.01, 0.01))

# Camera movement: slight dolly while the tape extends.
key_obj(cam, 1, loc=(4.2, -5.5, 2.7), rot=(math.radians(62), 0, math.radians(39)))
key_obj(cam, 110, loc=(4.55, -5.85, 2.85), rot=(math.radians(61), 0, math.radians(41)))
key_obj(cam, 220, loc=(4.2, -5.5, 2.7), rot=(math.radians(62), 0, math.radians(39)))

# Text remains stable.
key_obj(stage_text, 1, visible=True)
key_obj(stage_text, 220, visible=True)

set_interpolation("BEZIER")

# Timeline markers.
markers = [
    (1, "compact roll"),
    (25, "leader pulls outward"),
    (70, "unwinding"),
    (110, "fully unwound"),
    (140, "hold extended waterproof strip"),
    (175, "rewinding"),
    (205, "rewound"),
    (220, "final compact roll"),
]
for frame, name in markers:
    scene.timeline_markers.new(name, frame=frame)

for obj in bpy.context.scene.objects:
    obj.select_set(False)

print("Initialized waterproof tape scene with centered roll-pivot rotation and visible rotation indicator marks.")
