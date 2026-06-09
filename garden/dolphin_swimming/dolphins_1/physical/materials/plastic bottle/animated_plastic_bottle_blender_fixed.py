# animated_plastic_bottle_blender.py
# Run in Blender with:
# blender --python animated_plastic_bottle_blender.py
#
# Scene: a transparent plastic bottle is initialized, taken apart, reassembled,
# moved around, spun, wobbled, and finally displayed as a finished bottle.

import bpy
import math
from mathutils import Vector

# ------------------------------------------------------------
# Scene reset
# ------------------------------------------------------------
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete()

scene = bpy.context.scene
scene.frame_start = 1
scene.frame_end = 360
scene.frame_set(1)

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def make_material(name, color, roughness=0.35, alpha=1.0, transmission=0.0, metallic=0.0):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        if "Base Color" in bsdf.inputs:
            bsdf.inputs["Base Color"].default_value = color
        if "Alpha" in bsdf.inputs:
            bsdf.inputs["Alpha"].default_value = alpha
        if "Roughness" in bsdf.inputs:
            bsdf.inputs["Roughness"].default_value = roughness
        if "Metallic" in bsdf.inputs:
            bsdf.inputs["Metallic"].default_value = metallic
        if "Transmission Weight" in bsdf.inputs:
            bsdf.inputs["Transmission Weight"].default_value = transmission
        elif "Transmission" in bsdf.inputs:
            bsdf.inputs["Transmission"].default_value = transmission
        if "IOR" in bsdf.inputs:
            bsdf.inputs["IOR"].default_value = 1.46
    mat.blend_method = "BLEND"
    mat.use_screen_refraction = True
    mat.show_transparent_back = True
    return mat


def key(obj, frame, loc=None, rot=None, scale=None):
    scene.frame_set(frame)
    if loc is not None:
        obj.location = loc
        obj.keyframe_insert(data_path="location", frame=frame)
    if rot is not None:
        obj.rotation_euler = rot
        obj.keyframe_insert(data_path="rotation_euler", frame=frame)
    if scale is not None:
        obj.scale = scale
        obj.keyframe_insert(data_path="scale", frame=frame)


def set_visibility(obj, frame, visible):
    scene.frame_set(frame)
    obj.hide_viewport = not visible
    obj.hide_render = not visible
    obj.keyframe_insert(data_path="hide_viewport", frame=frame)
    obj.keyframe_insert(data_path="hide_render", frame=frame)


def _iter_action_fcurves(action):
    """Return f-curves from both older and newer Blender Action APIs."""
    if action is None:
        return []

    # Blender 2.x/3.x/early 4.x: Action directly exposes .fcurves.
    direct_fcurves = getattr(action, "fcurves", None)
    if direct_fcurves is not None:
        return list(direct_fcurves)

    # Newer Blender action system: f-curves may live inside layer/strip channelbags.
    found = []
    for layer in getattr(action, "layers", []):
        for strip in getattr(layer, "strips", []):
            channelbag = getattr(strip, "channelbag", None)
            if channelbag is not None and getattr(channelbag, "fcurves", None) is not None:
                found.extend(list(channelbag.fcurves))

            channelbags = getattr(strip, "channelbags", None)
            if channelbags is not None:
                for bag in channelbags:
                    if getattr(bag, "fcurves", None) is not None:
                        found.extend(list(bag.fcurves))
    return found


def set_interpolation(obj, interpolation="BEZIER"):
    if not obj.animation_data or not obj.animation_data.action:
        return

    for fc in _iter_action_fcurves(obj.animation_data.action):
        for kp in fc.keyframe_points:
            # BEZIER is supported broadly across Blender versions.
            kp.interpolation = interpolation


def add_cylinder(name, radius, depth, loc, material, vertices=96):
    bpy.ops.mesh.primitive_cylinder_add(vertices=vertices, radius=radius, depth=depth, location=loc)
    obj = bpy.context.object
    obj.name = name
    obj.data.materials.append(material)
    bpy.ops.object.shade_smooth()
    return obj

# ------------------------------------------------------------
# Materials
# ------------------------------------------------------------
clear_plastic = make_material(
    "clear slightly blue transparent PET plastic",
    (0.78, 0.92, 1.0, 0.28),
    roughness=0.08,
    alpha=0.28,
    transmission=0.65,
)
cap_plastic = make_material(
    "opaque blue cap plastic",
    (0.02, 0.18, 0.85, 1.0),
    roughness=0.42,
    alpha=1.0,
)
label_mat = make_material(
    "white paper label",
    (0.95, 0.95, 0.90, 1.0),
    roughness=0.75,
    alpha=1.0,
)
water_mat = make_material(
    "transparent water inside bottle",
    (0.35, 0.65, 1.0, 0.35),
    roughness=0.02,
    alpha=0.35,
    transmission=0.75,
)
dark_groove_mat = make_material(
    "shadowed bottle groove material",
    (0.35, 0.55, 0.65, 0.35),
    roughness=0.12,
    alpha=0.35,
    transmission=0.4,
)
text_mat = make_material(
    "blue printed label ink",
    (0.0, 0.15, 0.65, 1.0),
    roughness=0.55,
    alpha=1.0,
)
ground_mat = make_material(
    "matte light gray studio surface",
    (0.82, 0.82, 0.80, 1.0),
    roughness=0.65,
    alpha=1.0,
)

# ------------------------------------------------------------
# Parent empty for whole bottle movement
# ------------------------------------------------------------
bottle_rig = bpy.data.objects.new("Bottle whole-object animation rig", None)
bpy.context.collection.objects.link(bottle_rig)

# ------------------------------------------------------------
# Bottle body profile mesh
# ------------------------------------------------------------
profile = [
    (0.30, -1.80),
    (0.52, -1.72),
    (0.62, -1.55),
    (0.66, -1.20),
    (0.68, -0.60),
    (0.67,  0.05),
    (0.63,  0.55),
    (0.52,  0.85),
    (0.35,  1.05),
    (0.24,  1.25),
    (0.21,  1.55),
    (0.21,  1.82),
]
verts = []
faces = []
segments = 128
for radius, z in profile:
    for s in range(segments):
        angle = 2 * math.pi * s / segments
        verts.append((radius * math.cos(angle), radius * math.sin(angle), z))
for i in range(len(profile) - 1):
    for s in range(segments):
        a = i * segments + s
        b = i * segments + (s + 1) % segments
        c = (i + 1) * segments + (s + 1) % segments
        d = (i + 1) * segments + s
        faces.append((a, b, c, d))
mesh = bpy.data.meshes.new("Plastic_Bottle_Body_Mesh")
mesh.from_pydata(verts, [], faces)
mesh.update()
body = bpy.data.objects.new("transparent plastic bottle body", mesh)
bpy.context.collection.objects.link(body)
body.data.materials.append(clear_plastic)
body.parent = bottle_rig
bpy.context.view_layer.objects.active = body
body.select_set(True)
bpy.ops.object.shade_smooth()
solidify = body.modifiers.new("thin bottle wall", "SOLIDIFY")
solidify.thickness = 0.035
solidify.offset = 0
solidify.use_quality_normals = True
subsurf = body.modifiers.new("smooth rounded plastic", "SUBSURF")
subsurf.levels = 1
subsurf.render_levels = 1
body.select_set(False)

# ------------------------------------------------------------
# Parts
# ------------------------------------------------------------
base_outer = add_cylinder("raised circular base rim", 0.58, 0.045, (0, 0, -1.78), clear_plastic, 128)
base_inner = add_cylinder("inset bottom punt circle", 0.32, 0.035, (0, 0, -1.745), dark_groove_mat, 128)

all_parts = [body, base_outer, base_inner]

grooves = []
for idx, z in enumerate([-0.95, -0.72, -0.49, -0.26]):
    bpy.ops.mesh.primitive_torus_add(
        major_radius=0.675,
        minor_radius=0.018,
        major_segments=128,
        minor_segments=12,
        location=(0, 0, z),
    )
    groove = bpy.context.object
    groove.name = f"recessed grip groove {idx + 1}"
    groove.scale.z = 0.18
    groove.data.materials.append(dark_groove_mat)
    groove.parent = bottle_rig
    bpy.ops.object.shade_smooth()
    grooves.append(groove)
    all_parts.append(groove)

label = add_cylinder("paper label wrapped around bottle", 0.692, 0.50, (0, 0, 0.03), label_mat, 128)
water = add_cylinder("water volume inside transparent bottle", 0.58, 1.65, (0, 0, -0.70), water_mat, 128)
neck_ring_1 = add_cylinder("clear neck support ring", 0.255, 0.07, (0, 0, 1.35), clear_plastic, 128)
neck_ring_2 = add_cylinder("blue tamper ring below cap", 0.265, 0.08, (0, 0, 1.55), cap_plastic, 128)
cap = add_cylinder("blue screw cap", 0.30, 0.38, (0, 0, 1.82), cap_plastic, 128)

for obj in [base_outer, base_inner, label, water, neck_ring_1, neck_ring_2, cap]:
    obj.parent = bottle_rig
    all_parts.append(obj)

cap_ridges = []
for s in range(32):
    angle = 2 * math.pi * s / 32
    x = 0.307 * math.cos(angle)
    y = 0.307 * math.sin(angle)
    bpy.ops.mesh.primitive_cube_add(location=(x, y, 1.82))
    ridge = bpy.context.object
    ridge.name = f"vertical cap grip ridge {s + 1:02d}"
    ridge.dimensions = (0.025, 0.065, 0.32)
    ridge.rotation_euler[2] = angle
    ridge.data.materials.append(cap_plastic)
    ridge.parent = cap
    cap_ridges.append(ridge)

# Label text
bpy.ops.object.text_add(location=(0, -0.705, 0.05), rotation=(math.radians(90), 0, 0))
text = bpy.context.object
text.name = "front label text WATER"
text.data.body = "WATER"
text.data.align_x = "CENTER"
text.data.align_y = "CENTER"
text.data.size = 0.22
text.data.extrude = 0.003
text.data.materials.append(text_mat)
text.parent = bottle_rig
all_parts.append(text)

# Small arrow markers showing motion path / playfulness
arrow_mat = make_material("soft yellow motion marker", (1.0, 0.85, 0.25, 0.7), roughness=0.4, alpha=0.7)
for i, x in enumerate([-1.5, -0.75, 0.0, 0.75, 1.5]):
    bpy.ops.mesh.primitive_cone_add(vertices=3, radius1=0.10, radius2=0, depth=0.22, location=(x, -1.8, -1.2), rotation=(math.radians(90), 0, math.radians(30)))
    arrow = bpy.context.object
    arrow.name = f"floor motion arrow {i + 1}"
    arrow.data.materials.append(arrow_mat)

# ------------------------------------------------------------
# Lighting, camera, floor
# ------------------------------------------------------------
bpy.ops.object.light_add(type="AREA", location=(0, -4, 5))
key_light = bpy.context.object
key_light.name = "large softbox reflection light"
key_light.data.energy = 720
key_light.data.size = 5

bpy.ops.object.light_add(type="POINT", location=(3, 2, 3))
rim_light = bpy.context.object
rim_light.name = "small rim highlight light"
rim_light.data.energy = 100

bpy.ops.mesh.primitive_plane_add(size=7, location=(0, 0, -1.83))
ground = bpy.context.object
ground.name = "studio ground plane"
ground.data.materials.append(ground_mat)

bpy.ops.object.camera_add(location=(3.6, -5.4, 2.4), rotation=(math.radians(66), 0, math.radians(34)))
camera = bpy.context.object
camera.name = "camera plastic bottle animation view"
camera.data.lens = 48
scene.camera = camera

# ------------------------------------------------------------
# Animation design
# ------------------------------------------------------------
# Frames:
# 1-45: complete bottle shown
# 45-110: bottle taken apart outward
# 110-165: parts hover/spin separately
# 165-220: bottle reassembles
# 220-285: whole bottle moves, spins, wobbles
# 285-360: final bottle settles and displays

# Initial complete bottle
for obj in all_parts:
    key(obj, 1, loc=obj.location.copy(), rot=obj.rotation_euler.copy(), scale=obj.scale.copy())
    key(obj, 45, loc=obj.location.copy(), rot=obj.rotation_euler.copy(), scale=obj.scale.copy())

# Take apart motion
explode_map = {
    body.name: Vector((0.0, 0.0, 0.0)),
    base_outer.name: Vector((-1.35, 0.25, -0.05)),
    base_inner.name: Vector((-1.15, -0.35, 0.15)),
    label.name: Vector((1.35, -0.25, 0.25)),
    water.name: Vector((0.0, 1.15, 0.15)),
    neck_ring_1.name: Vector((-0.9, 0.75, 0.45)),
    neck_ring_2.name: Vector((0.95, 0.75, 0.55)),
    cap.name: Vector((0.0, 0.0, 1.05)),
    text.name: Vector((1.35, -0.35, 0.25)),
}
for i, groove in enumerate(grooves):
    explode_map[groove.name] = Vector((-1.1 + i * 0.7, -1.0, 0.25 + i * 0.08))

original = {obj.name: (obj.location.copy(), obj.rotation_euler.copy(), obj.scale.copy()) for obj in all_parts}

for obj in all_parts:
    loc0, rot0, scale0 = original[obj.name]
    offset = explode_map.get(obj.name, Vector((0, 0, 0)))
    apart_loc = loc0 + offset
    apart_rot = (rot0.x + 0.15, rot0.y + 0.4, rot0.z + 0.6)
    if obj == body:
        apart_rot = (0.0, 0.0, 0.0)
    key(obj, 110, loc=apart_loc, rot=apart_rot, scale=scale0)
    key(obj, 165, loc=apart_loc + Vector((0.0, 0.0, 0.15 * math.sin(len(obj.name)))), rot=(apart_rot[0], apart_rot[1], apart_rot[2] + math.radians(120)), scale=scale0)

# Reassemble
for obj in all_parts:
    loc0, rot0, scale0 = original[obj.name]
    key(obj, 220, loc=loc0, rot=rot0, scale=scale0)

# Parent rig movement after reassembly
key(bottle_rig, 1, loc=(0, 0, 0), rot=(0, 0, 0), scale=(1, 1, 1))
key(bottle_rig, 220, loc=(0, 0, 0), rot=(0, 0, 0), scale=(1, 1, 1))
key(bottle_rig, 245, loc=(-1.25, -0.15, 0.0), rot=(0.12, 0.0, math.radians(-14)), scale=(1, 1, 1))
key(bottle_rig, 270, loc=(1.15, 0.15, 0.08), rot=(-0.10, 0.20, math.radians(375)), scale=(1, 1, 1))
key(bottle_rig, 295, loc=(0.0, 0.0, 0.0), rot=(0.16, -0.08, math.radians(720)), scale=(1, 1, 1))
key(bottle_rig, 320, loc=(0.0, 0.0, 0.0), rot=(-0.06, 0.04, math.radians(735)), scale=(1, 1, 1))
key(bottle_rig, 360, loc=(0.0, 0.0, 0.0), rot=(0.0, 0.0, math.radians(720)), scale=(1, 1, 1))

# Cap gets an extra unscrewing/re-screwing motion
cap_rot0 = original[cap.name][1]
key(cap, 45, loc=original[cap.name][0], rot=cap_rot0, scale=original[cap.name][2])
key(cap, 80, loc=original[cap.name][0] + Vector((0, 0, 0.55)), rot=(cap_rot0.x, cap_rot0.y, cap_rot0.z + math.radians(360)), scale=original[cap.name][2])
key(cap, 110, loc=original[cap.name][0] + explode_map[cap.name], rot=(cap_rot0.x, cap_rot0.y, cap_rot0.z + math.radians(720)), scale=original[cap.name][2])
key(cap, 190, loc=original[cap.name][0] + Vector((0, 0, 0.55)), rot=(cap_rot0.x, cap_rot0.y, cap_rot0.z + math.radians(1080)), scale=original[cap.name][2])
key(cap, 220, loc=original[cap.name][0], rot=(cap_rot0.x, cap_rot0.y, cap_rot0.z + math.radians(1440)), scale=original[cap.name][2])

# Water slosh scale wobble during movement
for f, sx, sy, sz in [(220, 1, 1, 1), (240, 1.05, 0.95, 0.98), (260, 0.95, 1.05, 1.02), (285, 1.04, 0.96, 0.99), (320, 1, 1, 1), (360, 1, 1, 1)]:
    key(water, f, scale=(sx, sy, sz))

# Label peels slightly away while disassembled, then returns
key(label, 45, loc=original[label.name][0], rot=original[label.name][1], scale=original[label.name][2])
key(label, 110, loc=original[label.name][0] + explode_map[label.name], rot=(math.radians(8), math.radians(30), math.radians(15)), scale=(1.02, 1.02, 1.0))
key(label, 165, loc=original[label.name][0] + explode_map[label.name] + Vector((0.1, 0.0, 0.1)), rot=(math.radians(-8), math.radians(55), math.radians(45)), scale=(1.08, 1.08, 1.0))
key(label, 220, loc=original[label.name][0], rot=original[label.name][1], scale=original[label.name][2])

# Camera gently changes composition
key(camera, 1, loc=(3.6, -5.4, 2.4), rot=(math.radians(66), 0, math.radians(34)))
key(camera, 120, loc=(4.2, -5.8, 2.7), rot=(math.radians(65), 0, math.radians(37)))
key(camera, 240, loc=(3.3, -5.2, 2.25), rot=(math.radians(66), 0, math.radians(32)))
key(camera, 360, loc=(3.6, -5.4, 2.4), rot=(math.radians(66), 0, math.radians(34)))

# Visibility stays on, but motion arrows fade out late by scale
for obj in bpy.context.scene.objects:
    set_interpolation(obj, "BEZIER")

# ------------------------------------------------------------
# Render settings
# ------------------------------------------------------------
scene.render.engine = "CYCLES"
scene.cycles.samples = 96
scene.render.fps = 24
scene.render.resolution_x = 1280
scene.render.resolution_y = 720
scene.view_settings.view_transform = "Filmic"
scene.view_settings.look = "Medium High Contrast"
scene.world.color = (1.0, 1.0, 1.0)

# Add timeline markers
for frame, name in [
    (1, "Complete bottle"),
    (45, "Begin disassembly"),
    (110, "Bottle taken apart"),
    (165, "Parts hover and spin"),
    (220, "Bottle reassembled"),
    (270, "Bottle moves and spins"),
    (360, "Final display"),
]:
    scene.timeline_markers.new(name, frame=frame)

# Set active object
for obj in bpy.context.scene.objects:
    obj.select_set(False)
body.select_set(True)
bpy.context.view_layer.objects.active = body
scene.frame_set(1)

print("Initialized animated plastic bottle scene: disassembly, hovering parts, reassembly, movement, spin, wobble, and final display.")
