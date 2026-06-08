import bpy
import math
from mathutils import Vector

# ============================================================
# Animated On/Off Switch Blender Script
# Clears the scene, builds a small toggle switch, and animates
# the lever flipping OFF -> ON -> OFF with light and signal flow.
# ============================================================

# ----------------------------
# Scene reset
# ----------------------------
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)

for block in list(bpy.data.meshes):
    if block.users == 0:
        bpy.data.meshes.remove(block)
for block in list(bpy.data.materials):
    if block.users == 0:
        bpy.data.materials.remove(block)
for block in list(bpy.data.curves):
    if block.users == 0:
        bpy.data.curves.remove(block)

scene = bpy.context.scene
scene.frame_start = 1
scene.frame_end = 180
scene.render.fps = 24

# ----------------------------
# Helpers
# ----------------------------
def make_material(name, color=(1, 1, 1, 1), roughness=0.45, metallic=0.0,
                  alpha=1.0, emission_strength=0.0):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = color
        if "Roughness" in bsdf.inputs:
            bsdf.inputs["Roughness"].default_value = roughness
        if "Metallic" in bsdf.inputs:
            bsdf.inputs["Metallic"].default_value = metallic
        if "Alpha" in bsdf.inputs:
            bsdf.inputs["Alpha"].default_value = alpha
        if "Emission Color" in bsdf.inputs:
            bsdf.inputs["Emission Color"].default_value = color
        if "Emission Strength" in bsdf.inputs:
            bsdf.inputs["Emission Strength"].default_value = emission_strength

    if alpha < 1.0:
        mat.blend_method = "BLEND"
        try:
            mat.shadow_method = "HASHED"
        except Exception:
            pass
    return mat


def assign_material(obj, mat):
    if obj.data and hasattr(obj.data, "materials"):
        obj.data.materials.clear()
        obj.data.materials.append(mat)


def set_interpolation(obj, interpolation="BEZIER"):
    ad = getattr(obj, "animation_data", None)
    action = getattr(ad, "action", None) if ad else None
    fcurves = getattr(action, "fcurves", None) if action else None
    if not fcurves:
        return
    for fc in fcurves:
        for kp in fc.keyframe_points:
            kp.interpolation = interpolation


def parent_keep_transform(child, parent):
    child.parent = parent
    child.matrix_parent_inverse = parent.matrix_world.inverted()


def add_cube(name, location=(0, 0, 0), scale=(1, 1, 1), rotation=(0, 0, 0), material=None):
    bpy.ops.mesh.primitive_cube_add(location=location, rotation=rotation)
    obj = bpy.context.object
    obj.name = name
    obj.scale = scale
    if material:
        assign_material(obj, material)
    return obj


def add_cylinder(name, radius=0.1, depth=1.0, location=(0, 0, 0), rotation=(0, 0, 0),
                 vertices=48, material=None):
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=vertices,
        radius=radius,
        depth=depth,
        location=location,
        rotation=rotation,
    )
    obj = bpy.context.object
    obj.name = name
    bpy.ops.object.shade_smooth()
    if material:
        assign_material(obj, material)
    return obj


def add_uv_sphere(name, radius=0.1, location=(0, 0, 0), material=None):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=radius, location=location)
    obj = bpy.context.object
    obj.name = name
    bpy.ops.object.shade_smooth()
    if material:
        assign_material(obj, material)
    return obj


def add_curve(name, points, bevel_depth=0.02, material=None):
    curve = bpy.data.curves.new(name + "_Curve", type="CURVE")
    curve.dimensions = "3D"
    curve.resolution_u = 18
    curve.bevel_depth = bevel_depth
    curve.bevel_resolution = 5

    spline = curve.splines.new("BEZIER")
    spline.bezier_points.add(len(points) - 1)

    for i, point in enumerate(points):
        bp = spline.bezier_points[i]
        bp.co = Vector(point)
        bp.handle_left_type = "AUTO"
        bp.handle_right_type = "AUTO"

    obj = bpy.data.objects.new(name, curve)
    bpy.context.collection.objects.link(obj)
    if material:
        assign_material(obj, material)
    return obj


# ----------------------------
# Materials
# ----------------------------
mat_base = make_material("matte black switch base", (0.05, 0.05, 0.055, 1), roughness=0.75)
mat_plate = make_material("brushed metal plate", (0.72, 0.73, 0.74, 1), roughness=0.23, metallic=0.85)
mat_toggle = make_material("silver toggle lever", (0.82, 0.83, 0.84, 1), roughness=0.19, metallic=0.9)
mat_tip = make_material("white plastic toggle tip", (0.92, 0.92, 0.88, 1), roughness=0.35)
mat_red = make_material("red OFF indicator", (1.0, 0.08, 0.05, 1), roughness=0.25, emission_strength=0.2)
mat_green = make_material("green ON indicator", (0.08, 1.0, 0.20, 1), roughness=0.25, emission_strength=0.2)
mat_green_glow = make_material("active green glow", (0.08, 1.0, 0.20, 1), roughness=0.15, emission_strength=3.0)
mat_red_glow = make_material("active red glow", (1.0, 0.08, 0.05, 1), roughness=0.15, emission_strength=2.2)
mat_wire_red = make_material("red wire", (0.86, 0.05, 0.04, 1), roughness=0.52)
mat_wire_black = make_material("black wire", (0.02, 0.02, 0.025, 1), roughness=0.68)
mat_signal = make_material("blue signal pulse", (0.2, 0.8, 1.0, 1), roughness=0.1, emission_strength=3.0)
mat_floor = make_material("light studio floor", (0.93, 0.93, 0.95, 1), roughness=0.85)

# ----------------------------
# Root rig
# ----------------------------
bpy.ops.object.empty_add(type="PLAIN_AXES", location=(0, 0, 0))
rig = bpy.context.object
rig.name = "On_Off_Switch_Rig"

# ----------------------------
# Switch body
# ----------------------------
base = add_cube("rectangular switch housing", location=(0, 0, 0.22), scale=(1.35, 0.72, 0.22), material=mat_base)
plate = add_cube("top metal face plate", location=(0, 0, 0.48), scale=(1.12, 0.54, 0.045), material=mat_plate)

# Rounded pivot washer
washer = add_cylinder("toggle pivot washer", radius=0.24, depth=0.07, location=(0, 0, 0.56),
                      rotation=(0, 0, 0), vertices=64, material=mat_plate)

# Screw heads
for x in [-0.85, 0.85]:
    screw = add_cylinder("plate screw head", radius=0.075, depth=0.025, location=(x, 0, 0.54),
                         vertices=32, material=mat_plate)
    slot = add_cube("screw slot", location=(x, 0, 0.558), scale=(0.065, 0.010, 0.006), material=mat_base)
    parent_keep_transform(screw, rig)
    parent_keep_transform(slot, rig)

# Indicator lights
off_light = add_uv_sphere("OFF red indicator light", radius=0.075, location=(-0.48, -0.39, 0.60), material=mat_red_glow)
on_light = add_uv_sphere("ON green indicator light", radius=0.075, location=(0.48, -0.39, 0.60), material=mat_green)

# Text labels
def add_text(name, body, location, size, material):
    bpy.ops.object.text_add(location=location, rotation=(math.radians(75), 0, 0))
    obj = bpy.context.object
    obj.name = name
    obj.data.body = body
    obj.data.align_x = "CENTER"
    obj.data.align_y = "CENTER"
    obj.data.size = size
    obj.data.extrude = 0.002
    assign_material(obj, material)
    return obj

off_label = add_text("OFF label", "OFF", (-0.48, -0.58, 0.58), 0.13, mat_red)
on_label = add_text("ON label", "ON", (0.48, -0.58, 0.58), 0.13, mat_green)

# Toggle pivot rig, located at the center of the switch plate
bpy.ops.object.empty_add(type="PLAIN_AXES", location=(0, 0, 0.62))
toggle_pivot = bpy.context.object
toggle_pivot.name = "Toggle_Pivot_Controller"
parent_keep_transform(toggle_pivot, rig)

# Toggle lever built upward from the pivot, parented to the pivot empty
lever = add_cylinder("toggle metal lever", radius=0.055, depth=0.72, location=(0, 0, 0.98),
                     rotation=(0, 0, 0), vertices=32, material=mat_toggle)
tip = add_uv_sphere("rounded toggle tip", radius=0.115, location=(0, 0, 1.37), material=mat_tip)
parent_keep_transform(lever, toggle_pivot)
parent_keep_transform(tip, toggle_pivot)

# Wires and terminals underneath / behind
terminal_left = add_cylinder("left terminal lug", radius=0.07, depth=0.12, location=(-0.45, 0.68, 0.25),
                             rotation=(math.radians(90), 0, 0), material=mat_plate)
terminal_right = add_cylinder("right terminal lug", radius=0.07, depth=0.12, location=(0.45, 0.68, 0.25),
                              rotation=(math.radians(90), 0, 0), material=mat_plate)

wire_in = add_curve("incoming red wire", [(-1.25, 1.05, 0.18), (-0.90, 0.90, 0.22), (-0.45, 0.68, 0.25)],
                    bevel_depth=0.028, material=mat_wire_red)
wire_out = add_curve("outgoing black wire", [(0.45, 0.68, 0.25), (0.90, 0.90, 0.22), (1.25, 1.05, 0.18)],
                     bevel_depth=0.028, material=mat_wire_black)

for obj in [base, plate, washer, off_light, on_light, off_label, on_label, terminal_left, terminal_right, wire_in, wire_out]:
    parent_keep_transform(obj, rig)

# ----------------------------
# Signal flow path and particles
# ----------------------------
signal_path = add_curve(
    "hidden switched circuit signal path",
    [(-1.25, 1.05, 0.18), (-0.45, 0.68, 0.25), (0, 0.0, 0.62), (0.45, 0.68, 0.25), (1.25, 1.05, 0.18)],
    bevel_depth=0.0,
    material=None,
)
signal_path.hide_viewport = True
signal_path.hide_render = True
parent_keep_transform(signal_path, rig)

signal_particles = []
for i, start in enumerate([0.0, 0.22, 0.44, 0.66]):
    p = add_uv_sphere(f"signal pulse particle {i+1}", radius=0.045, location=(-1.25, 1.05, 0.18), material=mat_signal)
    constraint = p.constraints.new(type="FOLLOW_PATH")
    constraint.target = signal_path
    constraint.use_fixed_location = True
    constraint.offset_factor = start
    constraint.keyframe_insert(data_path="offset_factor", frame=70)
    constraint.offset_factor = min(start + 0.85, 1.0)
    constraint.keyframe_insert(data_path="offset_factor", frame=128)
    p.scale = (0.01, 0.01, 0.01)
    p.keyframe_insert(data_path="scale", frame=1)
    p.keyframe_insert(data_path="scale", frame=58)
    p.scale = (1, 1, 1)
    p.keyframe_insert(data_path="scale", frame=72)
    p.keyframe_insert(data_path="scale", frame=132)
    p.scale = (0.01, 0.01, 0.01)
    p.keyframe_insert(data_path="scale", frame=150)
    parent_keep_transform(p, rig)
    signal_particles.append(p)

# ----------------------------
# Floor, camera, lighting
# ----------------------------
bpy.ops.mesh.primitive_plane_add(size=7, location=(0, 0, -0.02))
floor = bpy.context.object
floor.name = "studio floor"
assign_material(floor, mat_floor)

bpy.ops.object.light_add(type="AREA", location=(3.8, -4.4, 5.5))
key = bpy.context.object
key.name = "large soft key light"
key.data.energy = 1600
key.data.size = 4.5

bpy.ops.object.light_add(type="AREA", location=(-3.5, 3.2, 3.4))
fill = bpy.context.object
fill.name = "soft fill light"
fill.data.energy = 650
fill.data.size = 4.0

bpy.ops.object.camera_add(location=(4.3, -5.2, 3.0), rotation=(math.radians(62), 0, math.radians(39)))
camera = bpy.context.object
camera.name = "camera switch view"
camera.data.lens = 55
scene.camera = camera

# ----------------------------
# Animation
# ----------------------------
# Toggle rotates around local Y axis: OFF lean left, ON lean right.
for fr, yrot in [
    (1, math.radians(-25)),
    (42, math.radians(-25)),
    (72, math.radians(25)),
    (124, math.radians(25)),
    (154, math.radians(-25)),
    (180, math.radians(-25)),
]:
    toggle_pivot.rotation_euler = (0, yrot, 0)
    toggle_pivot.keyframe_insert(data_path="rotation_euler", frame=fr)

# Indicator light switching
for fr, off_s, on_s in [
    (1, 1.25, 0.55),
    (58, 1.25, 0.55),
    (74, 0.55, 1.35),
    (124, 0.55, 1.35),
    (154, 1.25, 0.55),
    (180, 1.25, 0.55),
]:
    off_light.scale = (off_s, off_s, off_s)
    on_light.scale = (on_s, on_s, on_s)
    off_light.keyframe_insert(data_path="scale", frame=fr)
    on_light.keyframe_insert(data_path="scale", frame=fr)

# Swap materials for visible OFF/ON state changes at keyframes
# This uses scale pulses for compatibility; material is left stable to avoid driver complexity.

# Small switch body motion
for fr, loc, rot in [
    (1, (0, 0, 0), (0, 0, 0)),
    (72, (0.03, -0.01, 0.03), (math.radians(1), math.radians(-2), math.radians(1))),
    (124, (-0.02, 0.02, 0.02), (math.radians(-1), math.radians(2), math.radians(-1))),
    (180, (0, 0, 0), (0, 0, 0)),
]:
    rig.location = loc
    rig.rotation_euler = rot
    rig.keyframe_insert(data_path="location", frame=fr)
    rig.keyframe_insert(data_path="rotation_euler", frame=fr)

# Camera drift
for fr, loc, rot in [
    (1, (4.3, -5.2, 3.0), (math.radians(62), 0, math.radians(39))),
    (100, (4.0, -4.8, 2.8), (math.radians(61), 0, math.radians(38))),
    (180, (4.2, -5.1, 3.0), (math.radians(62), 0, math.radians(39))),
]:
    camera.location = loc
    camera.rotation_euler = rot
    camera.keyframe_insert(data_path="location", frame=fr)
    camera.keyframe_insert(data_path="rotation_euler", frame=fr)

# Interpolation
for obj in [toggle_pivot, off_light, on_light, rig, camera] + signal_particles:
    set_interpolation(obj, "BEZIER")

# Follow-path constraint interpolation must be handled through constraint animation data.
for particle in signal_particles:
    ad = getattr(particle, "animation_data", None)
    action = getattr(ad, "action", None) if ad else None
    fcurves = getattr(action, "fcurves", None) if action else None
    if fcurves:
        for fc in fcurves:
            for kp in fc.keyframe_points:
                kp.interpolation = "LINEAR"

# ----------------------------
# Render settings
# ----------------------------
scene.render.engine = "CYCLES"
scene.cycles.samples = 96
try:
    scene.view_settings.view_transform = "Filmic"
except Exception:
    pass
scene.world.color = (1.0, 1.0, 1.0)

print("Animated on/off switch scene created: toggle flips OFF to ON to OFF with indicator lights and signal pulses.")
