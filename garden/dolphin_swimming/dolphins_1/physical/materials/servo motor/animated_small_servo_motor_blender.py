import bpy
import math
from mathutils import Vector

# ============================================================
# Animated Small Hobby Servo Motor Blender Script
# Initializes a small servo motor and animates the servo horn
# sweeping back and forth like a working micro servo.
# ============================================================

# ----------------------------
# Scene reset
# ----------------------------
bpy.ops.object.select_all(action='SELECT')
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
def make_material(name, color=(1, 1, 1, 1), roughness=0.45, metallic=0.0, alpha=1.0):
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get('Principled BSDF')
    if bsdf:
        bsdf.inputs['Base Color'].default_value = color
        if 'Roughness' in bsdf.inputs:
            bsdf.inputs['Roughness'].default_value = roughness
        if 'Metallic' in bsdf.inputs:
            bsdf.inputs['Metallic'].default_value = metallic
        if 'Alpha' in bsdf.inputs:
            bsdf.inputs['Alpha'].default_value = alpha
    if alpha < 1.0:
        mat.blend_method = 'BLEND'
        try:
            mat.shadow_method = 'HASHED'
        except Exception:
            pass
    return mat


def assign_material(obj, mat):
    if obj.data and hasattr(obj.data, 'materials'):
        if len(obj.data.materials) == 0:
            obj.data.materials.append(mat)
        else:
            obj.data.materials[0] = mat


def parent_keep_transform(child, parent):
    child.parent = parent
    child.matrix_parent_inverse = parent.matrix_world.inverted()


def set_bezier_interpolation(obj):
    ad = getattr(obj, 'animation_data', None)
    if ad and ad.action:
        fcurves = getattr(ad.action, 'fcurves', None)
        if fcurves:
            for fc in fcurves:
                for kp in fc.keyframe_points:
                    kp.interpolation = 'BEZIER'


def add_cube(name, location, scale, material=None, rotation=(0, 0, 0)):
    bpy.ops.mesh.primitive_cube_add(location=location, rotation=rotation)
    obj = bpy.context.object
    obj.name = name
    obj.scale = scale
    bpy.ops.object.shade_smooth()
    if material:
        assign_material(obj, material)
    return obj


def add_cylinder(name, radius, depth, location, rotation=(0, 0, 0), material=None, vertices=64):
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


def add_curve(name, points, bevel_depth, material=None):
    curve = bpy.data.curves.new(name + '_Curve', type='CURVE')
    curve.dimensions = '3D'
    curve.resolution_u = 16
    curve.bevel_depth = bevel_depth
    curve.bevel_resolution = 4
    spline = curve.splines.new('POLY')
    spline.points.add(len(points) - 1)
    for p, co in zip(spline.points, points):
        p.co = (co[0], co[1], co[2], 1.0)
    obj = bpy.data.objects.new(name, curve)
    bpy.context.collection.objects.link(obj)
    if material:
        assign_material(obj, material)
    return obj

# ----------------------------
# Materials
# ----------------------------
mat_blue = make_material('Blue Plastic Servo Case', (0.05, 0.18, 0.80, 1.0), roughness=0.45)
mat_dark = make_material('Black Plastic Details', (0.02, 0.02, 0.025, 1.0), roughness=0.65)
mat_white = make_material('White Nylon Servo Horn', (0.92, 0.92, 0.88, 1.0), roughness=0.38)
mat_metal = make_material('Metal Screws and Shaft', (0.72, 0.72, 0.70, 1.0), roughness=0.22, metallic=0.85)
mat_red = make_material('Red Wire', (0.9, 0.02, 0.02, 1.0), roughness=0.45)
mat_brown = make_material('Brown Wire', (0.28, 0.10, 0.02, 1.0), roughness=0.5)
mat_yellow = make_material('Signal Yellow Wire', (1.0, 0.78, 0.05, 1.0), roughness=0.45)
mat_floor = make_material('Light Studio Floor', (0.94, 0.94, 0.96, 1.0), roughness=0.85)
mat_green = make_material('Angle Indicator Arc', (0.0, 0.75, 0.35, 1.0), roughness=0.35)

# ----------------------------
# Root and horn pivot
# ----------------------------
bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0, 0, 0))
servo_root = bpy.context.object
servo_root.name = 'Servo_Motor_Root'

# The horn rotates around this empty, centered on the servo shaft.
bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0, 0, 0.72))
horn_pivot = bpy.context.object
horn_pivot.name = 'Servo_Horn_Rotation_Pivot'
parent_keep_transform(horn_pivot, servo_root)

# ----------------------------
# Servo case
# ----------------------------
case = add_cube('Small Rectangular Servo Case', (0, 0, 0.22), (0.62, 0.38, 0.42), mat_blue)
case.modifiers.new('Slightly Rounded Case', 'BEVEL').width = 0.045
case.modifiers.new('Smooth Case', 'WEIGHTED_NORMAL')
parent_keep_transform(case, servo_root)

# Top cap / gearbox housing
gearbox = add_cube('Raised Gearbox Housing', (0.03, 0, 0.63), (0.38, 0.30, 0.12), mat_blue)
gearbox.modifiers.new('Rounded Gearbox', 'BEVEL').width = 0.035
gearbox.modifiers.new('Weighted Normal', 'WEIGHTED_NORMAL')
parent_keep_transform(gearbox, servo_root)

# Mounting ears
left_ear = add_cube('Left Mounting Ear', (0, 0.55, 0.22), (0.75, 0.08, 0.12), mat_blue)
right_ear = add_cube('Right Mounting Ear', (0, -0.55, 0.22), (0.75, 0.08, 0.12), mat_blue)
for ear in [left_ear, right_ear]:
    ear.modifiers.new('Rounded Mount Ear', 'BEVEL').width = 0.025
    ear.modifiers.new('Weighted Normal', 'WEIGHTED_NORMAL')
    parent_keep_transform(ear, servo_root)

# Screw holes and screw heads on mounting tabs
for y in [0.55, -0.55]:
    for x in [-0.42, 0.42]:
        screw = add_cylinder('Mounting Screw Head', 0.055, 0.018, (x, y, 0.35), material=mat_metal, vertices=40)
        parent_keep_transform(screw, servo_root)
        hole = add_cylinder('Dark Screw Hole', 0.034, 0.021, (x, y, 0.361), material=mat_dark, vertices=32)
        parent_keep_transform(hole, servo_root)

# Output shaft
shaft = add_cylinder('Metal Output Shaft', 0.085, 0.10, (0, 0, 0.72), material=mat_metal, vertices=64)
parent_keep_transform(shaft, horn_pivot)

# ----------------------------
# Servo horn assembly
# ----------------------------
horn_center = add_cylinder('Servo Horn Center Hub', 0.13, 0.055, (0, 0, 0.80), material=mat_white, vertices=64)
parent_keep_transform(horn_center, horn_pivot)

horn_arm_main = add_cube('Servo Horn Long Arm', (0.43, 0, 0.80), (0.44, 0.055, 0.028), mat_white)
horn_arm_main.modifiers.new('Rounded Horn Arm', 'BEVEL').width = 0.025
horn_arm_main.modifiers.new('Weighted Normal', 'WEIGHTED_NORMAL')
parent_keep_transform(horn_arm_main, horn_pivot)

horn_arm_short = add_cube('Servo Horn Short Arm', (-0.22, 0, 0.80), (0.21, 0.045, 0.024), mat_white)
horn_arm_short.modifiers.new('Rounded Short Horn Arm', 'BEVEL').width = 0.022
horn_arm_short.modifiers.new('Weighted Normal', 'WEIGHTED_NORMAL')
parent_keep_transform(horn_arm_short, horn_pivot)

# Hole pattern on horn arm
for i, x in enumerate([0.18, 0.35, 0.52, 0.69]):
    hole = add_cylinder('Servo Horn Linkage Hole', 0.026, 0.008, (x, 0, 0.834), material=mat_metal, vertices=24)
    parent_keep_transform(hole, horn_pivot)

center_screw = add_cylinder('Center Horn Screw', 0.055, 0.018, (0, 0, 0.835), material=mat_metal, vertices=40)
parent_keep_transform(center_screw, horn_pivot)

# Pointer dot on the horn so rotation is obvious
pointer = add_cylinder('White Horn Pointer Dot', 0.035, 0.012, (0.75, 0, 0.84), material=mat_green, vertices=32)
parent_keep_transform(pointer, horn_pivot)

# ----------------------------
# Wires leaving back of servo
# ----------------------------
wire_red = add_curve('Red Power Wire', [(-0.60, 0.10, 0.20), (-1.00, 0.20, 0.12), (-1.55, 0.16, 0.08), (-2.05, 0.22, 0.10)], 0.018, mat_red)
wire_brown = add_curve('Brown Ground Wire', [(-0.60, 0.00, 0.16), (-1.02, -0.02, 0.07), (-1.54, -0.06, 0.04), (-2.05, -0.02, 0.06)], 0.018, mat_brown)
wire_yellow = add_curve('Yellow Signal Wire', [(-0.60, -0.10, 0.20), (-0.98, -0.20, 0.12), (-1.50, -0.18, 0.09), (-2.05, -0.24, 0.10)], 0.018, mat_yellow)
for wire in [wire_red, wire_brown, wire_yellow]:
    parent_keep_transform(wire, servo_root)

# Small connector block
connector = add_cube('Three Pin Servo Connector', (-2.18, -0.02, 0.08), (0.12, 0.19, 0.07), mat_dark)
connector.modifiers.new('Rounded Connector', 'BEVEL').width = 0.012
connector.modifiers.new('Weighted Normal', 'WEIGHTED_NORMAL')
parent_keep_transform(connector, servo_root)

# ----------------------------
# Visual angle sweep arc above servo
# ----------------------------
arc_points = []
arc_radius = 0.92
for deg in range(-70, 71, 8):
    rad = math.radians(deg)
    arc_points.append((arc_radius * math.cos(rad), arc_radius * math.sin(rad), 0.93))
arc = add_curve('Servo Sweep Angle Arc', arc_points, 0.009, mat_green)
parent_keep_transform(arc, servo_root)

# End stops
for deg in [-70, 70]:
    rad = math.radians(deg)
    stop = add_cylinder('Servo Angle End Stop Marker', 0.025, 0.08, (arc_radius * math.cos(rad), arc_radius * math.sin(rad), 0.93), rotation=(math.radians(90), 0, 0), material=mat_green, vertices=24)
    parent_keep_transform(stop, servo_root)

# ----------------------------
# Floor, lighting, camera
# ----------------------------
bpy.ops.mesh.primitive_plane_add(size=7, location=(0, 0, -0.22))
floor = bpy.context.object
floor.name = 'Ground Plane'
assign_material(floor, mat_floor)

bpy.ops.object.light_add(type='AREA', location=(3.6, -4.2, 5.0))
key = bpy.context.object
key.name = 'Large Softbox Key Light'
key.data.energy = 1600
key.data.size = 4.8

bpy.ops.object.light_add(type='AREA', location=(-3.2, 2.5, 3.2))
fill = bpy.context.object
fill.name = 'Fill Light'
fill.data.energy = 650
fill.data.size = 4.0

bpy.ops.object.camera_add(location=(4.5, -5.0, 2.8), rotation=(math.radians(63), 0, math.radians(41)))
camera = bpy.context.object
camera.name = 'Camera Servo View'
camera.data.lens = 55
scene.camera = camera

# ----------------------------
# Render settings
# ----------------------------
scene.render.engine = 'CYCLES'
scene.cycles.samples = 96
try:
    scene.view_settings.view_transform = 'Filmic'
except Exception:
    pass
scene.world.color = (1.0, 1.0, 1.0)

# ----------------------------
# Animation
# ----------------------------
# Servo body subtle vibration while powered.
servo_root.location = (0, 0, 0)
servo_root.rotation_euler = (0, 0, 0)
servo_root.keyframe_insert(data_path='location', frame=1)
servo_root.keyframe_insert(data_path='rotation_euler', frame=1)

for fr, loc, rot in [
    (30,  (0.00, 0.00, 0.00), (0, 0, 0)),
    (60,  (0.02, 0.00, 0.00), (0, 0, math.radians(1.0))),
    (90,  (-0.01, 0.01, 0.00), (0, 0, math.radians(-0.8))),
    (120, (0.01, -0.01, 0.00), (0, 0, math.radians(0.7))),
    (150, (0.00, 0.00, 0.00), (0, 0, math.radians(-0.4))),
    (180, (0.00, 0.00, 0.00), (0, 0, 0)),
]:
    servo_root.location = loc
    servo_root.rotation_euler = rot
    servo_root.keyframe_insert(data_path='location', frame=fr)
    servo_root.keyframe_insert(data_path='rotation_euler', frame=fr)

# Main horn sweep animation around the vertical Z axis.
# The horn pivots at the exact center of the output shaft.
for fr, angle in [
    (1, 0),
    (25, 0),
    (45, -60),
    (70, 60),
    (95, -45),
    (120, 45),
    (145, -70),
    (165, 70),
    (180, 0),
]:
    horn_pivot.rotation_euler = (0, 0, math.radians(angle))
    horn_pivot.keyframe_insert(data_path='rotation_euler', frame=fr)

# Animate green pointer slightly brighter by scaling it during movement.
for fr, sc in [(1, 1.0), (45, 1.35), (70, 1.25), (95, 1.35), (120, 1.25), (145, 1.4), (165, 1.4), (180, 1.0)]:
    pointer.scale = (sc, sc, sc)
    pointer.keyframe_insert(data_path='scale', frame=fr)

# Camera drift
for fr, loc, rot in [
    (1, (4.5, -5.0, 2.8), (math.radians(63), 0, math.radians(41))),
    (100, (4.0, -4.6, 2.55), (math.radians(62), 0, math.radians(39))),
    (180, (4.4, -4.9, 2.75), (math.radians(63), 0, math.radians(41))),
]:
    camera.location = loc
    camera.rotation_euler = rot
    camera.keyframe_insert(data_path='location', frame=fr)
    camera.keyframe_insert(data_path='rotation_euler', frame=fr)

for obj in [servo_root, horn_pivot, pointer, camera]:
    set_bezier_interpolation(obj)

print('Animated small servo motor scene created: servo horn sweeps back and forth around the output shaft.')
