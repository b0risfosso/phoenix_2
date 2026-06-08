import bpy
import math
from mathutils import Vector

# ============================================================
# Animated Balloon Blender Script
# Initializes a balloon and animates it inflating, lifting,
# floating, bobbing, and gently swaying.
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
def make_material(name, color=(1, 1, 1, 1), roughness=0.4, metallic=0.0,
                  alpha=1.0, transmission=0.0):
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
        if 'Transmission Weight' in bsdf.inputs:
            bsdf.inputs['Transmission Weight'].default_value = transmission
        elif 'Transmission' in bsdf.inputs:
            bsdf.inputs['Transmission'].default_value = transmission
    if alpha < 1.0:
        mat.blend_method = 'BLEND'
        try:
            mat.shadow_method = 'HASHED'
        except Exception:
            pass
    return mat


def assign_material(obj, mat):
    if hasattr(obj.data, 'materials'):
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

# ----------------------------
# Materials
# ----------------------------
mat_balloon = make_material('BalloonRed', (0.93, 0.15, 0.18, 1.0), roughness=0.18)
mat_knot = make_material('BalloonKnot', (0.80, 0.07, 0.09, 1.0), roughness=0.25)
mat_string = make_material('String', (0.92, 0.91, 0.85, 1.0), roughness=0.8)
mat_floor = make_material('Floor', (0.94, 0.94, 0.96, 1.0), roughness=0.85)

# ----------------------------
# Root controller
# ----------------------------
bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0, 0, 0))
rig = bpy.context.object
rig.name = 'Balloon_Rig'

# ----------------------------
# Balloon body
# ----------------------------
bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0, location=(0, 0, 1.25))
balloon = bpy.context.object
balloon.name = 'Balloon_Body'
balloon.scale = (0.85, 0.85, 1.08)
bpy.ops.object.shade_smooth()
assign_material(balloon, mat_balloon)
parent_keep_transform(balloon, rig)

# Add a subdivision modifier for smoothness
sub = balloon.modifiers.new(name='Subsurf', type='SUBSURF')
sub.levels = 2
sub.render_levels = 2

# Knot at bottom of balloon
bpy.ops.mesh.primitive_cone_add(radius1=0.12, radius2=0.03, depth=0.22, location=(0, 0, 0.20), rotation=(math.radians(180), 0, 0))
knot = bpy.context.object
knot.name = 'Balloon_Knot'
bpy.ops.object.shade_smooth()
assign_material(knot, mat_knot)
parent_keep_transform(knot, rig)

# ----------------------------
# Balloon string
# ----------------------------
curve_data = bpy.data.curves.new('Balloon_String_Curve', type='CURVE')
curve_data.dimensions = '3D'
curve_data.resolution_u = 16
curve_data.bevel_depth = 0.008
curve_data.bevel_resolution = 4
spline = curve_data.splines.new('BEZIER')
spline.bezier_points.add(3)

string_points = [
    (0.0, 0.0, 0.18),
    (0.07, 0.02, -0.35),
    (-0.08, -0.03, -0.95),
    (0.02, 0.00, -1.55),
]
for i, p in enumerate(string_points):
    bp = spline.bezier_points[i]
    bp.co = Vector(p)
    bp.handle_left_type = 'AUTO'
    bp.handle_right_type = 'AUTO'

string_obj = bpy.data.objects.new('Balloon_String', curve_data)
bpy.context.collection.objects.link(string_obj)
assign_material(string_obj, mat_string)
parent_keep_transform(string_obj, rig)

# ----------------------------
# Ground plane
# ----------------------------
bpy.ops.mesh.primitive_plane_add(size=8, location=(0, 0, -1.55))
floor = bpy.context.object
floor.name = 'Ground'
assign_material(floor, mat_floor)

# ----------------------------
# Lighting
# ----------------------------
bpy.ops.object.light_add(type='AREA', location=(4.5, -4.5, 6.0))
key = bpy.context.object
key.name = 'Key_Light'
key.data.energy = 1800
key.data.size = 5.0

bpy.ops.object.light_add(type='AREA', location=(-4.0, 2.5, 3.5))
fill = bpy.context.object
fill.name = 'Fill_Light'
fill.data.energy = 700
fill.data.size = 4.0

bpy.ops.object.light_add(type='POINT', location=(1.5, -1.5, 2.5))
rim = bpy.context.object
rim.name = 'Rim_Light'
rim.data.energy = 160

# ----------------------------
# Camera
# ----------------------------
bpy.ops.object.camera_add(location=(6.0, -6.2, 3.8), rotation=(math.radians(66), 0, math.radians(42)))
camera = bpy.context.object
camera.name = 'Camera'
camera.data.lens = 50
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
# Start small/deflated near ground
rig.location = (0.0, 0.0, 0.0)
rig.rotation_euler = (0.0, 0.0, 0.0)
rig.scale = (0.18, 0.18, 0.18)
rig.keyframe_insert(data_path='location', frame=1)
rig.keyframe_insert(data_path='rotation_euler', frame=1)
rig.keyframe_insert(data_path='scale', frame=1)

# Inflation phase
rig.scale = (0.55, 0.55, 0.55)
rig.keyframe_insert(data_path='scale', frame=20)

rig.scale = (1.0, 1.0, 1.0)
rig.keyframe_insert(data_path='scale', frame=42)

# Hold briefly after inflation
rig.location = (0.0, 0.0, 0.0)
rig.rotation_euler = (0.0, 0.0, 0.0)
rig.scale = (1.0, 1.0, 1.0)
rig.keyframe_insert(data_path='location', frame=52)
rig.keyframe_insert(data_path='rotation_euler', frame=52)
rig.keyframe_insert(data_path='scale', frame=52)

# Lift off and drift
motion_keys = [
    (72,  (0.10, 0.05, 0.55), (math.radians(2),  math.radians(-4), math.radians(5))),
    (92,  (-0.15, 0.15, 1.10), (math.radians(-2), math.radians(5),  math.radians(-7))),
    (112, (0.22, -0.10, 1.65), (math.radians(3),  math.radians(-6), math.radians(6))),
    (132, (-0.18, -0.05, 2.10), (math.radians(-3), math.radians(4),  math.radians(-5))),
    (152, (0.08, 0.12, 2.45), (math.radians(2),  math.radians(-3), math.radians(4))),
    (180, (0.00, 0.00, 2.20), (0.0, 0.0, 0.0)),
]
for fr, loc, rot in motion_keys:
    rig.location = loc
    rig.rotation_euler = rot
    rig.keyframe_insert(data_path='location', frame=fr)
    rig.keyframe_insert(data_path='rotation_euler', frame=fr)

# Gentle squash/stretch to suggest buoyancy
balloon_scale_keys = [
    (1,   (0.85, 0.85, 1.08)),
    (20,  (0.92, 0.92, 1.15)),
    (42,  (0.85, 0.85, 1.08)),
    (72,  (0.87, 0.87, 1.05)),
    (92,  (0.84, 0.84, 1.10)),
    (112, (0.88, 0.88, 1.04)),
    (132, (0.84, 0.84, 1.10)),
    (152, (0.87, 0.87, 1.05)),
    (180, (0.85, 0.85, 1.08)),
]
for fr, sc in balloon_scale_keys:
    balloon.scale = sc
    balloon.keyframe_insert(data_path='scale', frame=fr)

# Subtle camera drift
camera_keys = [
    (1,   (6.0, -6.2, 3.8), (math.radians(66), 0, math.radians(42))),
    (100, (5.7, -5.7, 3.5), (math.radians(64), 0, math.radians(40))),
    (180, (5.9, -6.0, 3.7), (math.radians(65), 0, math.radians(41))),
]
for fr, loc, rot in camera_keys:
    camera.location = loc
    camera.rotation_euler = rot
    camera.keyframe_insert(data_path='location', frame=fr)
    camera.keyframe_insert(data_path='rotation_euler', frame=fr)

# Interpolation
for obj in [rig, balloon, camera]:
    set_bezier_interpolation(obj)

print('Animated balloon scene created: balloon inflates, lifts off, floats, and gently bobs in the air.')
