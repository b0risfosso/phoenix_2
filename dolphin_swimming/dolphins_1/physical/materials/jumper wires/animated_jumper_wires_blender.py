import bpy
import math
from mathutils import Vector

# ============================================================
# Animated Jumper Wires Blender Script
# Initializes colored jumper wires and animates them moving,
# bending, connecting to header pins, and carrying signal pulses.
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
scene.frame_end = 220
scene.render.fps = 24

# ----------------------------
# Helpers
# ----------------------------
def make_material(name, color=(1, 1, 1, 1), roughness=0.45, metallic=0.0, alpha=1.0, emission_strength=0.0):
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
        if 'Emission Color' in bsdf.inputs:
            bsdf.inputs['Emission Color'].default_value = color
        if 'Emission Strength' in bsdf.inputs:
            bsdf.inputs['Emission Strength'].default_value = emission_strength
        elif 'Emission' in bsdf.inputs and emission_strength > 0:
            bsdf.inputs['Emission'].default_value = color
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


def safe_set_interpolation(obj, interpolation='BEZIER'):
    ad = getattr(obj, 'animation_data', None)
    action = getattr(ad, 'action', None) if ad else None
    fcurves = getattr(action, 'fcurves', None) if action else None
    if fcurves:
        for fc in fcurves:
            for kp in fc.keyframe_points:
                kp.interpolation = interpolation


def safe_set_data_interpolation(data_block, interpolation='BEZIER'):
    ad = getattr(data_block, 'animation_data', None)
    action = getattr(ad, 'action', None) if ad else None
    fcurves = getattr(action, 'fcurves', None) if action else None
    if fcurves:
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
    bpy.ops.object.shade_smooth()
    if material:
        assign_material(obj, material)
    return obj


def add_cylinder(name, radius=0.1, depth=1.0, location=(0, 0, 0), rotation=(0, 0, 0), vertices=48, material=None):
    bpy.ops.mesh.primitive_cylinder_add(vertices=vertices, radius=radius, depth=depth, location=location, rotation=rotation)
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


def make_bezier_curve(name, points, bevel_depth=0.035, material=None):
    curve = bpy.data.curves.new(name + '_Curve', type='CURVE')
    curve.dimensions = '3D'
    curve.resolution_u = 24
    curve.bevel_depth = bevel_depth
    curve.bevel_resolution = 6
    spline = curve.splines.new('BEZIER')
    spline.bezier_points.add(len(points) - 1)
    for i, p in enumerate(points):
        bp = spline.bezier_points[i]
        bp.co = Vector(p)
        bp.handle_left_type = 'AUTO'
        bp.handle_right_type = 'AUTO'
    obj = bpy.data.objects.new(name, curve)
    bpy.context.collection.objects.link(obj)
    if material:
        assign_material(obj, material)
    return obj


def set_curve_points(curve_obj, points, frame):
    spline = curve_obj.data.splines[0]
    for i, p in enumerate(points):
        bp = spline.bezier_points[i]
        bp.co = Vector(p)
        bp.keyframe_insert(data_path='co', frame=frame)


def add_path_pulse(name, path_obj, material, start_offset, start_frame, end_frame, radius=0.055):
    pulse = add_uv_sphere(name, radius=radius, location=(0, 0, 0), material=material)
    con = pulse.constraints.new(type='FOLLOW_PATH')
    con.target = path_obj
    con.use_fixed_location = True
    con.forward_axis = 'FORWARD_X'
    con.up_axis = 'UP_Z'
    con.offset_factor = start_offset
    con.keyframe_insert(data_path='offset_factor', frame=start_frame)
    con.offset_factor = start_offset + 1.0
    con.keyframe_insert(data_path='offset_factor', frame=end_frame)
    return pulse

# ----------------------------
# Materials
# ----------------------------
mat_red = make_material('Red_Insulation', (0.95, 0.08, 0.08, 1.0), roughness=0.52)
mat_black = make_material('Black_Insulation', (0.02, 0.02, 0.025, 1.0), roughness=0.72)
mat_yellow = make_material('Yellow_Insulation', (1.0, 0.82, 0.05, 1.0), roughness=0.50)
mat_blue = make_material('Blue_Insulation', (0.05, 0.30, 0.95, 1.0), roughness=0.50)
mat_green = make_material('Green_Insulation', (0.05, 0.70, 0.25, 1.0), roughness=0.50)
mat_pin = make_material('Metal_Pin', (0.82, 0.75, 0.48, 1.0), roughness=0.20, metallic=0.85)
mat_housing = make_material('Black_Pin_Housing', (0.03, 0.03, 0.035, 1.0), roughness=0.68)
mat_board = make_material('Demo_Header_Board', (0.08, 0.36, 0.18, 1.0), roughness=0.55)
mat_floor = make_material('Floor', (0.94, 0.94, 0.96, 1.0), roughness=0.85)
mat_signal = make_material('Signal_Pulse', (0.25, 0.85, 1.0, 1.0), roughness=0.15, emission_strength=3.0)

# ----------------------------
# Root rig
# ----------------------------
bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0, 0, 0))
rig = bpy.context.object
rig.name = 'Jumper_Wires_Rig'

# ----------------------------
# Demo boards / pin rows
# ----------------------------
left_board = add_cube('Left_Header_Board', location=(-2.2, 0, 0.05), scale=(0.72, 1.1, 0.05), material=mat_board)
right_board = add_cube('Right_Header_Board', location=(2.2, 0, 0.05), scale=(0.72, 1.1, 0.05), material=mat_board)
parent_keep_transform(left_board, rig)
parent_keep_transform(right_board, rig)

left_pins = []
right_pins = []
for i, y in enumerate([-0.72, -0.36, 0.0, 0.36, 0.72]):
    lp = add_cylinder(f'Left_Header_Pin_{i+1}', radius=0.045, depth=0.34, location=(-2.2, y, 0.28), material=mat_pin)
    rp = add_cylinder(f'Right_Header_Pin_{i+1}', radius=0.045, depth=0.34, location=(2.2, y, 0.28), material=mat_pin)
    lh = add_cube(f'Left_Black_Header_Housing_{i+1}', location=(-2.2, y, 0.13), scale=(0.09, 0.09, 0.07), material=mat_housing)
    rh = add_cube(f'Right_Black_Header_Housing_{i+1}', location=(2.2, y, 0.13), scale=(0.09, 0.09, 0.07), material=mat_housing)
    for obj in [lp, rp, lh, rh]:
        parent_keep_transform(obj, rig)
    left_pins.append(lp)
    right_pins.append(rp)

# ----------------------------
# Jumper wire curves
# ----------------------------
wire_specs = [
    ('Red_Jumper_Wire', mat_red, -0.72, 0.55),
    ('Black_Jumper_Wire', mat_black, -0.36, 0.20),
    ('Yellow_Jumper_Wire', mat_yellow, 0.00, 0.85),
    ('Blue_Jumper_Wire', mat_blue, 0.36, 0.35),
    ('Green_Jumper_Wire', mat_green, 0.72, 0.65),
]

wires = []
end_parts = []
for idx, (name, mat, y, arc) in enumerate(wire_specs):
    start_points = [
        (-2.2, y, 0.58),
        (-1.15, y + 0.12, 1.05 + arc * 0.10),
        (1.15, y - 0.12, 1.05 + arc * 0.10),
        (2.2, y, 0.58),
    ]
    wire = make_bezier_curve(name, start_points, bevel_depth=0.035, material=mat)
    parent_keep_transform(wire, rig)
    wires.append((wire, y, arc))

    # Connector housings at both ends
    left_conn = add_cube(name + '_Left_Connector', location=(-2.2, y, 0.50), scale=(0.13, 0.10, 0.11), material=mat_housing)
    right_conn = add_cube(name + '_Right_Connector', location=(2.2, y, 0.50), scale=(0.13, 0.10, 0.11), material=mat_housing)
    left_tip = add_cylinder(name + '_Left_Metal_Tip', radius=0.035, depth=0.18, location=(-2.2, y, 0.35), material=mat_pin)
    right_tip = add_cylinder(name + '_Right_Metal_Tip', radius=0.035, depth=0.18, location=(2.2, y, 0.35), material=mat_pin)
    for obj in [left_conn, right_conn, left_tip, right_tip]:
        parent_keep_transform(obj, rig)
        end_parts.append((obj, y, idx))

# ----------------------------
# Signal pulses moving along wire curves
# ----------------------------
pulses = []
for idx, (wire, y, arc) in enumerate(wires):
    pulse = add_path_pulse(f'Signal_Pulse_{idx+1}', wire, mat_signal, start_offset=0.0, start_frame=72 + idx * 8, end_frame=205 + idx * 8, radius=0.055)
    parent_keep_transform(pulse, rig)
    pulses.append(pulse)

# ----------------------------
# Ground, camera, lights
# ----------------------------
bpy.ops.mesh.primitive_plane_add(size=8, location=(0, 0, -0.02))
floor = bpy.context.object
floor.name = 'Ground'
assign_material(floor, mat_floor)

bpy.ops.object.camera_add(location=(5.3, -5.8, 3.2), rotation=(math.radians(62), 0, math.radians(41)))
camera = bpy.context.object
camera.name = 'Camera'
camera.data.lens = 52
scene.camera = camera

bpy.ops.object.light_add(type='AREA', location=(3.8, -4.4, 5.6))
key = bpy.context.object
key.name = 'Key_Light'
key.data.energy = 1600
key.data.size = 5.0

bpy.ops.object.light_add(type='AREA', location=(-4.2, 3.0, 3.6))
fill = bpy.context.object
fill.name = 'Fill_Light'
fill.data.energy = 700
fill.data.size = 4.2

bpy.ops.object.light_add(type='POINT', location=(0, -2, 2.1))
rim = bpy.context.object
rim.name = 'Signal_Glow_Light'
rim.data.energy = 160

# ----------------------------
# Render settings
# ----------------------------
scene.render.engine = 'CYCLES'
scene.cycles.samples = 96
try:
    scene.view_settings.view_transform = 'Filmic'
except Exception:
    pass
scene.world.color = (1, 1, 1)

# ----------------------------
# Animation
# ----------------------------
# Rig lift and slight presentation motion
for fr, loc, rot in [
    (1, (0, 0, 0), (0, 0, 0)),
    (50, (0, 0, 0.04), (math.radians(1), 0, 0)),
    (120, (0.05, -0.02, 0.08), (math.radians(2), math.radians(-3), math.radians(1))),
    (180, (-0.03, 0.02, 0.06), (math.radians(-1), math.radians(2), math.radians(-1))),
    (220, (0, 0, 0), (0, 0, 0)),
]:
    rig.location = loc
    rig.rotation_euler = rot
    rig.keyframe_insert(data_path='location', frame=fr)
    rig.keyframe_insert(data_path='rotation_euler', frame=fr)

# Wires start spread apart/high, then settle into connected jumper shape, then wiggle.
for idx, (wire, y, arc) in enumerate(wires):
    spread = 0.35 + idx * 0.04
    start_shape = [
        (-2.55, y - spread, 0.72),
        (-1.25, y - 0.20, 1.55 + arc * 0.10),
        (1.25, y + 0.20, 1.55 + arc * 0.10),
        (2.55, y + spread, 0.72),
    ]
    connected_shape = [
        (-2.2, y, 0.58),
        (-1.15, y + 0.12, 1.05 + arc * 0.10),
        (1.15, y - 0.12, 1.05 + arc * 0.10),
        (2.2, y, 0.58),
    ]
    wiggle_shape = [
        (-2.2, y, 0.58),
        (-1.05, y + 0.18, 1.12 + arc * 0.10),
        (1.05, y - 0.18, 1.00 + arc * 0.10),
        (2.2, y, 0.58),
    ]
    final_shape = [
        (-2.2, y, 0.58),
        (-1.15, y + 0.10, 1.04 + arc * 0.10),
        (1.15, y - 0.10, 1.04 + arc * 0.10),
        (2.2, y, 0.58),
    ]
    set_curve_points(wire, start_shape, 1)
    set_curve_points(wire, start_shape, 30 + idx * 3)
    set_curve_points(wire, connected_shape, 70 + idx * 4)
    set_curve_points(wire, wiggle_shape, 130 + idx * 5)
    set_curve_points(wire, connected_shape, 170 + idx * 3)
    set_curve_points(wire, final_shape, 220)
    safe_set_data_interpolation(wire.data, 'BEZIER')

# Connector ends move toward pins
for obj, y, idx in end_parts:
    is_left = 'Left' in obj.name
    z_base = 0.50 if 'Connector' in obj.name else 0.35
    x_target = -2.2 if is_left else 2.2
    x_start = -2.55 if is_left else 2.55
    y_start = y - (0.35 + idx * 0.04) if is_left else y + (0.35 + idx * 0.04)
    obj.location = (x_start, y_start, z_base + 0.18)
    obj.keyframe_insert(data_path='location', frame=1)
    obj.location = (x_start, y_start, z_base + 0.18)
    obj.keyframe_insert(data_path='location', frame=34 + idx * 3)
    obj.location = (x_target, y, z_base)
    obj.keyframe_insert(data_path='location', frame=74 + idx * 4)
    obj.location = (x_target, y, z_base + 0.035)
    obj.keyframe_insert(data_path='location', frame=132 + idx * 3)
    obj.location = (x_target, y, z_base)
    obj.keyframe_insert(data_path='location', frame=170 + idx * 3)
    obj.location = (x_target, y, z_base)
    obj.keyframe_insert(data_path='location', frame=220)

# Header pins pulse as signal reaches them
for idx, (lp, rp) in enumerate(zip(left_pins, right_pins)):
    for pin, start in [(lp, 70 + idx * 8), (rp, 110 + idx * 8)]:
        for fr, s in [(1, 1.0), (start, 1.0), (start + 10, 1.35), (start + 22, 1.0), (220, 1.0)]:
            pin.scale = (s, s, 1.0)
            pin.keyframe_insert(data_path='scale', frame=fr)

# Camera drift
for fr, loc, rot in [
    (1, (5.3, -5.8, 3.2), (math.radians(62), 0, math.radians(41))),
    (120, (4.8, -5.1, 3.0), (math.radians(61), 0, math.radians(39))),
    (220, (5.1, -5.6, 3.15), (math.radians(62), 0, math.radians(40))),
]:
    camera.location = loc
    camera.rotation_euler = rot
    camera.keyframe_insert(data_path='location', frame=fr)
    camera.keyframe_insert(data_path='rotation_euler', frame=fr)

# Interpolation
for obj in [rig, camera] + [p for p in pulses] + [obj for obj, _, _ in end_parts] + left_pins + right_pins:
    safe_set_interpolation(obj, 'BEZIER')
for pulse in pulses:
    safe_set_interpolation(pulse, 'LINEAR')

print('Animated jumper wires scene created: wires move into place, connect to headers, wiggle, and show signal pulses.')
