import bpy
import math
from mathutils import Vector

# ============================================================
# Animated Battery Pack with Electron Flow
# Clears the current scene, builds a small battery pack, and
# animates visible electron flow through the pack and wires.
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
scene.frame_end = 240
scene.render.fps = 24

# ----------------------------
# Helpers
# ----------------------------
def make_material(name, color=(1, 1, 1, 1), roughness=0.4, metallic=0.0,
                  alpha=1.0, transmission=0.0, emission_strength=0.0):
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


def parent_keep_transform(child, parent):
    child.parent = parent
    child.matrix_parent_inverse = parent.matrix_world.inverted()


def iter_action_fcurves(action):
    """Return fcurves safely across Blender action API versions."""
    if action is None:
        return []
    direct_fcurves = getattr(action, 'fcurves', None)
    if direct_fcurves is not None:
        return direct_fcurves

    # Blender 4.4+ can expose layered/slot-based actions.
    curves = []
    layers = getattr(action, 'layers', None)
    if layers is not None:
        for layer in layers:
            strips = getattr(layer, 'strips', [])
            for strip in strips:
                channelbag = getattr(strip, 'channelbag', None)
                if channelbag is not None:
                    bag_fcurves = getattr(channelbag, 'fcurves', None)
                    if bag_fcurves is not None:
                        curves.extend(list(bag_fcurves))
                channelbags = getattr(strip, 'channelbags', None)
                if channelbags is not None:
                    for bag in channelbags:
                        bag_fcurves = getattr(bag, 'fcurves', None)
                        if bag_fcurves is not None:
                            curves.extend(list(bag_fcurves))
    return curves


def set_action_interpolation(animation_data_owner, interpolation='BEZIER'):
    ad = getattr(animation_data_owner, 'animation_data', None)
    action = getattr(ad, 'action', None) if ad else None
    for fc in iter_action_fcurves(action):
        for kp in fc.keyframe_points:
            kp.interpolation = interpolation


def set_bezier_interpolation(obj):
    set_action_interpolation(obj, 'BEZIER')


def set_linear_interpolation(obj):
    set_action_interpolation(obj, 'LINEAR')


def set_data_linear_interpolation(data_block):
    set_action_interpolation(data_block, 'LINEAR')


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


def add_curve(name, points, bevel_depth=0.02, resolution=16, material=None, cyclic=False):
    curve_data = bpy.data.curves.new(name + '_curve', type='CURVE')
    curve_data.dimensions = '3D'
    curve_data.resolution_u = resolution
    curve_data.bevel_depth = bevel_depth
    curve_data.bevel_resolution = 6
    spline = curve_data.splines.new('BEZIER')
    spline.bezier_points.add(len(points) - 1)
    for i, p in enumerate(points):
        bp = spline.bezier_points[i]
        bp.co = Vector(p)
        bp.handle_left_type = 'AUTO'
        bp.handle_right_type = 'AUTO'
    spline.use_cyclic_u = cyclic
    obj = bpy.data.objects.new(name, curve_data)
    bpy.context.collection.objects.link(obj)
    if material:
        assign_material(obj, material)
    return obj


def add_path_follower_sphere(name, path_obj, radius, color_mat, start_offset, end_offset, start_frame, end_frame, parent=None):
    sphere = add_uv_sphere(name, radius=radius, location=(0, 0, 0), material=color_mat)
    con = sphere.constraints.new(type='FOLLOW_PATH')
    con.target = path_obj
    con.use_fixed_location = True
    con.forward_axis = 'FORWARD_X'
    con.up_axis = 'UP_Z'
    con.offset_factor = start_offset
    con.keyframe_insert(data_path='offset_factor', frame=start_frame)
    con.offset_factor = end_offset
    con.keyframe_insert(data_path='offset_factor', frame=end_frame)
    set_linear_interpolation(sphere)
    if parent is not None:
        parent_keep_transform(sphere, parent)
    return sphere

# ----------------------------
# Materials
# ----------------------------
mat_case = make_material('Case', (0.17, 0.18, 0.21, 1.0), roughness=0.55)
mat_cover = make_material('Cover', (0.82, 0.90, 0.98, 0.22), roughness=0.08, alpha=0.22, transmission=0.8)
mat_cell_blue = make_material('CellBlue', (0.22, 0.45, 0.95, 1.0), roughness=0.35)
mat_cell_green = make_material('CellGreen', (0.18, 0.72, 0.35, 1.0), roughness=0.35)
mat_metal = make_material('Metal', (0.76, 0.78, 0.82, 1.0), roughness=0.22, metallic=0.85)
mat_red = make_material('RedWire', (0.92, 0.12, 0.14, 1.0), roughness=0.5)
mat_black = make_material('BlackWire', (0.08, 0.08, 0.09, 1.0), roughness=0.7)
mat_yellow = make_material('ChargeBar', (0.95, 0.82, 0.18, 1.0), roughness=0.32)
mat_floor = make_material('Floor', (0.94, 0.94, 0.96, 1.0), roughness=0.85)
mat_electron = make_material('Electron', (0.25, 0.80, 1.0, 1.0), roughness=0.15, emission_strength=3.0)
mat_glow = make_material('PowerGlow', (0.25, 0.80, 1.0, 1.0), roughness=0.15, emission_strength=1.2)

# ----------------------------
# Root rig
# ----------------------------
bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0, 0, 0))
rig = bpy.context.object
rig.name = 'BatteryPack_Rig'

# ----------------------------
# Battery pack body
# ----------------------------
case_bottom = add_cube('Case_Bottom', location=(0, 0, 0.42), scale=(1.60, 0.80, 0.38), material=mat_case)
case_back = add_cube('Case_Back', location=(-1.34, 0, 0.84), scale=(0.09, 0.76, 0.36), material=mat_case)
case_front = add_cube('Case_Front', location=(1.34, 0, 0.84), scale=(0.09, 0.76, 0.36), material=mat_case)
case_left = add_cube('Case_Left', location=(0, -0.71, 0.84), scale=(1.34, 0.09, 0.36), material=mat_case)
case_right = add_cube('Case_Right', location=(0, 0.71, 0.84), scale=(1.34, 0.09, 0.36), material=mat_case)
cover = add_cube('Transparent_Cover', location=(0, 0, 1.28), scale=(1.32, 0.69, 0.07), material=mat_cover)

# Cells
cell_1 = add_cylinder('Cell_1', radius=0.30, depth=1.45, location=(-0.52, 0, 0.78), rotation=(math.radians(90), 0, 0), material=mat_cell_blue)
cell_2 = add_cylinder('Cell_2', radius=0.30, depth=1.45, location=(0.52, 0, 0.78), rotation=(math.radians(90), 0, 0), material=mat_cell_green)

# Terminals and internal straps
neg_terminal = add_cylinder('Negative_Terminal', radius=0.09, depth=0.12, location=(-1.05, 0.42, 1.05), rotation=(0, math.radians(90), 0), material=mat_metal)
pos_terminal = add_cylinder('Positive_Terminal', radius=0.09, depth=0.12, location=(1.05, -0.42, 1.05), rotation=(0, math.radians(90), 0), material=mat_metal)
strap_mid = add_cube('Cell_Strap', location=(0.0, 0.0, 1.05), scale=(0.30, 0.08, 0.03), material=mat_metal)
strap_left = add_cube('Left_Contact', location=(-0.82, 0.36, 1.03), scale=(0.18, 0.06, 0.03), material=mat_metal)
strap_right = add_cube('Right_Contact', location=(0.82, -0.36, 1.03), scale=(0.18, 0.06, 0.03), material=mat_metal)

# Charge bars
charge_bars = []
for i in range(4):
    bar = add_cube(
        f'Charge_Bar_{i+1}',
        location=(-0.45 + i * 0.30, -0.92, 0.48),
        scale=(0.10, 0.04, 0.22 + i * 0.02),
        material=mat_yellow,
    )
    charge_bars.append(bar)

# External wires
red_wire_pts = [
    (1.05, -0.42, 1.05),
    (1.45, -0.65, 1.15),
    (1.95, -0.90, 1.12),
    (2.45, -0.65, 1.00),
]
black_wire_pts = [
    (-1.05, 0.42, 1.05),
    (-1.45, 0.72, 1.14),
    (-1.95, 0.98, 1.08),
    (-2.45, 0.72, 0.96),
]
red_wire = add_curve('Red_Wire', red_wire_pts, bevel_depth=0.035, material=mat_red)
black_wire = add_curve('Black_Wire', black_wire_pts, bevel_depth=0.035, material=mat_black)

# Electron-flow paths
# Closed outer loop: negative terminal -> black wire -> far side -> red wire -> positive terminal.
outer_flow_pts = [
    (-1.05, 0.42, 1.05),
    (-1.55, 0.84, 1.12),
    (-2.45, 0.72, 0.96),
    (0.00, 1.60, 1.20),
    (2.45, -0.65, 1.00),
    (1.55, -0.80, 1.12),
    (1.05, -0.42, 1.05),
]
outer_flow_path = add_curve('Outer_Electron_Path', outer_flow_pts, bevel_depth=0.0, material=None, cyclic=True)
outer_flow_path.hide_viewport = True
outer_flow_path.hide_render = True

# Internal electron flow path through the pack
internal_flow_pts = [
    (-1.05, 0.42, 1.05),
    (-0.82, 0.32, 1.03),
    (-0.52, 0.08, 1.00),
    (0.00, 0.00, 1.05),
    (0.52, -0.08, 1.00),
    (0.82, -0.32, 1.03),
    (1.05, -0.42, 1.05),
]
internal_flow_path = add_curve('Internal_Electron_Path', internal_flow_pts, bevel_depth=0.0, material=None, cyclic=False)
internal_flow_path.hide_viewport = True
internal_flow_path.hide_render = True

# Parent visible pack objects to rig
for obj in [case_bottom, case_back, case_front, case_left, case_right, cover,
            cell_1, cell_2, neg_terminal, pos_terminal, strap_mid, strap_left,
            strap_right, red_wire, black_wire] + charge_bars:
    parent_keep_transform(obj, rig)

# Parent hidden paths to rig too
for obj in [outer_flow_path, internal_flow_path]:
    parent_keep_transform(obj, rig)

# Power glow indicators near terminals
neg_glow = add_uv_sphere('Negative_Glow', radius=0.05, location=(-1.05, 0.42, 1.05), material=mat_glow)
pos_glow = add_uv_sphere('Positive_Glow', radius=0.05, location=(1.05, -0.42, 1.05), material=mat_glow)
for obj in [neg_glow, pos_glow]:
    parent_keep_transform(obj, rig)

# ----------------------------
# Electron particles
# ----------------------------
# Outer loop electrons
outer_electrons = []
outer_offsets = [0.00, 0.14, 0.28, 0.42, 0.56, 0.70]
for i, start in enumerate(outer_offsets):
    elec = add_path_follower_sphere(
        f'Outer_Electron_{i+1}', outer_flow_path, 0.055, mat_electron,
        start, start + 1.2, 1, 240
    )
    parent_keep_transform(elec, rig)
    outer_electrons.append(elec)

# Internal electrons
internal_electrons = []
internal_offsets = [0.00, 0.22, 0.44]
for i, start in enumerate(internal_offsets):
    elec = add_path_follower_sphere(
        f'Internal_Electron_{i+1}', internal_flow_path, 0.045, mat_electron,
        start, 1.0, 20 + i * 6, 160 + i * 6
    )
    parent_keep_transform(elec, rig)
    internal_electrons.append(elec)

# ----------------------------
# Ground, camera, lights
# ----------------------------
bpy.ops.mesh.primitive_plane_add(size=10, location=(0, 0, 0))
floor = bpy.context.object
floor.name = 'Ground'
assign_material(floor, mat_floor)

bpy.ops.object.camera_add(location=(6.6, -6.6, 4.1), rotation=(math.radians(64), 0, math.radians(42)))
camera = bpy.context.object
camera.name = 'Camera'
camera.data.lens = 52
scene.camera = camera

bpy.ops.object.light_add(type='AREA', location=(4.7, -4.2, 6.0))
key = bpy.context.object
key.data.energy = 1800
key.data.size = 5.2

bpy.ops.object.light_add(type='AREA', location=(-4.0, 3.0, 3.5))
fill = bpy.context.object
fill.data.energy = 850
fill.data.size = 4.0

bpy.ops.object.light_add(type='POINT', location=(1.5, -1.2, 2.8))
rim = bpy.context.object
rim.data.energy = 200

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
# Cover opens and closes slightly to reveal internals
for fr, loc, rot in [
    (1,   (0.0, 0.0, 1.28), (0.0, 0.0, 0.0)),
    (36,  (0.0, 0.0, 1.28), (0.0, 0.0, 0.0)),
    (70,  (0.0, -0.10, 1.42), (math.radians(-18), 0.0, 0.0)),
    (150, (0.0, -0.10, 1.42), (math.radians(-18), 0.0, 0.0)),
    (210, (0.0, 0.0, 1.28), (0.0, 0.0, 0.0)),
    (240, (0.0, 0.0, 1.28), (0.0, 0.0, 0.0)),
]:
    cover.location = loc
    cover.rotation_euler = rot
    cover.keyframe_insert(data_path='location', frame=fr)
    cover.keyframe_insert(data_path='rotation_euler', frame=fr)

# Rig motion: slight lift and tilt while energized
for fr, loc, rot in [
    (1,   (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
    (40,  (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
    (90,  (0.12, -0.05, 0.20), (math.radians(4), math.radians(-5), math.radians(3))),
    (140, (-0.10, 0.07, 0.28), (math.radians(-3), math.radians(5), math.radians(-3))),
    (190, (0.08, 0.02, 0.18), (math.radians(2), math.radians(-4), math.radians(2))),
    (240, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
]:
    rig.location = loc
    rig.rotation_euler = rot
    rig.keyframe_insert(data_path='location', frame=fr)
    rig.keyframe_insert(data_path='rotation_euler', frame=fr)

# Charge bars pulse to indicate activity
bar_scales = [1.0, 1.15, 1.28, 1.38]
for idx, bar in enumerate(charge_bars):
    base_scale = bar.scale.copy()
    frames = [1, 45 + idx * 8, 85 + idx * 8, 140 + idx * 5, 200 + idx * 3, 240]
    heights = [1.0, 1.0, bar_scales[idx], 1.05 + idx * 0.05, 1.18 + idx * 0.04, 1.0]
    for fr, h in zip(frames, heights):
        bar.scale = (base_scale.x, base_scale.y, base_scale.z * h)
        bar.keyframe_insert(data_path='scale', frame=fr)

# Terminal glow pulse
for glow in [neg_glow, pos_glow]:
    for fr, s in [(1, 0.7), (40, 0.7), (80, 1.4), (120, 1.0), (160, 1.5), (210, 1.1), (240, 0.7)]:
        glow.scale = (s, s, s)
        glow.keyframe_insert(data_path='scale', frame=fr)

# Camera drift
for fr, loc, rot in [
    (1,   (6.6, -6.6, 4.1), (math.radians(64), 0, math.radians(42))),
    (140, (6.0, -6.0, 3.8), (math.radians(63), 0, math.radians(40))),
    (240, (6.3, -6.3, 4.0), (math.radians(64), 0, math.radians(41))),
]:
    camera.location = loc
    camera.rotation_euler = rot
    camera.keyframe_insert(data_path='location', frame=fr)
    camera.keyframe_insert(data_path='rotation_euler', frame=fr)

# Interpolation
for obj in [cover, rig, camera, neg_glow, pos_glow] + charge_bars:
    set_bezier_interpolation(obj)

# Follow path constraint animation should be linear.
for elec in outer_electrons + internal_electrons:
    set_linear_interpolation(elec)

print('Animated battery pack created with visible electron flow through the battery pack and external wires.')
