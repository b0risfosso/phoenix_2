import bpy
import math
from mathutils import Vector

# ============================================================
# Animated Hot Glue Gun + Glue Loading + Realistic Curved Glue
# Strand + Glue Drips
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
def make_material(name, color=(0.8, 0.8, 0.8, 1.0), roughness=0.45, metallic=0.0,
                  alpha=1.0, transmission=0.0, emission_strength=0.0):
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    bsdf = nodes.get('Principled BSDF')
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


def set_bezier_interpolation(obj):
    ad = getattr(obj, 'animation_data', None)
    if ad and ad.action:
        fcurves = getattr(ad.action, 'fcurves', None)
        if fcurves:
            for fc in fcurves:
                for kp in fc.keyframe_points:
                    kp.interpolation = 'BEZIER'


def set_datablock_bezier_interpolation(data_block):
    ad = getattr(data_block, 'animation_data', None)
    if ad and ad.action:
        fcurves = getattr(ad.action, 'fcurves', None)
        if fcurves:
            for fc in fcurves:
                for kp in fc.keyframe_points:
                    kp.interpolation = 'BEZIER'


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


def add_curve_line(name, points, bevel_depth=0.02, resolution=18, material=None, radii=None):
    curve_data = bpy.data.curves.new(name=name + '_curve', type='CURVE')
    curve_data.dimensions = '3D'
    curve_data.resolution_u = resolution
    curve_data.bevel_depth = bevel_depth
    curve_data.bevel_resolution = 8
    curve_data.fill_mode = 'FULL'
    spline = curve_data.splines.new('BEZIER')
    spline.bezier_points.add(len(points) - 1)
    for i, p in enumerate(points):
        bp = spline.bezier_points[i]
        bp.co = Vector(p)
        bp.handle_left_type = 'AUTO'
        bp.handle_right_type = 'AUTO'
        if radii and i < len(radii):
            bp.radius = radii[i]
    obj = bpy.data.objects.new(name, curve_data)
    bpy.context.collection.objects.link(obj)
    if material:
        assign_material(obj, material)
    return obj


# ----------------------------
# Materials
# ----------------------------
mat_body = make_material('GlueGunBody', (0.96, 0.60, 0.11, 1.0), roughness=0.5)
mat_gray_rubber = make_material('GlueGunGrayRubber', (0.48, 0.48, 0.50, 1.0), roughness=0.62)
mat_metal = make_material('GlueGunMetal', (0.72, 0.74, 0.76, 1.0), roughness=0.22, metallic=0.9)
mat_trigger = make_material('TriggerBlue', (0.1, 0.35, 0.95, 1.0), roughness=0.4)
mat_glue_stick = make_material('GlueStick', (0.93, 0.95, 1.0, 0.38), roughness=0.07, alpha=0.38, transmission=0.65)
mat_hot_glue = make_material('HotGlue', (1.0, 0.84, 0.36, 0.70), roughness=0.06, alpha=0.70, transmission=0.35)
mat_glow = make_material('GlueGlow', (1.0, 0.65, 0.15, 1.0), roughness=0.2, emission_strength=0.4)
mat_floor = make_material('Floor', (0.93, 0.93, 0.95, 1.0), roughness=0.85)

# ----------------------------
# Root controller
# ----------------------------
bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0, 0, 0))
rig = bpy.context.object
rig.name = 'HotGlueGun_Rig'

# ----------------------------
# Hot glue gun model
# ----------------------------
body_main = add_cube('Body_Main', location=(0.0, 0.0, 1.12), scale=(0.82, 0.26, 0.32), material=mat_body)
body_top = add_cube('Body_Top', location=(-0.05, 0.0, 1.42), scale=(0.55, 0.22, 0.16), material=mat_body)
back_tube = add_cylinder('Rear_Glue_Channel_Gray', radius=0.12, depth=1.05, location=(-1.05, 0.0, 1.10), rotation=(0, math.radians(90), 0), material=mat_gray_rubber)
handle = add_cube('Handle', location=(-0.28, 0.0, 0.38), scale=(0.21, 0.14, 0.62), rotation=(0, math.radians(16), 0), material=mat_body)
grip = add_cube('Gray_Grip_Panel', location=(-0.24, 0.0, 0.30), scale=(0.12, 0.12, 0.42), rotation=(0, math.radians(16), 0), material=mat_gray_rubber)
trigger = add_cube('Trigger', location=(0.08, 0.0, 0.72), scale=(0.10, 0.07, 0.24), rotation=(math.radians(18), 0, 0), material=mat_trigger)
stand_front = add_cube('Stand_Front', location=(0.82, 0.0, 0.28), scale=(0.03, 0.13, 0.18), material=mat_metal)
stand_foot = add_cube('Stand_Foot', location=(0.93, 0.0, 0.13), scale=(0.13, 0.025, 0.025), rotation=(0, math.radians(6), 0), material=mat_metal)
nozzle_base = add_cylinder('Nozzle_Base', radius=0.10, depth=0.22, location=(0.93, 0.0, 1.10), rotation=(0, math.radians(90), 0), material=mat_metal)
nozzle_tip = add_cylinder('Nozzle_Tip', radius=0.055, depth=0.26, location=(1.14, 0.0, 1.10), rotation=(0, math.radians(90), 0), material=mat_metal)
nozzle_orifice = add_uv_sphere('Nozzle_Orifice', radius=0.034, location=(1.28, 0.0, 1.10), material=mat_glow)
side_panel = add_cube('Gray_Side_Panel', location=(0.16, 0.0, 1.18), scale=(0.24, 0.03, 0.20), material=mat_gray_rubber)
loading_port = add_cylinder('Loading_Port_Ring', radius=0.135, depth=0.06, location=(-0.48, 0.0, 1.10), rotation=(0, math.radians(90), 0), material=mat_metal)
indicator = add_uv_sphere('Power_Indicator', radius=0.04, location=(0.46, 0.18, 1.22), material=mat_trigger)

for obj in [body_main, body_top, back_tube, handle, grip, trigger, stand_front, stand_foot,
            nozzle_base, nozzle_tip, nozzle_orifice, side_panel, loading_port, indicator]:
    parent_keep_transform(obj, rig)

# ----------------------------
# Glue stick
# ----------------------------
glue_stick = add_cylinder(
    'Glue_Stick', radius=0.095, depth=2.4,
    location=(-2.45, 0.0, 1.10),
    rotation=(0, math.radians(90), 0),
    material=mat_glue_stick
)
parent_keep_transform(glue_stick, rig)
rear_mark = add_cube('Glue_Stick_End_Mark', location=(-3.63, 0.0, 1.10), scale=(0.03, 0.10, 0.10), material=mat_glow)
parent_keep_transform(rear_mark, glue_stick)

# ----------------------------
# Main curved glue strand
# ----------------------------
main_glue_points = [
    (1.28,  0.00, 1.10),
    (1.42,  0.01, 1.09),
    (1.62,  0.03, 1.05),
    (1.87,  0.05, 0.98),
    (2.10,  0.03, 0.90),
    (2.30, -0.01, 0.81),
    (2.48, -0.05, 0.72),
]
main_glue_radii = [1.00, 0.98, 0.95, 0.90, 0.82, 0.74, 0.58]

glue_strip = add_curve_line(
    'Hot_Glue_Strip', main_glue_points,
    bevel_depth=0.033,
    resolution=24,
    material=mat_hot_glue,
    radii=main_glue_radii,
)
parent_keep_transform(glue_strip, rig)
glue_strip.data.bevel_factor_mapping_start = 'SPLINE'
glue_strip.data.bevel_factor_mapping_end = 'SPLINE'
glue_strip.data.bevel_factor_start = 0.0
glue_strip.data.bevel_factor_end = 0.0

# End droplet for the main strand
end_droplet = add_uv_sphere('Glue_Droplet_End', radius=0.075, location=(2.48, -0.05, 0.72), material=mat_hot_glue)
parent_keep_transform(end_droplet, rig)
end_droplet.scale = (0.01, 0.01, 0.01)

# ----------------------------
# Additional hanging drips
# ----------------------------
drip1_points = [
    (1.84, 0.05, 0.99),
    (1.87, 0.06, 0.88),
    (1.90, 0.06, 0.73),
]
drip2_points = [
    (2.14, 0.02, 0.89),
    (2.17, 0.01, 0.76),
    (2.20, 0.00, 0.60),
]

drip1 = add_curve_line('Glue_Drip_1', drip1_points, bevel_depth=0.015, resolution=18, material=mat_hot_glue, radii=[0.65, 0.55, 0.40])
drip2 = add_curve_line('Glue_Drip_2', drip2_points, bevel_depth=0.013, resolution=18, material=mat_hot_glue, radii=[0.55, 0.46, 0.32])
for drip in [drip1, drip2]:
    parent_keep_transform(drip, rig)
    drip.data.bevel_factor_mapping_start = 'SPLINE'
    drip.data.bevel_factor_mapping_end = 'SPLINE'
    drip.data.bevel_factor_start = 0.0
    drip.data.bevel_factor_end = 0.0

drip1_drop = add_uv_sphere('Glue_Drip1_Bulb', radius=0.045, location=(1.90, 0.06, 0.73), material=mat_hot_glue)
drip2_drop = add_uv_sphere('Glue_Drip2_Bulb', radius=0.040, location=(2.20, 0.00, 0.60), material=mat_hot_glue)
for drop in [drip1_drop, drip2_drop]:
    parent_keep_transform(drop, rig)
    drop.scale = (0.01, 0.01, 0.01)

# A tiny nozzle drip
nozzle_drip = add_uv_sphere('Nozzle_Drip', radius=0.03, location=(1.30, 0.0, 1.06), material=mat_hot_glue)
parent_keep_transform(nozzle_drip, rig)
nozzle_drip.scale = (0.01, 0.01, 0.01)

# ----------------------------
# Ground plane
# ----------------------------
bpy.ops.mesh.primitive_plane_add(size=9, location=(0, 0, 0))
floor = bpy.context.object
floor.name = 'Ground'
assign_material(floor, mat_floor)

# ----------------------------
# Camera and lights
# ----------------------------
bpy.ops.object.camera_add(location=(6.2, -6.6, 3.6), rotation=(math.radians(66), 0, math.radians(43)))
camera = bpy.context.object
camera.name = 'Camera'
camera.data.lens = 50
scene.camera = camera

bpy.ops.object.light_add(type='AREA', location=(4.8, -4.6, 6.3))
key = bpy.context.object
key.data.energy = 1800
key.data.size = 5.5

bpy.ops.object.light_add(type='AREA', location=(-3.5, 3.0, 3.4))
fill = bpy.context.object
fill.data.energy = 800
fill.data.size = 4.0

bpy.ops.object.light_add(type='POINT', location=(1.7, -1.4, 2.3))
rim = bpy.context.object
rim.data.energy = 180

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
# Gun movement
for fr, loc, rot in [
    (1,   (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
    (40,  (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
    (75,  (0.15, 0.0, 0.0), (0.0, math.radians(-5), 0.0)),
    (115, (0.55, 0.12, 0.0), (0.0, math.radians(-12), math.radians(2.5))),
    (145, (0.30, -0.08, 0.0), (0.0, math.radians(-9), math.radians(-2.0))),
    (180, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
]:
    rig.location = loc
    rig.rotation_euler = rot
    rig.keyframe_insert(data_path='location', frame=fr)
    rig.keyframe_insert(data_path='rotation_euler', frame=fr)

# Glue stick loading animation
for fr, x in [(1, -2.45), (20, -2.45), (42, -1.55), (58, -0.98), (75, -0.70), (112, -0.50), (145, -0.38), (180, -0.30)]:
    glue_stick.location = (x, 0.0, 1.10)
    glue_stick.keyframe_insert(data_path='location', frame=fr)

# Trigger squeeze / release
for fr, rx in [(1, math.radians(18)), (52, math.radians(18)), (70, math.radians(4)), (105, math.radians(1)), (132, math.radians(8)), (160, math.radians(18)), (180, math.radians(18))]:
    trigger.rotation_euler = (rx, 0.0, 0.0)
    trigger.keyframe_insert(data_path='rotation_euler', frame=fr)

# Main glue strand growth
for fr, v in [(1, 0.0), (58, 0.0), (70, 0.10), (82, 0.30), (94, 0.55), (108, 0.82), (122, 1.0), (180, 1.0)]:
    glue_strip.data.bevel_factor_end = v
    glue_strip.data.keyframe_insert(data_path='bevel_factor_end', frame=fr)

# Drips appear later as the main strand gets heavier
for fr, v in [(1, 0.0), (90, 0.0), (102, 0.45), (116, 0.82), (126, 1.0), (180, 1.0)]:
    drip1.data.bevel_factor_end = v
    drip1.data.keyframe_insert(data_path='bevel_factor_end', frame=fr)

for fr, v in [(1, 0.0), (100, 0.0), (112, 0.38), (124, 0.78), (136, 1.0), (180, 1.0)]:
    drip2.data.bevel_factor_end = v
    drip2.data.keyframe_insert(data_path='bevel_factor_end', frame=fr)

# Droplet growth
for fr, s in [(1, 0.01), (70, 0.02), (92, 0.35), (110, 0.75), (124, 1.0), (180, 1.0)]:
    end_droplet.scale = (s, s * 1.15, s * 1.25)
    end_droplet.keyframe_insert(data_path='scale', frame=fr)

for fr, s in [(1, 0.01), (100, 0.01), (114, 0.42), (126, 0.92), (180, 0.92)]:
    drip1_drop.scale = (s, s, s * 1.35)
    drip1_drop.keyframe_insert(data_path='scale', frame=fr)

for fr, s in [(1, 0.01), (112, 0.01), (124, 0.35), (138, 0.82), (180, 0.82)]:
    drip2_drop.scale = (s, s, s * 1.30)
    drip2_drop.keyframe_insert(data_path='scale', frame=fr)

for fr, s in [(1, 0.01), (74, 0.20), (88, 0.32), (106, 0.24), (140, 0.18), (180, 0.14)]:
    nozzle_drip.scale = (s, s, s * 1.1)
    nozzle_drip.keyframe_insert(data_path='scale', frame=fr)

# Warm nozzle pulse
for fr, sc in [(1, 1.0), (60, 1.0), (78, 1.22), (98, 1.12), (118, 1.28), (138, 1.14), (160, 1.0), (180, 1.0)]:
    nozzle_orifice.scale = (sc, sc, sc)
    nozzle_orifice.keyframe_insert(data_path='scale', frame=fr)

# Subtle camera drift
for fr, loc, rot in [
    (1,   (6.2, -6.6, 3.6), (math.radians(66), 0, math.radians(43))),
    (120, (5.8, -6.0, 3.3), (math.radians(65), 0, math.radians(41))),
    (180, (6.0, -6.4, 3.5), (math.radians(66), 0, math.radians(42))),
]:
    camera.location = loc
    camera.rotation_euler = rot
    camera.keyframe_insert(data_path='location', frame=fr)
    camera.keyframe_insert(data_path='rotation_euler', frame=fr)

# Interpolation
for obj in [rig, glue_stick, trigger, end_droplet, drip1_drop, drip2_drop, nozzle_drip, nozzle_orifice, camera]:
    set_bezier_interpolation(obj)
for datablock in [glue_strip.data, drip1.data, drip2.data]:
    set_datablock_bezier_interpolation(datablock)

print('Animated hot glue gun scene created with a more realistic curved glue strand and multiple glue drips.')
