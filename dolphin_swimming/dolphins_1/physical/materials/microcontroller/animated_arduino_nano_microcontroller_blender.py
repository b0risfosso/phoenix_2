
import bpy
import math
from mathutils import Vector

# ============================================================
# Animated Arduino Nano Style Microcontroller
# Clears scene, initializes a small microcontroller board, and
# animates power-up, LED blinking, signal flow, pin activity,
# and subtle board motion.
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
                  alpha=1.0, emission_strength=0.0):
    mat = bpy.data.materials.new(name=name)
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
        elif "Emission" in bsdf.inputs and emission_strength > 0:
            bsdf.inputs["Emission"].default_value = color

    if alpha < 1.0:
        mat.blend_method = "BLEND"
        try:
            mat.shadow_method = "HASHED"
        except Exception:
            pass
    return mat


def assign_material(obj, mat):
    if obj.data and hasattr(obj.data, "materials"):
        if len(obj.data.materials) == 0:
            obj.data.materials.append(mat)
        else:
            obj.data.materials[0] = mat


def parent_keep_transform(child, parent):
    child.parent = parent
    child.matrix_parent_inverse = parent.matrix_world.inverted()


def get_action_fcurves(action):
    if action is None:
        return []
    fcurves = getattr(action, "fcurves", None)
    if fcurves is not None:
        return fcurves
    layers = getattr(action, "layers", None)
    if layers:
        found = []
        for layer in layers:
            strips = getattr(layer, "strips", [])
            for strip in strips:
                channelbag = getattr(strip, "channelbag", None)
                if channelbag:
                    fc = getattr(channelbag, "fcurves", None)
                    if fc:
                        found.extend(fc)
        return found
    return []


def set_interpolation(obj, interpolation="BEZIER"):
    ad = getattr(obj, "animation_data", None)
    if not ad:
        return
    for fc in get_action_fcurves(getattr(ad, "action", None)):
        for kp in fc.keyframe_points:
            kp.interpolation = interpolation


def set_data_interpolation(data_block, interpolation="BEZIER"):
    ad = getattr(data_block, "animation_data", None)
    if not ad:
        return
    for fc in get_action_fcurves(getattr(ad, "action", None)):
        for kp in fc.keyframe_points:
            kp.interpolation = interpolation


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


def add_curve(name, points, bevel_depth=0.01, resolution=16, material=None):
    curve_data = bpy.data.curves.new(name + "_curve", type="CURVE")
    curve_data.dimensions = "3D"
    curve_data.resolution_u = resolution
    curve_data.bevel_depth = bevel_depth
    curve_data.bevel_resolution = 4
    spline = curve_data.splines.new("BEZIER")
    spline.bezier_points.add(len(points) - 1)
    for i, p in enumerate(points):
        bp = spline.bezier_points[i]
        bp.co = Vector(p)
        bp.handle_left_type = "AUTO"
        bp.handle_right_type = "AUTO"
    obj = bpy.data.objects.new(name, curve_data)
    bpy.context.collection.objects.link(obj)
    if material:
        assign_material(obj, material)
    return obj


def add_text(name, body, location, size=0.14, rotation=(math.radians(90), 0, 0), material=None):
    bpy.ops.object.text_add(location=location, rotation=rotation)
    obj = bpy.context.object
    obj.name = name
    obj.data.body = body
    obj.data.align_x = "CENTER"
    obj.data.align_y = "CENTER"
    obj.data.size = size
    obj.data.extrude = 0.002
    if material:
        assign_material(obj, material)
    return obj


# ----------------------------
# Materials
# ----------------------------
mat_board = make_material("ArduinoNano_Blue_PCB", (0.02, 0.18, 0.70, 1.0), roughness=0.46)
mat_board_edge = make_material("PCB_Edge_Darker_Blue", (0.01, 0.08, 0.32, 1.0), roughness=0.55)
mat_metal = make_material("Silver_Metal", (0.78, 0.80, 0.82, 1.0), roughness=0.22, metallic=0.9)
mat_pin_black = make_material("Black_Pin_Header_Plastic", (0.025, 0.025, 0.03, 1.0), roughness=0.70)
mat_chip = make_material("Black_IC_Chip", (0.035, 0.035, 0.04, 1.0), roughness=0.62)
mat_usb = make_material("USB_Mini_Port_Metal", (0.72, 0.74, 0.76, 1.0), roughness=0.28, metallic=0.75)
mat_gold = make_material("Gold_Contacts", (1.0, 0.70, 0.20, 1.0), roughness=0.28, metallic=0.9)
mat_green_led = make_material("Green_LED_Off", (0.08, 0.35, 0.12, 1.0), roughness=0.25)
mat_green_led_on = make_material("Green_LED_Glow", (0.10, 1.00, 0.25, 1.0), roughness=0.08, emission_strength=3.5)
mat_red_led_on = make_material("Red_LED_Glow", (1.00, 0.10, 0.08, 1.0), roughness=0.08, emission_strength=3.2)
mat_signal = make_material("Cyan_Signal_Glow", (0.10, 0.80, 1.00, 1.0), roughness=0.14, emission_strength=2.5)
mat_trace = make_material("Pale_Copper_Traces", (0.95, 0.62, 0.22, 1.0), roughness=0.38, metallic=0.35)
mat_text = make_material("White_Silkscreen_Text", (0.95, 0.95, 0.92, 1.0), roughness=0.75)
mat_floor = make_material("Floor", (0.94, 0.94, 0.96, 1.0), roughness=0.85)

# ----------------------------
# Root controller
# ----------------------------
bpy.ops.object.empty_add(type="PLAIN_AXES", location=(0, 0, 0))
rig = bpy.context.object
rig.name = "Arduino_Nano_Style_Microcontroller_Rig"

# ----------------------------
# Board body
# ----------------------------
board = add_cube("Blue_PCB_Board", location=(0, 0, 0.18), scale=(1.65, 0.62, 0.055), material=mat_board)
board_edge_front = add_cube("PCB_Front_Edge", location=(0, -0.63, 0.18), scale=(1.66, 0.018, 0.06), material=mat_board_edge)
board_edge_back = add_cube("PCB_Back_Edge", location=(0, 0.63, 0.18), scale=(1.66, 0.018, 0.06), material=mat_board_edge)
for obj in [board, board_edge_front, board_edge_back]:
    parent_keep_transform(obj, rig)

# Mounting holes as silver rings with dark centers
for i, x in enumerate([-1.32, 1.32]):
    for j, y in enumerate([-0.42, 0.42]):
        ring = add_cylinder(f"Mounting_Hole_Ring_{i}_{j}", radius=0.075, depth=0.012, location=(x, y, 0.245), material=mat_metal, vertices=36)
        ring.rotation_euler[0] = 0
        center = add_cylinder(f"Mounting_Hole_Dark_Center_{i}_{j}", radius=0.045, depth=0.014, location=(x, y, 0.253), material=mat_chip, vertices=32)
        parent_keep_transform(ring, rig)
        parent_keep_transform(center, rig)

# USB mini/micro connector at one end
usb_body = add_cube("USB_Port_Metal_Shell", location=(-1.62, 0, 0.33), scale=(0.20, 0.30, 0.13), material=mat_usb)
usb_dark = add_cube("USB_Port_Dark_Insert", location=(-1.75, 0, 0.33), scale=(0.035, 0.22, 0.075), material=mat_chip)
for obj in [usb_body, usb_dark]:
    parent_keep_transform(obj, rig)

# Main microcontroller chip and small components
chip = add_cube("ATmega_Style_Main_Chip", location=(0.02, 0.0, 0.34), scale=(0.48, 0.28, 0.075), material=mat_chip)
regulator = add_cube("Voltage_Regulator", location=(-0.78, -0.18, 0.32), scale=(0.18, 0.12, 0.055), material=mat_chip)
oscillator = add_cube("Crystal_Oscillator", location=(0.68, 0.20, 0.31), scale=(0.18, 0.08, 0.045), material=mat_metal)
reset_button = add_cube("Reset_Button", location=(-0.82, 0.24, 0.34), scale=(0.13, 0.13, 0.045), material=mat_metal)
for obj in [chip, regulator, oscillator, reset_button]:
    parent_keep_transform(obj, rig)

# Chip pins as tiny metal legs
for side_y in [-0.32, 0.32]:
    for i in range(8):
        x = -0.32 + i * 0.092
        leg = add_cube(f"Chip_Pin_{side_y}_{i}", location=(x, side_y, 0.34), scale=(0.020, 0.060, 0.013), material=mat_metal)
        parent_keep_transform(leg, rig)

# Pin headers down both sides
pin_objects = []
for side_y in [-0.52, 0.52]:
    for i in range(15):
        x = -1.20 + i * 0.17
        plastic = add_cube(f"Header_Plastic_{side_y}_{i}", location=(x, side_y, 0.32), scale=(0.052, 0.052, 0.055), material=mat_pin_black)
        metal_pin = add_cylinder(f"Header_Pin_{side_y}_{i}", radius=0.015, depth=0.30, location=(x, side_y, 0.48), material=mat_metal, vertices=16)
        pin_objects.extend([plastic, metal_pin])
        parent_keep_transform(plastic, rig)
        parent_keep_transform(metal_pin, rig)

# Copper/gold pads near pins
for side_y in [-0.43, 0.43]:
    for i in range(15):
        x = -1.20 + i * 0.17
        pad = add_cube(f"Gold_Pin_Pad_{side_y}_{i}", location=(x, side_y, 0.255), scale=(0.046, 0.028, 0.006), material=mat_gold)
        parent_keep_transform(pad, rig)

# PCB traces
trace_points = [
    [(-1.20, -0.40, 0.262), (-0.60, -0.30, 0.262), (-0.20, -0.18, 0.262), (0.02, -0.15, 0.262)],
    [(-0.50, 0.40, 0.263), (-0.10, 0.28, 0.263), (0.20, 0.16, 0.263), (0.45, 0.10, 0.263)],
    [(0.95, -0.42, 0.264), (0.72, -0.25, 0.264), (0.48, -0.12, 0.264), (0.16, -0.05, 0.264)],
    [(1.18, 0.42, 0.264), (0.90, 0.30, 0.264), (0.68, 0.20, 0.264)],
]
traces = []
for i, pts in enumerate(trace_points):
    t = add_curve(f"PCB_Trace_{i+1}", pts, bevel_depth=0.006, material=mat_trace)
    parent_keep_transform(t, rig)
    traces.append(t)

# LEDs
power_led = add_uv_sphere("Power_LED", radius=0.055, location=(-0.46, 0.22, 0.36), material=mat_green_led)
tx_led = add_uv_sphere("TX_LED", radius=0.040, location=(0.82, -0.18, 0.35), material=mat_green_led)
rx_led = add_uv_sphere("RX_LED", radius=0.040, location=(0.82, -0.02, 0.35), material=mat_green_led)
for obj in [power_led, tx_led, rx_led]:
    parent_keep_transform(obj, rig)

# Silkscreen labels
labels = [
    ("Nano_Label", "NANO", (0.00, -0.02, 0.405), 0.16),
    ("USB_Label", "USB", (-1.18, 0.22, 0.305), 0.075),
    ("VIN_Label", "VIN", (1.15, -0.36, 0.305), 0.060),
    ("GND_Label", "GND", (1.15, 0.36, 0.305), 0.055),
    ("D_Label", "D0-D13", (0.0, 0.55, 0.305), 0.055),
    ("A_Label", "A0-A7", (0.0, -0.55, 0.305), 0.055),
]
for name, body, loc, size in labels:
    txt = add_text(name, body, loc, size=size, material=mat_text)
    parent_keep_transform(txt, rig)

# Signal path curves hidden for electron/signal followers
signal_paths = []
path_defs = [
    [(-1.55, 0.0, 0.42), (-0.95, 0.0, 0.43), (-0.45, 0.12, 0.43), (0.02, 0.05, 0.43), (0.82, -0.18, 0.43), (1.15, -0.52, 0.50)],
    [(0.02, 0.05, 0.43), (0.48, 0.15, 0.43), (0.82, -0.02, 0.43), (1.15, 0.52, 0.50)],
    [(-0.46, 0.22, 0.43), (-0.20, 0.18, 0.43), (0.02, 0.05, 0.43), (0.55, -0.20, 0.43), (0.95, -0.52, 0.50)],
]
for i, pts in enumerate(path_defs):
    p = add_curve(f"Hidden_Signal_Path_{i+1}", pts, bevel_depth=0.0, material=None)
    p.hide_viewport = True
    p.hide_render = True
    parent_keep_transform(p, rig)
    signal_paths.append(p)

# Visible signal pulses as glowing spheres following paths
pulse_objects = []
for path_i, path in enumerate(signal_paths):
    for j in range(3):
        pulse = add_uv_sphere(f"Signal_Pulse_{path_i+1}_{j+1}", radius=0.040, location=(0, 0, 0), material=mat_signal)
        con = pulse.constraints.new(type="FOLLOW_PATH")
        con.target = path
        con.use_fixed_location = True
        con.forward_axis = "FORWARD_X"
        con.up_axis = "UP_Z"
        start_frame = 45 + path_i * 12 + j * 20
        end_frame = start_frame + 90
        con.offset_factor = 0.0
        con.keyframe_insert(data_path="offset_factor", frame=start_frame)
        con.offset_factor = 1.0
        con.keyframe_insert(data_path="offset_factor", frame=end_frame)
        parent_keep_transform(pulse, rig)
        pulse_objects.append(pulse)

# ----------------------------
# Ground, camera, lights
# ----------------------------
bpy.ops.mesh.primitive_plane_add(size=8, location=(0, 0, 0))
floor = bpy.context.object
floor.name = "Ground"
assign_material(floor, mat_floor)

bpy.ops.object.camera_add(location=(4.8, -5.2, 3.1), rotation=(math.radians(61), 0, math.radians(41)))
camera = bpy.context.object
camera.name = "Camera"
camera.data.lens = 58
scene.camera = camera

bpy.ops.object.light_add(type="AREA", location=(3.8, -4.3, 5.3))
key = bpy.context.object
key.name = "Key_Light"
key.data.energy = 1700
key.data.size = 4.8

bpy.ops.object.light_add(type="AREA", location=(-3.5, 2.8, 3.2))
fill = bpy.context.object
fill.name = "Fill_Light"
fill.data.energy = 750
fill.data.size = 4.0

bpy.ops.object.light_add(type="POINT", location=(1.5, -1.2, 2.2))
rim = bpy.context.object
rim.name = "Signal_Glow_Light"
rim.data.energy = 120

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

# ----------------------------
# Animation
# ----------------------------
# Board motion: lift, tilt, settle
for fr, loc, rot in [
    (1,   (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
    (40,  (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
    (90,  (0.05, -0.04, 0.20), (math.radians(3), math.radians(-5), math.radians(2))),
    (140, (-0.05, 0.03, 0.25), (math.radians(-2), math.radians(5), math.radians(-2))),
    (190, (0.03, 0.00, 0.16), (math.radians(2), math.radians(-3), math.radians(1))),
    (240, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
]:
    rig.location = loc
    rig.rotation_euler = rot
    rig.keyframe_insert(data_path="location", frame=fr)
    rig.keyframe_insert(data_path="rotation_euler", frame=fr)

# Power LED grows/pulses
for fr, s, mat in [
    (1, 0.55, mat_green_led),
    (35, 0.55, mat_green_led),
    (48, 1.25, mat_green_led_on),
    (75, 0.95, mat_green_led_on),
    (120, 1.15, mat_green_led_on),
    (180, 1.00, mat_green_led_on),
    (240, 0.85, mat_green_led_on),
]:
    power_led.scale = (s, s, s)
    power_led.keyframe_insert(data_path="scale", frame=fr)
    if fr in [1, 48]:
        assign_material(power_led, mat)

# TX/RX blinking LEDs
for led, on_mat, offset in [(tx_led, mat_green_led_on, 0), (rx_led, mat_red_led_on, 10)]:
    for fr, s in [
        (1, 0.45),
        (55 + offset, 0.45),
        (65 + offset, 1.15),
        (78 + offset, 0.45),
        (108 + offset, 1.20),
        (122 + offset, 0.45),
        (150 + offset, 1.10),
        (166 + offset, 0.45),
        (210 + offset, 1.05),
        (230 + offset, 0.45),
    ]:
        led.scale = (s, s, s)
        led.keyframe_insert(data_path="scale", frame=fr)
    assign_material(led, on_mat)

# Pin headers pulse upward/downward slightly
for idx, obj in enumerate(pin_objects):
    if "Header_Pin" in obj.name:
        original_z = obj.location.z
        delay = idx % 18
        for fr, zadd in [
            (1, 0.0),
            (70 + delay, 0.0),
            (82 + delay, 0.07),
            (100 + delay, 0.0),
            (150 + delay, 0.05),
            (170 + delay, 0.0),
            (240, 0.0),
        ]:
            obj.location.z = original_z + zadd
            obj.keyframe_insert(data_path="location", frame=min(fr, 240))

# Signal pulse visibility/scaling
for i, pulse in enumerate(pulse_objects):
    start = 45 + (i % 3) * 18
    for fr, s in [
        (1, 0.01),
        (start, 0.01),
        (start + 8, 1.0),
        (start + 78, 1.0),
        (start + 92, 0.01),
        (240, 0.01),
    ]:
        pulse.scale = (s, s, s)
        pulse.keyframe_insert(data_path="scale", frame=min(fr, 240))

# Camera drift
for fr, loc, rot in [
    (1, (4.8, -5.2, 3.1), (math.radians(61), 0, math.radians(41))),
    (120, (4.5, -4.8, 2.9), (math.radians(60), 0, math.radians(39))),
    (240, (4.7, -5.0, 3.0), (math.radians(61), 0, math.radians(40))),
]:
    camera.location = loc
    camera.rotation_euler = rot
    camera.keyframe_insert(data_path="location", frame=fr)
    camera.keyframe_insert(data_path="rotation_euler", frame=fr)

# Interpolation
for obj in [rig, camera, power_led, tx_led, rx_led] + pin_objects + pulse_objects:
    set_interpolation(obj, "BEZIER")

# Constraint keyframes for signal pulses should move linearly
for pulse in pulse_objects:
    ad = getattr(pulse, "animation_data", None)
    if ad and ad.action:
        for fc in get_action_fcurves(ad.action):
            if "offset_factor" in fc.data_path:
                for kp in fc.keyframe_points:
                    kp.interpolation = "LINEAR"

print("Animated Arduino Nano style microcontroller scene created: board powers up, LEDs blink, pins pulse, and signal particles travel across the board.")
