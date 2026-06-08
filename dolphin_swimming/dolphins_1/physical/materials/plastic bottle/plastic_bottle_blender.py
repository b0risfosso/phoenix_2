# plastic_bottle_blender.py
# Run in Blender with:
# blender --python plastic_bottle_blender.py

import bpy
import math

# ----------------------------
# Scene reset
# ----------------------------
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete()

scene = bpy.context.scene
scene.frame_start = 1
scene.frame_end = 120

# ----------------------------
# Helpers
# ----------------------------
def make_material(name, color, roughness=0.35, alpha=1.0, transmission=0.0):
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
        if "Transmission Weight" in bsdf.inputs:
            bsdf.inputs["Transmission Weight"].default_value = transmission
        elif "Transmission" in bsdf.inputs:
            bsdf.inputs["Transmission"].default_value = transmission
        if "Metallic" in bsdf.inputs:
            bsdf.inputs["Metallic"].default_value = 0.0

    mat.blend_method = "BLEND"
    mat.use_screen_refraction = True
    mat.show_transparent_back = True
    return mat


def add_cylinder(name, radius, depth, z, material, vertices=96):
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=vertices,
        radius=radius,
        depth=depth,
        location=(0, 0, z),
    )
    obj = bpy.context.object
    obj.name = name
    obj.data.materials.append(material)
    bpy.ops.object.shade_smooth()
    return obj

# ----------------------------
# Materials
# ----------------------------
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

# ----------------------------
# Bottle body from revolved profile
# ----------------------------
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
segments = 96

for radius, z in profile:
    for s in range(segments):
        angle = 2 * math.pi * s / segments
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        verts.append((x, y, z))

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

# ----------------------------
# Base rings
# ----------------------------
add_cylinder("raised circular base rim", 0.58, 0.045, -1.78, clear_plastic)
add_cylinder("inset bottom punt circle", 0.32, 0.035, -1.745, dark_groove_mat)

# ----------------------------
# Horizontal grip grooves
# Uses torus only because Blender includes mesh.primitive_torus_add.
# This is not VPython torus.
# ----------------------------
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
    bpy.ops.object.shade_smooth()

# ----------------------------
# Label band
# ----------------------------
label = add_cylinder("paper label wrapped around bottle", 0.692, 0.50, 0.03, label_mat, vertices=128)

# ----------------------------
# Water volume
# ----------------------------
water = add_cylinder("water volume inside transparent bottle", 0.58, 1.65, -0.70, water_mat, vertices=96)
water.name = "simple water fill volume"

# ----------------------------
# Neck rings and cap
# ----------------------------
add_cylinder("neck support ring", 0.255, 0.07, 1.35, clear_plastic)
add_cylinder("tamper ring below cap", 0.265, 0.08, 1.55, cap_plastic)
cap = add_cylinder("blue screw cap", 0.30, 0.38, 1.82, cap_plastic, vertices=96)

# Cap ridges
for s in range(32):
    angle = 2 * math.pi * s / 32
    x = 0.307 * math.cos(angle)
    y = 0.307 * math.sin(angle)
    bpy.ops.mesh.primitive_cube_add(location=(x, y, 1.82), rotation=(0, 0, angle))
    ridge = bpy.context.object
    ridge.name = "vertical cap grip ridge"
    ridge.dimensions = (0.025, 0.065, 0.32)
    ridge.data.materials.append(cap_plastic)

# ----------------------------
# Label text
# ----------------------------
bpy.ops.object.text_add(location=(0, -0.705, 0.05), rotation=(math.radians(90), 0, 0))
text = bpy.context.object
text.name = "front label text"
text.data.body = "WATER"
text.data.align_x = "CENTER"
text.data.align_y = "CENTER"
text.data.size = 0.22
text.data.extrude = 0.003
text_mat = make_material("blue printed label ink", (0.0, 0.15, 0.65, 1.0), roughness=0.55, alpha=1.0)
text.data.materials.append(text_mat)

# ----------------------------
# Lighting
# ----------------------------
bpy.ops.object.light_add(type="AREA", location=(0, -4, 5))
key_light = bpy.context.object
key_light.name = "large softbox reflection light"
key_light.data.energy = 650
key_light.data.size = 5

bpy.ops.object.light_add(type="POINT", location=(3, 2, 3))
rim_light = bpy.context.object
rim_light.name = "small rim highlight light"
rim_light.data.energy = 90

# ----------------------------
# Camera
# ----------------------------
bpy.ops.object.camera_add(location=(3.0, -5.0, 2.0), rotation=(math.radians(68), 0, math.radians(32)))
camera = bpy.context.object
camera.name = "camera plastic bottle view"
bpy.context.scene.camera = camera
camera.data.lens = 55

# ----------------------------
# Ground plane
# ----------------------------
ground_mat = make_material("matte light gray studio surface", (0.82, 0.82, 0.80, 1.0), roughness=0.65, alpha=1.0)
bpy.ops.mesh.primitive_plane_add(size=6, location=(0, 0, -1.83))
ground = bpy.context.object
ground.name = "studio ground plane"
ground.data.materials.append(ground_mat)

# ----------------------------
# Render settings
# ----------------------------
try:
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 96
except Exception:
    scene.render.engine = "BLENDER_EEVEE_NEXT"

scene.view_settings.view_transform = "Filmic"
scene.view_settings.look = "Medium High Contrast"
scene.world.color = (1.0, 1.0, 1.0)

for obj in bpy.context.scene.objects:
    obj.select_set(False)

body.select_set(True)
bpy.context.view_layer.objects.active = body

print("Initialized plastic bottle scene with transparent PET body, cap, label, water volume, grooves, lights, and camera.")
