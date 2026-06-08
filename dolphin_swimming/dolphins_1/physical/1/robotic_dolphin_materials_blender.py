#!/usr/bin/env python3
"""
Blender scene: materials-only layout for a robotic dolphin-inspired swimmer.

This script initializes a clean scene and shows only the materials needed for the
build. It does not animate an assembly stage, construction sequence, or play stage.

Run in Blender:
    blender --python robotic_dolphin_materials_blender.py

Or open Blender > Scripting > paste/run this file.
"""

import math
import bpy
from mathutils import Vector


# -----------------------------------------------------------------------------
# Scene utilities
# -----------------------------------------------------------------------------

def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def make_mat(name, color, roughness=0.55, metallic=0.0, alpha=1.0):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = color
        bsdf.inputs["Roughness"].default_value = roughness
        bsdf.inputs["Metallic"].default_value = metallic
        bsdf.inputs["Alpha"].default_value = alpha
    mat.diffuse_color = color
    if alpha < 1.0:
        mat.blend_method = "BLEND"
        mat.use_screen_refraction = True
        mat.show_transparent_back = True
    return mat


def add_cube(name, loc, scale, mat, bevel=0.0):
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=loc)
    obj = bpy.context.object
    obj.name = name
    obj.dimensions = scale
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    if mat:
        obj.data.materials.append(mat)
    if bevel > 0:
        mod = obj.modifiers.new("small rounded edges", "BEVEL")
        mod.width = bevel
        mod.segments = 5
        obj.modifiers.new("soft material shading", "WEIGHTED_NORMAL")
    return obj


def add_cylinder(name, loc, radius, depth, mat, vertices=48, rotation=(0, 0, 0), bevel=False):
    bpy.ops.mesh.primitive_cylinder_add(vertices=vertices, radius=radius, depth=depth, location=loc, rotation=rotation)
    obj = bpy.context.object
    obj.name = name
    if mat:
        obj.data.materials.append(mat)
    if bevel:
        mod = obj.modifiers.new("rounded rim", "BEVEL")
        mod.width = 0.035
        mod.segments = 5
        obj.modifiers.new("weighted normals", "WEIGHTED_NORMAL")
    return obj


def add_uvsphere(name, loc, scale, mat, segments=48, rings=24):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=segments, ring_count=rings, radius=1.0, location=loc)
    obj = bpy.context.object
    obj.name = name
    obj.scale = scale
    if mat:
        obj.data.materials.append(mat)
    obj.modifiers.new("smooth surface", "WEIGHTED_NORMAL")
    return obj


def add_text(name, text, loc, size=0.32, mat=None, align="CENTER"):
    bpy.ops.object.text_add(location=loc, rotation=(math.radians(72), 0, 0))
    obj = bpy.context.object
    obj.name = name
    obj.data.body = text
    obj.data.align_x = align
    obj.data.align_y = "CENTER"
    obj.data.size = size
    obj.data.font = obj.data.font
    if mat:
        obj.data.materials.append(mat)
    return obj


def add_label(text, loc, mat):
    return add_text("label - " + text[:32], text, loc, size=0.25, mat=mat)


def add_table_card(name, loc, size, mat):
    return add_cube(name, loc, size, mat, bevel=0.04)


def add_wire_curve(name, points, mat, bevel_depth=0.025):
    curve = bpy.data.curves.new(name, type="CURVE")
    curve.dimensions = "3D"
    curve.resolution_u = 10
    curve.bevel_depth = bevel_depth
    curve.bevel_resolution = 3
    spl = curve.splines.new("POLY")
    spl.points.add(len(points) - 1)
    for p, co in zip(spl.points, points):
        p.co = (co[0], co[1], co[2], 1.0)
    obj = bpy.data.objects.new(name, curve)
    bpy.context.collection.objects.link(obj)
    if mat:
        obj.data.materials.append(mat)
    return obj


# -----------------------------------------------------------------------------
# Material kit objects
# -----------------------------------------------------------------------------

def create_plastic_body_shell(mats):
    # Bottle-like waterproof shell: hollow-looking translucent cylinder plus nose cap.
    add_cylinder(
        "waterproof plastic body shell / bottle",
        loc=(-4.7, 2.0, 0.8),
        radius=0.48,
        depth=2.7,
        mat=mats["clear_blue"],
        vertices=64,
        rotation=(0, math.radians(90), 0),
        bevel=True,
    )
    add_uvsphere("rounded dolphin-nose cap", (-6.08, 2.0, 0.8), (0.50, 0.50, 0.50), mats["clear_blue"])
    add_cylinder("rear removable access cap", (-3.30, 2.0, 0.8), 0.50, 0.18, mats["dark_plastic"], rotation=(0, math.radians(90), 0), bevel=True)
    add_label("waterproof body shell", (-4.7, 2.0, 1.55), mats["ink"])


def create_electronics_group(mats):
    # Board, battery, switch, wires, and inner waterproof pouch.
    add_cube("inner waterproof electronics pouch", (-1.1, 2.0, 0.72), (2.0, 1.05, 0.12), mats["transparent_bag"], bevel=0.05)
    board = add_cube("microcontroller board", (-1.45, 2.0, 0.86), (0.90, 0.55, 0.08), mats["pcb_green"], bevel=0.02)
    # Pin headers and chips.
    for y in [1.76, 2.24]:
        for i in range(6):
            add_cube("pin header", (-1.80 + i * 0.14, y, 0.94), (0.045, 0.045, 0.045), mats["metal"], bevel=0.005)
    add_cube("black processor chip", (-1.45, 2.0, 0.94), (0.24, 0.20, 0.045), mats["black"], bevel=0.01)
    add_cube("battery pack", (-0.52, 2.0, 0.89), (0.72, 0.40, 0.24), mats["battery_black"], bevel=0.04)
    add_cube("small on off switch", (0.05, 2.0, 0.92), (0.30, 0.18, 0.12), mats["red"], bevel=0.02)
    add_wire_curve("red power wire", [(-0.35, 2.18, 0.97), (-0.85, 2.35, 0.98), (-1.35, 2.25, 0.96)], mats["red"], 0.014)
    add_wire_curve("black ground wire", [(-0.35, 1.82, 0.97), (-0.95, 1.65, 0.98), (-1.55, 1.77, 0.96)], mats["black"], 0.014)
    add_label("electronics + battery + switch", (-0.95, 2.0, 1.45), mats["ink"])


def create_tail_system(mats):
    # Servo, horn, linkage, flexible tail, and flukes.
    add_cube("servo motor case", (2.3, 2.0, 0.76), (0.75, 0.55, 0.48), mats["servo_blue"], bevel=0.05)
    add_cylinder("servo output shaft", (2.30, 2.0, 1.05), 0.10, 0.08, mats["metal"], vertices=32, bevel=True)
    add_cube("white servo horn", (2.30, 2.0, 1.15), (1.00, 0.10, 0.045), mats["white_plastic"], bevel=0.025)
    add_wire_curve("thin linkage rod", [(2.80, 2.0, 1.15), (3.35, 2.0, 1.02), (3.85, 2.0, 0.95)], mats["metal"], 0.018)
    add_cube("flexible tail strip", (4.40, 2.0, 0.78), (1.25, 0.16, 0.10), mats["rubber"], bevel=0.025)
    # Fluke pair angled as two soft paddles.
    left = add_cube("left soft tail fluke", (5.25, 1.72, 0.78), (0.72, 0.36, 0.07), mats["foam_tail"], bevel=0.04)
    left.rotation_euler[2] = math.radians(-23)
    right = add_cube("right soft tail fluke", (5.25, 2.28, 0.78), (0.72, 0.36, 0.07), mats["foam_tail"], bevel=0.04)
    right.rotation_euler[2] = math.radians(23)
    add_label("servo + flexible tail + soft flukes", (3.85, 2.0, 1.55), mats["ink"])


def create_waterproofing_group(mats):
    # Tape roll, silicone tube, gasket/rubber bands.
    add_cylinder("waterproof tape roll outer", (-4.6, -0.4, 0.75), 0.48, 0.26, mats["tape_blue"], vertices=64, bevel=True)
    add_cylinder("waterproof tape roll inner hole", (-4.6, -0.4, 0.76), 0.24, 0.285, mats["table"], vertices=64, bevel=True)
    add_cylinder("silicone sealant tube", (-3.35, -0.35, 0.75), 0.18, 1.25, mats["white_plastic"], vertices=48, rotation=(0, math.radians(78), 0), bevel=True)
    add_cylinder("silicone nozzle", (-2.73, -0.35, 0.92), 0.07, 0.42, mats["clear_silicone"], vertices=32, rotation=(0, math.radians(78), 0), bevel=True)
    for i, x in enumerate([-2.2, -1.95, -1.70]):
        add_cylinder(f"rubber gasket {i+1}", (x, -0.35, 0.75), 0.20, 0.05, mats["rubber"], vertices=48, bevel=True)
    add_label("waterproof tape, silicone, gaskets", (-3.25, -0.35, 1.42), mats["ink"])


def create_flotation_balance_group(mats):
    # Foam blocks and ballast washers/weights.
    for i, x in enumerate([-0.8, -0.28, 0.24]):
        add_cube(f"foam flotation block {i+1}", (x, -0.45, 0.78 + 0.03 * i), (0.42, 0.58, 0.25), mats["foam"], bevel=0.035)
    for i, x in enumerate([1.05, 1.25, 1.45, 1.65]):
        add_cylinder(f"ballast washer {i+1}", (x, -0.45, 0.70 + i * 0.025), 0.16, 0.035, mats["metal"], vertices=48, bevel=True)
    add_cube("small clay ballast", (2.10, -0.45, 0.74), (0.36, 0.24, 0.18), mats["clay"], bevel=0.06)
    add_label("foam flotation + ballast weights", (0.65, -0.45, 1.36), mats["ink"])


def create_testing_items(mats):
    # Shallow test tub and towel; these are materials/tools, not a play stage.
    add_cube("shallow water test tub", (4.2, -0.45, 0.55), (2.30, 1.25, 0.18), mats["clear_blue"], bevel=0.08)
    add_cube("dry towel", (4.2, -1.35, 0.61), (2.05, 0.55, 0.07), mats["towel"], bevel=0.03)
    add_label("test tub + towel", (4.2, -0.85, 1.18), mats["ink"])


def create_reference_cards(mats):
    add_text(
        "title",
        "Robotic Dolphin Swimmer — Materials Only",
        (0.0, 3.35, 0.05),
        size=0.42,
        mat=mats["ink"],
    )
    add_text(
        "note no animation",
        "Scene shows loose build materials: shell, electronics, servo, flexible tail, flukes, waterproofing, flotation, ballast, and testing tools. No assembly animation is included.",
        (0.0, -2.15, 0.05),
        size=0.20,
        mat=mats["ink"],
    )


# -----------------------------------------------------------------------------
# Scene setup
# -----------------------------------------------------------------------------

def setup_camera_and_lights():
    bpy.ops.object.light_add(type="AREA", location=(0, -5.5, 7.0))
    light = bpy.context.object
    light.name = "large softbox light"
    light.data.energy = 520
    light.data.size = 6.5

    bpy.ops.object.camera_add(location=(0.0, -7.4, 5.2), rotation=(math.radians(58), 0, 0))
    bpy.context.scene.camera = bpy.context.object
    bpy.context.object.name = "camera - materials overview"

    bpy.context.scene.render.engine = "CYCLES"
    bpy.context.scene.cycles.samples = 80
    bpy.context.scene.view_settings.view_transform = "Filmic"
    bpy.context.scene.view_settings.look = "Medium High Contrast"
    bpy.context.scene.render.resolution_x = 1800
    bpy.context.scene.render.resolution_y = 1200


def build_scene():
    clear_scene()

    mats = {
        "table": make_mat("mat warm light table", (0.84, 0.88, 0.86, 1), roughness=0.7),
        "ink": make_mat("mat dark label ink", (0.04, 0.07, 0.09, 1), roughness=0.8),
        "clear_blue": make_mat("mat translucent waterproof plastic", (0.45, 0.75, 1.0, 0.38), roughness=0.16, alpha=0.38),
        "transparent_bag": make_mat("mat clear electronics pouch", (0.90, 0.97, 1.0, 0.28), roughness=0.12, alpha=0.28),
        "dark_plastic": make_mat("mat dark plastic cap", (0.05, 0.07, 0.08, 1), roughness=0.45),
        "pcb_green": make_mat("mat green circuit board", (0.02, 0.38, 0.17, 1), roughness=0.46),
        "battery_black": make_mat("mat black battery pack", (0.02, 0.02, 0.025, 1), roughness=0.5),
        "black": make_mat("mat black rubber plastic", (0.0, 0.0, 0.0, 1), roughness=0.55),
        "red": make_mat("mat red electrical part", (0.92, 0.07, 0.04, 1), roughness=0.4),
        "metal": make_mat("mat dull metal", (0.62, 0.62, 0.60, 1), roughness=0.35, metallic=0.75),
        "servo_blue": make_mat("mat blue servo case", (0.08, 0.30, 0.75, 1), roughness=0.42),
        "white_plastic": make_mat("mat white plastic", (0.96, 0.96, 0.91, 1), roughness=0.45),
        "rubber": make_mat("mat flexible black rubber", (0.015, 0.016, 0.014, 1), roughness=0.82),
        "foam_tail": make_mat("mat soft gray tail foam", (0.55, 0.62, 0.66, 1), roughness=0.9),
        "tape_blue": make_mat("mat blue waterproof tape", (0.07, 0.44, 0.86, 1), roughness=0.38),
        "clear_silicone": make_mat("mat clear silicone", (0.85, 0.95, 1.0, 0.55), roughness=0.2, alpha=0.55),
        "foam": make_mat("mat white flotation foam", (0.96, 0.98, 0.93, 1), roughness=0.95),
        "clay": make_mat("mat gray modeling clay", (0.50, 0.48, 0.44, 1), roughness=0.85),
        "towel": make_mat("mat folded towel", (0.68, 0.82, 0.91, 1), roughness=0.98),
    }

    add_table_card("single display table", (0, 0, 0.43), (11.8, 5.4, 0.12), mats["table"])
    create_plastic_body_shell(mats)
    create_electronics_group(mats)
    create_tail_system(mats)
    create_waterproofing_group(mats)
    create_flotation_balance_group(mats)
    create_testing_items(mats)
    create_reference_cards(mats)
    setup_camera_and_lights()

    # Set origin and view clipping for convenience.
    for obj in bpy.context.scene.objects:
        obj.select_set(False)
    bpy.context.scene.frame_set(1)


if __name__ == "__main__":
    build_scene()
