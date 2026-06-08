"""
Blender Script: Suspended Bird Puppet - Raw Materials to Flight Space

Purpose:
    Create a staged Blender scene showing a suspended bird puppet build beginning
    from raw physical materials: cardboard, paper, string, tape, glue, dowels,
    balance weights, labels, and frame pieces.

    The scene progresses visually from:
        Stage 0 - raw materials on a work table
        Stage 1 - cut body, wing, and tail pieces
        Stage 2 - assembled bird puppet
        Stage 3 - strings and balance weights
        Stage 4 - overhead support frame
        Stage 5 - 3D coordinate space with origin, axes, and octant labels
        Stage 6 - bird suspended and ready for flight animation

Run:
    Open Blender -> Scripting -> paste this file -> Run Script

Notes:
    This script uses only standard Blender Python APIs.
"""

import bpy
import math
from mathutils import Vector


# ============================================================
# Configuration
# ============================================================

SCENE_NAME = "Suspended Bird Puppet - Raw Materials to Flight Space"

# Set this to a value from 0 to 6 to display only part of the build.
# Leave as 6 to include the full staged build.
BUILD_STAGE = 6

# If True, the timeline shows:
# Stage 0 appears, disappears -> Stage 1 appears, disappears -> ...
# Stage 6 appears -> finished build remains -> bird movement animation begins.
ANIMATE_BUILD_SEQUENCE = True

# Frame windows for each construction stage.
# Each stage is visible only inside its window, except final setup stages can be held.
STAGE_FRAME_WINDOWS = {
    0: (1, 35),       # raw materials
    1: (36, 70),      # cut pieces
    2: (71, 105),     # assembled bird
    3: (106, 140),    # strings and balance
    4: (141, 175),    # overhead frame
    5: (176, 210),    # coordinate space
    6: (211, 245),    # flight path guides / completed scene reveal
}

# Finished build and flight animation timing.
FINISHED_BUILD_HOLD_START = 246
BIRD_FLIGHT_START_FRAME = 270
BIRD_FLIGHT_END_FRAME = 420

# Scene scale.
TABLE_Z = 0.0
BIRD_Z = 3.0
FRAME_TOP_Z = 5.8

# Coordinate-space dimensions.
AXIS_LENGTH = 5.0
OCTANT_LABEL_DISTANCE = 4.1


# ============================================================
# Basic Helpers
# ============================================================

def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def set_units_and_scene():
    bpy.context.scene.name = SCENE_NAME
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = 420
    bpy.context.scene.render.fps = 24

    # Light, readable viewport/render setup.
    bpy.context.scene.world = bpy.data.worlds.new("Light Workshop World") if bpy.context.scene.world is None else bpy.context.scene.world
    bpy.context.scene.world.color = (0.93, 0.95, 0.98)


def make_material(name, color, roughness=0.55, metallic=0.0, alpha=1.0):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = color

    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        try:
            bsdf.inputs["Base Color"].default_value = color
            bsdf.inputs["Roughness"].default_value = roughness
            bsdf.inputs["Metallic"].default_value = metallic
            bsdf.inputs["Alpha"].default_value = alpha
        except Exception:
            pass

    if alpha < 1.0:
        mat.blend_method = "BLEND"
        mat.use_screen_refraction = True
        mat.show_transparent_back = True

    return mat


def create_materials():
    mats = {}

    # Raw build materials.
    mats["cardboard"] = make_material("Raw cardboard / foam board", (0.78, 0.62, 0.39, 1.0))
    mats["paper"] = make_material("White paper sheets", (0.96, 0.95, 0.88, 1.0))
    mats["wing_paper"] = make_material("Thin wing paper", (0.92, 0.93, 0.86, 1.0))
    mats["tail_paper"] = make_material("Tail paper", (0.88, 0.90, 0.78, 1.0))
    mats["tape"] = make_material("Semi transparent tape", (0.95, 0.88, 0.55, 0.42), alpha=0.42)
    mats["string"] = make_material("Fishing line / string", (0.08, 0.08, 0.08, 0.55), roughness=0.2, alpha=0.55)
    mats["glue"] = make_material("Glue bottle plastic", (0.85, 0.90, 1.0, 1.0))
    mats["glue_cap"] = make_material("Glue cap", (0.16, 0.32, 0.95, 1.0))
    mats["wood"] = make_material("Wooden dowel / frame", (0.58, 0.38, 0.18, 1.0))
    mats["metal"] = make_material("Metal hook / balance weights", (0.55, 0.55, 0.57, 1.0), metallic=0.5)
    mats["black"] = make_material("Dark label text", (0.02, 0.02, 0.02, 1.0))

    # Coordinate materials.
    mats["floor"] = make_material("Light floor", (0.88, 0.90, 0.89, 1.0))
    mats["grid"] = make_material("Soft grid line", (0.45, 0.45, 0.45, 1.0))
    mats["origin"] = make_material("Origin marker", (1.0, 0.95, 0.2, 1.0))
    mats["x_axis"] = make_material("X axis", (0.95, 0.20, 0.20, 1.0))
    mats["y_axis"] = make_material("Y axis", (0.20, 0.75, 0.25, 1.0))
    mats["z_axis"] = make_material("Z axis", (0.20, 0.40, 1.0, 1.0))
    mats["target"] = make_material("Octant target marker", (1.0, 0.55, 0.15, 1.0))

    return mats


def assign_material(obj, mat):
    obj.data.materials.clear()
    obj.data.materials.append(mat)


def add_cube(name, location, scale, mat=None):
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=location)
    obj = bpy.context.object
    obj.name = name
    obj.dimensions = scale
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    if mat:
        assign_material(obj, mat)
    return obj


def add_cylinder(name, location, radius, depth, mat=None, vertices=32, rotation=(0, 0, 0)):
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=vertices,
        radius=radius,
        depth=depth,
        location=location,
        rotation=rotation,
    )
    obj = bpy.context.object
    obj.name = name
    if mat:
        assign_material(obj, mat)
    return obj


def add_uv_sphere(name, location, scale=(1, 1, 1), mat=None, segments=32, rings=16):
    bpy.ops.mesh.primitive_uv_sphere_add(
        segments=segments,
        ring_count=rings,
        radius=1.0,
        location=location,
    )
    obj = bpy.context.object
    obj.name = name
    obj.scale = scale
    if mat:
        assign_material(obj, mat)
    return obj


def add_torus(name, location, major_radius, minor_radius, mat=None, rotation=(0, 0, 0)):
    bpy.ops.mesh.primitive_torus_add(
        major_radius=major_radius,
        minor_radius=minor_radius,
        major_segments=64,
        minor_segments=12,
        location=location,
        rotation=rotation,
    )
    obj = bpy.context.object
    obj.name = name
    if mat:
        assign_material(obj, mat)
    return obj


def create_curve_line(name, points, mat=None, bevel_depth=0.015):
    curve = bpy.data.curves.new(name, "CURVE")
    curve.dimensions = "3D"
    curve.resolution_u = 2
    curve.bevel_depth = bevel_depth
    curve.bevel_resolution = 3

    polyline = curve.splines.new("POLY")
    polyline.points.add(len(points) - 1)

    for point, co in zip(polyline.points, points):
        point.co = (co[0], co[1], co[2], 1.0)

    obj = bpy.data.objects.new(name, curve)
    bpy.context.collection.objects.link(obj)
    if mat:
        curve.materials.append(mat)
    return obj


def add_text(name, text, location, size=0.25, mat=None, rotation=(math.radians(70), 0, 0)):
    bpy.ops.object.text_add(location=location, rotation=rotation)
    obj = bpy.context.object
    obj.name = name
    obj.data.body = text
    obj.data.align_x = "CENTER"
    obj.data.align_y = "CENTER"
    obj.data.size = size
    obj.data.extrude = 0.004
    if mat:
        assign_material(obj, mat)
    return obj


def create_collection(name):
    collection = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(collection)
    return collection


def move_to_collection(obj, collection):
    for c in obj.users_collection:
        c.objects.unlink(obj)
    collection.objects.link(obj)


def put_many_in_collection(objects, collection):
    for obj in objects:
        move_to_collection(obj, collection)


# ============================================================
# Mesh Helpers for Cut Shapes
# ============================================================

def make_flat_polygon(name, verts_2d, location, mat=None, thickness=0.025, rotation=(0, 0, 0)):
    """
    Creates a flat polygon mesh in the local XY plane, then gives it a thin solidify modifier.
    """
    verts = [(x, y, 0.0) for x, y in verts_2d]
    faces = [list(range(len(verts)))]
    mesh = bpy.data.meshes.new(name + "_Mesh")
    mesh.from_pydata(verts, [], faces)
    mesh.update()

    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    obj.location = location
    obj.rotation_euler = rotation

    if mat:
        assign_material(obj, mat)

    solid = obj.modifiers.new("thin_material_thickness", "SOLIDIFY")
    solid.thickness = thickness
    solid.offset = 0.0

    return obj


def make_bird_body_shape(name, location, mat):
    verts = [
        (-0.90, 0.00),
        (-0.65, 0.24),
        (-0.15, 0.34),
        (0.45, 0.25),
        (0.85, 0.04),
        (0.65, -0.16),
        (0.10, -0.30),
        (-0.55, -0.25),
    ]
    return make_flat_polygon(name, verts, location, mat, thickness=0.05, rotation=(math.radians(90), 0, 0))


def make_wing_shape(name, side, location, mat):
    sign = 1 if side == "left" else -1
    verts = [
        (0.00, 0.00),
        (1.25 * sign, 0.35),
        (1.80 * sign, 0.05),
        (1.05 * sign, -0.35),
    ]
    return make_flat_polygon(name, verts, location, mat, thickness=0.025, rotation=(math.radians(90), 0, 0))


def make_tail_shape(name, location, mat):
    verts = [
        (0.00, 0.00),
        (-0.35, 0.42),
        (-0.62, 0.20),
        (-0.48, 0.00),
        (-0.62, -0.20),
        (-0.35, -0.42),
    ]
    return make_flat_polygon(name, verts, location, mat, thickness=0.025, rotation=(math.radians(90), 0, 0))


def make_label_card(name, location, text, mats):
    card = add_cube(name + "_Card", location, (0.75, 0.04, 0.35), mats["paper"])
    label = add_text(
        name + "_Text",
        text,
        (location[0], location[1] - 0.035, location[2] + 0.005),
        size=0.16,
        mat=mats["black"],
        rotation=(math.radians(90), 0, 0),
    )
    return [card, label]


# ============================================================
# Stage 0: Raw Materials
# ============================================================

def stage_0_raw_materials(mats):
    collection = create_collection("Stage 0 - Raw Materials")
    objs = []

    # Work table.
    objs.append(add_cube("Workshop_Table", (0, -4.4, TABLE_Z - 0.08), (6.5, 3.0, 0.16), mats["wood"]))

    # Raw cardboard / foam sheets.
    objs.append(add_cube("Raw_Cardboard_Sheet_A", (-2.15, -4.6, TABLE_Z + 0.05), (1.9, 1.2, 0.035), mats["cardboard"]))
    objs.append(add_cube("Raw_Cardboard_Sheet_B", (-1.85, -4.25, TABLE_Z + 0.10), (1.6, 1.0, 0.03), mats["cardboard"]))

    # Paper sheets.
    for i in range(4):
        objs.append(add_cube(
            f"Raw_Paper_Sheet_{i+1}",
            (0.15 + 0.04*i, -4.55 + 0.035*i, TABLE_Z + 0.05 + 0.012*i),
            (1.25, 0.85, 0.012),
            mats["paper"],
        ))

    # Tape roll.
    tape_roll = add_torus("Tape_Roll", (1.65, -4.55, TABLE_Z + 0.14), 0.24, 0.055, mats["tape"], rotation=(math.radians(90), 0, 0))
    objs.append(tape_roll)

    # String coil: several rings.
    for i, radius in enumerate([0.25, 0.20, 0.15]):
        objs.append(add_torus(
            f"String_Coil_Ring_{i+1}",
            (2.35, -4.55, TABLE_Z + 0.08 + i*0.012),
            radius,
            0.012,
            mats["string"],
            rotation=(math.radians(90), 0, 0),
        ))

    # Glue bottle.
    objs.append(add_cylinder("Glue_Bottle_Body", (2.95, -4.5, TABLE_Z + 0.28), 0.15, 0.48, mats["glue"], vertices=32))
    objs.append(add_cylinder("Glue_Bottle_Cap", (2.95, -4.5, TABLE_Z + 0.58), 0.10, 0.12, mats["glue_cap"], vertices=32))

    # Wooden dowels / skewers.
    for i in range(4):
        objs.append(add_cylinder(
            f"Raw_Wooden_Dowel_{i+1}",
            (-2.6 + i*0.22, -3.32, TABLE_Z + 0.12),
            0.018,
            1.45,
            mats["wood"],
            vertices=12,
            rotation=(0, math.radians(90), math.radians(12)),
        ))

    # Balance weights / paper clips as small metal rings.
    for i in range(5):
        objs.append(add_torus(
            f"Small_Balance_Weight_{i+1}",
            (0.95 + 0.16*i, -3.35, TABLE_Z + 0.09),
            0.055,
            0.010,
            mats["metal"],
            rotation=(math.radians(90), 0, 0),
        ))

    # Octant label cards waiting to be placed.
    labels = ["O1 +++", "O2 -++", "O3 --+", "O4 +-+", "O5 ++-", "O6 -+-", "O7 ---", "O8 +--"]
    for i, label in enumerate(labels):
        objs.extend(make_label_card(
            f"Raw_Octant_Label_{i+1}",
            (-2.75 + (i % 4) * 0.62, -5.5 - (i // 4) * 0.38, TABLE_Z + 0.12),
            label,
            mats,
        ))

    put_many_in_collection(objs, collection)
    return collection


# ============================================================
# Stage 1: Cut Pieces
# ============================================================

def stage_1_cut_pieces(mats):
    collection = create_collection("Stage 1 - Cut Puppet Pieces")
    objs = []

    # Cut outline scraps.
    objs.append(add_cube("Cardboard_Sheet_With_Cutout_Outlines", (-2.65, -1.6, TABLE_Z + 0.04), (1.8, 1.15, 0.035), mats["cardboard"]))
    add_text("Cutout_Outline_Label", "cut outlines", (-2.65, -2.19, TABLE_Z + 0.09), 0.15, mats["black"], rotation=(math.radians(90), 0, 0))

    # Cut body, wings, tail laid out separately.
    objs.append(make_bird_body_shape("Cut_Bird_Body_Piece", (-0.7, -1.6, TABLE_Z + 0.11), mats["cardboard"]))
    objs.append(make_wing_shape("Cut_Left_Wing_Piece", "left", (0.85, -1.48, TABLE_Z + 0.10), mats["wing_paper"]))
    objs.append(make_wing_shape("Cut_Right_Wing_Piece", "right", (0.85, -1.98, TABLE_Z + 0.10), mats["wing_paper"]))
    objs.append(make_tail_shape("Cut_Tail_Piece", (2.45, -1.6, TABLE_Z + 0.10), mats["tail_paper"]))

    # Beak piece.
    beak_verts = [(0, 0), (0.36, 0.10), (0.36, -0.10)]
    objs.append(make_flat_polygon("Cut_Beak_Piece", beak_verts, (2.9, -1.6, TABLE_Z + 0.11), mats["cardboard"], thickness=0.025, rotation=(math.radians(90), 0, 0)))

    # Stage label.
    objs.append(add_text(
        "Stage_1_Label",
        "Stage 1: cardboard, paper, and tail pieces are cut",
        (0.0, -2.55, TABLE_Z + 0.08),
        size=0.18,
        mat=mats["black"],
        rotation=(math.radians(90), 0, 0),
    ))

    put_many_in_collection([o for o in objs if o.name in bpy.data.objects], collection)
    return collection


# ============================================================
# Stage 2: Assembled Bird Puppet
# ============================================================

def create_bird_puppet(mats, location=(0, 0, BIRD_Z), prefix="Bird"):
    """
    Create a simple assembled bird puppet.
    Returns the rig empty and all related objects.
    """
    objs = []

    rig = bpy.data.objects.new(prefix + "_Rig_Empty", None)
    rig.empty_display_type = "ARROWS"
    rig.empty_display_size = 0.6
    rig.location = location
    bpy.context.collection.objects.link(rig)
    objs.append(rig)

    # Body as an ellipsoid.
    body = add_uv_sphere(prefix + "_Body_Cardboard", location, (0.70, 0.22, 0.28), mats["cardboard"])
    objs.append(body)

    head = add_uv_sphere(prefix + "_Head", (location[0] + 0.72, location[1], location[2] + 0.08), (0.22, 0.18, 0.18), mats["cardboard"])
    objs.append(head)

    # Beak as small cone.
    bpy.ops.mesh.primitive_cone_add(
        vertices=24,
        radius1=0.10,
        radius2=0.0,
        depth=0.35,
        location=(location[0] + 1.0, location[1], location[2] + 0.08),
        rotation=(0, math.radians(90), 0),
    )
    beak = bpy.context.object
    beak.name = prefix + "_Beak"
    assign_material(beak, mats["tail_paper"])
    objs.append(beak)

    # Wing panels as polygons, angled upward.
    left_wing = make_wing_shape(prefix + "_Left_Wing_Paper", "left", (location[0] - 0.05, location[1], location[2] + 0.04), mats["wing_paper"])
    left_wing.rotation_euler = (math.radians(90), math.radians(-8), math.radians(4))
    objs.append(left_wing)

    right_wing = make_wing_shape(prefix + "_Right_Wing_Paper", "right", (location[0] - 0.05, location[1], location[2] + 0.04), mats["wing_paper"])
    right_wing.rotation_euler = (math.radians(90), math.radians(8), math.radians(-4))
    objs.append(right_wing)

    # Tail.
    tail = make_tail_shape(prefix + "_Tail_Fan", (location[0] - 0.78, location[1], location[2]), mats["tail_paper"])
    tail.rotation_euler = (math.radians(90), 0, 0)
    objs.append(tail)

    # Wing support dowels.
    left_rod = add_cylinder(
        prefix + "_Left_Wing_Leading_Edge_Dowel",
        (location[0] + 0.22, location[1] + 0.58, location[2] + 0.06),
        0.018,
        1.45,
        mats["wood"],
        vertices=12,
        rotation=(math.radians(90), math.radians(88), 0),
    )
    objs.append(left_rod)

    right_rod = add_cylinder(
        prefix + "_Right_Wing_Leading_Edge_Dowel",
        (location[0] + 0.22, location[1] - 0.58, location[2] + 0.06),
        0.018,
        1.45,
        mats["wood"],
        vertices=12,
        rotation=(math.radians(90), math.radians(92), 0),
    )
    objs.append(right_rod)

    # Tape strips.
    objs.append(add_cube(prefix + "_Tape_Left_Wing_Hinge", (location[0] - 0.05, location[1] + 0.28, location[2] + 0.08), (0.28, 0.06, 0.025), mats["tape"]))
    objs.append(add_cube(prefix + "_Tape_Right_Wing_Hinge", (location[0] - 0.05, location[1] - 0.28, location[2] + 0.08), (0.28, 0.06, 0.025), mats["tape"]))
    objs.append(add_cube(prefix + "_Tape_Tail_Hinge", (location[0] - 0.58, location[1], location[2] + 0.04), (0.16, 0.32, 0.025), mats["tape"]))

    # Parent everything to rig.
    for obj in objs:
        if obj != rig:
            obj.parent = rig

    return rig, objs


def stage_2_assembled_bird(mats):
    collection = create_collection("Stage 2 - Assembled Bird Puppet")
    rig, objs = create_bird_puppet(mats, location=(0, 0, BIRD_Z), prefix="Assembled_Bird")
    label = add_text(
        "Stage_2_Label",
        "Stage 2: lightweight bird body, wings, tail, tape hinges, and dowel supports",
        (0, -1.65, BIRD_Z - 0.85),
        size=0.18,
        mat=mats["black"],
        rotation=(math.radians(70), 0, 0),
    )
    objs.append(label)
    put_many_in_collection(objs, collection)
    return rig, collection


# ============================================================
# Stage 3: Strings and Balance
# ============================================================

def stage_3_strings_and_balance(mats, bird_location=(0, 0, BIRD_Z)):
    collection = create_collection("Stage 3 - Suspension Strings and Balance")
    objs = []

    anchor_z = FRAME_TOP_Z
    anchors = {
        "Main": (0.00, 0.00, anchor_z),
        "Left": (-0.30, 0.95, anchor_z),
        "Right": (-0.30, -0.95, anchor_z),
        "Front": (0.90, 0.00, anchor_z),
    }
    bird_points = {
        "Main": (bird_location[0], bird_location[1], bird_location[2] + 0.35),
        "Left": (bird_location[0] - 0.10, bird_location[1] + 0.62, bird_location[2] + 0.10),
        "Right": (bird_location[0] - 0.10, bird_location[1] - 0.62, bird_location[2] + 0.10),
        "Front": (bird_location[0] + 0.72, bird_location[1], bird_location[2] + 0.15),
    }

    for name in anchors:
        objs.append(create_curve_line(
            f"{name}_Suspension_String",
            [anchors[name], bird_points[name]],
            mats["string"],
            bevel_depth=0.012,
        ))
        objs.append(add_uv_sphere(
            f"{name}_String_Anchor_Bead",
            anchors[name],
            (0.045, 0.045, 0.045),
            mats["metal"],
            segments=16,
            rings=8,
        ))

    # Balance weights.
    objs.append(add_torus("Nose_Balance_Weight", (bird_location[0] + 0.47, bird_location[1], bird_location[2] - 0.24), 0.07, 0.012, mats["metal"], rotation=(math.radians(90), 0, 0)))
    objs.append(add_torus("Tail_Balance_Weight", (bird_location[0] - 0.64, bird_location[1], bird_location[2] - 0.18), 0.055, 0.010, mats["metal"], rotation=(math.radians(90), 0, 0)))

    # Instructional label.
    objs.append(add_text(
        "Stage_3_Label",
        "Stage 3: main, side, and front strings make the puppet steerable",
        (0, 1.75, BIRD_Z - 0.65),
        size=0.18,
        mat=mats["black"],
        rotation=(math.radians(70), 0, math.radians(180)),
    ))

    put_many_in_collection(objs, collection)
    return collection


# ============================================================
# Stage 4: Overhead Support Frame
# ============================================================

def stage_4_overhead_frame(mats):
    collection = create_collection("Stage 4 - Overhead Support Frame")
    objs = []

    # Four vertical posts.
    post_positions = [
        (-2.2, -2.2, FRAME_TOP_Z / 2),
        (2.2, -2.2, FRAME_TOP_Z / 2),
        (-2.2, 2.2, FRAME_TOP_Z / 2),
        (2.2, 2.2, FRAME_TOP_Z / 2),
    ]
    for i, pos in enumerate(post_positions):
        objs.append(add_cylinder(
            f"Support_Frame_Post_{i+1}",
            pos,
            0.045,
            FRAME_TOP_Z,
            mats["wood"],
            vertices=16,
        ))

    # Top bars.
    objs.append(add_cylinder("Support_Frame_Top_Bar_X_Front", (0, -2.2, FRAME_TOP_Z), 0.045, 4.4, mats["wood"], vertices=16, rotation=(0, math.radians(90), 0)))
    objs.append(add_cylinder("Support_Frame_Top_Bar_X_Back", (0, 2.2, FRAME_TOP_Z), 0.045, 4.4, mats["wood"], vertices=16, rotation=(0, math.radians(90), 0)))
    objs.append(add_cylinder("Support_Frame_Top_Bar_Y_Left", (-2.2, 0, FRAME_TOP_Z), 0.045, 4.4, mats["wood"], vertices=16, rotation=(math.radians(90), 0, 0)))
    objs.append(add_cylinder("Support_Frame_Top_Bar_Y_Right", (2.2, 0, FRAME_TOP_Z), 0.045, 4.4, mats["wood"], vertices=16, rotation=(math.radians(90), 0, 0)))

    # Central hook.
    objs.append(add_torus("Ceiling_Hook_Main_Ring", (0, 0, FRAME_TOP_Z + 0.08), 0.14, 0.015, mats["metal"], rotation=(math.radians(90), 0, 0)))
    objs.append(add_cylinder("Ceiling_Hook_Stem", (0, 0, FRAME_TOP_Z + 0.23), 0.022, 0.24, mats["metal"], vertices=16))

    objs.append(add_text(
        "Stage_4_Label",
        "Stage 4: overhead frame holds the suspension strings",
        (0, 2.55, FRAME_TOP_Z - 0.35),
        size=0.18,
        mat=mats["black"],
        rotation=(math.radians(70), 0, math.radians(180)),
    ))

    put_many_in_collection(objs, collection)
    return collection


# ============================================================
# Stage 5: Coordinate Space, Axes, Octants
# ============================================================

def stage_5_coordinate_space(mats):
    collection = create_collection("Stage 5 - 3D Coordinate Space and Octants")
    objs = []

    # Floor and grid.
    objs.append(add_cube("Flight_Space_Floor", (0, 0, -0.035), (8.7, 8.7, 0.03), mats["floor"]))

    # Grid lines.
    grid_extent = 4
    for i in range(-grid_extent, grid_extent + 1):
        objs.append(add_cube(f"Grid_X_Line_y_{i}", (0, i, 0.005), (8.4, 0.012, 0.012), mats["grid"]))
        objs.append(add_cube(f"Grid_Y_Line_x_{i}", (i, 0, 0.006), (0.012, 8.4, 0.012), mats["grid"]))

    # Origin marker.
    objs.append(add_cylinder("Origin_Marker", (0, 0, 0.045), 0.18, 0.05, mats["origin"], vertices=40))
    objs.append(add_text("Origin_Label", "ORIGIN", (0, -0.42, 0.08), 0.16, mats["black"], rotation=(math.radians(90), 0, 0)))

    # Axes as cylinders.
    objs.append(add_cylinder("X_Axis_Positive_Negative", (0, 0, 0.10), 0.022, AXIS_LENGTH * 2, mats["x_axis"], vertices=16, rotation=(0, math.radians(90), 0)))
    objs.append(add_cylinder("Y_Axis_Positive_Negative", (0, 0, 0.11), 0.022, AXIS_LENGTH * 2, mats["y_axis"], vertices=16, rotation=(math.radians(90), 0, 0)))
    objs.append(add_cylinder("Z_Axis_Positive_Negative", (0, 0, AXIS_LENGTH / 2), 0.022, AXIS_LENGTH, mats["z_axis"], vertices=16))

    objs.append(add_text("X_Pos_Label", "+X", (AXIS_LENGTH + 0.25, 0, 0.18), 0.22, mats["x_axis"], rotation=(math.radians(90), 0, 0)))
    objs.append(add_text("X_Neg_Label", "-X", (-AXIS_LENGTH - 0.25, 0, 0.18), 0.22, mats["x_axis"], rotation=(math.radians(90), 0, 0)))
    objs.append(add_text("Y_Pos_Label", "+Y", (0, AXIS_LENGTH + 0.25, 0.18), 0.22, mats["y_axis"], rotation=(math.radians(90), 0, 0)))
    objs.append(add_text("Y_Neg_Label", "-Y", (0, -AXIS_LENGTH - 0.25, 0.18), 0.22, mats["y_axis"], rotation=(math.radians(90), 0, 0)))
    objs.append(add_text("Z_Pos_Label", "+Z", (0.35, 0.35, AXIS_LENGTH + 0.15), 0.22, mats["z_axis"], rotation=(math.radians(65), 0, math.radians(-35))))

    # Octant labels and target markers.
    octants = [
        ("O1 +++",  1,  1,  1),
        ("O2 -++", -1,  1,  1),
        ("O3 --+", -1, -1,  1),
        ("O4 +-+",  1, -1,  1),
        ("O5 ++-",  1,  1, -1),
        ("O6 -+-", -1,  1, -1),
        ("O7 ---", -1, -1, -1),
        ("O8 +--",  1, -1, -1),
    ]

    for label, sx, sy, sz in octants:
        z = 3.25 if sz > 0 else 0.65
        loc = (sx * OCTANT_LABEL_DISTANCE, sy * OCTANT_LABEL_DISTANCE, z)
        objs.append(add_uv_sphere(f"{label}_Target_Marker", loc, (0.10, 0.10, 0.10), mats["target"], segments=16, rings=8))
        objs.append(add_text(
            f"{label}_Label",
            label,
            (loc[0], loc[1], loc[2] + 0.28),
            size=0.24,
            mat=mats["black"],
            rotation=(math.radians(65), 0, math.atan2(loc[1], loc[0]) - math.pi/2),
        ))

    objs.append(add_text(
        "Stage_5_Label",
        "Stage 5: centered 3D coordinate space with eight octants",
        (0, -4.15, 0.22),
        size=0.20,
        mat=mats["black"],
        rotation=(math.radians(90), 0, 0),
    ))

    put_many_in_collection(objs, collection)
    return collection


# ============================================================
# Stage 6: Animation Guides
# ============================================================

def stage_6_animation_paths(mats):
    collection = create_collection("Stage 6 - Flight Path Guides")
    objs = []

    # Direct octant chain guide.
    direct_points = [
        (0, 0, BIRD_Z),
        (2.7, 2.7, 3.8),
        (-2.7, 2.7, 3.8),
        (-2.7, -2.7, 3.8),
        (2.7, -2.7, 3.8),
        (2.7, 2.7, 1.4),
        (-2.7, 2.7, 1.4),
        (-2.7, -2.7, 1.4),
        (2.7, -2.7, 1.4),
    ]
    objs.append(create_curve_line("Guide_Path_Direct_Octant_Chain", direct_points, mats["target"], bevel_depth=0.018))

    # Figure-eight guide path.
    fig8_points = []
    for i in range(96):
        t = 2 * math.pi * i / 95.0
        x = 2.1 * math.sin(t)
        y = 1.2 * math.sin(2 * t)
        z = BIRD_Z + 0.6 * math.sin(t + math.pi / 3)
        fig8_points.append((x, y, z))
    objs.append(create_curve_line("Guide_Path_Figure_Eight_All_Axes", fig8_points, mats["z_axis"], bevel_depth=0.014))

    # Spiral guide path.
    spiral_points = []
    for i in range(120):
        t = 6.0 * math.pi * i / 119.0
        r = 0.15 + 2.8 * i / 119.0
        x = r * math.cos(t)
        y = r * math.sin(t)
        z = BIRD_Z + 0.75 * math.sin(0.8 * t)
        spiral_points.append((x, y, z))
    objs.append(create_curve_line("Guide_Path_Spiral_Octant_Expansion", spiral_points, mats["y_axis"], bevel_depth=0.014))

    # Stage label.
    objs.append(add_text(
        "Stage_6_Label",
        "Stage 6: finished suspended bird is ready for direct, spiral, diagonal, chaotic, and figure-eight flight paths",
        (0, 4.15, 0.25),
        size=0.18,
        mat=mats["black"],
        rotation=(math.radians(90), 0, math.radians(180)),
    ))

    put_many_in_collection(objs, collection)
    return collection


# ============================================================
# Bird Rig Animation
# ============================================================

def animate_bird_rig(rig):
    """
    Animate the assembled bird rig only after the build sequence finishes.
    The early timeline is reserved for material-to-build staging.
    """
    if rig is None:
        return

    # Hide the bird during raw-material and cut-piece stages.
    # After Stage 2 begins, the bird remains visible as part of the accumulating finished build.
    if ANIMATE_BUILD_SEQUENCE:
        stage_2_start = STAGE_FRAME_WINDOWS[2][0]
        set_object_visibility_key(rig, 1, False)
        set_object_visibility_key(rig, stage_2_start, True)
        set_object_visibility_key(rig, BIRD_FLIGHT_END_FRAME, True)

    keyframes = [
        (BIRD_FLIGHT_START_FRAME,       (0.0, 0.0, BIRD_Z),        (0, 0, 0)),
        (BIRD_FLIGHT_START_FRAME + 25,  (2.2, 2.2, 3.8),          (0, 0, math.radians(12))),
        (BIRD_FLIGHT_START_FRAME + 50,  (-2.3, -2.2, 1.5),        (math.radians(10), 0, math.radians(-20))),
        (BIRD_FLIGHT_START_FRAME + 75,  (0.0, 0.0, 3.1),          (0, math.radians(8), 0)),
        (BIRD_FLIGHT_START_FRAME + 100, (-2.0, 2.2, 3.6),         (math.radians(-8), 0, math.radians(22))),
        (BIRD_FLIGHT_START_FRAME + 125, (2.1, -2.0, 1.6),         (math.radians(9), 0, math.radians(-18))),
        (BIRD_FLIGHT_END_FRAME,         (0.0, 0.0, BIRD_Z),       (0, 0, 0)),
    ]

    for frame, loc, rot in keyframes:
        bpy.context.scene.frame_set(frame)
        rig.location = loc
        rig.rotation_euler = rot
        rig.keyframe_insert(data_path="location", frame=frame)
        rig.keyframe_insert(data_path="rotation_euler", frame=frame)

    # Smooth interpolation when Blender exposes action.fcurves.
    action = rig.animation_data.action if rig.animation_data else None
    if action and hasattr(action, "fcurves"):
        for fc in action.fcurves:
            for kp in fc.keyframe_points:
                kp.interpolation = "BEZIER"


# ============================================================
# Lighting and Camera
# ============================================================

def create_lighting_and_camera(mats):
    # Area light.
    bpy.ops.object.light_add(type="AREA", location=(0, -5.5, 7.5))
    light = bpy.context.object
    light.name = "Large_Workshop_Area_Light"
    light.data.energy = 650
    light.data.size = 5.0

    # Secondary light.
    bpy.ops.object.light_add(type="POINT", location=(-3.2, 3.2, 4.5))
    point = bpy.context.object
    point.name = "Soft_Fill_Light"
    point.data.energy = 120

    # Camera.
    bpy.ops.object.camera_add(location=(6.3, -7.4, 5.4), rotation=(math.radians(62), 0, math.radians(40)))
    cam = bpy.context.object
    bpy.context.scene.camera = cam
    cam.name = "Camera_Overview_Workshop_To_Flight_Space"
    cam.data.lens = 28

    # Aim camera at origin.
    direction = Vector((0, 0, 2.1)) - cam.location
    cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()

    # Add main title.
    add_text(
        "Scene_Title",
        "Suspended Bird Puppet: raw materials -> assembled puppet -> 3D octant flight space",
        (0, -5.9, 1.25),
        size=0.25,
        mat=mats["black"],
        rotation=(math.radians(75), 0, 0),
    )



# ============================================================
# Staged Build Visibility Animation
# ============================================================

def set_object_visibility_key(obj, frame, visible):
    """
    Keyframe object visibility for both viewport and render.
    visible=True means the object is shown at this frame.
    """
    obj.hide_viewport = not visible
    obj.hide_render = not visible
    obj.keyframe_insert(data_path="hide_viewport", frame=frame)
    obj.keyframe_insert(data_path="hide_render", frame=frame)


def keyframe_collection_visibility(collection, start_frame, end_frame, visible_before=False, visible_during=True, visible_after=False):
    """
    Animate every object in a collection so the collection appears only inside
    a chosen frame range.
    """
    for obj in collection.objects:
        # Before stage appears.
        set_object_visibility_key(obj, max(1, start_frame - 1), visible_before)

        # Stage appears.
        set_object_visibility_key(obj, start_frame, visible_during)

        # Stage remains through end.
        set_object_visibility_key(obj, end_frame, visible_during)

        # Stage disappears after end.
        set_object_visibility_key(obj, end_frame + 1, visible_after)


def keyframe_object_visibility_range(obj, start_frame, end_frame, visible_after=False):
    set_object_visibility_key(obj, max(1, start_frame - 1), False)
    set_object_visibility_key(obj, start_frame, True)
    set_object_visibility_key(obj, end_frame, True)
    set_object_visibility_key(obj, end_frame + 1, visible_after)


def animate_stage_sequence():
    """
    Make the construction stages appear in sequence.

    Raw/non-final material stages:
        Stage 0 raw materials appear, then disappear.
        Stage 1 cut pieces appear, then disappear.

    Finished-build stages:
        Stage 2 assembled bird appears and remains.
        Stage 3 strings/balance appear and remain.
        Stage 4 support frame appears and remains.
        Stage 5 coordinate space appears and remains.
        Stage 6 flight path guides appear and remain.

    Result:
        The finished build is accumulated over time. Only objects that are not
        part of the finished build are removed.
    """
    if not ANIMATE_BUILD_SEQUENCE:
        return

    # These are process/workshop-only views. They are not part of the final installation.
    temporary_stage_nums = [0, 1]

    # These are part of the finished build and should remain once constructed.
    persistent_stage_nums = [2, 3, 4, 5, 6]

    # Stage 0 and Stage 1 appear only during their construction windows.
    for stage_num in temporary_stage_nums:
        start, end = STAGE_FRAME_WINDOWS[stage_num]
        collection_name_prefix = f"Stage {stage_num} -"
        for collection in bpy.data.collections:
            if collection.name.startswith(collection_name_prefix):
                keyframe_collection_visibility(
                    collection,
                    start,
                    end,
                    visible_before=False,
                    visible_during=True,
                    visible_after=False,
                )

    # Finished-build stages accumulate: once visible, they stay visible through flight.
    for stage_num in persistent_stage_nums:
        start, end = STAGE_FRAME_WINDOWS[stage_num]
        collection_name_prefix = f"Stage {stage_num} -"
        for collection in bpy.data.collections:
            if collection.name.startswith(collection_name_prefix):
                for obj in collection.objects:
                    set_object_visibility_key(obj, max(1, start - 1), False)
                    set_object_visibility_key(obj, start, True)
                    set_object_visibility_key(obj, end, True)
                    set_object_visibility_key(obj, BIRD_FLIGHT_END_FRAME, True)

    # Keep title, camera, and lights visible through the whole animation.
    always_visible_names = [
        "Scene_Title",
        "Large_Workshop_Area_Light",
        "Soft_Fill_Light",
        "Camera_Overview_Workshop_To_Flight_Space",
    ]
    for name in always_visible_names:
        obj = bpy.data.objects.get(name)
        if obj:
            set_object_visibility_key(obj, 1, True)
            set_object_visibility_key(obj, BIRD_FLIGHT_END_FRAME, True)

    # Make visibility changes constant rather than fading/interpolating.
    for obj in bpy.data.objects:
        if obj.animation_data and obj.animation_data.action:
            action = obj.animation_data.action
            if hasattr(action, "fcurves"):
                for fc in action.fcurves:
                    if "hide_viewport" in fc.data_path or "hide_render" in fc.data_path:
                        for kp in fc.keyframe_points:
                            kp.interpolation = "CONSTANT"


# ============================================================
# Visibility by Stage
# ============================================================

def apply_build_stage_visibility():
    """
    If BUILD_STAGE is less than 6, hide later stage collections.
    """
    for collection in bpy.data.collections:
        if collection.name.startswith("Stage "):
            try:
                stage_num = int(collection.name.split(" ")[1])
            except Exception:
                continue

            hidden = stage_num > BUILD_STAGE
            collection.hide_viewport = hidden
            collection.hide_render = hidden


# ============================================================
# Main
# ============================================================

def main():
    clear_scene()
    set_units_and_scene()
    mats = create_materials()

    if BUILD_STAGE >= 0:
        stage_0_raw_materials(mats)

    if BUILD_STAGE >= 1:
        stage_1_cut_pieces(mats)

    bird_rig = None
    if BUILD_STAGE >= 2:
        bird_rig, _ = stage_2_assembled_bird(mats)

    if BUILD_STAGE >= 3:
        stage_3_strings_and_balance(mats, bird_location=(0, 0, BIRD_Z))

    if BUILD_STAGE >= 4:
        stage_4_overhead_frame(mats)

    if BUILD_STAGE >= 5:
        stage_5_coordinate_space(mats)

    if BUILD_STAGE >= 6:
        stage_6_animation_paths(mats)

    create_lighting_and_camera(mats)

    if ANIMATE_BUILD_SEQUENCE:
        animate_stage_sequence()
    else:
        apply_build_stage_visibility()

    animate_bird_rig(bird_rig)

    # Set origin view frame.
    bpy.context.scene.frame_set(1)

    # Set clipping and color management.
    bpy.context.scene.view_settings.view_transform = "Filmic"
    bpy.context.scene.view_settings.look = "Medium High Contrast"
    bpy.context.scene.view_settings.exposure = 0
    bpy.context.scene.view_settings.gamma = 1

    print("Created Blender scene:", SCENE_NAME)
    print("BUILD_STAGE =", BUILD_STAGE)
    print("Use the timeline to see the assembled bird rig move through a mixed 3D path.")


if __name__ == "__main__":
    main()
