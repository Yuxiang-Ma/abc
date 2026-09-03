"""Render an exported ABC shot in Blender (Cycles or EEVEE) with authored materials, lighting and cameras.

    blender -b --python-exit-code 1 --python render_usd.py -- --usd shot/frames/frame_93.usdc \
        --meta shot/geoms.json --out renders/shot --cams top,wrist,front --samples 64 --device OPTIX

Cameras: top, left, right, wrist (the active wrist from geoms.json), front, front_left, front_right,
overhead, left_side, right_side, close.  Robot cameras keep the enclosure walls; outside cameras hide
them and see a plain room.  Writes <out>/<cam>/frame_#####.png and skips frames that already exist.
"""

import argparse
import json
import math
import os
import socket
import sys
import time

import bmesh
import bpy
from mathutils import Matrix, Vector
import numpy as np

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--usd", required=True)
ap.add_argument("--meta", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--cams", default="top,wrist,front")
ap.add_argument("--frames", default="", help="start:end[:step] in USD time codes, or a,b,c; default all")
ap.add_argument("--size", default="1440x1080", help="WxH; 4:3 keeps the sim cameras' 58 degree fovy")
ap.add_argument("--wrist-size", default="", help="WxH for the wrist camera, default --size")
ap.add_argument("--samples", type=int, default=64)
ap.add_argument("--device", default="OPTIX", choices=["CPU", "CUDA", "OPTIX"])
ap.add_argument("--engine", default="CYCLES", choices=["CYCLES", "EEVEE"])
ap.add_argument("--denoiser", default="OPENIMAGEDENOISE", choices=["OPENIMAGEDENOISE", "OPTIX", "NONE"])
ap.add_argument("--threads", type=int, default=0)
ap.add_argument("--exposure", type=float, default=-0.3)
ap.add_argument("--hdri-strength", type=float, default=0.30)
ap.add_argument("--room", type=float, default=0.62, help="room wall albedo")
ap.add_argument("--floor", type=float, default=0.42, help="floor tile albedo")
ap.add_argument("--table", type=float, default=0.68, help="table albedo")
ap.add_argument("--key-energy", type=float, default=260.0, help="key light watts")
ap.add_argument("--view-transform", default="AgX")
ap.add_argument("--look", default="AgX - Punchy")
ap.add_argument("--hdri", default=os.environ.get("ABC_HDRI", ""), help="equirectangular .hdr; default $ABC_HDRI")
ap.add_argument("--fstop", type=float, default=0.0, help="depth of field on outside cameras, 0 = off")
ap.add_argument("--motion-blur", action="store_true")
ap.add_argument("--glare", type=float, default=0.0, help="compositor bloom mix, 0 = off")
ap.add_argument("--no-bevel", action="store_true", help="skip the bevel shader on edges (faster)")
ap.add_argument("--keep-long-tris", action="store_true", help="do not bisect the aluminium frame mesh")
ap.add_argument("--cache", default=os.path.expanduser("~/.cache/abc_blender"))
ap.add_argument("--save-blend", default="")
ap.add_argument("--serve", default="", help="unix socket: render poses streamed by abc_sim.rendering.blender.live")
args = ap.parse_args(argv)

meta = json.load(open(args.meta))
info = {g["usd"]: g for g in meta["geoms"]}
W, H = (int(v) for v in args.size.lower().split("x"))
WW, WH = (int(v) for v in args.wrist_size.lower().split("x")) if args.wrist_size else (W, H)

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.wm.usd_import(filepath=args.usd, import_cameras=True, import_lights=True, import_materials=True,
                      import_textures_mode="IMPORT_PACK", set_frame_range=True)
scene = bpy.context.scene
scene.render.fps = int(meta.get("fps", 30))
for obj in scene.objects:
    if obj.type == "LIGHT":
        obj.hide_render = True                       # the exporter's lights are black sphere lights

if not args.frames:
    frames = list(range(scene.frame_start, scene.frame_end))
elif ":" in args.frames:
    parts = [int(v) for v in args.frames.split(":")]
    frames = list(range(parts[0], parts[1], parts[2] if len(parts) > 2 else 1))
else:
    frames = [int(v) for v in args.frames.split(",")]
scene.frame_set(frames[0])

# --------------------------------------------------------------------------- what each geom is
PROP_KINDS = {"bead", "abs", "bin_plastic", "ceramic", "paper", "textured_prop", "prop", "wire_rack",
              "steel", "rubber", "painted_metal", "bristle"}


def classify(g):
    geom, mat, mesh = (g.get(k) or "" for k in ("geom", "material", "mesh"))
    if geom in ("back_wall", "left_wall", "right_wall"):
        return "wall"
    if geom == "floor":
        return "floor"
    if mesh == "base_visual_gate":
        return "aluminium"
    if geom == "play_table":
        return "table"
    if mesh == "camera_d405":
        return "camera"
    if mesh.startswith("model2"):
        return "robot_white" if mat == "white" else "robot_black"
    if geom.startswith("bead_"):
        return "bead"
    if mat.startswith(("lego_", "count_item")):
        return "abs"
    if mat.endswith("_bin_material") or mat.startswith(("conveyor_bin", "opaque_count_box", "grab_clutter_box")) or mat in (
            "garbage_can_mat", "conveyor_05_plastico_preto_001"):
        return "bin_plastic"
    if mat == "conveyor_00_borracha_001":
        return "rubber"
    if mat in ("conveyor_06_rolete", "conveyor_02_metal_motor_001", "nut_steel", "bolt_steel"):
        return "steel"
    if mat in ("conveyor_03_metal_pintado_002", "conveyor_04_parto_do_motor"):
        return "painted_metal"
    if "dish_rack" in mat:
        return "wire_rack"
    if "plate" in mat or mat in ("mug_1_color", "cup_body_color"):
        return "ceramic"
    if mat.startswith("trash_"):
        return "paper"
    if mat == "brush_handle_mat":                    # exporter swaps the brush names; this is the bristle colour
        return "bristle"
    return "textured_prop" if g["textured"] else "prop"


# --------------------------------------------------------------------------- materials
def bsdf_of(mat):
    return next((n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None)


def set_in(bsdf, name, value):
    if name in bsdf.inputs and not bsdf.inputs[name].is_linked:
        bsdf.inputs[name].default_value = value


def object_noise(nt, scale, detail=4.0):
    coord = nt.nodes.new("ShaderNodeTexCoord")
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = scale
    noise.inputs["Detail"].default_value = detail
    nt.links.new(coord.outputs["Object"], noise.inputs["Vector"])
    return noise


def roughness_variation(mat, bsdf, lo, hi, scale):
    nt = mat.node_tree
    mr = nt.nodes.new("ShaderNodeMapRange")
    mr.inputs["To Min"].default_value, mr.inputs["To Max"].default_value = lo, hi
    nt.links.new(object_noise(nt, scale).outputs["Fac"], mr.inputs["Value"])
    nt.links.new(mr.outputs["Result"], bsdf.inputs["Roughness"])


def color_variation(mat, bsdf, base, amount, scale, bump=0.0):
    """Object-space noise blends the base colour toward a darker tone; no UVs needed."""
    if bsdf.inputs["Base Color"].is_linked:
        return
    nt = mat.node_tree
    noise = object_noise(nt, scale, detail=6.0)
    mr = nt.nodes.new("ShaderNodeMapRange")
    mr.inputs["To Min"].default_value, mr.inputs["To Max"].default_value = 1.0 - amount, 1.0
    nt.links.new(noise.outputs["Fac"], mr.inputs["Value"])
    mix = nt.nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    sock = {s.identifier: s for s in mix.inputs}
    sock["A_Color"].default_value = tuple(c * 0.72 for c in base[:3]) + (1.0,)
    sock["B_Color"].default_value = base
    nt.links.new(mr.outputs["Result"], sock["Factor_Float"])
    nt.links.new(next(o for o in mix.outputs if o.identifier == "Result_Color"), bsdf.inputs["Base Color"])
    if bump > 0:
        b = nt.nodes.new("ShaderNodeBump")
        b.inputs["Strength"].default_value = bump
        nt.links.new(noise.outputs["Fac"], b.inputs["Height"])
        nt.links.new(b.outputs["Normal"], bsdf.inputs["Normal"])


def bevel(mat, bsdf, radius):
    if args.no_bevel:
        return
    nt = mat.node_tree
    b = nt.nodes.new("ShaderNodeBevel")
    b.inputs["Radius"].default_value = radius
    b.samples = 4
    if bsdf.inputs["Normal"].is_linked:
        nt.links.new(bsdf.inputs["Normal"].links[0].from_socket, b.inputs["Normal"])
    nt.links.new(b.outputs["Normal"], bsdf.inputs["Normal"])


def floor_tiles(mat, bsdf):
    nt = mat.node_tree
    coord = nt.nodes.new("ShaderNodeTexCoord")
    brick = nt.nodes.new("ShaderNodeTexBrick")
    brick.offset = 0.0
    brick.inputs["Scale"].default_value = 1.0
    brick.inputs["Mortar Size"].default_value = 0.004
    brick.inputs["Mortar Smooth"].default_value = 0.3
    brick.inputs["Brick Width"].default_value = 0.6
    brick.inputs["Row Height"].default_value = 0.6
    f = args.floor
    brick.inputs["Color1"].default_value = (f, f, f * 1.03, 1.0)
    brick.inputs["Color2"].default_value = (f * 0.9, f * 0.9, f * 0.93, 1.0)
    brick.inputs["Mortar"].default_value = (f * 0.5, f * 0.5, f * 0.5, 1.0)
    nt.links.new(coord.outputs["Object"], brick.inputs["Vector"])
    noise = object_noise(nt, 2.0, detail=6.0)
    mix = nt.nodes.new("ShaderNodeMix")
    mix.data_type, mix.blend_type = "RGBA", "MULTIPLY"
    sock = {s.identifier: s for s in mix.inputs}
    sock["Factor_Float"].default_value = 0.25
    nt.links.new(brick.outputs["Color"], sock["A_Color"])
    nt.links.new(noise.outputs["Color"], sock["B_Color"])
    nt.links.new(next(o for o in mix.outputs if o.identifier == "Result_Color"), bsdf.inputs["Base Color"])
    mr = nt.nodes.new("ShaderNodeMapRange")
    mr.inputs["To Min"].default_value, mr.inputs["To Max"].default_value = 0.45, 0.65
    nt.links.new(noise.outputs["Fac"], mr.inputs["Value"])
    nt.links.new(mr.outputs["Result"], bsdf.inputs["Roughness"])
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.25
    bump.invert = True
    nt.links.new(brick.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])


def apply_material(mat, kind):
    bsdf = bsdf_of(mat)
    if bsdf is None:
        return
    base = tuple(bsdf.inputs["Base Color"].default_value)
    set_in(bsdf, "Metallic", 0.0)                    # the exporter writes MuJoCo shininess into metallic
    set_in(bsdf, "Alpha", 1.0)
    set_in(bsdf, "Specular IOR Level", 0.5)
    if kind == "wall":
        set_in(bsdf, "Base Color", (0.74, 0.74, 0.72, 1.0))
        set_in(bsdf, "Roughness", 0.82)
        color_variation(mat, bsdf, (0.74, 0.74, 0.72, 1.0), 0.05, 1.2)
    elif kind == "floor":
        floor_tiles(mat, bsdf)
    elif kind == "table":
        set_in(bsdf, "Coat Weight", 0.15)
        color_variation(mat, bsdf, (args.table, args.table, args.table - 0.01, 1.0), 0.05, 18.0, bump=0.012)
        roughness_variation(mat, bsdf, 0.30, 0.50, 45.0)
    elif kind == "aluminium":
        set_in(bsdf, "Base Color", (0.72, 0.73, 0.75, 1.0))
        set_in(bsdf, "Metallic", 1.0)
        set_in(bsdf, "Anisotropic", 0.5)
        roughness_variation(mat, bsdf, 0.25, 0.42, 220.0)
        bevel(mat, bsdf, 0.0012)
    elif kind == "steel":
        set_in(bsdf, "Base Color", (0.66, 0.66, 0.67, 1.0))
        set_in(bsdf, "Metallic", 1.0)
        set_in(bsdf, "Anisotropic", 0.6)
        roughness_variation(mat, bsdf, 0.24, 0.36, 150.0)
    elif kind == "camera":
        set_in(bsdf, "Base Color", (0.03, 0.03, 0.035, 1.0))
        set_in(bsdf, "Metallic", 0.3)
        set_in(bsdf, "Roughness", 0.45)
        bevel(mat, bsdf, 0.001)
    elif kind == "robot_white":
        set_in(bsdf, "Base Color", (0.82, 0.82, 0.83, 1.0))
        set_in(bsdf, "Coat Weight", 0.3)
        set_in(bsdf, "Coat Roughness", 0.08)
        roughness_variation(mat, bsdf, 0.33, 0.48, 60.0)
        bevel(mat, bsdf, 0.0015)
    elif kind == "robot_black":
        set_in(bsdf, "Base Color", (0.04, 0.04, 0.045, 1.0))
        set_in(bsdf, "Specular IOR Level", 0.35)
        roughness_variation(mat, bsdf, 0.45, 0.62, 60.0)
        bevel(mat, bsdf, 0.0015)
    elif kind == "bead":
        set_in(bsdf, "Roughness", 0.12)
        set_in(bsdf, "Coat Weight", 1.0)
        set_in(bsdf, "Coat Roughness", 0.03)
    elif kind == "abs":
        set_in(bsdf, "Roughness", 0.25)
        set_in(bsdf, "Coat Weight", 0.35)
        set_in(bsdf, "Coat Roughness", 0.05)
        bevel(mat, bsdf, 0.0004)
    elif kind == "bin_plastic":
        set_in(bsdf, "Coat Weight", 0.2)
        roughness_variation(mat, bsdf, 0.38, 0.52, 40.0)
        bevel(mat, bsdf, 0.001)
    elif kind == "rubber":
        set_in(bsdf, "Base Color", (0.03, 0.03, 0.03, 1.0))
        set_in(bsdf, "Roughness", 0.75)
    elif kind == "painted_metal":
        set_in(bsdf, "Roughness", 0.35)
        set_in(bsdf, "Coat Weight", 0.3)
        bevel(mat, bsdf, 0.001)
    elif kind == "wire_rack":
        set_in(bsdf, "Metallic", 0.85)
        set_in(bsdf, "Roughness", 0.3)
    elif kind == "ceramic":
        set_in(bsdf, "Roughness", 0.18)
        set_in(bsdf, "Coat Weight", 1.0)
        set_in(bsdf, "Coat Roughness", 0.02)
        bevel(mat, bsdf, 0.0008)
    elif kind == "paper":
        set_in(bsdf, "Base Color", tuple(min(1.0, c * 1.35) for c in base[:3]) + (1.0,))
        set_in(bsdf, "Roughness", 0.85)
        set_in(bsdf, "Sheen Weight", 0.6)
    elif kind == "bristle":
        set_in(bsdf, "Roughness", 0.7)
        set_in(bsdf, "Sheen Weight", 0.4)
    elif kind == "textured_prop":
        set_in(bsdf, "Roughness", 0.45)
        set_in(bsdf, "Coat Weight", 0.15)
    else:
        set_in(bsdf, "Roughness", 0.45)
        set_in(bsdf, "Coat Weight", 0.1)
        bevel(mat, bsdf, 0.001)


walls, props, floor_obj, by_body = {}, [], None, {}
for obj in scene.objects:
    g = info.get(obj.name[len("Mesh_"):]) if obj.type == "MESH" and obj.name.startswith("Mesh_") else None
    if g is None:
        continue
    kind = classify(g)
    by_body.setdefault(g.get("body"), []).append((obj, g))
    if g["rgba"][3] < 0.05 or "_inner" in (g.get("mesh") or ""):   # hollow assets ship a flipped inner shell for MuJoCo's back-face culling; Cycles is double-sided and the coincident copy z-fights
        obj.hide_render = True
        continue
    if kind == "wall":
        walls[g["geom"]] = obj
    if kind == "floor":
        floor_obj = obj
        obj.hide_render = True                       # replaced by a large tiled ground plane
    if kind in PROP_KINDS:
        props.append(obj)
    obj.data.shade_smooth()
    if kind not in ("wall", "floor", "table"):
        obj.modifiers.new("EdgeSplit", "EDGE_SPLIT").split_angle = math.radians(32.0)
    for slot in obj.material_slots:
        if slot.material and slot.material.node_tree:
            apply_material(slot.material, kind)


# --------------------------------------------------------------------------- geometry fixes
def cut_long_faces(obj, step=0.05):
    """The extruded frame comes out of MuJoCo as metre-long slivers, which make ray traversal
    several times slower; bisect every `step` metres along each axis."""
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    lo = [min(v.co[i] for v in bm.verts) for i in range(3)]
    hi = [max(v.co[i] for v in bm.verts) for i in range(3)]
    for axis in range(3):
        normal = Vector((0.0, 0.0, 0.0))
        normal[axis] = 1.0
        x = lo[axis] + step
        while x < hi[axis]:
            co = Vector((0.0, 0.0, 0.0))
            co[axis] = x
            bmesh.ops.bisect_plane(bm, geom=bm.verts[:] + bm.edges[:] + bm.faces[:],
                                   plane_co=co, plane_no=normal, dist=1e-6)
            x += step
    bm.to_mesh(obj.data)
    bm.free()


def centre(obj):
    return sum((obj.matrix_world @ Vector(c) for c in obj.bound_box), Vector()) / 8.0


if not args.keep_long_tris:
    for obj, g in [x for pairs in by_body.values() for x in pairs]:
        if g.get("mesh") != "base_visual_gate":
            continue
        if args.serve:                                   # 4k faces: the live renders do not resolve the profile
            obj.modifiers.new("Decimate", "DECIMATE").ratio = 0.01
            obj.modifiers.move(1, 0)
            continue
        cache = os.path.join(args.cache, "base_visual_gate_cut.blend")
        if os.path.exists(cache):
            with bpy.data.libraries.load(cache) as (src, dst):
                dst.meshes = ["base_visual_gate_cut"]
            obj.data = dst.meshes[0]
        else:
            cut_long_faces(obj)
            obj.data.name = "base_visual_gate_cut"
            os.makedirs(args.cache, exist_ok=True)
            tmp = cache.replace('.blend', f'.{os.getpid()}.blend')       # parallel renders race for the first write
            bpy.data.libraries.write(tmp, {obj.data}, fake_user=True)
            os.replace(tmp, cache)

# the sim has no bracket between gripper and wrist camera: add a black bar so the D405 does not float
mount = bpy.data.materials.new("CameraMount")
mount.use_nodes = True
set_in(bsdf_of(mount), "Base Color", (0.02, 0.02, 0.022, 1.0))
set_in(bsdf_of(mount), "Roughness", 0.75)
for side in ("left", "right"):
    cams = [o for o, g in by_body.get(f"{side}_camera_d405", []) if g.get("mesh") == "camera_d405"]
    bases = [o for o, g in by_body.get(f"{side}_link_6", []) if g.get("mesh") == "model2__12"]
    anchor = bpy.data.objects.get(f"Camera_{side}")
    if not bases or not (cams or anchor):
        continue
    if not cams:
        # scenes that hide the D405 housing: build one behind the recorded camera (42 x 42 x 23 mm),
        # lens along local +Z like the exported mesh, so the bar code below applies unchanged
        bpy.ops.mesh.primitive_cube_add(size=1.0)
        body = bpy.context.active_object
        body.name = f"{side}_camera_body"
        body.data.transform(Matrix.Diagonal((0.042, 0.042, 0.023, 1.0)) @ Matrix.Translation((0.0, 0.0, -0.5)))
        loc, rot, _ = anchor.matrix_world.decompose()          # ignore any import scale on the camera
        body.matrix_world = Matrix.Translation(loc) @ rot.to_matrix().to_4x4() @ Matrix.Rotation(math.pi, 4, "X")
        body.parent = anchor
        body.matrix_parent_inverse = anchor.matrix_world.inverted()
        body.data.materials.append(mount)
        cams = [body]
    cam_obj, base_obj = cams[0], bases[0]
    coords = [v.co for v in cam_obj.data.vertices]          # bound_box lags behind mesh edits
    lo = Vector(min(c[i] for c in coords) for i in range(3))
    hi = Vector(max(c[i] for c in coords) for i in range(3))
    back = cam_obj.matrix_world @ Vector(((lo.x + hi.x) / 2.0, (lo.y + hi.y) / 2.0, lo.z))   # back face: lens is +Z
    b = centre(base_obj)
    d = (b - back).normalized()
    a, e = back - d * 0.004, b - d * 0.02                    # one bar, from just inside the back face into the base
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(a + e) / 2.0)
    bar = bpy.context.active_object
    bar.name = f"{side}_camera_mount"
    bar.scale = (0.022, 0.016, (e - a).length)
    bar.rotation_euler = (e - a).to_track_quat("Z", "Y").to_euler()
    bar.data.materials.append(mount)
    bar.parent = cam_obj
    bar.matrix_parent_inverse = cam_obj.matrix_world.inverted()


def in_workspace(p):
    return -0.4 <= p.x <= 1.3 and -1.0 <= p.y <= 1.0 and -0.05 <= p.z <= 1.7


# props the task parks outside the enclosure are invisible to sim cameras but not to outside ones
for f in frames if not args.serve else []:
    scene.frame_set(f)
    for obj in props:
        obj.hide_render = not in_workspace(centre(obj))
        obj.keyframe_insert("hide_render", frame=f)
scene.frame_set(frames[0])

# --------------------------------------------------------------------------- room, world, lights
bpy.ops.mesh.primitive_plane_add(size=40.0, location=(0.0, 0.0, 0.0))
ground = bpy.context.active_object
if floor_obj is not None and floor_obj.material_slots:
    ground.data.materials.append(floor_obj.material_slots[0].material)
room = bpy.data.materials.new("Room")
room.use_nodes = True
set_in(bsdf_of(room), "Base Color", (args.room, args.room, args.room - 0.01, 1.0))
set_in(bsdf_of(room), "Roughness", 0.9)
color_variation(room, bsdf_of(room), (args.room, args.room, args.room - 0.01, 1.0), 0.08, 0.8)
for loc, rot, scale in (((-2.6, 0.0, 1.6), (0.0, math.radians(90.0), 0.0), (3.2, 9.0, 1.0)),
                        ((0.0, -2.6, 1.6), (math.radians(90.0), 0.0, 0.0), (9.0, 3.2, 1.0)),
                        ((0.0, 2.6, 1.6), (math.radians(90.0), 0.0, 0.0), (9.0, 3.2, 1.0))):
    bpy.ops.mesh.primitive_plane_add(size=1.0, location=loc, rotation=rot)
    bpy.context.active_object.scale = scale
    bpy.context.active_object.data.materials.append(room)

world = bpy.data.worlds.new("Studio")
world.use_nodes = True
nt = world.node_tree
nt.nodes.clear()
bg_light = nt.nodes.new("ShaderNodeBackground")       # lights and reflections
bg_camera = nt.nodes.new("ShaderNodeBackground")      # what the camera sees where no geometry is
bg_camera.inputs["Color"].default_value = (0.34, 0.34, 0.36, 1.0)
if args.hdri:
    coord = nt.nodes.new("ShaderNodeTexCoord")
    mapping = nt.nodes.new("ShaderNodeMapping")
    mapping.inputs["Rotation"].default_value = (0.0, 0.0, math.radians(200.0))
    env = nt.nodes.new("ShaderNodeTexEnvironment")
    env.image = bpy.data.images.load(args.hdri)
    nt.links.new(coord.outputs["Generated"], mapping.inputs["Vector"])
    nt.links.new(mapping.outputs["Vector"], env.inputs["Vector"])
    nt.links.new(env.outputs["Color"], bg_light.inputs["Color"])
    bg_light.inputs["Strength"].default_value = args.hdri_strength
else:
    bg_light.inputs["Color"].default_value = (0.5, 0.5, 0.52, 1.0)
    bg_light.inputs["Strength"].default_value = 0.3
mix = nt.nodes.new("ShaderNodeMixShader")
out = nt.nodes.new("ShaderNodeOutputWorld")
nt.links.new(nt.nodes.new("ShaderNodeLightPath").outputs["Is Camera Ray"], mix.inputs["Fac"])
nt.links.new(bg_light.outputs["Background"], mix.inputs[1])
nt.links.new(bg_camera.outputs["Background"], mix.inputs[2])
nt.links.new(mix.outputs["Shader"], out.inputs["Surface"])
scene.world = world


def point_at(obj, target):
    obj.rotation_euler = (Vector(target) - obj.location).to_track_quat("-Z", "Y").to_euler()


def area(name, loc, target, energy, size_x, size_y, color=(1.0, 1.0, 1.0)):
    data = bpy.data.lights.new(name, "AREA")
    data.energy, data.color = energy, color
    data.shape, data.size, data.size_y = "RECTANGLE", size_x, size_y
    obj = bpy.data.objects.new(name, data)
    scene.collection.objects.link(obj)
    obj.location = loc
    point_at(obj, target)


focus = (0.45, 0.0, 0.85)
area("Key", (1.6, -1.2, 2.4), focus, args.key_energy, 1.2, 0.8, (1.0, 0.965, 0.92))
area("Fill", (0.4, 1.3, 2.1), focus, 50.0, 2.0, 1.4, (0.94, 0.96, 1.0))
area("Rim", (-1.1, 0.35, 2.3), focus, 120.0, 0.9, 0.5)

# --------------------------------------------------------------------------- cameras
visible = [centre(o) for o in props if not o.hide_render]
table_centre = Vector((0.58, 0.0, 0.80))
aim = 0.6 * (sum(visible, Vector()) / len(visible) if visible else table_centre) + 0.4 * table_centre
close_aim = (aim.x, aim.y, max(0.80, min(aim.z, 0.95)))   # table-level detail
wide_aim = (aim.x, aim.y, 0.98)                            # props and both arms


def lens_for_fovy(fovy_deg, sensor_h=24.0):
    return (sensor_h / 2.0) / math.tan(math.radians(fovy_deg) / 2.0)


robot_fovy = 58.0 if W / H < 1.5 else 45.0        # 58 on 4:3 is the sim camera; 45 matches a 16:9 crop of it
for name in ("Camera_top", "Camera_left", "Camera_right"):
    cam = bpy.data.objects.get(name)
    if cam and cam.type == "CAMERA":
        cam.data.sensor_fit, cam.data.sensor_height = "VERTICAL", 24.0
        cam.data.lens = lens_for_fovy(robot_fovy)
        cam.data.clip_start, cam.data.clip_end = 0.01, 100.0


def camera(name, loc, target, lens, fstop):
    data = bpy.data.cameras.new(name)
    data.lens, data.sensor_width, data.sensor_fit, data.clip_start = lens, 36.0, "HORIZONTAL", 0.01
    if args.fstop > 0:
        data.dof.use_dof = True
        data.dof.focus_distance = (Vector(loc) - Vector(target)).length
        data.dof.aperture_fstop = args.fstop if args.fstop > 0 else fstop
    obj = bpy.data.objects.new(name, data)
    scene.collection.objects.link(obj)
    obj.location = loc
    point_at(obj, target)
    return obj


OUTSIDE = {
    "front": ((2.15, 0.35, 1.45), wide_aim, 42.0, 4.0),
    "front_left": ((1.8, 1.3, 1.55), wide_aim, 38.0, 4.0),
    "front_right": ((1.8, -1.3, 1.55), wide_aim, 38.0, 4.0),
    "overhead": ((1.35, 0.0, 2.35), (0.5, 0.0, 0.8), 35.0, 5.6),
    "left_side": ((0.55, 1.75, 1.35), (0.55, 0.0, 0.85), 45.0, 4.0),
    "right_side": ((0.55, -1.75, 1.35), (0.55, 0.0, 0.85), 45.0, 4.0),
    "close": ((1.75, -0.6, 1.12), close_aim, 70.0, 2.8),
}
for side in ("left", "right"):
    anchor = bpy.data.objects.get(f"Camera_{side}")
    if anchor is not None:
        c = anchor.matrix_world.translation
        OUTSIDE[f"mount_{side}"] = (tuple(c + Vector((0.42, 0.28 if side == "left" else -0.28, 0.22))), tuple(c), 70.0, 8.0)
wrist = meta.get("active_wrist", "right")
CAMERAS = {
    "top": (bpy.data.objects.get("Camera_top"), False, (W, H)),
    "left": (bpy.data.objects.get("Camera_left"), False, (WW, WH)),
    "right": (bpy.data.objects.get("Camera_right"), False, (WW, WH)),
    "wrist": (bpy.data.objects.get(f"Camera_{wrist}"), False, (WW, WH)),
}
for key, (loc, target, lens, fstop) in OUTSIDE.items():
    CAMERAS[key] = (camera(f"Cam_{key}", loc, target, lens, fstop), True, (W, H))

# --------------------------------------------------------------------------- render settings
if args.engine == "EEVEE":
    scene.render.engine = "BLENDER_EEVEE"
    scene.eevee.taa_render_samples = args.samples
    scene.eevee.use_shadows = scene.eevee.use_raytracing = scene.eevee.use_fast_gi = True
    scene.eevee.shadow_ray_count, scene.eevee.shadow_step_count = 4, 8
else:
    scene.render.engine = "CYCLES"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    if args.device != "CPU":
        prefs.compute_device_type = args.device
        prefs.get_devices()
        for d in prefs.devices:
            d.use = d.type == args.device
        scene.cycles.device = "GPU"
        scene.cycles.denoising_use_gpu = True
    scene.cycles.samples = args.samples
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.adaptive_threshold = 0.03
    scene.cycles.use_denoising = args.denoiser != "NONE"
    if args.denoiser != "NONE":
        scene.cycles.denoiser = args.denoiser
    scene.cycles.denoising_input_passes = "RGB_ALBEDO_NORMAL"
    scene.cycles.caustics_reflective = scene.cycles.caustics_refractive = False
    scene.cycles.max_bounces, scene.cycles.diffuse_bounces, scene.cycles.glossy_bounces = 8, 4, 4
if args.threads:
    scene.render.threads_mode, scene.render.threads = "FIXED", args.threads
scene.render.use_motion_blur = args.motion_blur
scene.render.motion_blur_shutter = 0.5
scene.render.filter_size = 1.5
scene.render.use_persistent_data = True
scene.render.use_overwrite = False
scene.render.use_placeholder = False
scene.render.image_settings.file_format, scene.render.image_settings.color_mode = "PNG", "RGB"
scene.view_settings.view_transform, scene.view_settings.look = args.view_transform, args.look
scene.view_settings.exposure = args.exposure
if args.glare > 0:                                # bloom; compositor API differs before/after Blender 4.5
    tree = getattr(scene, "node_tree", None)
    if tree is None:
        tree = bpy.data.node_groups.new("Compositing", "CompositorNodeTree")
        tree.interface.new_socket("Image", in_out="OUTPUT", socket_type="NodeSocketColor")
        scene.compositing_node_group = tree
        output = tree.nodes.new("NodeGroupOutput")
    else:
        scene.use_nodes = True
        tree.nodes.clear()
        output = tree.nodes.new("CompositorNodeComposite")
    scene.render.use_compositing = True
    layers = tree.nodes.new("CompositorNodeRLayers")
    glare = tree.nodes.new("CompositorNodeGlare")
    if "Type" in glare.inputs:                       # Blender 5: type and settings are sockets
        glare.inputs["Type"].default_value = "Bloom"
        glare.inputs["Threshold"].default_value = 1.0
        glare.inputs["Strength"].default_value = args.glare
    else:
        glare.glare_type, glare.threshold, glare.mix = "BLOOM", 1.0, args.glare * 2.0 - 1.0
    tree.links.new(layers.outputs["Image"], glare.inputs["Image"])
    tree.links.new(glare.outputs["Image"], output.inputs[0])



def serve(path):
    """Set the poses streamed over `path` (see abc_sim.rendering.blender.live) and render the policy cameras."""
    for o in scene.objects:
        if o.parent is not None and o.parent.type == "EMPTY":    # the importer's root: keep world matrices, drop the hierarchy
            mw = o.matrix_world.copy()
            o.parent = None
            o.matrix_world = mw
    dg = bpy.context.evaluated_depsgraph_get()
    for o, _ in [x for pairs in by_body.values() for x in pairs]:
        if o.modifiers:                                  # Cycles instances a modifier-free mesh with a second user, so moving it only refits the top-level BVH
            o.data = bpy.data.meshes.new_from_object(o.evaluated_get(dg))
            o.modifiers.clear()
        user = bpy.data.objects.new(o.name + "_user", o.data)
        user.location.z = -100.0
        scene.collection.objects.link(user)
    scene.cycles.max_bounces = 4
    scene.render.image_settings.file_format = "BMP"
    scene.render.use_overwrite = True
    scene.render.resolution_x, scene.render.resolution_y, scene.render.resolution_percentage = W, H, 100
    cams = [(name, CAMERAS[name][0]) for name in meta["cameras"]]
    objs = [bpy.data.objects[f"Mesh_{g['usd']}"] for g in meta["geoms"]] + [cam for _, cam in cams]
    fix = None
    srv = socket.socket(socket.AF_UNIX)
    srv.bind(path)
    srv.listen(1)
    f = srv.accept()[0].makefile("rwb")
    while buf := f.read(len(objs) * 48):
        poses = [Matrix.Translation(p[:3]) @ Matrix((p[3:6], p[6:9], p[9:12])).to_4x4()
                 for p in np.frombuffer(buf, np.float32).reshape(-1, 12).tolist()]
        if fix is None:                                  # the exported state: keep whatever the importer baked into each object
            fix = [m.inverted() @ o.matrix_world for o, m in zip(objs, poses)]
        for o, m, s in zip(objs, poses, fix):
            o.matrix_world = m @ s
        for name, cam in cams:
            scene.camera = cam
            scene.render.filepath = f"{args.out}/{name}.bmp"
            bpy.ops.render.render(write_still=True)
            f.write(f"{scene.render.filepath}\n".encode())
        f.flush()


if args.serve:
    serve(args.serve)
for key in args.cams.split(",") if not args.serve else []:
    cam, hide_walls, (rw, rh) = CAMERAS[key]
    if cam is None:
        print("SKIP", key, flush=True)
        continue
    for w in walls.values():
        w.hide_render = hide_walls
    scene.camera = cam
    scene.render.resolution_x, scene.render.resolution_y, scene.render.resolution_percentage = rw, rh, 100
    scene.render.filepath = f"{args.out}/{key}/frame_#####"
    t0 = time.time()
    if frames == list(range(frames[0], frames[-1] + 1)):
        scene.frame_start, scene.frame_end, scene.frame_step = frames[0], frames[-1], 1
        bpy.ops.render.render(animation=True)
    else:
        for f in frames:
            scene.frame_start = scene.frame_end = f
            bpy.ops.render.render(animation=True)
    print(f"DONE {key} {len(frames)} frames {(time.time() - t0) / len(frames):.2f}s/frame", flush=True)
if args.save_blend:
    bpy.ops.wm.save_as_mainfile(filepath=args.save_blend)
