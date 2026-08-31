"""In-hand transfer asset discovery and scene XML assembly."""

from __future__ import annotations

import xml.etree.ElementTree as _ET
from typing import Any

import numpy as np

from ..core import _scaled_mesh_attr
from .paths import _MODELS_DIR, _Path

_BASE_SCENE_XML = _MODELS_DIR / "yam_inhand_transfer_base.xml"
_INHAND_TASK_ASSET_ROOT = _MODELS_DIR / "assets" / "task_inhand_transfer"

# Approved object categories for the inhand_transfer task.
# Edit this list to add/remove objects from the randomization pool.
INHAND_OBJECT_CATEGORIES: list[str] = [
    # copied from the lightwheel RoboCasa pack
    "dish_brush", "whisk", "salt_and_pepper_shaker", "cream_cheese_stick",
    "cheese_grater", "pizza_cutter",
    # copied from the objaverse RoboCasa pack
    "rolling_pin", "water_bottle", "can", "ladle",
]
_INHAND_CATEGORIES = INHAND_OBJECT_CATEGORIES  # internal alias
_INHAND_OBJECT_MASS_KG_BY_CATEGORY: dict[str, float] = {
    "cream_cheese_stick": 0.02,
    "ladle": 0.03,
    "whisk": 0.035,
    "pizza_cutter": 0.04,
    "dish_brush": 0.05,
    "water_bottle": 0.05,
    "rolling_pin": 0.06,
    "salt_and_pepper_shaker": 0.07,
    "can": 0.08,
    "cheese_grater": 0.10,
}
_CATEGORY_MESH_SCALE: dict[str, str] = {"wooden_spoon": "1 1 2.5"}
_X_MIN, _X_MAX = 0.42, 0.78
_Y_LEFT_MIN, _Y_LEFT_MAX = 0.10, 0.40
_Y_RIGHT_MIN, _Y_RIGHT_MAX = -0.40, -0.10
_OBJ_Z = 0.82
_INHAND_SCALE_FACTOR_RANGE = (0.90, 1.20)

def _inhand_asset_base(category: str) -> _Path:
    return _INHAND_TASK_ASSET_ROOT


def _inhand_get_variants(category: str) -> list[_Path]:
    cat_dir = _inhand_asset_base(category) / category
    variants = []
    for d in sorted(cat_dir.iterdir()):
        if not d.is_dir():
            continue
        try:
            parsed = _inhand_parse_model_xml(d)
            if parsed["col_geoms"]:
                variants.append(d)
        except Exception:
            pass
    return variants


def _inhand_parse_model_xml(variant_dir: _Path) -> dict:
    root = _ET.parse(str(variant_dir / "model.xml")).getroot()
    asset_elem = root.find("asset")
    meshes, textures, materials = [], [], []
    if asset_elem is not None:
        for mesh in asset_elem.findall("mesh"):
            extra = {k: v for k, v in mesh.attrib.items() if k not in ("name", "file")}
            meshes.append({"name": mesh.get("name", ""), "file": mesh.get("file", ""), "extra": extra})
        for tex in asset_elem.findall("texture"):
            rel = tex.get("file", "")
            if rel:
                textures.append({"name": tex.get("name", ""), "file": rel, "type": tex.get("type", "2d")})
        for mat in asset_elem.findall("material"):
            materials.append({
                "name": mat.get("name", ""), "texture": mat.get("texture", ""),
                "rgba": mat.get("rgba", ""), "shininess": mat.get("shininess", ""),
                "specular": mat.get("specular", ""),
            })
    vis_geoms, col_geoms = [], []
    bbox_pos: list[float] | None = None
    bbox_size: list[float] | None = None
    worldbody = root.find("worldbody")
    if worldbody is not None:
        for geom in worldbody.iter("geom"):
            cls = geom.get("class", "")
            if geom.get("name") == "reg_bbox":
                bbox_pos = [float(value) for value in geom.get("pos", "0 0 0").split()]
                bbox_size = [float(value) for value in geom.get("size", "").split()]
                continue
            is_visual = cls == "visual" or (geom.get("contype") == "0" and cls != "region")
            if is_visual:
                vis_geoms.append({"mesh": geom.get("mesh", ""), "material": geom.get("material", "")})
            elif cls == "collision":
                attrs = {k: v for k, v in geom.attrib.items() if k not in ("name", "class")}
                attrs.setdefault("mesh", geom.get("mesh", ""))
                attrs.setdefault("type", geom.get("type", "mesh"))
                col_geoms.append(attrs)
    return {"meshes": meshes, "textures": textures, "materials": materials,
            "vis_geoms": vis_geoms, "col_geoms": col_geoms,
            "bbox_pos": bbox_pos, "bbox_size": bbox_size}


def _inhand_apply_scene_transforms(xml: str, options: Any = None) -> str:
    if options is None:
        return xml
    if not (options.clean or options.mocap or options.flexible_gripper):
        return xml

    from abc_sim.scene_xml import transform_scene_xml

    transformed_xml, _ = transform_scene_xml(xml, options=options)
    return transformed_xml


def _inhand_build_xml(
    category: str,
    variant_dir: _Path,
    x: float,
    y: float,
    z: float,
    yaw: float,
    *,
    scale_factor: float = 1.0,
) -> str:
    base_text = _BASE_SCENE_XML.read_text()
    parsed = _inhand_parse_model_xml(variant_dir)
    prefix = "obj"
    cat_scale = _CATEGORY_MESH_SCALE.get(category)

    lines_asset = [f"    <!-- Object: {category}/{variant_dir.name} -->"]
    for m in parsed["meshes"]:
        extra = {**m["extra"]}
        base_scale = extra.get("scale") or cat_scale
        if base_scale or abs(scale_factor - 1.0) > 1e-9:
            extra["scale"] = _scaled_mesh_attr(base_scale, scale_factor)
        extra_str = "".join(f' {k}="{v}"' for k, v in extra.items())
        lines_asset.append(f'    <mesh file="{variant_dir / m["file"]}" name="{prefix}_{m["name"]}"{extra_str}/>')
    for t in parsed["textures"]:
        lines_asset.append(f'    <texture file="{variant_dir / t["file"]}" name="{prefix}_{t["name"]}" type="{t["type"]}"/>')
    for mat in parsed["materials"]:
        attrs = f'name="{prefix}_{mat["name"]}"'
        if mat["texture"]: attrs += f' texture="{prefix}_{mat["texture"]}"'
        if mat["rgba"]:    attrs += f' rgba="{mat["rgba"]}"'
        if mat["shininess"]: attrs += f' shininess="{mat["shininess"]}"'
        if mat["specular"]:  attrs += f' specular="{mat["specular"]}"'
        lines_asset.append(f"    <material {attrs}/>")

    w, s = np.cos(yaw / 2), np.sin(yaw / 2)
    lines_body = [
        f"    <!-- Task object: {category}/{variant_dir.name} -->",
        f'    <body name="task_object" pos="{x:.4f} {y:.4f} {z:.4f}" quat="{w:.6f} 0 0 {s:.6f}">',
        f'      <freejoint name="task_object_joint"/>',
    ]
    for vg in parsed["vis_geoms"]:
        mesh_attr = f'mesh="{prefix}_{vg["mesh"]}"' if vg["mesh"] else ""
        mat_attr = f' material="{prefix}_{vg["material"]}"' if vg["material"] else ""
        lines_body.append(
            f'      <geom type="mesh" {mesh_attr}{mat_attr} contype="0" conaffinity="0" group="2"'
            f' density="0" solimp="0.998 0.998 0.001" solref="0.001 1"/>')
    object_mass = _INHAND_OBJECT_MASS_KG_BY_CATEGORY.get(category, 0.05)
    collision_mass = object_mass / max(len(parsed["col_geoms"]), 1)
    for cg in parsed["col_geoms"]:
        mesh_attr = f'mesh="{prefix}_{cg["mesh"]}"' if cg["mesh"] else ""
        lines_body.append(
            f'      <geom type="mesh" {mesh_attr} group="3" density="0" mass="{collision_mass:.8f}"'
            f' condim="6" friction="3.0 0.03 0.003" solimp="0.998 0.998 0.001"'
            f' solref="0.004 1" priority="1"/>')
    lines_body.append("    </body>")

    result = base_text.replace("<!-- TASK_ASSETS_PLACEHOLDER -->", "\n".join(lines_asset))
    result = result.replace("<!-- TASK_BODY_PLACEHOLDER -->", "\n".join(lines_body))
    result = result.replace('file="i2rt_yam/', f'file="{_MODELS_DIR}/assets/i2rt_yam/')
    return result


__all__ = [name for name in globals() if (name.startswith('_') and not name.startswith('__')) or name == 'INHAND_OBJECT_CATEGORIES']
