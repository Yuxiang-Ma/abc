"""Chess tin asset discovery and scene XML assembly."""

from __future__ import annotations

import copy
import xml.etree.ElementTree as _ET
from functools import lru_cache as _lru_cache

import mujoco
import numpy as np

from ..core import (
    _apply_object_scales_to_scene_xml,
    _format_float_list,
    _resolve_scene_xml_paths,
    _scale_body_subtree,
)
from .paths import _MODELS_DIR, _Path
from .xml_common import (
    _dishrack_absolutize_file_attr,
    _dishrack_asset_local_name,
    _dishrack_compiled_model_bounds,
    _dishrack_find_object_body,
    _dishrack_prefixed_name,
    _dishrack_prefix_body_names,
    _dishrack_remove_dynamic_joints,
    _dishrack_rewrite_asset_refs,
    _dishrack_shift_body,
)

_CHESS_BASE_SCENE_XML = _MODELS_DIR / "yam_chess_scene.xml"
_CHESS_TIN_ASSET_ROOT = _MODELS_DIR / "assets" / "task_chess" / "tin_box"
_CHESS_TIN_DEFAULT_VARIANT = "tin_2"
_CHESS_TIN_TARGET_FOOTPRINT_MAX = 0.357
_CHESS_TIN_VISUAL_MARGIN_M = 0.004
_CHESS_TIN_WALL_THICKNESS_M = 0.022
_CHESS_TIN_FLOOR_HALF_Z = 0.006
_CHESS_TABLE_Z = 0.75
_CHESS_TIN_JOINT = "tin_box_joint"

def _chess_tin_variant_names() -> list[str]:
    if not (_CHESS_TIN_ASSET_ROOT / _CHESS_TIN_DEFAULT_VARIANT / "body" / "body.xml").exists():
        raise FileNotFoundError(
            "Missing chess tin body.xml: "
            f"{_CHESS_TIN_ASSET_ROOT / _CHESS_TIN_DEFAULT_VARIANT / 'body' / 'body.xml'}"
        )
    return [_CHESS_TIN_DEFAULT_VARIANT]


def _chess_tin_canonical_variant_name(variant_name: str) -> str:
    if variant_name == "current":
        return _CHESS_TIN_DEFAULT_VARIANT
    if variant_name.isdigit():
        return f"tin_{variant_name}"
    if variant_name.startswith("tin") and variant_name[3:].isdigit():
        return f"tin_{variant_name[3:]}"
    return variant_name


def _chess_tin_variant_dir(variant_name: str) -> _Path:
    variant_name = _chess_tin_canonical_variant_name(variant_name)
    path = _CHESS_TIN_ASSET_ROOT / variant_name
    if not (path / "body" / "body.xml").exists():
        raise FileNotFoundError(f"Missing chess tin body.xml: {path / 'body' / 'body.xml'}")
    return path


@_lru_cache(maxsize=None)
def _chess_tin_compiled_metadata(variant_name: str) -> tuple[float, float, float, float, float, float]:
    variant_dir = _chess_tin_variant_dir(variant_name)
    model = mujoco.MjModel.from_xml_path(str(variant_dir / "body" / "body.xml"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    min_xyz, max_xyz = _dishrack_compiled_model_bounds(model, data)
    half_xy = 0.5 * (max_xyz[:2] - min_xyz[:2])
    center_xy = 0.5 * (min_xyz[:2] + max_xyz[:2])
    return (
        float(half_xy[0]),
        float(half_xy[1]),
        float(min_xyz[2]),
        float(max_xyz[2]),
        float(center_xy[0]),
        float(center_xy[1]),
    )


@_lru_cache(maxsize=None)
def _chess_tin_variant_scale(variant_name: str) -> float:
    half_x, half_y, *_ = _chess_tin_compiled_metadata(variant_name)
    max_extent = 2.0 * max(half_x, half_y)
    if max_extent <= 0.0:
        return 1.0
    return max(1.0, _CHESS_TIN_TARGET_FOOTPRINT_MAX / max_extent)


@_lru_cache(maxsize=None)
def _chess_tin_scaled_outer_half_xy(variant_name: str) -> tuple[float, float]:
    half_x, half_y, *_ = _chess_tin_compiled_metadata(variant_name)
    scale = _chess_tin_variant_scale(variant_name)
    return (
        float(half_x * scale + _CHESS_TIN_VISUAL_MARGIN_M),
        float(half_y * scale + _CHESS_TIN_VISUAL_MARGIN_M),
    )


@_lru_cache(maxsize=None)
def _chess_tin_scaled_inner_half_xy(variant_name: str) -> tuple[float, float]:
    outer_x, outer_y = _chess_tin_scaled_outer_half_xy(variant_name)
    wall = _CHESS_TIN_WALL_THICKNESS_M
    return (
        max(0.040, float(outer_x - wall)),
        max(0.040, float(outer_y - wall)),
    )


def _chess_tin_instance_prefix(variant_name: str) -> str:
    return f"chess_tin_{_chess_tin_canonical_variant_name(variant_name)}"


def _chess_tin_prepare_imported_visual_geoms(body: _ET.Element) -> None:
    for parent in body.iter():
        for child in list(parent):
            if child.tag != "geom":
                continue

            mesh_name = child.get("mesh", "")
            is_visual = (
                child.get("class") == "visual"
                or (
                    child.get("contype") == "0"
                    and child.get("conaffinity") == "0"
                    and "collision" not in mesh_name
                )
            )
            if not is_visual:
                parent.remove(child)
                continue

            child.attrib.pop("class", None)
            child.set("type", "mesh")
            if child.get("group") is None:
                child.set("group", "2")
            child.set("contype", "0")
            child.set("conaffinity", "0")
            child.set("density", "0")
            child.attrib.pop("mass", None)


def _chess_tin_visual_anchor_offset(variant_name: str) -> np.ndarray:
    scale = _chess_tin_variant_scale(variant_name)
    _half_x, _half_y, min_z, _max_z, center_x, center_y = _chess_tin_compiled_metadata(variant_name)
    return np.array(
        [
            -center_x * scale,
            -center_y * scale,
            -_CHESS_TIN_FLOOR_HALF_Z - min_z * scale,
        ],
        dtype=np.float64,
    )


def _chess_tin_build_visual_body_block(
    *,
    variant_name: str,
) -> tuple[list[_ET.Element], _ET.Element]:
    variant_name = _chess_tin_canonical_variant_name(variant_name)
    variant_dir = _chess_tin_variant_dir(variant_name)
    body_dir = variant_dir / "body"
    root = _ET.parse(str(body_dir / "body.xml")).getroot()
    object_body = _dishrack_find_object_body(root)

    prefix = _chess_tin_instance_prefix(variant_name)
    mesh_map: dict[str, str] = {}
    texture_map: dict[str, str] = {}
    material_map: dict[str, str] = {}
    temp_asset = _ET.Element("asset")

    asset_root = root.find("asset")
    if asset_root is not None:
        asset_children = []
        visual_mesh_names = {
            geom.get("mesh", "")
            for geom in object_body.iter("geom")
            if geom.get("mesh")
            and (
                geom.get("class") == "visual"
                or (geom.get("contype") == "0" and geom.get("conaffinity") == "0")
            )
        }
        for asset_child in list(asset_root):
            local_name = _dishrack_asset_local_name(asset_child)
            if (
                asset_child.tag == "mesh"
                and "collision" in local_name
                and local_name not in visual_mesh_names
            ):
                continue
            asset_children.append(asset_child)
            if asset_child.tag == "mesh":
                mesh_map[local_name] = _dishrack_prefixed_name(prefix, local_name)
            elif asset_child.tag == "texture":
                texture_map[local_name] = _dishrack_prefixed_name(prefix, local_name)
            elif asset_child.tag == "material":
                material_map[local_name] = _dishrack_prefixed_name(prefix, local_name)

        for asset_child in asset_children:
            cloned = copy.deepcopy(asset_child)
            local_name = _dishrack_asset_local_name(asset_child)
            if asset_child.tag == "mesh":
                cloned.set("name", mesh_map[local_name])
                _dishrack_absolutize_file_attr(cloned, body_dir)
            elif asset_child.tag == "texture":
                cloned.set("name", texture_map[local_name])
                _dishrack_absolutize_file_attr(cloned, body_dir)
            elif asset_child.tag == "material":
                cloned.set("name", material_map[local_name])
                texture_name = asset_child.get("texture")
                if texture_name and texture_name in texture_map:
                    cloned.set("texture", texture_map[texture_name])
            temp_asset.append(cloned)

    cloned_body = copy.deepcopy(object_body)
    _dishrack_remove_dynamic_joints(cloned_body)
    _dishrack_prefix_body_names(cloned_body, prefix)
    _dishrack_rewrite_asset_refs(
        cloned_body,
        mesh_map=mesh_map,
        material_map=material_map,
    )
    _chess_tin_prepare_imported_visual_geoms(cloned_body)

    scale_factor = _chess_tin_variant_scale(variant_name)
    if abs(scale_factor - 1.0) > 1e-9:
        mesh_assets = {
            mesh.get("name", ""): mesh
            for mesh in temp_asset.findall("mesh")
            if mesh.get("name")
        }
        _scale_body_subtree(
            body=cloned_body,
            factor=scale_factor,
            target_name=_CHESS_TIN_JOINT,
            asset_elem=temp_asset,
            mesh_assets=mesh_assets,
        )

    _dishrack_shift_body(cloned_body, _chess_tin_visual_anchor_offset(variant_name))
    return list(temp_asset), cloned_body


def _chess_tin_collision_geoms(variant_name: str) -> list[_ET.Element]:
    outer_x, outer_y = _chess_tin_scaled_outer_half_xy(variant_name)
    inner_x, inner_y = _chess_tin_scaled_inner_half_xy(variant_name)
    wall_x = max(0.006, outer_x - inner_x)
    wall_y = max(0.006, outer_y - inner_y)
    _half_x, _half_y, min_z, max_z, _center_x, _center_y = _chess_tin_compiled_metadata(variant_name)
    visual_height = max(0.0, (max_z - min_z) * _chess_tin_variant_scale(variant_name))
    wall_half_z = max(0.035, min(0.095, 0.5 * visual_height))
    floor_half_z = _CHESS_TIN_FLOOR_HALF_Z
    wall_z = floor_half_z + wall_half_z

    specs = (
        ("tin_box_floor", [outer_x, outer_y, floor_half_z], [0.0, 0.0, 0.0], 0.060),
        ("tin_box_wall_y_pos", [outer_x, 0.5 * wall_y, wall_half_z], [0.0, inner_y + 0.5 * wall_y, wall_z], 0.015),
        ("tin_box_wall_y_neg", [outer_x, 0.5 * wall_y, wall_half_z], [0.0, -inner_y - 0.5 * wall_y, wall_z], 0.015),
        ("tin_box_wall_x_pos", [0.5 * wall_x, inner_y, wall_half_z], [inner_x + 0.5 * wall_x, 0.0, wall_z], 0.015),
        ("tin_box_wall_x_neg", [0.5 * wall_x, inner_y, wall_half_z], [-inner_x - 0.5 * wall_x, 0.0, wall_z], 0.015),
    )
    geoms: list[_ET.Element] = []
    for name, size, pos, mass in specs:
        geoms.append(
            _ET.Element(
                "geom",
                name=name,
                type="box",
                size=_format_float_list(size),
                pos=_format_float_list(pos),
                material="tin_box_mat",
                mass=f"{mass:.3f}",
                group="3",
                rgba="0 0 0 0",
                friction="2.0 0.1 0.01",
                condim="6",
                solref="0.004 1",
                solimp="0.998 0.998 0.001",
                priority="1",
            )
        )
    return geoms


def _chess_remove_body_by_name(root: _ET.Element, body_name: str) -> tuple[_ET.Element, int] | None:
    for parent in root.iter():
        children = list(parent)
        for index, child in enumerate(children):
            if child.tag == "body" and child.get("name") == body_name:
                parent.remove(child)
                return parent, index
    return None


def _build_chess_tin_scene_xml(
    *,
    tin_variant: str,
    scale_states: dict[str, float],
    base_scene_xml: str | None,
    base_scene_dir: _Path | None,
) -> str:
    tin_variant = _chess_tin_canonical_variant_name(tin_variant)
    base_text = base_scene_xml or _CHESS_BASE_SCENE_XML.read_text()
    base_dir = base_scene_dir or _CHESS_BASE_SCENE_XML.parent
    root = _ET.fromstring(base_text)
    asset_elem = root.find("asset")
    if asset_elem is None:
        raise ValueError("Chess scene XML is missing <asset>")
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("Chess scene XML is missing <worldbody>")

    for child in list(asset_elem):
        name = child.get("name", "")
        if name.startswith("chess_tin_"):
            asset_elem.remove(child)

    removed = _chess_remove_body_by_name(root, "tin_box")
    parent, insert_index = removed if removed is not None else (worldbody, len(list(worldbody)))

    tin_assets, visual_body = _chess_tin_build_visual_body_block(variant_name=tin_variant)
    for asset in tin_assets:
        asset_elem.append(asset)

    wrapper = _ET.Element(
        "body",
        name="tin_box",
        pos=_format_float_list([0.6, 0.460, _CHESS_TABLE_Z + _CHESS_TIN_FLOOR_HALF_Z + 0.001]),
    )
    _ET.SubElement(wrapper, "freejoint", name=_CHESS_TIN_JOINT)
    for geom in _chess_tin_collision_geoms(tin_variant):
        wrapper.append(geom)
    wrapper.append(visual_body)
    parent.insert(insert_index, wrapper)

    xml = _ET.tostring(root, encoding="unicode")
    if scale_states:
        xml = _apply_object_scales_to_scene_xml(xml, scale_states)
    return _resolve_scene_xml_paths(xml, base_dir)




__all__ = [name for name in globals() if name.startswith('_') and not name.startswith('__')]
