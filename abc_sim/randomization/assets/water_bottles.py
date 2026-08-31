"""Water-bottle asset discovery and scene XML assembly."""

from __future__ import annotations

import copy
import xml.etree.ElementTree as _ET
from functools import lru_cache as _lru_cache

import mujoco
import numpy as np

from ..core import (
    _apply_object_scales_to_scene_xml,
    _format_float_list,
    _quat_from_axis_angle,
    _quat_from_yaw,
    _quat_mul,
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
    _dishrack_serialize_elements,
    _dishrack_shift_body,
)

_WATER_BOTTLE_BASE_SCENE_XML = _MODELS_DIR / "yam_put_bottles_scene.xml"
_WATER_BOTTLE_TASK_ASSET_ROOT = _MODELS_DIR / "assets" / "task_water_bottles" / "bottle"
_WATER_BOTTLE_DEFAULT_VARIANT = "bottle_0"
_WATER_BOTTLE_MIN_COUNT = 2
_WATER_BOTTLE_MAX_COUNT = 6
_WATER_BOTTLE_MASS_KG = 0.05
_WATER_BOTTLE_TABLE_Z = 0.75
_WATER_BOTTLE_FLAT_SPAWN_CLEARANCE_M = 0.002
_WATER_BOTTLE_WRAPPER_XY: tuple[tuple[float, float], ...] = (
    (0.46, -0.38),
    (0.62, -0.38),
    (0.78, -0.38),
    (0.46, 0.38),
    (0.62, 0.38),
    (0.78, 0.38),
)
_WATER_BOTTLE_WRAPPER_Z = 0.754

def _water_bottle_variant_names() -> list[str]:
    variants = [
        path.name
        for path in _WATER_BOTTLE_TASK_ASSET_ROOT.iterdir()
        if path.is_dir() and (path / "model.xml").exists()
    ]
    variants.sort(
        key=lambda name: (
            0,
            int(name[len("bottle_") :]),
        )
        if name.startswith("bottle_") and name[len("bottle_") :].isdigit()
        else (1, name)
    )
    if not variants:
        raise FileNotFoundError(f"No water bottle variants found under {_WATER_BOTTLE_TASK_ASSET_ROOT}")
    return variants


def _water_bottle_canonical_variant_name(variant_name: str) -> str:
    if variant_name == "current":
        return _WATER_BOTTLE_DEFAULT_VARIANT
    if variant_name.isdigit():
        return f"bottle_{variant_name}"
    if variant_name.startswith("bottle") and variant_name[6:].isdigit():
        return f"bottle_{variant_name[6:]}"
    return variant_name


def _water_bottle_variant_dir(variant_name: str) -> _Path:
    variant_name = _water_bottle_canonical_variant_name(variant_name)
    path = _WATER_BOTTLE_TASK_ASSET_ROOT / variant_name
    if not (path / "model.xml").exists():
        raise FileNotFoundError(f"Missing water bottle variant model.xml: {path}")
    return path


def _water_bottle_body_name(index: int) -> str:
    return f"bottle_{index + 1}"


def _water_bottle_joint_name(index: int) -> str:
    return f"{_water_bottle_body_name(index)}_joint"


def _water_bottle_instance_prefix(variant_name: str, instance_index: int) -> str:
    return f"bottle_{instance_index}_{variant_name}"


@_lru_cache(maxsize=None)
def _water_bottle_compiled_raw_metadata(
    variant_name: str,
) -> tuple[float, float, float, float, float, float]:
    variant_dir = _water_bottle_variant_dir(variant_name)
    model = mujoco.MjModel.from_xml_path(str(variant_dir / "model.xml"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    min_xyz, max_xyz = _dishrack_compiled_model_bounds(model, data)
    half_xy = 0.5 * (max_xyz[:2] - min_xyz[:2])
    center_xy = 0.5 * (min_xyz[:2] + max_xyz[:2])
    return (
        float(half_xy[0]),
        float(half_xy[1]),
        -float(center_xy[0]),
        -float(center_xy[1]),
        -float(min_xyz[2]),
        float(max_xyz[2] - min_xyz[2]),
    )


@_lru_cache(maxsize=None)
def _water_bottle_compiled_metadata(variant_name: str) -> tuple[float, float, float, float]:
    half_x, half_y, _offset_x, _offset_y, _offset_z, height = _water_bottle_compiled_raw_metadata(variant_name)
    return half_x, half_y, 0.0, height


@_lru_cache(maxsize=None)
def _water_bottle_compiled_anchor_offset(variant_name: str) -> tuple[float, float, float]:
    _half_x, _half_y, offset_x, offset_y, offset_z, _height = _water_bottle_compiled_raw_metadata(variant_name)
    return offset_x, offset_y, offset_z


@_lru_cache(maxsize=None)
def _water_bottle_flat_compiled_metadata(variant_name: str) -> tuple[float, float, float]:
    half_x, half_y, _offset_x, _offset_y, _offset_z, height = _water_bottle_compiled_raw_metadata(variant_name)
    return 0.5 * height, half_y, half_x


def _quat_rotate_vector(quat: np.ndarray, vector: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64)
    v = np.asarray(vector, dtype=np.float64)
    q_conj = np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)
    rotated = _quat_mul(_quat_mul(q, np.array([0.0, v[0], v[1], v[2]], dtype=np.float64)), q_conj)
    return rotated[1:]


def _water_bottle_flat_quat(yaw: float) -> np.ndarray:
    flat_quat = _quat_from_axis_angle(np.array([0.0, 1.0, 0.0], dtype=np.float64), np.pi / 2.0)
    quat = _quat_mul(_quat_from_yaw(yaw), flat_quat)
    norm = float(np.linalg.norm(quat))
    if norm > 0.0:
        quat = quat / norm
    return quat


def _water_bottle_flat_yaw_from_quat(quat: np.ndarray) -> float:
    long_axis = _quat_rotate_vector(np.asarray(quat, dtype=np.float64), np.array([0.0, 0.0, 1.0]))
    return float(np.arctan2(long_axis[1], long_axis[0]))


def _water_bottle_flat_center_from_pose(
    *,
    pos: list[float] | tuple[float, ...],
    quat: list[float] | tuple[float, ...] | np.ndarray,
    variant_name: str,
    scale_factor: float,
) -> np.ndarray:
    yaw = _water_bottle_flat_yaw_from_quat(np.asarray(quat, dtype=np.float64))
    half_length, _half_radius, _vertical_radius = _water_bottle_flat_compiled_metadata(variant_name)
    offset = np.array([np.cos(yaw), np.sin(yaw)], dtype=np.float64) * half_length * float(scale_factor)
    return np.asarray(pos[:2], dtype=np.float64) + offset


def _water_bottle_prepare_imported_geoms(body: _ET.Element) -> None:
    for parent in body.iter():
        for child in list(parent):
            if child.tag != "geom":
                continue

            name = child.get("name", "")
            child_class = child.get("class", "")
            mesh_name = child.get("mesh")
            if (
                child_class == "region"
                or name.startswith("reg_")
                or mesh_name is None
            ):
                parent.remove(child)
                continue

            is_visual = child.get("contype") == "0" and child.get("conaffinity") == "0"
            child.attrib.pop("class", None)
            child.attrib.pop("mass", None)
            child.set("density", "0")
            if is_visual:
                child.set("group", "2")
                child.set("contype", "0")
                child.set("conaffinity", "0")
                continue

            child.set("group", "3")
            child.set("rgba", "0 0 0 0")
            child.set("friction", "3.0 0.03 0.003")
            child.set("condim", "6")
            child.set("solref", "0.004 1")
            child.set("solimp", "0.998 0.998 0.001")
            child.set("priority", "1")


def _water_bottle_add_inertial(
    body: _ET.Element,
    *,
    variant_name: str,
    scale_factor: float,
) -> None:
    for child in list(body):
        if child.tag == "inertial":
            body.remove(child)

    half_x, half_y, offset_x, offset_y, offset_z, height = _water_bottle_compiled_raw_metadata(variant_name)
    scale_factor = float(scale_factor)
    mass = _WATER_BOTTLE_MASS_KG
    size_x = 2.0 * half_x * scale_factor
    size_y = 2.0 * half_y * scale_factor
    size_z = height * scale_factor
    inertia = [
        mass * (size_y * size_y + size_z * size_z) / 12.0,
        mass * (size_x * size_x + size_z * size_z) / 12.0,
        mass * (size_x * size_x + size_y * size_y) / 12.0,
    ]
    com_pos = [
        -offset_x * scale_factor,
        -offset_y * scale_factor,
        -offset_z * scale_factor + 0.5 * size_z,
    ]
    inertial = _ET.Element(
        "inertial",
        pos=_format_float_list(com_pos),
        mass=_format_float_list([mass]),
        diaginertia=_format_float_list(inertia),
    )
    body.insert(0, inertial)


def _water_bottle_build_object_block(
    *,
    variant_name: str,
    instance_index: int,
    scale_factor: float,
    object_name: str,
    joint_name: str,
) -> tuple[list[_ET.Element], _ET.Element]:
    variant_name = _water_bottle_canonical_variant_name(variant_name)
    variant_dir = _water_bottle_variant_dir(variant_name)
    root = _ET.parse(str(variant_dir / "model.xml")).getroot()
    object_body = _dishrack_find_object_body(root)
    offset = np.asarray(_water_bottle_compiled_anchor_offset(variant_name), dtype=np.float64) * float(scale_factor)

    prefix = _water_bottle_instance_prefix(variant_name, instance_index)
    mesh_map: dict[str, str] = {}
    texture_map: dict[str, str] = {}
    material_map: dict[str, str] = {}
    temp_asset = _ET.Element("asset")

    asset_root = root.find("asset")
    if asset_root is not None:
        asset_children = list(asset_root)
        for asset_child in asset_children:
            local_name = _dishrack_asset_local_name(asset_child)
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
                _dishrack_absolutize_file_attr(cloned, variant_dir)
            elif asset_child.tag == "texture":
                cloned.set("name", texture_map[local_name])
                _dishrack_absolutize_file_attr(cloned, variant_dir)
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
    _water_bottle_prepare_imported_geoms(cloned_body)
    _dishrack_shift_body(cloned_body, offset)

    if abs(scale_factor - 1.0) > 1e-9:
        mesh_assets = {
            mesh.get("name", ""): mesh
            for mesh in temp_asset.findall("mesh")
            if mesh.get("name")
        }
        _scale_body_subtree(
            body=cloned_body,
            factor=scale_factor,
            target_name=joint_name,
            asset_elem=temp_asset,
            mesh_assets=mesh_assets,
        )

    _water_bottle_add_inertial(
        cloned_body,
        variant_name=variant_name,
        scale_factor=scale_factor,
    )

    x, y = _WATER_BOTTLE_WRAPPER_XY[instance_index]
    wrapper = _ET.Element(
        "body",
        name=object_name,
        pos=_format_float_list([x, y, _WATER_BOTTLE_WRAPPER_Z]),
    )
    _ET.SubElement(wrapper, "freejoint", name=joint_name)
    wrapper.append(cloned_body)
    return list(temp_asset), wrapper


def _build_water_bottle_scene_xml(
    *,
    bottle_variants: list[str] | tuple[str, ...],
    scale_states: dict[str, float],
    base_scene_xml: str | None,
    base_scene_dir: _Path | None,
) -> str:
    if not _WATER_BOTTLE_MIN_COUNT <= len(bottle_variants) <= _WATER_BOTTLE_MAX_COUNT:
        raise ValueError(
            "put_bottles requires between "
            f"{_WATER_BOTTLE_MIN_COUNT} and "
            f"{_WATER_BOTTLE_MAX_COUNT} bottle variants, got {len(bottle_variants)}"
        )

    base_text = base_scene_xml or _WATER_BOTTLE_BASE_SCENE_XML.read_text()
    base_dir = base_scene_dir or _WATER_BOTTLE_BASE_SCENE_XML.parent
    task_assets: list[_ET.Element] = []
    task_bodies: list[_ET.Element] = []

    for index, variant_name in enumerate(bottle_variants):
        body_name = _water_bottle_body_name(index)
        joint_name = _water_bottle_joint_name(index)
        variant_assets, bottle_body = _water_bottle_build_object_block(
            variant_name=variant_name,
            instance_index=index,
            scale_factor=float(scale_states.get(joint_name, 1.0)),
            object_name=body_name,
            joint_name=joint_name,
        )
        task_assets.extend(variant_assets)
        task_bodies.append(bottle_body)

    xml = base_text.replace("<!-- TASK_DEFAULTS_PLACEHOLDER -->", "")
    xml = xml.replace("<!-- TASK_ASSETS_PLACEHOLDER -->", _dishrack_serialize_elements(task_assets))
    xml = xml.replace("<!-- TASK_BODY_PLACEHOLDER -->", _dishrack_serialize_elements(task_bodies))
    bin_scale = float(scale_states.get("bin_joint", 1.0))
    if abs(bin_scale - 1.0) > 1e-9:
        xml = _apply_object_scales_to_scene_xml(xml, {"bin_joint": bin_scale})
    return _resolve_scene_xml_paths(xml, base_dir)




__all__ = [name for name in globals() if name.startswith('_') and not name.startswith('__')]
