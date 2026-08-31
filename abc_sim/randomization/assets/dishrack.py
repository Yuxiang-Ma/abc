"""Dishrack asset discovery and scene XML assembly."""

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
    _dishrack_prepare_imported_geoms,
    _dishrack_prefix_body_names,
    _dishrack_remove_dynamic_joints,
    _dishrack_rewrite_asset_refs,
    _dishrack_serialize_elements,
    _dishrack_shift_body,
)

_DISHRACK_BASE_SCENE_XML = _MODELS_DIR / "yam_dishrack_base.xml"
_DISHRACK_TASK_ASSET_ROOT = _MODELS_DIR / "assets" / "task_dishrack"
_DISHRACK_MAX_PLATE_COUNT = 4
_DISHRACK_VARIANT_ROOTS: dict[str, _Path] = {
    "plate": _DISHRACK_TASK_ASSET_ROOT / "plate",
    "dish_rack": _DISHRACK_TASK_ASSET_ROOT / "dish_rack",
}
_DISHRACK_DEFAULT_VARIANTS: dict[str, str] = {
    "dish_rack": "dish_rack_0",
    "plate": "plate_0",
}
_DISHRACK_EXCLUDED_VARIANTS: dict[str, set[str]] = {
    # Broken asset that consistently causes failed dishrack eval episodes.
    # Keep it loadable through _dishrack_variant_dir so old traces can replay,
    # but exclude it from random sampling, cycling, and explicit eval requests.
    "dish_rack": {"dish_rack_10"},
    "plate": set(),
}
_DISHRACK_VARIANT_ALIASES: dict[str, dict[str, str]] = {
    "dish_rack": {
        "current": "dish_rack_0",
        "DishRack026": "dish_rack_1",
        "DishRack027": "dish_rack_2",
        "DishRack028": "dish_rack_3",
        "DishRack030": "dish_rack_4",
        "DishRack038": "dish_rack_5",
        "DishRack039": "dish_rack_6",
        "DishRack040": "dish_rack_7",
        "DishRack041": "dish_rack_8",
        "DishRack043": "dish_rack_9",
        "DishRack044": "dish_rack_10",
        "DishRack047": "dish_rack_11",
        "DishRack050": "dish_rack_12",
    },
    "plate": {
        "current": "plate_0",
    },
}
_DISHRACK_RACK_WRAPPER_POSITION: tuple[float, float, float] = (0.62, -0.24, 0.75)
_DISHRACK_PLATE_WRAPPER_XY: tuple[tuple[float, float], ...] = (
    (0.50, 0.20),
    (0.72, 0.20),
    (0.50, 0.44),
    (0.72, 0.44),
)
_DISHRACK_PLATE_WRAPPER_Z: float = 0.75

def _dishrack_variant_names(kind: str) -> list[str]:
    root = _DISHRACK_VARIANT_ROOTS[kind]
    excluded = _DISHRACK_EXCLUDED_VARIANTS.get(kind, set())
    variants = [
        path.name
        for path in root.iterdir()
        if path.is_dir() and (path / "model.xml").exists() and path.name not in excluded
    ]
    prefix = "plate_" if kind == "plate" else "dish_rack_"
    variants.sort(
        key=lambda name: (
            0,
            int(name[len(prefix) :]),
        )
        if name.startswith(prefix) and name[len(prefix) :].isdigit()
        else (1, name)
    )
    return variants


def _dishrack_canonical_variant_name(kind: str, variant_name: str) -> str:
    return _DISHRACK_VARIANT_ALIASES.get(kind, {}).get(variant_name, variant_name)


def _dishrack_variant_dir(kind: str, variant_name: str) -> _Path:
    variant_name = _dishrack_canonical_variant_name(kind, variant_name)
    path = _DISHRACK_VARIANT_ROOTS[kind] / variant_name
    if not (path / "model.xml").exists():
        raise FileNotFoundError(f"Missing {kind} variant model.xml: {path}")
    return path


def _dishrack_sample_variant_name(kind: str, rng: np.random.Generator) -> str:
    variants = _dishrack_variant_names(kind)
    return variants[int(rng.integers(0, len(variants)))]


def _dishrack_plate_body_name(index: int) -> str:
    if index == 0:
        return "plate"
    return f"plate_{index}"


def _dishrack_plate_joint_name(index: int) -> str:
    if index == 0:
        return "plate_joint"
    return f"plate_joint_{index}"


def _dishrack_instance_prefix(kind: str, variant_name: str, instance_index: int) -> str:
    return f"{kind}_{instance_index}_{variant_name}"


def _dishrack_normalize_plate_variants(
    plate_variants: str | list[str] | tuple[str, ...],
) -> list[str]:
    if isinstance(plate_variants, str):
        normalized = [_dishrack_canonical_variant_name("plate", plate_variants)]
    else:
        normalized = [_dishrack_canonical_variant_name("plate", str(value)) for value in plate_variants]

    if not 1 <= len(normalized) <= _DISHRACK_MAX_PLATE_COUNT:
        raise ValueError(
            "DishRack requires between 1 and "
            f"{_DISHRACK_MAX_PLATE_COUNT} plate variants, got {len(normalized)}"
        )

    available = set(_dishrack_variant_names("plate"))
    invalid = [name for name in normalized if name not in available]
    if invalid:
        raise ValueError(
            f"Unknown plate variants {invalid}. Available: {', '.join(sorted(available))}"
        )
    return normalized



def _dishrack_compiled_metadata(
    kind: str,
    variant_name: str,
) -> tuple[float, float, float, float, float]:
    variant_dir = _dishrack_variant_dir(kind, variant_name)
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
    )


@_lru_cache(maxsize=None)
def _dishrack_compiled_xy_half_extents(kind: str, variant_name: str) -> tuple[float, float]:
    half_x, half_y, _, _, _ = _dishrack_compiled_metadata(kind, variant_name)
    return half_x, half_y


@_lru_cache(maxsize=None)
def _dishrack_compiled_anchor_offset(kind: str, variant_name: str) -> tuple[float, float, float]:
    _, _, offset_x, offset_y, offset_z = _dishrack_compiled_metadata(kind, variant_name)
    return offset_x, offset_y, offset_z



def _dishrack_wrapper_position(
    kind: str,
    variant_name: str,
    instance_index: int = 0,
) -> tuple[float, float, float]:
    if kind == "dish_rack":
        return _DISHRACK_RACK_WRAPPER_POSITION
    if kind != "plate":
        raise ValueError(f"Unsupported dishrack object kind {kind!r}")
    if not 0 <= instance_index < len(_DISHRACK_PLATE_WRAPPER_XY):
        raise ValueError(
            f"Plate instance index must be in [0, {len(_DISHRACK_PLATE_WRAPPER_XY) - 1}], "
            f"got {instance_index}"
        )
    x, y = _DISHRACK_PLATE_WRAPPER_XY[instance_index]
    return (x, y, _DISHRACK_PLATE_WRAPPER_Z)


def _dishrack_build_object_block(
    *,
    kind: str,
    variant_name: str,
    scale_factor: float,
    object_name: str,
    joint_name: str | None,
    instance_index: int,
) -> tuple[list[_ET.Element], _ET.Element]:
    variant_name = _dishrack_canonical_variant_name(kind, variant_name)
    variant_dir = _dishrack_variant_dir(kind, variant_name)
    root = _ET.parse(str(variant_dir / "model.xml")).getroot()
    object_body = _dishrack_find_object_body(root)
    offset = np.asarray(_dishrack_compiled_anchor_offset(kind, variant_name), dtype=np.float64) * float(scale_factor)

    prefix = _dishrack_instance_prefix(kind, variant_name, instance_index)
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
    _dishrack_prepare_imported_geoms(cloned_body)
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
            target_name=joint_name or object_name,
            asset_elem=temp_asset,
            mesh_assets=mesh_assets,
        )

    wrapper = _ET.Element(
        "body",
        name=object_name,
        pos=_format_float_list(list(_dishrack_wrapper_position(kind, variant_name, instance_index))),
    )
    if joint_name is not None:
        _ET.SubElement(wrapper, "freejoint", name=joint_name)
    wrapper.append(cloned_body)
    return list(temp_asset), wrapper


def _build_dishrack_scene_xml(
    *,
    dish_rack_variant: str,
    plate_variant: str | None = None,
    plate_variants: str | list[str] | tuple[str, ...] | None = None,
    scale_states: dict[str, float],
    base_scene_xml: str | None,
    base_scene_dir: _Path | None,
) -> str:
    base_text = base_scene_xml or _DISHRACK_BASE_SCENE_XML.read_text()
    base_dir = base_scene_dir or _DISHRACK_BASE_SCENE_XML.parent
    normalized_plate_variants = _dishrack_normalize_plate_variants(
        plate_variants if plate_variants is not None else (plate_variant or _DISHRACK_DEFAULT_VARIANTS["plate"])
    )
    dish_rack_variant = _dishrack_canonical_variant_name("dish_rack", dish_rack_variant)

    rack_assets, rack_body = _dishrack_build_object_block(
        kind="dish_rack",
        variant_name=dish_rack_variant,
        scale_factor=float(scale_states.get("dishrack", 1.0)),
        object_name="dishrack",
        joint_name="dishrack",
        instance_index=0,
    )
    task_assets = list(rack_assets)
    task_bodies = [rack_body]
    for index, current_plate_variant in enumerate(normalized_plate_variants):
        plate_assets, plate_body = _dishrack_build_object_block(
            kind="plate",
            variant_name=current_plate_variant,
            scale_factor=float(scale_states.get(_dishrack_plate_joint_name(index), scale_states.get("plate_joint", 1.0))),
            object_name=_dishrack_plate_body_name(index),
            joint_name=_dishrack_plate_joint_name(index),
            instance_index=index,
        )
        task_assets.extend(plate_assets)
        task_bodies.append(plate_body)

    xml = base_text.replace("<!-- TASK_DEFAULTS_PLACEHOLDER -->", "")
    xml = xml.replace("<!-- TASK_ASSETS_PLACEHOLDER -->", _dishrack_serialize_elements(task_assets))
    xml = xml.replace("<!-- TASK_BODY_PLACEHOLDER -->", _dishrack_serialize_elements(task_bodies))
    bin_scale = float(scale_states.get("bin_joint", 1.0))
    if abs(bin_scale - 1.0) > 1e-9:
        xml = _apply_object_scales_to_scene_xml(xml, {"bin_joint": bin_scale})
    return _resolve_scene_xml_paths(xml, base_dir)

__all__ = [name for name in globals() if name.startswith('_') and not name.startswith('__')]
