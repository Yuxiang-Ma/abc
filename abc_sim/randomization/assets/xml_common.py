"""Shared XML helpers for task asset builders."""

from __future__ import annotations

import copy
import re
import xml.etree.ElementTree as _ET
from functools import lru_cache as _lru_cache
from pathlib import Path as _Path

import mujoco
import numpy as np

from ..core import _format_float_list, _parse_float_list

def _dishrack_asset_local_name(elem: _ET.Element) -> str:
    name = elem.get("name")
    if name:
        return name
    file_attr = elem.get("file", "")
    stem = _Path(file_attr).stem
    if stem:
        return stem
    raise ValueError(f"Unable to infer asset name for element: {elem.tag}")


def _dishrack_prefixed_name(prefix: str, local_name: str) -> str:
    return f"{prefix}_{local_name.replace('.', '_')}"


def _dishrack_find_object_body(root: _ET.Element) -> _ET.Element:
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("Variant model.xml is missing <worldbody>")

    for candidate_name in ("object", "model"):
        body = worldbody.find(f".//body[@name='{candidate_name}']")
        if body is not None:
            return body

    for body in worldbody.iter("body"):
        if any(child.tag == "geom" for child in body.iter()):
            return body
    raise ValueError("Variant model.xml does not contain a body with geoms")


def _dishrack_find_bbox_geom(body: _ET.Element) -> _ET.Element | None:
    for geom in body.iter("geom"):
        if geom.get("name") == "reg_bbox":
            return geom
    return None


def _dishrack_bbox_anchor_offset(body: _ET.Element) -> np.ndarray:
    bbox = _dishrack_find_bbox_geom(body)
    if bbox is None:
        return np.zeros(3, dtype=np.float64)
    pos = np.asarray(_parse_float_list(bbox.get("pos", "0 0 0")), dtype=np.float64)
    size = np.asarray(_parse_float_list(bbox.get("size", "0 0 0")), dtype=np.float64)
    return np.array([-pos[0], -pos[1], -(pos[2] - size[2])], dtype=np.float64)


def _dishrack_geom_world_bounds(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    geom_id: int,
) -> tuple[np.ndarray, np.ndarray]:
    geom_type = int(model.geom_type[geom_id])
    pos = np.asarray(data.geom_xpos[geom_id], dtype=np.float64)
    rot = np.asarray(data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)

    if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
        size = np.asarray(model.geom_size[geom_id][:3], dtype=np.float64)
        half = np.abs(rot) @ size
        return pos - half, pos + half

    if geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
        radius = float(model.geom_size[geom_id][0])
        half = np.full(3, radius, dtype=np.float64)
        return pos - half, pos + half

    if geom_type == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
        size = np.asarray(model.geom_size[geom_id][:3], dtype=np.float64)
        half = np.abs(rot) @ size
        return pos - half, pos + half

    if geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
        radius = float(model.geom_size[geom_id][0])
        half_len = float(model.geom_size[geom_id][1])
        half = np.abs(rot) @ np.array([radius, radius, half_len], dtype=np.float64)
        return pos - half, pos + half

    if geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
        radius = float(model.geom_size[geom_id][0])
        half_len = float(model.geom_size[geom_id][1]) + radius
        half = np.abs(rot) @ np.array([radius, radius, half_len], dtype=np.float64)
        return pos - half, pos + half

    if geom_type == mujoco.mjtGeom.mjGEOM_MESH:
        mesh_id = int(model.geom_dataid[geom_id])
        if mesh_id < 0:
            raise ValueError(f"Mesh geom {geom_id} is missing compiled mesh data")
        start = int(model.mesh_vertadr[mesh_id])
        count = int(model.mesh_vertnum[mesh_id])
        verts = np.asarray(model.mesh_vert[start : start + count], dtype=np.float64)
        world = verts @ rot.T + pos
        return world.min(axis=0), world.max(axis=0)

    radius = float(model.geom_rbound[geom_id])
    half = np.full(3, radius, dtype=np.float64)
    return pos - half, pos + half


def _dishrack_compiled_model_bounds(
    model: mujoco.MjModel,
    data: mujoco.MjData,
) -> tuple[np.ndarray, np.ndarray]:
    mins: list[np.ndarray] = []
    maxs: list[np.ndarray] = []
    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        if name.startswith("reg_"):
            continue
        lower, upper = _dishrack_geom_world_bounds(model, data, geom_id)
        mins.append(lower)
        maxs.append(upper)
    if not mins:
        raise ValueError("Variant model did not contain any non-region geoms")
    return np.vstack(mins).min(axis=0), np.vstack(maxs).max(axis=0)


def _dishrack_absolutize_file_attr(elem: _ET.Element, variant_dir: _Path) -> None:
    file_attr = elem.get("file")
    if file_attr and not _Path(file_attr).is_absolute():
        elem.set("file", str((variant_dir / file_attr).resolve()))


def _dishrack_remove_dynamic_joints(body: _ET.Element) -> None:
    for parent in body.iter():
        for child in list(parent):
            if child.tag in {"freejoint", "joint"}:
                parent.remove(child)


def _dishrack_prefix_body_names(elem: _ET.Element, prefix: str) -> None:
    for child in elem.iter():
        name = child.get("name")
        if name and child.tag in {"body", "geom", "site"}:
            child.set("name", _dishrack_prefixed_name(prefix, name))


def _dishrack_rewrite_asset_refs(
    body: _ET.Element,
    *,
    mesh_map: dict[str, str],
    material_map: dict[str, str],
) -> None:
    for child in body.iter():
        mesh_name = child.get("mesh")
        if mesh_name and mesh_name in mesh_map:
            child.set("mesh", mesh_map[mesh_name])
        material_name = child.get("material")
        if material_name and material_name in material_map:
            child.set("material", material_map[material_name])


def _dishrack_prepare_imported_geoms(body: _ET.Element) -> None:
    for parent in body.iter():
        for child in list(parent):
            if child.tag != "geom":
                continue
            if child.get("name", "").endswith("_reg_bbox") or child.get("class") == "region":
                parent.remove(child)
                continue

            child.attrib.pop("class", None)


def _dishrack_shift_body(body: _ET.Element, offset: np.ndarray) -> None:
    base_pos = np.zeros(3, dtype=np.float64)
    if body.get("pos"):
        base_pos = np.asarray(_parse_float_list(body.get("pos", "0 0 0")), dtype=np.float64)
    body.set("pos", _format_float_list((base_pos + offset).tolist()))


def _dishrack_serialize_elements(elements: list[_ET.Element], indent: str = "    ") -> str:
    rendered: list[str] = []
    for elem in elements:
        if hasattr(_ET, "indent"):
            _ET.indent(elem, space="  ")
        xml = _ET.tostring(elem, encoding="unicode")
        rendered.append("\n".join(f"{indent}{line}" if line else line for line in xml.splitlines()))
    return "\n".join(rendered)


_SORTING_TASK_ARM_QPOS = "0 1.047 1.047 0 0 0 0 0  0 1.047 1.047 0 0 0 0 0"
_LEGO_COLORS = ("red", "yellow", "blue")
_BIN_VISUAL_STYLE_MESHES = {
    "stackable": "stackable_bin_visual",
    "low_poly_crate": "low_poly_crate_bin_visual",
}
_BIN_COLLISION_STYLE_MESHES = {
    "stackable": tuple(f"stackable_bin_collision_{index}" for index in range(16)),
    "low_poly_crate": tuple(f"low_poly_crate_bin_collision_{index}" for index in range(10)),
}
_BIN_VISUAL_STYLES = tuple(_BIN_VISUAL_STYLE_MESHES)
_NUT_BOLT_BODY_RE = re.compile(r"^(nut|bolt)_(\d+)$")
_LEGO_BODY_RE = re.compile(r"^(red|yellow|blue)_lego_(\d+)$")
def _find_xml_body(root: _ET.Element, body_name: str) -> _ET.Element:
    for body in root.iter("body"):
        if body.get("name") == body_name:
            return body
    raise RuntimeError(f"Scene XML is missing body {body_name!r}")


def _find_xml_body_parent(root: _ET.Element, body_name: str) -> tuple[_ET.Element, _ET.Element]:
    for parent in root.iter():
        for child in list(parent):
            if child.tag == "body" and child.get("name") == body_name:
                return parent, child
    raise RuntimeError(f"Scene XML is missing body {body_name!r}")


def _sorting_candidate_template_body_name(body_name: str) -> str | None:
    match = _NUT_BOLT_BODY_RE.fullmatch(body_name)
    if match is not None:
        return f"{match.group(1)}_1"
    match = _LEGO_BODY_RE.fullmatch(body_name)
    if match is not None:
        return f"{match.group(1)}_lego_1"
    return None


def _sorting_candidate_generated_pose(body_name: str) -> tuple[list[float], list[float]] | None:
    match = _NUT_BOLT_BODY_RE.fullmatch(body_name)
    if match is not None:
        part_type, index_str = match.groups()
        slot_index = 2 * (int(index_str) - 1) + (0 if part_type == "nut" else 1)
        col = slot_index % 5
        row = slot_index // 5
        yaw = 0.35 * slot_index
        return (
            [0.360 + 0.045 * col, -0.365 + 0.105 * row, 0.75576],
            [float(np.cos(yaw * 0.5)), 0.0, 0.0, float(np.sin(yaw * 0.5))],
        )

    match = _LEGO_BODY_RE.fullmatch(body_name)
    if match is not None:
        color, index_str = match.groups()
        color_index = _LEGO_COLORS.index(color)
        slot_index = 3 * (int(index_str) - 1) + color_index
        col = slot_index % 3
        row = slot_index // 3
        yaw = 0.50 * slot_index
        return (
            [0.355 + 0.105 * col, -0.410 + 0.1025 * row, 0.7572],
            [float(np.cos(yaw * 0.5)), 0.0, 0.0, float(np.sin(yaw * 0.5))],
        )

    return None


def _rename_xml_subtree_names(elem: _ET.Element, old_prefix: str, new_prefix: str) -> None:
    for child in elem.iter():
        name = child.get("name")
        if name:
            child.set("name", name.replace(old_prefix, new_prefix))


def _ensure_sorting_candidate_bodies(
    root: _ET.Element,
    candidate_object_body_names: tuple[str, ...],
) -> None:
    existing_body_names = {
        body.get("name", "")
        for body in root.iter("body")
        if body.get("name")
    }
    for body_name in candidate_object_body_names:
        template_body_name = _sorting_candidate_template_body_name(body_name)
        generated_pose = _sorting_candidate_generated_pose(body_name)
        if template_body_name is None or generated_pose is None:
            raise RuntimeError(f"Cannot generate sorting candidate body {body_name!r}")
        pos, quat = generated_pose

        if body_name in existing_body_names:
            body = _find_xml_body(root, body_name)
            body.set("pos", _format_float_list(pos))
            body.set("quat", _format_float_list(quat))
            continue

        parent, template_body = _find_xml_body_parent(root, template_body_name)
        cloned_body = copy.deepcopy(template_body)
        _rename_xml_subtree_names(cloned_body, template_body_name, body_name)
        cloned_body.set("pos", _format_float_list(pos))
        cloned_body.set("quat", _format_float_list(quat))
        parent.append(cloned_body)
        existing_body_names.add(body_name)


def _body_default_free_qpos(root: _ET.Element, body_name: str) -> str:
    body = _find_xml_body(root, body_name)
    pos = _parse_float_list(body.get("pos", "0 0 0"))
    quat = _parse_float_list(body.get("quat", "1 0 0 0"))
    if len(pos) != 3:
        raise RuntimeError(f"Body {body_name!r} has invalid pos")
    if len(quat) != 4:
        raise RuntimeError(f"Body {body_name!r} has invalid quat")
    return _format_float_list([*pos, *quat])


def _apply_xml_body_pose_sources(
    root: _ET.Element,
    body_pose_source_names: dict[str, str],
) -> None:
    source_poses = {
        source_name: (
            _find_xml_body(root, source_name).get("pos", "0 0 0"),
            _find_xml_body(root, source_name).get("quat", "1 0 0 0"),
        )
        for source_name in set(body_pose_source_names.values())
    }
    for body_name, source_name in body_pose_source_names.items():
        body = _find_xml_body(root, body_name)
        pos, quat = source_poses[source_name]
        body.set("pos", pos)
        body.set("quat", quat)


def _sample_bin_visual_style(rng: np.random.Generator) -> str:
    return str(_BIN_VISUAL_STYLES[int(rng.integers(0, len(_BIN_VISUAL_STYLES)))])


def _sample_bin_visual_styles(
    body_names: tuple[str, ...],
    rng: np.random.Generator,
) -> dict[str, str]:
    return {body_name: _sample_bin_visual_style(rng) for body_name in body_names}


def _apply_bin_visual_styles(
    root: _ET.Element,
    bin_visual_styles: dict[str, str],
) -> None:
    asset_elem = root.find("asset")
    available_mesh_names = {
        mesh.get("name", "")
        for mesh in asset_elem.findall("mesh")
    } if asset_elem is not None else set()
    for body_name, style in bin_visual_styles.items():
        mesh_name = _BIN_VISUAL_STYLE_MESHES.get(style)
        if mesh_name is None:
            raise RuntimeError(f"Unknown bin visual style {style!r}")
        body = _find_xml_body(root, body_name)
        visual_geom = body.find(f"./geom[@name='{body_name}_visual']")
        if visual_geom is None:
            raise RuntimeError(f"Body {body_name!r} is missing visual bin geom")
        visual_geom.set("mesh", mesh_name)
        collision_mesh_names = _BIN_COLLISION_STYLE_MESHES.get(style)
        if collision_mesh_names is None or not set(collision_mesh_names) <= available_mesh_names:
            continue
        collision_prefix = f"{body_name}_collision_"
        for child in list(body):
            if child.tag == "geom" and child.get("name", "").startswith(collision_prefix):
                body.remove(child)
        for index, collision_mesh_name in enumerate(collision_mesh_names):
            body.append(
                _ET.Element(
                    "geom",
                    {
                        "name": f"{body_name}_collision_{index}",
                        "type": "mesh",
                        "mesh": collision_mesh_name,
                        "group": "3",
                        "rgba": "0 0 0 0",
                        "friction": "1.3 0.1 0.01",
                    },
                )
            )


def _apply_bin_visual_styles_to_xml(xml: str, bin_visual_styles: dict[str, str]) -> str:
    if not bin_visual_styles:
        return xml
    root = _ET.fromstring(xml)
    _apply_bin_visual_styles(root, bin_visual_styles)
    return _ET.tostring(root, encoding="unicode")


def _filter_sorting_scene_xml(
    xml: str,
    *,
    candidate_object_body_names: tuple[str, ...],
    active_object_body_names: tuple[str, ...],
    home_body_names: tuple[str, ...],
    body_pose_source_names: dict[str, str] | None = None,
    bin_visual_styles: dict[str, str] | None = None,
) -> str:
    root = _ET.fromstring(xml)
    _ensure_sorting_candidate_bodies(root, candidate_object_body_names)
    if body_pose_source_names:
        _apply_xml_body_pose_sources(root, body_pose_source_names)
    if bin_visual_styles:
        _apply_bin_visual_styles(root, bin_visual_styles)

    inactive_body_names = set(candidate_object_body_names) - set(active_object_body_names)
    for parent in root.iter():
        for child in list(parent):
            if child.tag == "body" and child.get("name") in inactive_body_names:
                parent.remove(child)

    home_key = root.find("./keyframe/key[@name='home']")
    if home_key is None:
        raise RuntimeError("Sorting scene XML is missing home keyframe")
    qpos_lines = [_body_default_free_qpos(root, body_name) for body_name in home_body_names]
    qpos_lines.append(_SORTING_TASK_ARM_QPOS)
    home_key.set("qpos", "\n            ".join(qpos_lines))
    return _ET.tostring(root, encoding="unicode")
__all__ = [name for name in globals() if name.startswith('_') and not name.startswith('__')]
