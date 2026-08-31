"""Bottle-in-bin task evaluators.

Both evaluators score bottle mass centres against geometry-derived bin
interiors. The put bin uses a tapered radial profile; the throw bin uses the
inner wall faces. Height extends to the rim plus each bottle's collision reach.
"""

from __future__ import annotations

import re
from typing import Any

import mujoco
import numpy as np

from abc_sim.task_eval.base import TaskEvalResult
from abc_sim.task_specs import SimTaskSpec

# Azimuth bins for the median wall radius; the same measurement the pour
# evaluator uses, so a handle or local dent cannot masquerade as the wall.
_AZIMUTH_BINS = 24


def _quat_to_rotmat_batch(quat_batch: np.ndarray) -> np.ndarray:
    """Convert wxyz quaternions into rotation matrices."""

    quat_batch = np.asarray(quat_batch, dtype=np.float32)
    if quat_batch.ndim != 2 or quat_batch.shape[1] != 4:
        raise ValueError(f"Expected quaternion batch shape (B, 4), got {quat_batch.shape}")

    norm = np.linalg.norm(quat_batch, axis=1, keepdims=True)
    norm = np.where(norm > 0.0, norm, 1.0)
    q = quat_batch / norm
    w, x, y, z = q.T

    return np.stack(
        [
            np.stack([1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)], axis=-1),
            np.stack([2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)], axis=-1),
            np.stack([2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)], axis=-1),
        ],
        axis=1,
    ).astype(np.float32)


def _body_child_ids(model: mujoco.MjModel, parent_body_id: int) -> list[int]:
    return [
        int(body_id)
        for body_id in range(model.nbody)
        if int(model.body_parentid[body_id]) == int(parent_body_id)
    ]


def _quat_to_rotmat(quat: np.ndarray) -> np.ndarray:
    return _quat_to_rotmat_batch(np.asarray(quat, dtype=np.float32)[None, :])[0]


def _subtree_body_transforms(
    model: mujoco.MjModel,
    root_body_id: int,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    transforms: dict[int, tuple[np.ndarray, np.ndarray]] = {
        int(root_body_id): (np.zeros(3, dtype=np.float32), np.eye(3, dtype=np.float32))
    }
    stack = [int(root_body_id)]
    while stack:
        body_id = stack.pop()
        parent_pos, parent_rot = transforms[body_id]
        for child_id in _body_child_ids(model, body_id):
            child_rot = _quat_to_rotmat(np.asarray(model.body_quat[child_id], dtype=np.float32))
            child_pos = np.asarray(model.body_pos[child_id], dtype=np.float32)
            transforms[child_id] = (
                parent_pos + parent_rot @ child_pos,
                parent_rot @ child_rot,
            )
            stack.append(child_id)
    return transforms


def _body_subtree_ids(model: mujoco.MjModel, root_body_id: int) -> set[int]:
    body_ids = {int(root_body_id)}
    stack = [int(root_body_id)]
    while stack:
        body_id = stack.pop()
        for child_id in _body_child_ids(model, body_id):
            body_ids.add(child_id)
            stack.append(child_id)
    return body_ids


def _resolve_bottle_qpos_addrs(
    model: mujoco.MjModel,
    *,
    allow_empty: bool = False,
) -> tuple[list[str], np.ndarray]:
    """Bottle freejoints by index. The put scene spawns its bottles from the
    randomizer, so its evaluator binds an empty set at scene load and rebinds
    after the reset reload; the throw scene bakes bottles into the XML, so an
    empty set there means the wrong model."""

    entries: list[tuple[int, str, int]] = []
    for joint_id in range(model.njnt):
        name = model.jnt(joint_id).name
        if not name:
            continue
        match = re.fullmatch(r"bottle_(\d+)_joint", name)
        if match is None:
            continue
        bottle_index = int(match.group(1))
        entries.append(
            (
                bottle_index,
                f"bottle_{bottle_index}",
                int(model.jnt_qposadr[joint_id]),
            )
        )

    if not entries and not allow_empty:
        raise ValueError("No bottle_*_joint freejoints found in model")

    entries.sort(key=lambda item: item[0])
    bottle_names = [name for _, name, _ in entries]
    addrs = np.asarray([adr for _, _, adr in entries], dtype=np.int32)
    return bottle_names, addrs


def _resolve_bottle_center_offsets(
    model: mujoco.MjModel,
    bottle_names: list[str],
) -> np.ndarray:
    """Mass-weighted centre of each bottle, in its own joint frame."""

    offsets: list[np.ndarray] = []
    for bottle_name in bottle_names:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bottle_name)
        if body_id < 0:
            raise ValueError(f"Bottle body {bottle_name!r} not found in model")
        transforms = _subtree_body_transforms(model, body_id)
        weighted_sum = np.zeros(3, dtype=np.float32)
        total_mass = 0.0
        for subtree_body_id, (local_pos, local_rot) in transforms.items():
            mass = float(model.body_mass[subtree_body_id])
            if mass <= 0.0:
                continue
            inertial_pos = np.asarray(model.body_ipos[subtree_body_id], dtype=np.float32)
            weighted_sum += mass * (local_pos + local_rot @ inertial_pos)
            total_mass += mass
        if total_mass <= 0.0:
            offsets.append(np.zeros(3, dtype=np.float32))
        else:
            offsets.append(weighted_sum / total_mass)
    if not offsets:
        return np.zeros((0, 3), dtype=np.float32)
    return np.stack(offsets, axis=0).astype(np.float32)


def _resolve_bottle_reaches(
    model: mujoco.MjModel,
    bottle_names: list[str],
    center_offsets: np.ndarray,
) -> np.ndarray:
    """How far each bottle's collision geometry extends from its mass centre.

    An orientation-free bound: geom centre distance plus MuJoCo's bounding
    radius per collision geom. Used as the over-rim allowance -- a centre more
    than one reach above the rim cannot have any part below it.
    """

    reaches: list[float] = []
    for bottle_name, center in zip(bottle_names, center_offsets):
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bottle_name)
        body_ids = _body_subtree_ids(model, body_id)
        transforms = _subtree_body_transforms(model, body_id)
        reach = 0.0
        for geom_id in range(model.ngeom):
            if int(model.geom_bodyid[geom_id]) not in body_ids:
                continue
            if (
                int(model.geom_contype[geom_id]) == 0
                and int(model.geom_conaffinity[geom_id]) == 0
            ):
                continue
            body_pos, body_rot = transforms[int(model.geom_bodyid[geom_id])]
            geom_pos = body_pos + body_rot @ np.asarray(
                model.geom_pos[geom_id], dtype=np.float32
            )
            reach = max(
                reach,
                float(np.linalg.norm(geom_pos - center))
                + float(model.geom_rbound[geom_id]),
            )
        reaches.append(reach)
    return np.asarray(reaches, dtype=np.float32)


def _azimuthal_median_radius(vertices: np.ndarray) -> float:
    """Median over azimuth bins of the farthest vertex radius in each bin."""

    radial = np.hypot(vertices[:, 0], vertices[:, 2])
    azimuth = np.arctan2(vertices[:, 2], vertices[:, 0])
    bin_index = np.clip(
        ((azimuth + np.pi) / (2.0 * np.pi) * _AZIMUTH_BINS).astype(np.int32),
        0,
        _AZIMUTH_BINS - 1,
    )
    maxima = [
        float(radial[bin_index == index].max())
        for index in range(_AZIMUTH_BINS)
        if np.any(bin_index == index)
    ]
    if not maxima:
        raise ValueError("No vertices to measure a wall radius from")
    return float(np.median(maxima))


class BottlesInBinEvaluator:
    """Score the throw-bottles task from bottle centres in the bin's frame.

    Membership uses the geometry-derived inner walls, floor, and rim height.
    """

    def __init__(
        self,
        *,
        model: mujoco.MjModel,
        spec: SimTaskSpec,
        success_count: int | None = None,
        floor_tolerance_m: float = 0.005,
    ) -> None:
        self.model = model
        self.spec = spec
        self.floor_tolerance_m = float(floor_tolerance_m)

        self.bottle_names, self._bottle_qpos_addrs = _resolve_bottle_qpos_addrs(model)
        if success_count is None:
            self.success_count = len(self.bottle_names)
        else:
            self.success_count = int(success_count)
        if self.success_count < 1:
            raise ValueError(f"success_count must be >= 1, got {self.success_count}")
        self._bottle_center_offsets = _resolve_bottle_center_offsets(model, self.bottle_names)
        self._bottle_reaches = _resolve_bottle_reaches(
            model, self.bottle_names, self._bottle_center_offsets
        )
        self._bin_qpos_adr = self._resolve_joint_qpos_adr(model, "bin_joint")
        self._bin_apothem = self._resolve_bin_inner_apothem(model)
        self._bin_floor_z = self._resolve_bin_floor_top_z(model)
        self._bin_top_z = self._resolve_bin_top_z(model)
        self._nworld = 1
        self._max_bottles_in_bin = np.zeros((1,), dtype=np.int32)
        self._ever_success = np.zeros((1,), dtype=bool)

    def reset(self, *, nworld: int = 1) -> None:
        self._nworld = int(nworld)
        self._max_bottles_in_bin = np.zeros((self._nworld,), dtype=np.int32)
        self._ever_success = np.zeros((self._nworld,), dtype=bool)

    @staticmethod
    def _resolve_joint_qpos_adr(model: mujoco.MjModel, joint_name: str) -> int:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise ValueError(f"Joint {joint_name!r} not found in model")
        return int(model.jnt_qposadr[joint_id])

    @staticmethod
    def _resolve_bottle_qpos_addrs(model: mujoco.MjModel) -> tuple[list[str], np.ndarray]:
        return _resolve_bottle_qpos_addrs(model)

    @staticmethod
    def _resolve_bin_inner_apothem(model: mujoco.MjModel) -> float:
        """Distance from the bin axis to the walls' inner faces.

        Each wall box sits with its radial half-thickness in geom_size[0], so
        the inner face is the centre distance minus that. The old criterion
        used the centre distance itself, accepting centres inside the wall
        material.
        """

        inner_faces: list[float] = []
        for geom_id in range(model.ngeom):
            name = model.geom(geom_id).name
            if name and name.startswith("bin_wall_"):
                center_distance = float(np.linalg.norm(model.geom_pos[geom_id][:2]))
                inner_faces.append(center_distance - float(model.geom_size[geom_id][0]))
        if not inner_faces:
            raise ValueError("No bin wall geoms found in model")
        return float(min(inner_faces))

    @staticmethod
    def _resolve_bin_floor_top_z(model: mujoco.MjModel) -> float:
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "bin_bottom")
        if geom_id < 0:
            return 0.0
        geom_type = int(model.geom_type[geom_id])
        # The plate ships as a cylinder (half-height in size[1]); accept a box
        # variant (half-height in size[2]) so a rebuilt scene self-measures.
        if geom_type == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
            half_height = float(model.geom_size[geom_id][1])
        elif geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
            half_height = float(model.geom_size[geom_id][2])
        else:
            half_height = 0.0
        return float(model.geom_pos[geom_id][2]) + half_height

    @staticmethod
    def _resolve_bin_top_z(model: mujoco.MjModel) -> float:
        top_values: list[float] = []
        for geom_id in range(model.ngeom):
            name = model.geom(geom_id).name
            if not name or not name.startswith("bin_wall_"):
                continue
            geom_type = int(model.geom_type[geom_id])
            if geom_type != mujoco.mjtGeom.mjGEOM_BOX:
                continue
            top_values.append(float(model.geom_pos[geom_id][2] + model.geom_size[geom_id][2]))

        if not top_values:
            raise ValueError("No bin wall box geoms found in model")
        return float(max(top_values))

    def evaluate_qpos_batch(self, qpos_batch: np.ndarray) -> TaskEvalResult:
        qpos_batch = np.asarray(qpos_batch, dtype=np.float32)
        if qpos_batch.ndim != 2 or qpos_batch.shape[1] != self.model.nq:
            raise ValueError(
                f"Expected qpos batch shape (B, {self.model.nq}), got {qpos_batch.shape}"
            )
        if qpos_batch.shape[0] != self._nworld:
            self.reset(nworld=qpos_batch.shape[0])

        bin_pos = qpos_batch[:, self._bin_qpos_adr : self._bin_qpos_adr + 3]
        bin_quat = qpos_batch[:, self._bin_qpos_adr + 3 : self._bin_qpos_adr + 7]

        bottle_pos = np.stack(
            [qpos_batch[:, adr : adr + 3] for adr in self._bottle_qpos_addrs],
            axis=1,
        )
        bottle_quat = np.stack(
            [qpos_batch[:, adr + 3 : adr + 7] for adr in self._bottle_qpos_addrs],
            axis=1,
        )
        bottle_rot = _quat_to_rotmat_batch(bottle_quat.reshape(-1, 4)).reshape(
            qpos_batch.shape[0],
            len(self.bottle_names),
            3,
            3,
        )
        bottle_centers = bottle_pos + np.einsum(
            "bnij,nj->bni",
            bottle_rot,
            self._bottle_center_offsets,
        )

        rel_world = bottle_centers - bin_pos[:, None, :]
        rot_world_from_local = _quat_to_rotmat_batch(bin_quat)
        rel_local = np.einsum("bij,bnj->bni", np.swapaxes(rot_world_from_local, 1, 2), rel_world)

        radial_xy = np.linalg.norm(rel_local[..., :2], axis=-1)
        local_z = rel_local[..., 2]
        radial_margin = self._bin_apothem - radial_xy
        lower_height_margin = local_z - (self._bin_floor_z - self.floor_tolerance_m)
        upper_height_margin = (
            self._bin_top_z + self._bottle_reaches[None, :] - local_z
        )
        in_bin_mask = (
            (radial_margin >= 0.0)
            & (lower_height_margin >= 0.0)
            & (upper_height_margin >= 0.0)
        )

        num_bottles_in_bin = in_bin_mask.sum(axis=1).astype(np.int32)
        self._max_bottles_in_bin = np.maximum(self._max_bottles_in_bin, num_bottles_in_bin)
        reward = np.clip(
            num_bottles_in_bin.astype(np.float32) / float(self.success_count),
            0.0,
            1.0,
        )
        success = num_bottles_in_bin >= self.success_count
        self._ever_success |= success
        bottles_in_bin = [
            [name for name, in_bin in zip(self.bottle_names, world_mask) if bool(in_bin)]
            for world_mask in in_bin_mask
        ]

        metrics: dict[str, Any] = {
            "num_bottles_in_bin": num_bottles_in_bin,
            "max_bottles_in_bin_so_far": self._max_bottles_in_bin.copy(),
            "ever_success": self._ever_success.copy(),
            "closest_radial_margin": radial_margin.max(axis=1).astype(np.float32),
            "closest_height_margin": lower_height_margin.max(axis=1).astype(np.float32),
            "closest_upper_height_margin": upper_height_margin.max(axis=1).astype(np.float32),
            "bottle_in_bin_mask": in_bin_mask,
            "bottles_in_bin": bottles_in_bin,
            "bottle_names": list(self.bottle_names),
            "bottle_center_offsets": self._bottle_center_offsets.copy(),
            "bottle_reaches": self._bottle_reaches.copy(),
            "success_count": self.success_count,
            "bin_apothem": self._bin_apothem,
            "bin_floor_z": self._bin_floor_z,
            "bin_top_z": self._bin_top_z,
        }
        return TaskEvalResult(reward=reward, success=success, metrics=metrics)


class PutBottlesInBinEvaluator:
    """Score put-bottles by requiring every bottle centre inside the tub.

    The tapered acceptance region is derived from the collision hull.
    """

    def __init__(
        self,
        *,
        model: mujoco.MjModel,
        spec: SimTaskSpec,
    ) -> None:
        self.model = model
        self.spec = spec
        self.bottle_names, self._bottle_qpos_addrs = _resolve_bottle_qpos_addrs(
            model, allow_empty=True
        )
        self._bottle_center_offsets = _resolve_bottle_center_offsets(model, self.bottle_names)
        self._bottle_reaches = _resolve_bottle_reaches(
            model, self.bottle_names, self._bottle_center_offsets
        )
        self._bin_qpos_adr = BottlesInBinEvaluator._resolve_joint_qpos_adr(model, "bin_joint")
        (
            self._bin_bottom_y,
            self._bin_top_y,
            self._bin_radius_bottom,
            self._bin_radius_top,
        ) = self._resolve_bin_cone(model)
        self.success_count = len(self.bottle_names)
        self._nworld = 1
        self._max_bottles_in_bin = np.zeros((1,), dtype=np.int32)
        self._ever_success = np.zeros((1,), dtype=bool)

    def reset(self, *, nworld: int = 1) -> None:
        self._nworld = int(nworld)
        self._max_bottles_in_bin = np.zeros((self._nworld,), dtype=np.int32)
        self._ever_success = np.zeros((self._nworld,), dtype=bool)

    @staticmethod
    def _resolve_bottle_qpos_addrs(model: mujoco.MjModel) -> tuple[list[str], np.ndarray]:
        return _resolve_bottle_qpos_addrs(model, allow_empty=True)

    @staticmethod
    def _resolve_bottle_center_offsets(
        model: mujoco.MjModel,
        bottle_names: list[str],
    ) -> np.ndarray:
        return _resolve_bottle_center_offsets(model, bottle_names)

    @staticmethod
    def _resolve_bin_cone(model: mujoco.MjModel) -> tuple[float, float, float, float]:
        """Measure the tub as (bottom y, top y, bottom radius, top radius).

        All from the ``bin_container`` subtree's group-3 collision meshes in
        the bin's own frame (local y is the tub's height axis), so the scale
        randomization carries through. Each radius is the azimuthal median of
        the farthest vertex per azimuth bin over the hull's bottom / top
        quarter, which tracks the wall and ignores handle-like outliers.
        """

        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "bin_container")
        if body_id < 0:
            raise ValueError("Body 'bin_container' not found in model")
        body_ids = _body_subtree_ids(model, body_id)
        transforms = _subtree_body_transforms(model, body_id)
        vertices: list[np.ndarray] = []
        for geom_id in range(model.ngeom):
            if int(model.geom_bodyid[geom_id]) not in body_ids:
                continue
            if int(model.geom_group[geom_id]) != 3:
                continue
            mesh_id = int(model.geom_dataid[geom_id])
            if mesh_id < 0:
                continue
            mesh_start = int(model.mesh_vertadr[mesh_id])
            mesh_count = int(model.mesh_vertnum[mesh_id])
            mesh_vertices = np.asarray(
                model.mesh_vert[mesh_start : mesh_start + mesh_count],
                dtype=np.float32,
            )
            geom_rot = _quat_to_rotmat(np.asarray(model.geom_quat[geom_id], dtype=np.float32))
            geom_pos = np.asarray(model.geom_pos[geom_id], dtype=np.float32)
            geom_body_id = int(model.geom_bodyid[geom_id])
            body_pos, body_rot = transforms[geom_body_id]
            geom_vertices = mesh_vertices @ geom_rot.T + geom_pos
            vertices.append(geom_vertices @ body_rot.T + body_pos)
        if not vertices:
            raise ValueError("No bin collision mesh vertices found in model")
        all_vertices = np.concatenate(vertices, axis=0)
        heights = all_vertices[:, 1]
        bottom = float(heights.min())
        top = float(heights.max())
        if top <= bottom:
            raise ValueError("Bin collision hull has no height extent")
        band = 0.25 * (top - bottom)
        radius_bottom = _azimuthal_median_radius(all_vertices[heights <= bottom + band])
        radius_top = _azimuthal_median_radius(all_vertices[heights >= top - band])
        if radius_bottom <= 0.0 or radius_top <= 0.0:
            raise ValueError("Bin measured a non-positive wall radius")
        return bottom, top, radius_bottom, radius_top

    def _allowed_radius(self, local_height: np.ndarray) -> np.ndarray:
        """The cone's wall radius at a local height, clamped to its ends."""

        span = max(self._bin_top_y - self._bin_bottom_y, 1.0e-9)
        fraction = np.clip((local_height - self._bin_bottom_y) / span, 0.0, 1.0)
        return self._bin_radius_bottom + fraction * (
            self._bin_radius_top - self._bin_radius_bottom
        )

    def evaluate_qpos_batch(self, qpos_batch: np.ndarray) -> TaskEvalResult:
        qpos_batch = np.asarray(qpos_batch, dtype=np.float32)
        if qpos_batch.ndim != 2 or qpos_batch.shape[1] != self.model.nq:
            raise ValueError(
                f"Expected qpos batch shape (B, {self.model.nq}), got {qpos_batch.shape}"
            )
        if qpos_batch.shape[0] != self._nworld:
            self.reset(nworld=qpos_batch.shape[0])

        if not self.bottle_names:
            reward = np.zeros((qpos_batch.shape[0],), dtype=np.float32)
            success = np.zeros((qpos_batch.shape[0],), dtype=bool)
            metrics = {
                "num_bottles_in_bin": np.zeros((qpos_batch.shape[0],), dtype=np.int32),
                "num_active_bottles": np.zeros((qpos_batch.shape[0],), dtype=np.int32),
                "max_bottles_in_bin_so_far": self._max_bottles_in_bin.copy(),
                "ever_success": self._ever_success.copy(),
                "bottle_in_bin_mask": np.zeros((qpos_batch.shape[0], 0), dtype=bool),
                "bottles_in_bin": [[] for _ in range(qpos_batch.shape[0])],
                "bottle_names": [],
                "success_count": 0,
            }
            return TaskEvalResult(reward=reward, success=success, metrics=metrics)

        bin_pos = qpos_batch[:, self._bin_qpos_adr : self._bin_qpos_adr + 3]
        bin_quat = qpos_batch[:, self._bin_qpos_adr + 3 : self._bin_qpos_adr + 7]
        bottle_pos = np.stack(
            [qpos_batch[:, adr : adr + 3] for adr in self._bottle_qpos_addrs],
            axis=1,
        )
        bottle_quat = np.stack(
            [qpos_batch[:, adr + 3 : adr + 7] for adr in self._bottle_qpos_addrs],
            axis=1,
        )
        bottle_rot = _quat_to_rotmat_batch(bottle_quat.reshape(-1, 4)).reshape(
            qpos_batch.shape[0],
            len(self.bottle_names),
            3,
            3,
        )
        bottle_centers = bottle_pos + np.einsum(
            "bnij,nj->bni",
            bottle_rot,
            self._bottle_center_offsets,
        )

        rot_world_from_bin = _quat_to_rotmat_batch(bin_quat)
        rel_world = bottle_centers - bin_pos[:, None, :]
        rel_bin = np.einsum("bij,bnj->bni", np.swapaxes(rot_world_from_bin, 1, 2), rel_world)

        local_height = rel_bin[..., 1]
        radial = np.hypot(rel_bin[..., 0], rel_bin[..., 2])
        radial_margin = self._allowed_radius(local_height) - radial
        lower_height_margin = local_height - self._bin_bottom_y
        upper_height_margin = (
            self._bin_top_y + self._bottle_reaches[None, :] - local_height
        )
        in_bin_mask = (
            (radial_margin >= 0.0)
            & (lower_height_margin >= 0.0)
            & (upper_height_margin >= 0.0)
        )

        num_bottles_in_bin = in_bin_mask.sum(axis=1).astype(np.int32)
        active_count = np.full(
            (qpos_batch.shape[0],),
            len(self.bottle_names),
            dtype=np.int32,
        )
        self._max_bottles_in_bin = np.maximum(self._max_bottles_in_bin, num_bottles_in_bin)
        reward = num_bottles_in_bin.astype(np.float32) / active_count.astype(np.float32)
        success = num_bottles_in_bin == active_count
        self._ever_success |= success
        bottles_in_bin = [
            [name for name, in_bin in zip(self.bottle_names, world_mask) if bool(in_bin)]
            for world_mask in in_bin_mask
        ]

        metrics: dict[str, Any] = {
            "num_bottles_in_bin": num_bottles_in_bin,
            "num_active_bottles": active_count,
            "max_bottles_in_bin_so_far": self._max_bottles_in_bin.copy(),
            "ever_success": self._ever_success.copy(),
            "closest_radial_margin": radial_margin.max(axis=1).astype(np.float32),
            "closest_height_margin": lower_height_margin.max(axis=1).astype(np.float32),
            "closest_upper_height_margin": upper_height_margin.max(axis=1).astype(np.float32),
            "bottle_in_bin_mask": in_bin_mask,
            "bottles_in_bin": bottles_in_bin,
            "bottle_names": list(self.bottle_names),
            "success_count": len(self.bottle_names),
            "bin_bottom_y": self._bin_bottom_y,
            "bin_top_y": self._bin_top_y,
            "bin_radius_bottom": self._bin_radius_bottom,
            "bin_radius_top": self._bin_radius_top,
            "bottle_center_offsets": self._bottle_center_offsets.copy(),
            "bottle_reaches": self._bottle_reaches.copy(),
        }
        return TaskEvalResult(reward=reward, success=success, metrics=metrics)
