"""Mug-on-rack task evaluator."""

from __future__ import annotations

import re

import mujoco
import numpy as np

from abc_sim.task_eval.base import TaskEvalResult
from abc_sim.task_specs import SimTaskSpec


class MugOnRackEvaluator:
    """Score hang_mug_on_mug_rack by mugs hanging in the tree's peg band.

    The scene deals one to three mugs on the table beside a peg tree the
    randomizer shifts and yaws per reset. A mug hangs when its centre sits
    inside a cylinder around the tree's trunk axis, within the height band
    from just above the tree's lowest point up past its top -- measured
    against the tree's live pose, so the same criterion follows the tree
    wherever a reset put it.

    The radial and vertical bounds distinguish mugs on pegs from mugs resting
    at table level. Success requires every dealt mug to be hanging at once.

    The tree is a static body whose pose the randomizer edits in place, so
    the evaluator re-measures the tree's world frame on every
    ``configure_from_randomization`` -- rebinding only on model identity
    would keep a stale pose after an in-place perturbation.
    """

    def __init__(
        self,
        *,
        model: mujoco.MjModel,
        spec: SimTaskSpec,
        tree_body_name: str = "mug_tree",
        mug_joint_regex: str = r"mug_(\d+)_jnt",
        hang_radius_m: float = 0.115,
        base_clearance_m: float = 0.005,
        top_clearance_m: float = 0.05,
    ) -> None:
        self.spec = spec
        self._tree_body_name = tree_body_name
        self._mug_joint_regex = mug_joint_regex
        self.hang_radius_m = float(hang_radius_m)
        self.base_clearance_m = float(base_clearance_m)
        self.top_clearance_m = float(top_clearance_m)
        self._bind_model(model)
        self.reset(nworld=1)

    def _bind_model(self, model: mujoco.MjModel) -> None:
        self.model = model
        self.mug_names, self._mug_qpos_addrs = self._resolve_mug_qpos_addrs(
            model, self._mug_joint_regex
        )
        self._tree_body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, self._tree_body_name
        )
        if self._tree_body_id < 0:
            raise ValueError(f"Body {self._tree_body_name!r} not found in model")
        if int(model.body_jntnum[self._tree_body_id]) != 0:
            raise ValueError(
                f"Body {self._tree_body_name!r} must be a static body; a jointed "
                "tree would need its pose read from qpos"
            )
        # Collision-mesh vertices in the tree's own frame, measured once; the
        # world pose composes in per reset.
        self._tree_local_vertices = self._collect_local_vertices(
            model, self._tree_body_id
        )
        self._measure_tree(model)

    def _measure_tree(self, model: mujoco.MjModel) -> None:
        """Compose the tree's current world pose into the hang band."""

        pos = np.asarray(model.body_pos[self._tree_body_id], dtype=np.float64)
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, model.body_quat[self._tree_body_id])
        world = self._tree_local_vertices @ rot.reshape(3, 3).T + pos
        self._axis_xy = world[:, :2].mean(axis=0)
        base_z = float(world[:, 2].min())
        top_z = float(world[:, 2].max())
        self._hang_z_lo = base_z + self.base_clearance_m
        self._hang_z_hi = top_z + self.top_clearance_m
        if self._hang_z_lo >= self._hang_z_hi:
            raise ValueError(
                f"Tree {self._tree_body_name!r} measured an empty hang band"
            )

    @staticmethod
    def _collect_local_vertices(model: mujoco.MjModel, body_id: int) -> np.ndarray:
        vertices = []
        for geom_id in range(model.ngeom):
            if int(model.geom_bodyid[geom_id]) != body_id:
                continue
            if int(model.geom_group[geom_id]) != 3:
                continue
            mesh_id = int(model.geom_dataid[geom_id])
            if mesh_id < 0:
                continue
            start = int(model.mesh_vertadr[mesh_id])
            count = int(model.mesh_vertnum[mesh_id])
            mesh_vertices = np.asarray(
                model.mesh_vert[start : start + count], dtype=np.float64
            )
            rot = np.zeros(9)
            mujoco.mju_quat2Mat(rot, model.geom_quat[geom_id])
            vertices.append(
                mesh_vertices @ rot.reshape(3, 3).T
                + np.asarray(model.geom_pos[geom_id], dtype=np.float64)
            )
        if not vertices:
            raise ValueError(
                f"Body id {body_id} has no collision-mesh vertices to measure"
            )
        return np.concatenate(vertices, axis=0)

    @staticmethod
    def _resolve_mug_qpos_addrs(
        model: mujoco.MjModel,
        mug_joint_regex: str,
    ) -> tuple[list[str], np.ndarray]:
        pattern = re.compile(mug_joint_regex)
        entries: list[tuple[int, str, int]] = []
        for joint_id in range(model.njnt):
            name = model.jnt(joint_id).name
            if not name:
                continue
            match = pattern.fullmatch(name)
            if match is None:
                continue
            if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE:
                raise ValueError(f"Joint {name!r} must be a free joint")
            entries.append(
                (
                    int(match.group(1)),
                    name.removesuffix("_jnt"),
                    int(model.jnt_qposadr[joint_id]),
                )
            )
        if not entries:
            raise ValueError(f"No joints matching {mug_joint_regex!r} found in model")
        entries.sort(key=lambda item: item[0])
        names = [name for _, name, _ in entries]
        addrs = np.asarray([addr for _, _, addr in entries], dtype=np.int32)
        return names, addrs

    def reset(self, *, nworld: int = 1) -> None:
        self._nworld = int(nworld)
        self._max_hanging = np.zeros((self._nworld,), dtype=np.int32)
        self._ever_success = np.zeros((self._nworld,), dtype=bool)

    def configure_from_randomization(
        self,
        model: mujoco.MjModel,
        randomization: object,
    ) -> None:
        """Re-measure the tree: its static pose is edited in place per reset."""

        if model is not self.model:
            self._bind_model(model)
            self.reset(nworld=self._nworld)
        else:
            self._measure_tree(model)

    def debug_spec(self) -> dict[str, object] | None:
        return None

    def evaluate_qpos_batch(self, qpos_batch: np.ndarray) -> TaskEvalResult:
        qpos_batch = np.asarray(qpos_batch, dtype=np.float32)
        if qpos_batch.ndim != 2 or qpos_batch.shape[1] != self.model.nq:
            raise ValueError(
                f"Expected qpos batch shape (B, {self.model.nq}), got {qpos_batch.shape}"
            )
        if qpos_batch.shape[0] != self._nworld:
            self.reset(nworld=qpos_batch.shape[0])

        mug_pos = np.stack(
            [qpos_batch[:, adr : adr + 3] for adr in self._mug_qpos_addrs], axis=1
        ).astype(np.float64)
        radial = np.linalg.norm(mug_pos[..., :2] - self._axis_xy[None, None, :], axis=-1)
        hanging = (
            (radial <= self.hang_radius_m)
            & (mug_pos[..., 2] >= self._hang_z_lo)
            & (mug_pos[..., 2] <= self._hang_z_hi)
        )

        num_hanging = hanging.sum(axis=1).astype(np.int32)
        total = len(self.mug_names)
        self._max_hanging = np.maximum(self._max_hanging, num_hanging)
        success = num_hanging >= total
        self._ever_success |= success

        mugs_hanging = [
            [name for name, ok in zip(self.mug_names, world) if bool(ok)]
            for world in hanging
        ]
        return TaskEvalResult(
            reward=num_hanging.astype(np.float32) / float(max(total, 1)),
            success=success,
            metrics={
                "num_mugs_hanging": num_hanging,
                "num_active_mugs": np.full_like(num_hanging, total),
                "max_mugs_hanging_so_far": self._max_hanging.copy(),
                "mugs_hanging": mugs_hanging,
                "mug_names": list(self.mug_names),
                "hang_radius_m": self.hang_radius_m,
                "hang_z_band": [self._hang_z_lo, self._hang_z_hi],
                "ever_success": self._ever_success.copy(),
            },
        )
