"""Grab-target-from-clutter task evaluator."""

from __future__ import annotations

import mujoco
import numpy as np

from abc_sim.task_eval.base import TaskEvalResult
from abc_sim.task_eval.bottles import _quat_to_rotmat_batch
from abc_sim.task_specs import SimTaskSpec


class GrabClutterTargetInBoxEvaluator:
    """Score success when the prompted target object is placed in the target box."""

    def __init__(
        self,
        *,
        model: mujoco.MjModel,
        spec: SimTaskSpec,
        goal_site_name: str = "grab_clutter_target_box_goal_region",
    ) -> None:
        self.model = model
        self.spec = spec
        self._target_qpos_addr = self._resolve_target_qpos_addr(model)
        self._box_qpos_addr = self._resolve_joint_qpos_addr(
            model,
            "grab_clutter_target_box_joint",
        )
        self._goal_local_center, self._goal_half_size = self._resolve_goal_site(
            model,
            goal_site_name,
        )
        self.reset(nworld=1)

    def reset(self, *, nworld: int = 1) -> None:
        self._nworld = int(nworld)
        self._ever_success = np.zeros((self._nworld,), dtype=bool)

    @staticmethod
    def _resolve_target_qpos_addr(model: mujoco.MjModel) -> int:
        joint_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            "target_object_joint",
        )
        if joint_id < 0:
            raise ValueError("Joint 'target_object_joint' not found in model")
        if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError("Joint 'target_object_joint' must be a free joint")
        return int(model.jnt_qposadr[joint_id])

    @staticmethod
    def _resolve_joint_qpos_addr(model: mujoco.MjModel, joint_name: str) -> int:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise ValueError(f"Joint {joint_name!r} not found in model")
        if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError(f"Joint {joint_name!r} must be a free joint")
        return int(model.jnt_qposadr[joint_id])

    @staticmethod
    def _resolve_goal_site(
        model: mujoco.MjModel,
        goal_site_name: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        site_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_SITE,
            goal_site_name,
        )
        if site_id < 0:
            raise ValueError(f"Site {goal_site_name!r} not found in model")
        return (
            np.asarray(model.site_pos[site_id], dtype=np.float32).copy(),
            np.asarray(model.site_size[site_id], dtype=np.float32).copy(),
        )

    def evaluate_qpos_batch(self, qpos_batch: np.ndarray) -> TaskEvalResult:
        qpos_batch = np.asarray(qpos_batch, dtype=np.float32)
        if qpos_batch.ndim != 2 or qpos_batch.shape[1] != self.model.nq:
            raise ValueError(
                f"Expected qpos batch shape (B, {self.model.nq}), got {qpos_batch.shape}"
            )
        if qpos_batch.shape[0] != self._nworld:
            self.reset(nworld=qpos_batch.shape[0])

        target_pos = qpos_batch[:, self._target_qpos_addr: self._target_qpos_addr + 3]
        box_pos = qpos_batch[:, self._box_qpos_addr: self._box_qpos_addr + 3]
        box_quat = qpos_batch[:, self._box_qpos_addr + 3: self._box_qpos_addr + 7]
        rot_world_from_box = _quat_to_rotmat_batch(box_quat)
        rel_world = target_pos - box_pos
        rel_box = np.einsum("bij,bj->bi", np.swapaxes(rot_world_from_box, 1, 2), rel_world)
        delta = np.abs(rel_box - self._goal_local_center[None, :])
        success = np.all(delta <= self._goal_half_size[None, :], axis=1)
        self._ever_success |= success

        world_goal_center = box_pos + np.einsum(
            "bij,j->bi",
            rot_world_from_box,
            self._goal_local_center,
        )
        xy_distance = np.linalg.norm(target_pos[:, :2] - world_goal_center[:, :2], axis=1)
        xy_radius = max(float(np.linalg.norm(self._goal_half_size[:2])), 1e-6)
        reward = np.clip(1.0 - (xy_distance / xy_radius), 0.0, 1.0).astype(np.float32)
        reward = np.where(success, 1.0, reward).astype(np.float32)
        return TaskEvalResult(
            reward=reward,
            success=success,
            metrics={
                "target_pos": target_pos.astype(np.float32),
                "target_box_pos": box_pos.astype(np.float32),
                "target_box_center": np.broadcast_to(
                    world_goal_center,
                    target_pos.shape,
                ).copy(),
                "target_box_half_size": np.broadcast_to(
                    self._goal_half_size,
                    target_pos.shape,
                ).copy(),
                "target_box_xy_distance_m": xy_distance.astype(np.float32),
                "target_in_box": success.copy(),
                "ever_success": self._ever_success.copy(),
            },
        )
