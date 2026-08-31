"""Multi-drawer search task evaluator."""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np

from abc_sim.task_eval.base import TaskEvalResult
from abc_sim.task_eval.bottles import _quat_to_rotmat_batch
from abc_sim.task_specs import SimTaskSpec


class MultiDrawerSearchTargetInBinEvaluator:
    """Score success when the sampled target drawer object is placed in the goal bin."""

    def __init__(
        self,
        *,
        model: mujoco.MjModel,
        spec: SimTaskSpec,
        target_joint_name: str = "drawer_search_target_joint",
        goal_bin_joint_name: str = "drawer_search_goal_bin_joint",
        goal_site_name: str = "drawer_search_goal_region",
    ) -> None:
        self.model = model
        self.spec = spec
        self._target_qpos_addrs = [
            self._resolve_free_joint_qpos_addr(model, target_joint_name)
        ]
        self._bin_qpos_addr = self._resolve_free_joint_qpos_addr(
            model,
            goal_bin_joint_name,
        )
        self._goal_local_center, self._goal_half_size = self._resolve_goal_site(
            model,
            goal_site_name,
        )
        self.reset(nworld=1)

    def configure_from_randomization(
        self,
        model: mujoco.MjModel,
        randomization: Any,
    ) -> None:
        raw_sequence = randomization.metadata.get("target_sequence", [])
        target_joint_names = [
            str(target["joint"])
            for target in raw_sequence
            if isinstance(target, dict) and target.get("joint")
        ]
        if not target_joint_names:
            target_joint_names = [
                str(
                    randomization.metadata.get(
                        "target_joint",
                        "drawer_search_target_joint",
                    )
                )
            ]
        if model is not self.model:
            self.model = model
            self._bin_qpos_addr = self._resolve_free_joint_qpos_addr(
                model,
                "drawer_search_goal_bin_joint",
            )
            self._goal_local_center, self._goal_half_size = self._resolve_goal_site(
                model,
                "drawer_search_goal_region",
            )
        self._target_qpos_addrs = [
            self._resolve_free_joint_qpos_addr(self.model, joint_name)
            for joint_name in target_joint_names
        ]
        self.reset(nworld=self._nworld)

    def reset(self, *, nworld: int = 1) -> None:
        self._nworld = int(nworld)
        self._ever_success = np.zeros((self._nworld,), dtype=bool)
        self._completed_target_count = np.zeros((self._nworld,), dtype=np.int32)

    @staticmethod
    def _resolve_free_joint_qpos_addr(model: mujoco.MjModel, joint_name: str) -> int:
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

        target_positions = np.stack(
            [
                qpos_batch[:, qpos_addr: qpos_addr + 3]
                for qpos_addr in self._target_qpos_addrs
            ],
            axis=1,
        )
        bin_pos = qpos_batch[:, self._bin_qpos_addr: self._bin_qpos_addr + 3]
        bin_quat = qpos_batch[:, self._bin_qpos_addr + 3: self._bin_qpos_addr + 7]
        rot_world_from_bin = _quat_to_rotmat_batch(bin_quat)
        rel_world = target_positions - bin_pos[:, None, :]
        rel_bin = np.einsum(
            "bij,btj->bti",
            np.swapaxes(rot_world_from_bin, 1, 2),
            rel_world,
        )
        delta = np.abs(rel_bin - self._goal_local_center[None, None, :])
        targets_in_bin = np.all(
            delta <= self._goal_half_size[None, None, :],
            axis=2,
        )

        active_indices = self._completed_target_count.copy()
        active_target_in_bin = np.zeros((qpos_batch.shape[0],), dtype=bool)
        target_count = len(self._target_qpos_addrs)
        for world_index in range(qpos_batch.shape[0]):
            current_index = int(active_indices[world_index])
            if current_index < target_count:
                active_target_in_bin[world_index] = targets_in_bin[
                    world_index,
                    current_index,
                ]
            while (
                self._completed_target_count[world_index] < target_count
                and targets_in_bin[
                    world_index,
                    self._completed_target_count[world_index],
                ]
            ):
                self._completed_target_count[world_index] += 1

        success = self._completed_target_count >= target_count
        self._ever_success |= success

        position_indices = np.minimum(active_indices, target_count - 1)
        target_pos = target_positions[
            np.arange(qpos_batch.shape[0]),
            position_indices,
        ]
        world_goal_center = bin_pos + np.einsum(
            "bij,j->bi",
            rot_world_from_bin,
            self._goal_local_center,
        )
        xy_distance = np.linalg.norm(target_pos[:, :2] - world_goal_center[:, :2], axis=1)
        xy_radius = max(float(np.linalg.norm(self._goal_half_size[:2])), 1e-6)
        target_progress = np.clip(
            1.0 - (xy_distance / xy_radius),
            0.0,
            1.0,
        )
        reward = (
            self._completed_target_count.astype(np.float32)
            + np.where(success, 0.0, target_progress)
        ) / float(target_count)
        reward = np.where(success, 1.0, reward).astype(np.float32)
        return TaskEvalResult(
            reward=reward,
            success=success,
            metrics={
                "target_pos": target_pos.astype(np.float32),
                "goal_bin_pos": bin_pos.astype(np.float32),
                "goal_bin_center": np.broadcast_to(
                    world_goal_center,
                    target_pos.shape,
                ).copy(),
                "goal_bin_half_size": np.broadcast_to(
                    self._goal_half_size,
                    target_pos.shape,
                ).copy(),
                "target_bin_xy_distance_m": xy_distance.astype(np.float32),
                "target_in_bin": active_target_in_bin,
                "target_sequence_in_bin": targets_in_bin,
                "current_target_index": np.minimum(
                    self._completed_target_count,
                    target_count,
                ).copy(),
                "completed_target_count": self._completed_target_count.copy(),
                "target_count": np.full(
                    (qpos_batch.shape[0],),
                    target_count,
                    dtype=np.int32,
                ),
                "ever_success": self._ever_success.copy(),
            },
        )
