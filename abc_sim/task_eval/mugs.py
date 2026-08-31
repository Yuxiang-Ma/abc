"""Mug task evaluators."""

from __future__ import annotations

import re
from typing import Any

import mujoco
import numpy as np

from abc_sim.task_eval.base import TaskEvalResult
from abc_sim.task_specs import SimTaskSpec


def _quat_to_rotmat_batch(quat_batch: np.ndarray) -> np.ndarray:
    quat_batch = np.asarray(quat_batch, dtype=np.float32)
    if quat_batch.ndim != 2 or quat_batch.shape[1] != 4:
        raise ValueError(f"Expected quaternion batch shape (B, 4), got {quat_batch.shape}")

    norm = np.linalg.norm(quat_batch, axis=1, keepdims=True)
    norm = np.where(norm > 0.0, norm, 1.0)
    q = quat_batch / norm
    w, x, y, z = q.T

    return np.stack(
        [
            np.stack(
                [
                    1.0 - 2.0 * (y * y + z * z),
                    2.0 * (x * y - z * w),
                    2.0 * (x * z + y * w),
                ],
                axis=-1,
            ),
            np.stack(
                [
                    2.0 * (x * y + z * w),
                    1.0 - 2.0 * (x * x + z * z),
                    2.0 * (y * z - x * w),
                ],
                axis=-1,
            ),
            np.stack(
                [
                    2.0 * (x * z - y * w),
                    2.0 * (y * z + x * w),
                    1.0 - 2.0 * (x * x + y * y),
                ],
                axis=-1,
            ),
        ],
        axis=1,
    ).astype(np.float32)


class MugFlipUprightEvaluator:
    """Score mug flip by requiring every spawned mug to be right-side-up."""

    def __init__(
        self,
        *,
        model: mujoco.MjModel,
        spec: SimTaskSpec,
        upright_z_min: float = 0.75,
    ) -> None:
        self.model = model
        self.spec = spec
        self.upright_z_min = float(upright_z_min)
        self.mug_names, self._mug_qpos_addrs = self._resolve_mug_qpos_addrs(model)
        self.success_count = len(self.mug_names)
        self._nworld = 1
        self._max_upright_mugs = np.zeros((1,), dtype=np.int32)
        self._ever_success = np.zeros((1,), dtype=bool)

    def reset(self, *, nworld: int = 1) -> None:
        self._nworld = int(nworld)
        self._max_upright_mugs = np.zeros((self._nworld,), dtype=np.int32)
        self._ever_success = np.zeros((self._nworld,), dtype=bool)

    @staticmethod
    def _resolve_mug_qpos_addrs(model: mujoco.MjModel) -> tuple[list[str], np.ndarray]:
        entries: list[tuple[int, str, int]] = []
        for joint_id in range(model.njnt):
            name = model.jnt(joint_id).name
            if not name:
                continue
            match = re.fullmatch(r"mug_(\d+)_jnt", name)
            if match is None:
                continue
            mug_index = int(match.group(1))
            entries.append((mug_index, f"mug_{mug_index}", int(model.jnt_qposadr[joint_id])))

        entries.sort(key=lambda item: item[0])
        return [name for _, name, _ in entries], np.asarray(
            [adr for _, _, adr in entries],
            dtype=np.int32,
        )

    def evaluate_qpos_batch(self, qpos_batch: np.ndarray) -> TaskEvalResult:
        qpos_batch = np.asarray(qpos_batch, dtype=np.float32)
        if qpos_batch.ndim != 2 or qpos_batch.shape[1] != self.model.nq:
            raise ValueError(
                f"Expected qpos batch shape (B, {self.model.nq}), got {qpos_batch.shape}"
            )
        if qpos_batch.shape[0] != self._nworld:
            self.reset(nworld=qpos_batch.shape[0])

        if not self.mug_names:
            reward = np.zeros((qpos_batch.shape[0],), dtype=np.float32)
            success = np.zeros((qpos_batch.shape[0],), dtype=bool)
            metrics = {
                "num_upright_mugs": np.zeros((qpos_batch.shape[0],), dtype=np.int32),
                "num_active_mugs": np.zeros((qpos_batch.shape[0],), dtype=np.int32),
                "max_upright_mugs_so_far": self._max_upright_mugs.copy(),
                "ever_success": self._ever_success.copy(),
                "mug_upright_mask": np.zeros((qpos_batch.shape[0], 0), dtype=bool),
                "mug_upright_z": np.zeros((qpos_batch.shape[0], 0), dtype=np.float32),
                "min_mug_upright_z": np.full(
                    (qpos_batch.shape[0],),
                    np.nan,
                    dtype=np.float32,
                ),
                "upright_mugs": [[] for _ in range(qpos_batch.shape[0])],
                "mug_names": [],
                "success_count": 0,
                "upright_z_min": self.upright_z_min,
            }
            return TaskEvalResult(reward=reward, success=success, metrics=metrics)

        mug_quat = np.stack(
            [qpos_batch[:, adr + 3 : adr + 7] for adr in self._mug_qpos_addrs],
            axis=1,
        )
        mug_rot = _quat_to_rotmat_batch(mug_quat.reshape(-1, 4)).reshape(
            qpos_batch.shape[0],
            len(self.mug_names),
            3,
            3,
        )
        local_z_world = mug_rot[..., :, 2]
        upright_z = local_z_world[..., 2]
        upright_mask = upright_z >= self.upright_z_min

        num_upright = upright_mask.sum(axis=1).astype(np.int32)
        active_count = np.full(
            (qpos_batch.shape[0],),
            len(self.mug_names),
            dtype=np.int32,
        )
        self._max_upright_mugs = np.maximum(self._max_upright_mugs, num_upright)
        reward = num_upright.astype(np.float32) / active_count.astype(np.float32)
        success = num_upright == active_count
        self._ever_success |= success
        upright_mugs = [
            [name for name, is_upright in zip(self.mug_names, world_mask) if bool(is_upright)]
            for world_mask in upright_mask
        ]

        metrics: dict[str, Any] = {
            "num_upright_mugs": num_upright,
            "num_active_mugs": active_count,
            "max_upright_mugs_so_far": self._max_upright_mugs.copy(),
            "ever_success": self._ever_success.copy(),
            "mug_upright_mask": upright_mask,
            "mug_upright_z": upright_z.astype(np.float32),
            "min_mug_upright_z": upright_z.min(axis=1).astype(np.float32),
            "upright_mugs": upright_mugs,
            "mug_names": list(self.mug_names),
            "success_count": len(self.mug_names),
            "upright_z_min": self.upright_z_min,
        }
        return TaskEvalResult(reward=reward, success=success, metrics=metrics)
