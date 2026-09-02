"""Nuts-and-bolts sorting task evaluator."""

from __future__ import annotations

import re

import mujoco
import numpy as np

from abc_sim.task_eval.base import TaskEvalResult
from abc_sim.task_eval.bottles import _quat_to_rotmat_batch
from abc_sim.task_specs import SimTaskSpec


class NutsBoltsSortingEvaluator:
    """Score success when the nuts and bolts end up cleanly partitioned.

    Success is a perfect partition: every nut in one bin and every bolt in
    the other, in either orientation. The demonstration data uses both bin
    conventions, so the evaluator judges separation rather than color assignment.

    The strict per-convention counts stay in the metrics
    (``num_correct_spec``, ``num_correct_inverted``, ``sorted_orientation``)
    so an eval can still report which convention a policy used.
    """

    def __init__(
        self,
        *,
        model: mujoco.MjModel,
        spec: SimTaskSpec,
    ) -> None:
        self.model = model
        self.spec = spec
        self.nut_names, self._nut_qpos_addrs = self._resolve_part_qpos_addrs(model, "nut")
        self.bolt_names, self._bolt_qpos_addrs = self._resolve_part_qpos_addrs(model, "bolt")
        self._nuts_bin_qpos_addr = self._resolve_joint_qpos_addr(model, "nuts_sorting_bin_joint")
        self._bolts_bin_qpos_addr = self._resolve_joint_qpos_addr(model, "bolts_sorting_bin_joint")
        self._nuts_goal_local_center, self._nuts_goal_size = self._resolve_goal_site(
            model,
            "nuts_bin_goal_region",
        )
        self._bolts_goal_local_center, self._bolts_goal_size = self._resolve_goal_site(
            model,
            "bolts_bin_goal_region",
        )
        self.reset(nworld=1)

    def reset(self, *, nworld: int = 1) -> None:
        self._nworld = int(nworld)
        self._ever_success = np.zeros((self._nworld,), dtype=bool)
        self._active_nuts = np.ones((self._nworld, len(self.nut_names)), dtype=bool)
        self._active_bolts = np.ones((self._nworld, len(self.bolt_names)), dtype=bool)

    def set_active_object_joints(self, active_joints_by_world: Any) -> None:
        """Score only each world's active parts; the rest sit parked off the table."""
        if self._nworld == 1 and active_joints_by_world and isinstance(active_joints_by_world[0], str):
            active_joints_by_world = [active_joints_by_world]
        for world, joints in enumerate(active_joints_by_world):
            active = set(joints)
            self._active_nuts[world] = [f"{name}_joint" in active for name in self.nut_names]
            self._active_bolts[world] = [f"{name}_joint" in active for name in self.bolt_names]

    @staticmethod
    def _resolve_part_qpos_addrs(
        model: mujoco.MjModel,
        prefix: str,
    ) -> tuple[list[str], np.ndarray]:
        entries: list[tuple[int, str, int]] = []
        pattern = re.compile(rf"{re.escape(prefix)}_(\d+)_joint")
        for joint_id in range(model.njnt):
            joint_name = model.jnt(joint_id).name
            if not joint_name:
                continue
            match = pattern.fullmatch(joint_name)
            if match is None:
                continue
            index = int(match.group(1))
            entries.append((index, f"{prefix}_{index}", int(model.jnt_qposadr[joint_id])))

        if not entries:
            raise ValueError(f"No {prefix}_*_joint freejoints found in model")

        entries.sort(key=lambda item: item[0])
        names = [name for _, name, _ in entries]
        addrs = np.asarray([addr for _, _, addr in entries], dtype=np.int32)
        return names, addrs

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
        site_name: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if site_id < 0:
            raise ValueError(f"Site {site_name!r} not found in model")
        return (
            np.asarray(model.site_pos[site_id], dtype=np.float32),
            np.asarray(model.site_size[site_id], dtype=np.float32),
        )

    @staticmethod
    @staticmethod
    def _inside_local_box(
        positions: np.ndarray,
        center: np.ndarray,
        size: np.ndarray,
    ) -> np.ndarray:
        return np.all(np.abs(positions - center[None, None, :]) <= size[None, None, :], axis=-1)

    def _inside_bin(
        self,
        part_positions: np.ndarray,
        qpos_batch: np.ndarray,
        bin_qpos_addr: int,
        goal_local_center: np.ndarray,
        goal_size: np.ndarray,
    ) -> np.ndarray:
        bin_pos = qpos_batch[:, bin_qpos_addr: bin_qpos_addr + 3]
        bin_quat = qpos_batch[:, bin_qpos_addr + 3: bin_qpos_addr + 7]
        rel_world = part_positions - bin_pos[:, None, :]
        rot_world_from_bin = _quat_to_rotmat_batch(bin_quat)
        rel_bin = np.einsum("bij,bnj->bni", np.swapaxes(rot_world_from_bin, 1, 2), rel_world)
        return self._inside_local_box(rel_bin, goal_local_center, goal_size)

    def evaluate_qpos_batch(self, qpos_batch: np.ndarray) -> TaskEvalResult:
        qpos_batch = np.asarray(qpos_batch, dtype=np.float32)
        if qpos_batch.ndim != 2 or qpos_batch.shape[1] != self.model.nq:
            raise ValueError(
                f"Expected qpos batch shape (B, {self.model.nq}), got {qpos_batch.shape}"
            )
        if qpos_batch.shape[0] != self._nworld:
            self.reset(nworld=qpos_batch.shape[0])

        nut_positions = np.stack(
            [qpos_batch[:, addr : addr + 3] for addr in self._nut_qpos_addrs],
            axis=1,
        )
        bolt_positions = np.stack(
            [qpos_batch[:, addr : addr + 3] for addr in self._bolt_qpos_addrs],
            axis=1,
        )

        nuts_in_nuts_bin = self._active_nuts & self._inside_bin(
            nut_positions,
            qpos_batch,
            self._nuts_bin_qpos_addr,
            self._nuts_goal_local_center,
            self._nuts_goal_size,
        )
        bolts_in_bolts_bin = self._active_bolts & self._inside_bin(
            bolt_positions,
            qpos_batch,
            self._bolts_bin_qpos_addr,
            self._bolts_goal_local_center,
            self._bolts_goal_size,
        )
        nuts_in_bolts_bin = self._active_nuts & self._inside_bin(
            nut_positions,
            qpos_batch,
            self._bolts_bin_qpos_addr,
            self._bolts_goal_local_center,
            self._bolts_goal_size,
        )
        bolts_in_nuts_bin = self._active_bolts & self._inside_bin(
            bolt_positions,
            qpos_batch,
            self._nuts_bin_qpos_addr,
            self._nuts_goal_local_center,
            self._nuts_goal_size,
        )

        num_nuts_sorted = nuts_in_nuts_bin.sum(axis=1).astype(np.int32)
        num_bolts_sorted = bolts_in_bolts_bin.sum(axis=1).astype(np.int32)
        num_nuts_inverted = nuts_in_bolts_bin.sum(axis=1).astype(np.int32)
        num_bolts_inverted = bolts_in_nuts_bin.sum(axis=1).astype(np.int32)
        total_parts = self._active_nuts.sum(axis=1) + self._active_bolts.sum(axis=1)
        # Two ways to partition the parts over the two bins; each rollout is
        # scored by whichever it is closer to, and succeeds on completing
        # either. A part in the wrong bin *for that orientation* counts
        # against it, so mixed bins never score.
        num_correct_spec = num_nuts_sorted + num_bolts_sorted
        num_correct_inverted = num_nuts_inverted + num_bolts_inverted
        spec_complete = num_correct_spec == total_parts
        inverted_complete = num_correct_inverted == total_parts
        success = spec_complete | inverted_complete
        self._ever_success |= success
        num_correct = np.maximum(num_correct_spec, num_correct_inverted)
        orientation = np.where(
            spec_complete, "spec", np.where(inverted_complete, "inverted", "none")
        )

        return TaskEvalResult(
            reward=(num_correct / np.maximum(total_parts, 1)).astype(np.float32),
            success=success,
            metrics={
                "num_nuts_sorted": num_nuts_sorted,
                "num_bolts_sorted": num_bolts_sorted,
                "num_correct": num_correct,
                "num_correct_spec": num_correct_spec,
                "num_correct_inverted": num_correct_inverted,
                "sorted_orientation": orientation.tolist(),
                "ever_success": self._ever_success.copy(),
            },
        )
