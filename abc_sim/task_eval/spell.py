"""Letter-block spelling task evaluator."""

from __future__ import annotations

import mujoco
import numpy as np

from abc_sim.task_eval.base import TaskEvalResult
from abc_sim.task_eval.bottles import _quat_to_rotmat_batch
from abc_sim.task_specs import SimTaskSpec


class SpellWordEvaluator:
    """Score spelling by the word's blocks forming a tight reading-order row.

    The blocks scene deals all 26 letter cubes; a spell task is done when the
    word's letters sit in a row that reads left to right from the operator's
    side -- which in table coordinates is *decreasing* y. Success requires,
    for each consecutive letter pair of the word:

    * reading order and spacing: dy in [-``gap_max_m``, -``gap_min_m``]
    * alignment: |dx| <= ``row_dx_max_m``
    * every letter resting at table height (within ``z_tolerance_m`` of the
      table plane plus the block's own half-height) with a letter face up
      (|block local x toward world up| >= ``upright_dot_min``): a block
      stacked on a neighbour or resting letter-sideways does not spell.

    Distractor blocks are ignored.

    The word arrives through the spec's evaluator options, so seven specs
    share this class; all seven released words have distinct letters, and the
    constructor rejects a word that repeats one rather than silently scoring
    a block twice.
    """

    def __init__(
        self,
        *,
        model: mujoco.MjModel,
        spec: SimTaskSpec,
        word: str,
        table_geom_name: str = "table_plane",
        gap_min_m: float = 0.015,
        gap_max_m: float = 0.060,
        row_dx_max_m: float = 0.03,
        z_tolerance_m: float = 0.02,
        upright_dot_min: float = 0.85,
    ) -> None:
        self.spec = spec
        self.word = str(word).upper()
        if not self.word.isalpha():
            raise ValueError(f"word must be alphabetic, got {word!r}")
        if len(set(self.word)) != len(self.word):
            raise ValueError(
                f"word {word!r} repeats a letter; one block per letter cannot spell it"
            )
        self.gap_min_m = float(gap_min_m)
        self.gap_max_m = float(gap_max_m)
        self.row_dx_max_m = float(row_dx_max_m)
        self.z_tolerance_m = float(z_tolerance_m)
        self.upright_dot_min = float(upright_dot_min)
        self._table_geom_name = table_geom_name
        self._bind_model(model)
        self.reset(nworld=1)

    def _bind_model(self, model: mujoco.MjModel) -> None:
        self.model = model
        self._letter_addrs = np.asarray(
            [
                self._resolve_free_joint_qpos_adr(model, f"block_{letter}_jnt")
                for letter in self.word
            ],
            dtype=np.int32,
        )
        self._rest_z = self._resolve_rest_heights(model)

    def reset(self, *, nworld: int = 1) -> None:
        self._nworld = int(nworld)
        self._ever_success = np.zeros((self._nworld,), dtype=bool)

    def configure_from_randomization(
        self,
        model: mujoco.MjModel,
        randomization: object,
    ) -> None:
        """Rebind to a model a scene reload swapped in under us."""

        if model is not self.model:
            self._bind_model(model)
            self.reset(nworld=self._nworld)

    def debug_spec(self) -> dict[str, object] | None:
        return None

    @staticmethod
    def _resolve_free_joint_qpos_adr(model: mujoco.MjModel, joint_name: str) -> int:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise ValueError(f"Joint {joint_name!r} not found in model")
        if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError(f"Joint {joint_name!r} must be a free joint")
        return int(model.jnt_qposadr[joint_id])

    def _resolve_rest_heights(self, model: mujoco.MjModel) -> np.ndarray:
        """Table plane z plus each word block's own collision half-height."""

        geom_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, self._table_geom_name
        )
        if geom_id < 0:
            raise ValueError(f"Geom {self._table_geom_name!r} not found in model")
        table_z = float(model.geom_pos[geom_id][2])

        heights = []
        for letter in self.word:
            body_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, f"block_{letter}"
            )
            half = None
            for gid in range(model.ngeom):
                if int(model.geom_bodyid[gid]) != body_id:
                    continue
                if int(model.geom_type[gid]) == int(mujoco.mjtGeom.mjGEOM_BOX):
                    half = float(model.geom_size[gid][2])
                    break
            if half is None:
                raise ValueError(
                    f"block_{letter} has no box collision geom to measure"
                )
            heights.append(table_z + half)
        return np.asarray(heights, dtype=np.float64)

    def evaluate_qpos_batch(self, qpos_batch: np.ndarray) -> TaskEvalResult:
        qpos_batch = np.asarray(qpos_batch, dtype=np.float32)
        if qpos_batch.ndim != 2 or qpos_batch.shape[1] != self.model.nq:
            raise ValueError(
                f"Expected qpos batch shape (B, {self.model.nq}), got {qpos_batch.shape}"
            )
        if qpos_batch.shape[0] != self._nworld:
            self.reset(nworld=qpos_batch.shape[0])

        pos = np.stack(
            [qpos_batch[:, adr : adr + 3] for adr in self._letter_addrs], axis=1
        ).astype(np.float64)
        quat = np.stack(
            [qpos_batch[:, adr + 3 : adr + 7] for adr in self._letter_addrs], axis=1
        )

        dy = np.diff(pos[..., 1], axis=1)
        dx = np.diff(pos[..., 0], axis=1)
        pair_ok = (
            (dy <= -self.gap_min_m)
            & (dy >= -self.gap_max_m)
            & (np.abs(dx) <= self.row_dx_max_m)
        )
        on_table = (
            np.abs(pos[..., 2] - self._rest_z[None, :]) <= self.z_tolerance_m
        )
        batch, count = quat.shape[:2]
        rot = _quat_to_rotmat_batch(quat.reshape(-1, 4))
        # The letter is printed on both x faces of the cube (a flipped block
        # still reads -- checked by rendering a demo that ends with one block
        # rotated 180 degrees), so a letter face is up whenever the block's
        # local x axis is vertical, either sign.
        face_up = np.abs(rot[:, 2, 0].reshape(batch, count)) >= self.upright_dot_min

        letter_ok = on_table & face_up
        num_pairs_ok = pair_ok.sum(axis=1).astype(np.int32)
        num_letters_ok = letter_ok.sum(axis=1).astype(np.int32)
        success = pair_ok.all(axis=1) & letter_ok.all(axis=1)
        self._ever_success |= success

        total = pair_ok.shape[1] + letter_ok.shape[1]
        reward = (
            (num_pairs_ok + num_letters_ok).astype(np.float32) / float(total)
        ).clip(0.0, 1.0)
        reward = np.where(success, 1.0, np.minimum(reward, 0.99)).astype(np.float32)

        return TaskEvalResult(
            reward=reward,
            success=success,
            metrics={
                "word": self.word,
                "num_pairs_in_row": num_pairs_ok,
                "num_letters_resting_face_up": num_letters_ok,
                "pair_dy_m": dy.astype(np.float32),
                "pair_dx_m": dx.astype(np.float32),
                "ever_success": self._ever_success.copy(),
            },
        )
