"""Chess board-setup task evaluator."""

from __future__ import annotations

import re
from typing import Any, Mapping

import mujoco
import numpy as np

from abc_sim.task_eval.base import TaskEvalResult
from abc_sim.task_eval.bottles import _quat_to_rotmat_batch
from abc_sim.task_specs import SimTaskSpec


class ChessBoardSetupEvaluator:
    """Score set_up_chess_pieces by the target pieces standing on their squares.

    Every reset the randomizer picks 5-10 target pieces, displaces them (onto
    the table, knocked near their squares, or into the tin), leaves the rest
    standing on their home squares, and records in its metadata exactly what
    finishing looks like: ``target_joints`` and each one's board pose in
    ``target_poses``. The evaluator adopts those, converting every goal into
    the *board's own frame* from the recorded board spawn pose -- the board is
    a free body the arms can nudge, so a piece is judged against the square
    under it now, not where that square was at reset.

    Identical pieces are interchangeable: the demonstrations put *a* black
    pawn on each vacated pawn square, not the specific pawn the randomizer
    happened to displace, and identity-keyed goals would fail a third of them
    for swaps no observer could see. Targets are therefore matched to their
    class's goal squares (nearest first within e.g. the displaced black
    pawns) before scoring. A matched piece counts as placed when it sits
    within ``position_tolerance_m`` of the square centre *in the board
    plane*, within ``height_tolerance_m`` along the board's normal, and
    standing up (``upright_z_min``): a piece lying across the right square is
    knocked over, not set up. The board's free-joint frame is not z-up, so
    the plane/normal split comes from the recorded spawn orientation (the
    board-local direction that pointed at world up), not from axis indices.
    Success is every target placed at once; ``ever_success`` latches for
    eval_policy's early-stop, and the metrics carry how many non-target
    pieces have strayed from their recorded poses so a rollout that
    bulldozes the rest of the board while placing its targets is at least
    visible.

    Until the first reset hands over metadata there is no target set to
    score, so the evaluator reports failure alongside ``goal_configured:
    False`` rather than guessing one.
    """

    def __init__(
        self,
        *,
        model: mujoco.MjModel,
        spec: SimTaskSpec,
        board_joint_name: str = "chessboard",
        position_tolerance_m: float = 0.028,
        height_tolerance_m: float = 0.05,
        upright_z_min: float = 0.85,
    ) -> None:
        self.spec = spec
        self._board_joint_name = board_joint_name
        self.position_tolerance_m = float(position_tolerance_m)
        self.height_tolerance_m = float(height_tolerance_m)
        self.upright_z_min = float(upright_z_min)
        self._directive: dict[str, Any] | None = None
        self._bind_model(model)
        self.reset(nworld=1)

    def _bind_model(self, model: mujoco.MjModel) -> None:
        self.model = model
        self._directive = None
        self._board_adr = self._resolve_free_joint_qpos_adr(
            model, self._board_joint_name
        )

    def reset(self, *, nworld: int = 1) -> None:
        self._nworld = int(nworld)
        self._ever_success = np.zeros((self._nworld,), dtype=bool)

    def configure_from_randomization(
        self,
        model: mujoco.MjModel,
        randomization: object,
    ) -> None:
        """Adopt the episode's target set (and any reloaded model)."""

        if model is not self.model:
            self._bind_model(model)
            self.reset(nworld=self._nworld)

        metadata = getattr(randomization, "metadata", None)
        if metadata is None and isinstance(randomization, Mapping):
            metadata = randomization.get("metadata")
        if not isinstance(metadata, Mapping):
            return
        target_joints = metadata.get("target_joints")
        target_poses = metadata.get("target_poses")
        if not target_joints or not isinstance(target_poses, Mapping):
            return

        object_states = getattr(randomization, "object_states", None)
        if object_states is None and isinstance(randomization, Mapping):
            object_states = randomization.get("object_states")
        if not isinstance(object_states, Mapping):
            return
        board_state = object_states.get(self._board_joint_name) or object_states.get(
            "chessboard"
        )
        if not isinstance(board_state, Mapping) or "pos" not in board_state:
            return

        board_pos = np.asarray(board_state["pos"], dtype=np.float64)
        board_quat = np.asarray(
            board_state.get("quat", (1.0, 0.0, 0.0, 0.0)), dtype=np.float64
        )
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, board_quat)
        rot_world_from_board = rot.reshape(3, 3)
        # The board-local direction that pointed at world up when the goals
        # were recorded; heights and planar distances split against it.
        normal_local = rot_world_from_board.T @ np.asarray([0.0, 0.0, 1.0])

        joints: list[str] = []
        addrs: list[int] = []
        goals_board: list[np.ndarray] = []
        for joint_name in target_joints:
            pose = target_poses.get(joint_name)
            if not isinstance(pose, Mapping) or "pos" not in pose:
                raise ValueError(
                    f"target_poses is missing a pose for target {joint_name!r}"
                )
            joints.append(str(joint_name))
            addrs.append(self._resolve_free_joint_qpos_adr(self.model, str(joint_name)))
            world = np.asarray(pose["pos"], dtype=np.float64)
            goals_board.append(rot_world_from_board.T @ (world - board_pos))

        non_targets = [
            str(name)
            for name in (metadata.get("non_target_pieces") or [])
            if isinstance(name, str)
        ]
        non_target_addrs = []
        non_target_goals = []
        board_poses = metadata.get("board_target_poses")
        if isinstance(board_poses, Mapping):
            for joint_name in non_targets:
                pose = board_poses.get(joint_name)
                if not isinstance(pose, Mapping) or "pos" not in pose:
                    continue
                non_target_addrs.append(
                    self._resolve_free_joint_qpos_adr(self.model, joint_name)
                )
                world = np.asarray(pose["pos"], dtype=np.float64)
                non_target_goals.append(rot_world_from_board.T @ (world - board_pos))

        classes = [re.sub(r"_\d+_jnt$", "", name) for name in joints]
        self._directive = {
            "joints": joints,
            "classes": classes,
            "normal_local": normal_local,
            "addrs": np.asarray(addrs, dtype=np.int32),
            "goals_board": np.asarray(goals_board, dtype=np.float64),
            "non_target_addrs": np.asarray(non_target_addrs, dtype=np.int32),
            "non_target_goals": (
                np.asarray(non_target_goals, dtype=np.float64)
                if non_target_goals
                else np.zeros((0, 3), dtype=np.float64)
            ),
            "scenario": str(metadata.get("scenario", "")),
        }

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

    def _split_plane_normal(
        self, delta: np.ndarray, normal_local: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Split board-frame deltas into (planar distance, height)."""

        height = delta @ normal_local
        planar_vec = delta - height[..., None] * normal_local[None, ...]
        return np.linalg.norm(planar_vec, axis=-1), height

    def _piece_placement(
        self,
        qpos_batch: np.ndarray,
        addrs: np.ndarray,
        goals_board: np.ndarray,
        rot_board_from_world: np.ndarray,
        board_pos: np.ndarray,
        normal_local: np.ndarray,
        classes: list[str] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Per piece: (on-a-matched-square mask, upright mask).

        With ``classes`` given, pieces are first matched greedily (nearest
        planar distance) to the goal squares of their own class, so
        interchangeable pieces score wherever their kind belongs.
        """

        pos = np.stack(
            [qpos_batch[:, adr : adr + 3] for adr in addrs], axis=1
        ).astype(np.float64)
        quat = np.stack(
            [qpos_batch[:, adr + 3 : adr + 7] for adr in addrs], axis=1
        ).astype(np.float64)
        rel = np.einsum(
            "bij,bnj->bni", rot_board_from_world, pos - board_pos[:, None, :]
        )
        batch, count = rel.shape[:2]

        in_square = np.zeros((batch, count), dtype=bool)
        if classes is None:
            planar, height = self._split_plane_normal(
                rel - goals_board[None, :, :], normal_local
            )
            in_square = (planar <= self.position_tolerance_m) & (
                np.abs(height) <= self.height_tolerance_m
            )
        else:
            class_indices: dict[str, list[int]] = {}
            for index, name in enumerate(classes):
                class_indices.setdefault(name, []).append(index)
            for indices in class_indices.values():
                idx = np.asarray(indices, dtype=np.int32)
                for b in range(batch):
                    # planar distance of every class piece to every class goal
                    delta = rel[b, idx][:, None, :] - goals_board[idx][None, :, :]
                    planar, height = self._split_plane_normal(delta, normal_local)
                    ok = (planar <= self.position_tolerance_m) & (
                        np.abs(height) <= self.height_tolerance_m
                    )
                    # greedy nearest-first assignment
                    order = np.argsort(planar, axis=None)
                    used_piece = np.zeros(len(idx), dtype=bool)
                    used_goal = np.zeros(len(idx), dtype=bool)
                    for flat in order:
                        pi, gi = divmod(int(flat), len(idx))
                        if used_piece[pi] or used_goal[gi]:
                            continue
                        used_piece[pi] = True
                        used_goal[gi] = True
                        if ok[pi, gi]:
                            in_square[b, idx[pi]] = True

        rot = _quat_to_rotmat_batch(quat.reshape(-1, 4).astype(np.float32))
        upright = rot[:, 2, 2].reshape(batch, count) >= self.upright_z_min
        return in_square, upright

    def evaluate_qpos_batch(self, qpos_batch: np.ndarray) -> TaskEvalResult:
        qpos_batch = np.asarray(qpos_batch, dtype=np.float32)
        if qpos_batch.ndim != 2 or qpos_batch.shape[1] != self.model.nq:
            raise ValueError(
                f"Expected qpos batch shape (B, {self.model.nq}), got {qpos_batch.shape}"
            )
        if qpos_batch.shape[0] != self._nworld:
            self.reset(nworld=qpos_batch.shape[0])

        batch = qpos_batch.shape[0]
        if self._directive is None:
            zeros = np.zeros((batch,), dtype=np.float32)
            return TaskEvalResult(
                reward=zeros,
                success=np.zeros((batch,), dtype=bool),
                metrics={
                    "goal_configured": False,
                    "ever_success": self._ever_success.copy(),
                },
            )

        directive = self._directive
        board_pos = qpos_batch[:, self._board_adr : self._board_adr + 3].astype(
            np.float64
        )
        board_quat = qpos_batch[:, self._board_adr + 3 : self._board_adr + 7]
        rot_board_from_world = np.swapaxes(
            _quat_to_rotmat_batch(board_quat), 1, 2
        ).astype(np.float64)

        in_square, upright = self._piece_placement(
            qpos_batch,
            directive["addrs"],
            directive["goals_board"],
            rot_board_from_world,
            board_pos,
            directive["normal_local"],
            classes=directive["classes"],
        )
        placed = in_square & upright
        num_placed = placed.sum(axis=1).astype(np.int32)
        target_count = len(directive["joints"])
        success = num_placed == target_count
        self._ever_success |= success

        if len(directive["non_target_addrs"]):
            nt_in, nt_up = self._piece_placement(
                qpos_batch,
                directive["non_target_addrs"],
                directive["non_target_goals"],
                rot_board_from_world,
                board_pos,
                directive["normal_local"],
            )
            num_non_target_disturbed = (~(nt_in & nt_up)).sum(axis=1).astype(np.int32)
        else:
            num_non_target_disturbed = np.zeros((batch,), dtype=np.int32)

        placed_names = [
            [name for name, ok in zip(directive["joints"], world) if bool(ok)]
            for world in placed
        ]
        return TaskEvalResult(
            reward=(num_placed.astype(np.float32) / float(max(target_count, 1))),
            success=success,
            metrics={
                "goal_configured": True,
                "scenario": directive["scenario"],
                "target_count": target_count,
                "num_targets_placed": num_placed,
                "targets_placed": placed_names,
                "num_non_target_disturbed": num_non_target_disturbed,
                "ever_success": self._ever_success.copy(),
            },
        )
