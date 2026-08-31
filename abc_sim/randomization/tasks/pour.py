from __future__ import annotations

from typing import Any

import mujoco
import numpy as np

from ..core import (
    _COLOR_RANDOMIZE_PROB,
    _MUG_COLOR_PALETTE,
    PerturbRange,
    RandomizationState,
    ScalePerturbRange,
    SceneRandomizer,
    _apply_mat_color,
    _quat_from_yaw,
    _quat_mul,
)

class PourRandomizer(SceneRandomizer):
    """Randomize mug, cup, and beads-inside-mug for the pour/screw scene.

    Beads translate with the mug (same dx/dy) so they stay inside it after reset.
    Mug and cup are placed independently; contacts checked between them only.
    """

    min_clearance_m = 0.08
    # Only mug and cup in perturbations — used for pairwise/contact checks.
    # Beads are injected by _sample_once below.
    perturbations = [
        PerturbRange("mug_1_jnt",  delta_x=(-0.12, 0.12), delta_y=(-0.15, 0.15)),
        PerturbRange("cup_1_jnt", delta_x=(-0.10, 0.10), delta_y=(-0.20, 0.20)),
    ]
    _bead_joints = [f"bead_{i}_jnt" for i in range(1, 11)]
    size_perturbations = [
        ScalePerturbRange("mug_1_jnt"),
        ScalePerturbRange("cup_1_jnt"),
        *[ScalePerturbRange(f"bead_{i}_jnt") for i in range(1, 11)],
    ]

    def _read_nominals(self, model: Any, data: Any) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        nominals = super()._read_nominals(model, data)
        for bead_name in self._bead_joints:
            jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, bead_name)
            if jnt_id >= 0:
                adr = int(model.jnt_qposadr[jnt_id])
                nominals[bead_name] = (
                    data.qpos[adr: adr + 3].copy(),
                    data.qpos[adr + 3: adr + 7].copy(),
                )
        return nominals

    def _sample_once(
        self,
        nominals: dict[str, tuple[np.ndarray, np.ndarray]],
        rng: np.random.Generator,
    ) -> dict[str, dict[str, list[float]]]:
        x_min, x_max, y_min, y_max = self.table_bounds
        states: dict[str, dict[str, list[float]]] = {}

        mug_p = next(p for p in self.perturbations if p.joint_name == "mug_1_jnt")
        mug_nom_pos, mug_nom_quat = nominals["mug_1_jnt"]
        eff_dx = (max(mug_p.delta_x[0], x_min - mug_nom_pos[0]),
                  min(mug_p.delta_x[1], x_max - mug_nom_pos[0]))
        eff_dy = (max(mug_p.delta_y[0], y_min - mug_nom_pos[1]),
                  min(mug_p.delta_y[1], y_max - mug_nom_pos[1]))
        mug_dx = rng.uniform(*eff_dx)
        mug_dy = rng.uniform(*eff_dy)
        mug_new_pos = mug_nom_pos + np.array([mug_dx, mug_dy, 0.0])
        mug_yaw = float(rng.uniform(*mug_p.delta_yaw))
        q_yaw = _quat_from_yaw(mug_yaw)
        states["mug_1_jnt"] = {
            "pos": mug_new_pos.tolist(),
            "quat": _quat_mul(q_yaw, mug_nom_quat).tolist(),
        }

        # Keep beads in the sampled mug frame. Translating only by (dx, dy)
        # leaves the bead pile behind when mug yaw/scale is randomized.
        mug_scale = float(self._current_scale_states.get("mug_1_jnt", 1.0))
        cos_yaw = float(np.cos(mug_yaw))
        sin_yaw = float(np.sin(mug_yaw))
        for bead_name in self._bead_joints:
            if bead_name not in nominals:
                continue
            bead_nom_pos, bead_nom_quat = nominals[bead_name]
            offset = bead_nom_pos - mug_nom_pos
            rotated_xy_offset = np.array(
                [
                    cos_yaw * offset[0] - sin_yaw * offset[1],
                    sin_yaw * offset[0] + cos_yaw * offset[1],
                ],
                dtype=np.float64,
            )
            bead_new_pos = mug_new_pos + np.array(
                [
                    rotated_xy_offset[0] * mug_scale,
                    rotated_xy_offset[1] * mug_scale,
                    offset[2] * mug_scale,
                ],
                dtype=np.float64,
            )
            states[bead_name] = {
                "pos": bead_new_pos.tolist(),
                "quat": _quat_mul(q_yaw, bead_nom_quat).tolist(),
            }

        cup_p = next(p for p in self.perturbations if p.joint_name == "cup_1_jnt")
        cup_nom_pos, cup_nom_quat = nominals["cup_1_jnt"]
        eff_dx = (max(cup_p.delta_x[0], x_min - cup_nom_pos[0]),
                  min(cup_p.delta_x[1], x_max - cup_nom_pos[0]))
        eff_dy = (max(cup_p.delta_y[0], y_min - cup_nom_pos[1]),
                  min(cup_p.delta_y[1], y_max - cup_nom_pos[1]))
        cup_new_pos = cup_nom_pos + np.array([
            rng.uniform(*eff_dx), rng.uniform(*eff_dy), 0.0,
        ])
        q_yaw_cup = _quat_from_yaw(rng.uniform(*cup_p.delta_yaw))
        states["cup_1_jnt"] = {
            "pos": cup_new_pos.tolist(),
            "quat": _quat_mul(q_yaw_cup, cup_nom_quat).tolist(),
        }

        return states

    def _pairwise_ok(self, states: dict[str, dict[str, list[float]]]) -> bool:
        # Beads are intentionally close together inside the mug. Only use the
        # independently sampled containers for placement rejection.
        container_states = {
            name: states[name]
            for name in ("mug_1_jnt", "cup_1_jnt")
            if name in states
        }
        return super()._pairwise_ok(container_states)

    def randomize(
        self,
        model: Any,
        data: Any,
        seed: int | None = None,
        request: Any | None = None,
    ) -> RandomizationState:
        state = super().randomize(model, data, seed, request=request)
        rng = np.random.default_rng(seed)
        target_model = self._env_ref.model if self._env_ref is not None else model
        if rng.random() < _COLOR_RANDOMIZE_PROB:
            for mat_name in ("mug_1_color", "cup_body_color"):
                _apply_mat_color(target_model, mat_name, _MUG_COLOR_PALETTE[rng.integers(len(_MUG_COLOR_PALETTE))])
        return state




__all__ = ["PourRandomizer"]
