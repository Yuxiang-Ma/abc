"""Runtime prompt sequencing for the multi-drawer memory task."""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np


class MultiDrawerSearchRuntime:
    """Advance through prompted objects as they are placed in the goal bin."""

    def __init__(self) -> None:
        self._goal_site_id = -1
        self._target_body_ids: list[int] = []
        self._targets: list[dict[str, str]] = []
        self._current_index = 0
        self._display_prompt = "find the target object"
        self._state = "waiting_for_target"

    def bind(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """Refresh model indices after a scene reload."""

        self._goal_site_id = int(
            mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_SITE,
                "drawer_search_goal_region",
            )
        )
        self._target_body_ids = [
            int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, target["body"]))
            for target in self._targets
        ]

    def after_reset(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        randomization: Any = None,
    ) -> None:
        """Load the reset's sampled target sequence."""

        metadata = getattr(randomization, "metadata", {})
        raw_targets = metadata.get("target_sequence", []) if isinstance(metadata, dict) else []
        if not raw_targets and isinstance(metadata, dict) and metadata.get("target_body"):
            raw_targets = [
                {
                    "body": metadata["target_body"],
                    "joint": metadata.get("target_joint", ""),
                    "label": metadata.get("target_label", "target object"),
                }
            ]
        self._targets = [
            {
                "body": str(target["body"]),
                "joint": str(target.get("joint", "")),
                "label": str(target["label"]),
            }
            for target in raw_targets
            if isinstance(target, dict) and target.get("body") and target.get("label")
        ]
        self._current_index = 0
        self._state = "waiting_for_target" if self._targets else "finished"
        self.bind(model, data)
        self._update_prompt()

    def before_step(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """No pre-step controls are needed."""

    def after_step(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """Advance every target that has reached the bin."""

        while self._current_index < len(self._targets):
            body_id = self._target_body_ids[self._current_index]
            if body_id < 0 or not self._body_is_in_goal(model, data, body_id):
                break
            self._current_index += 1
        self._state = (
            "finished"
            if self._current_index >= len(self._targets)
            else "waiting_for_target"
        )
        self._update_prompt()

    def observation_prompt(self) -> str:
        """Return the prompt that should be included in observations and recordings."""

        return self._display_prompt

    def debug_state(self) -> dict[str, Any]:
        """Return sequence progress for the VR streamer and task dashboard."""

        current_target = (
            None
            if self._current_index >= len(self._targets)
            else dict(self._targets[self._current_index])
        )
        return {
            "state": self._state,
            "display_prompt": self._display_prompt,
            "current_target_index": self._current_index,
            "completed_target_count": self._current_index,
            "target_count": len(self._targets),
            "current_target": current_target,
            "target_sequence": [dict(target) for target in self._targets],
            "complete": self._state == "finished",
        }

    def _body_is_in_goal(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        body_id: int,
    ) -> bool:
        if self._goal_site_id < 0:
            return False
        goal_center = np.asarray(data.site_xpos[self._goal_site_id], dtype=np.float64)
        goal_rotation = np.asarray(
            data.site_xmat[self._goal_site_id],
            dtype=np.float64,
        ).reshape(3, 3)
        body_pos = np.asarray(data.xpos[body_id], dtype=np.float64)
        local_pos = goal_rotation.T @ (body_pos - goal_center)
        half_size = np.asarray(model.site_size[self._goal_site_id], dtype=np.float64)
        return bool(np.all(np.abs(local_pos) <= half_size))

    def _update_prompt(self) -> None:
        if self._current_index >= len(self._targets):
            self._display_prompt = "finished"
            return
        self._display_prompt = f'find the {self._targets[self._current_index]["label"]}'
