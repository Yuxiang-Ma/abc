"""Anchored end-effector-relative pose composition."""

from __future__ import annotations

import numpy as np


def _normalize(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=float).reshape(4)
    norm = np.linalg.norm(quaternion)
    if norm <= 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return quaternion / norm


def _multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = left
    w2, x2, y2, z2 = right
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def _inverse(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = _normalize(quaternion)
    return np.array([w, -x, -y, -z])


def _scaled_rotation(quaternion: np.ndarray, scale: float) -> np.ndarray:
    quaternion = _normalize(quaternion)
    if quaternion[0] < 0:
        quaternion = -quaternion
    angle = 2.0 * np.arccos(np.clip(quaternion[0], -1.0, 1.0))
    sine = np.sin(angle / 2.0)
    if abs(sine) < 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = quaternion[1:] / sine
    scaled = angle * float(scale)
    return _normalize(
        np.concatenate(([np.cos(scaled / 2.0)], axis * np.sin(scaled / 2.0)))
    )


def apply_relative_pose_delta(
    *,
    tracked_pos: np.ndarray,
    tracked_wxyz: np.ndarray,
    anchor_tracked_pos: np.ndarray,
    anchor_tracked_wxyz: np.ndarray,
    anchor_output_pos: np.ndarray,
    anchor_output_wxyz: np.ndarray,
    position_scale: float = 1.0,
    rotation_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the tracked pose's anchor-relative SE(3) delta to an output pose."""
    delta_pos = (
        np.asarray(tracked_pos, dtype=float)
        - np.asarray(anchor_tracked_pos, dtype=float)
    ) * float(position_scale)
    delta_rotation = _multiply(_normalize(tracked_wxyz), _inverse(anchor_tracked_wxyz))
    delta_rotation = _scaled_rotation(delta_rotation, rotation_scale)
    output_rotation = _multiply(delta_rotation, _normalize(anchor_output_wxyz))
    return (
        np.asarray(anchor_output_pos, dtype=float) + delta_pos,
        _normalize(output_rotation),
    )
