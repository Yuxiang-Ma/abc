"""Torch-free reading of exported episode directories.

One episode directory holds ``states_actions.bin`` (float64 rows of
``2 * STATE_DIM``: 14 recorded robot dofs then the 14 commanded ones),
``episode_metadata.json``, ``combined_camera-images-rgb.mp4`` and, when the
replay sidecars were exported, ``scene_assembled.xml`` / ``initial_qpos.npy`` /
``scene_qpos.npy`` / ``randomization.json``. Everything here supports the
interactive episode viewer (abc_minimal/viz_episode.py) and must stay
importable without torch.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

STATE_DIM = 14
ARM_INDICES = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
GRIPPER_INDICES = [6, 13]
JOINT_LABELS = [
    *[f"left_joint{j}" for j in range(1, 7)],
    "left_gripper",
    *[f"right_joint{j}" for j in range(1, 7)],
    "right_gripper",
]

# The export grid is the recording's fixed 30 Hz frame clock; states, actions
# and video frames all live on it, whatever the task's control rate was.
DATA_FPS = 30.0

# The whole scene's qpos on that same grid, one row per exported frame. Added by
# the sim_224 re-cut; episodes cut before it carry only row 0, as initial_qpos.npy.
SCENE_QPOS_FILENAME = "scene_qpos.npy"

# Recorded scenes may hardcode the absolute asset root of the machine that
# collected them. Published (public-release) scenes instead carry roots already
# relative to ``assets/`` — those match nothing here and pass through unchanged,
# which is correct: abc_sim.make_env materializes scene strings inside
# ``abc_sim/models/``, where a relative ``assets/`` resolves.
HOST_ASSET_ROOT = re.compile(r'(?<=")(/[^"]*?/models/assets)(?=[/"])')


def load_episode(episode_dir: Path) -> tuple[dict, np.ndarray, np.ndarray]:
    """Read episode metadata and split states_actions.bin into states and actions."""
    metadata_path = episode_dir / "episode_metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}

    raw = np.fromfile(episode_dir / "states_actions.bin", dtype=np.float64)
    if raw.size == 0 or raw.size % (2 * STATE_DIM):
        raise ValueError(
            f"{episode_dir.name}: states_actions.bin holds {raw.size} float64s, "
            f"not a multiple of {2 * STATE_DIM}"
        )
    table = raw.reshape(-1, 2 * STATE_DIM)

    expected_steps = metadata.get("num_steps")
    if expected_steps is not None and len(table) != int(expected_steps):
        raise ValueError(
            f"{episode_dir.name}: states_actions.bin has {len(table)} steps, "
            f"metadata says {expected_steps}"
        )
    return metadata, table[:, :STATE_DIM], table[:, STATE_DIM:]


def localize_scene_assets(scene_xml: str, assets_dir: Path) -> tuple[str, list[str]]:
    """Repoint a recorded scene's absolute asset root at the loaded package's assets."""
    rewritten = sorted(
        {match.group(0) for match in HOST_ASSET_ROOT.finditer(scene_xml)}
    )
    for host_root in rewritten:
        scene_xml = scene_xml.replace(host_root, str(assets_dir))
    return scene_xml, rewritten


def load_scene_xml(
    episode_dir: Path, metadata: dict, assets_dir: Path
) -> tuple[str | None, str]:
    """Return the episode's own scene as a localized XML string, or None.

    The metadata pointer is legally null when no scene sidecar was exported —
    ``or``, not a ``.get`` default, so None still falls back to the filename.
    """
    scene_xml_path = episode_dir / (
        metadata.get("scene_xml_file") or "scene_assembled.xml"
    )
    if not scene_xml_path.exists():
        return None, "task registry scene"
    scene_xml, rewritten = localize_scene_assets(
        scene_xml_path.read_text(), assets_dir
    )
    return scene_xml, f"{scene_xml_path.name} (asset roots rewritten: {len(rewritten)})"


def seed_initial_state(env, episode_dir: Path, metadata: dict, states: np.ndarray) -> str:
    """Plant the episode's recorded starting state; fall back to arm-only placement."""
    import mujoco

    qpos_path = episode_dir / (metadata.get("initial_qpos_file") or "initial_qpos.npy")
    if qpos_path.exists():
        initial_qpos = np.load(qpos_path)
        if initial_qpos.shape == (env.model.nq,):
            env.data.qpos[:] = initial_qpos
            env.data.qvel[:] = 0.0
            mujoco.mj_forward(env.model, env.data)
            return f"initial_qpos.npy (full qpos, nq={env.model.nq})"
        reason = (
            f"initial_qpos.npy has {initial_qpos.size} dofs, model nq={env.model.nq}"
        )
    else:
        reason = "no initial_qpos.npy in episode dir"

    # Only the 14 robot dofs can be recovered from the policy-space recording; the rest
    # of the scene keeps whatever the reset produced. Reuse the env's own setter so the
    # gripper scaling stays in one place.
    env._set_qpos_from_state(np.asarray(states[0], dtype=np.float32))
    mujoco.mj_forward(env.model, env.data)
    return f"arm-only from states[0] ({reason})"


def load_scene_qpos(
    episode_dir: Path, metadata: dict, num_steps: int
) -> np.ndarray | None:
    """Return the recorded qpos of the whole scene per frame, or None.

    Where ``initial_qpos.npy`` pins frame 0 only, this is the ground-truth motion
    of every dof — objects as well as arms — on the export's 30 Hz grid, so a
    replay can pose the scene instead of reconstructing arms from the 14D states.
    Absent for episodes cut before the sim_224 re-cut; None means "arms only",
    not an error. The metadata pointer is legally null, so ``or``, not a ``.get``
    default. Widened to float64, the dtype ``data.qpos`` assignment wants.
    """
    qpos_path = episode_dir / (metadata.get("scene_qpos_file") or SCENE_QPOS_FILENAME)
    if not qpos_path.exists():
        return None
    scene_qpos = np.load(qpos_path)
    if scene_qpos.ndim != 2 or len(scene_qpos) != num_steps:
        raise ValueError(
            f"{episode_dir.name}: {qpos_path.name} has shape {scene_qpos.shape}, "
            f"expected ({num_steps}, nq) — one row per exported frame"
        )
    return np.asarray(scene_qpos, dtype=np.float64)


def discover_episodes(root: Path, limit: int | None = None) -> list[Path]:
    """Episode directories under ``root`` (a flat ``episode_*/`` pool), sorted by id."""
    if not root.is_dir():
        return []
    found = sorted(
        path.parent for path in root.glob("episode_*/states_actions.bin")
    )
    return found[:limit] if limit is not None else found
