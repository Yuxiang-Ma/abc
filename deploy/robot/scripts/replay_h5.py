"""Replay a synchronized teleop H5 trajectory on the real robot."""

import argparse
import os
import time
from pathlib import Path

import h5py
import numpy as np

from deploy.robot import launch
from deploy.robot.config import get_i2rt_config
from deploy.robot.gym.yam_env import YAMEnv
from deploy.robot.launch import ProcessSpec
from deploy.robot.specs import follower_specs


def load_actions(path: str, expected_hz: float) -> np.ndarray:
    if expected_hz <= 0:
        raise ValueError("rate must be positive")
    with h5py.File(path, "r") as file:
        streams = [file["data/q_des_left"][:], file["data/q_des_right"][:]]
        timestamps = [
            np.asarray(file[f"timestamps/q_des_{side}"], dtype=np.int64)
            for side in ("left", "right")
        ]
    if streams[0].shape != streams[1].shape or streams[0].ndim != 2:
        raise ValueError(
            "left/right q_des streams must have matching [time, dof] shapes"
        )
    actions = np.concatenate(streams, axis=1).astype(np.float32)
    if len(actions) < 2 or actions.shape[1] != 14 or not np.isfinite(actions).all():
        raise ValueError(f"expected finite 14-DoF actions, got {actions.shape}")
    if any(len(ts) != len(actions) for ts in timestamps) or not np.array_equal(
        *timestamps
    ):
        raise ValueError("q_des streams do not have synchronized timestamps")
    duration = (timestamps[0][-1] - timestamps[0][0]) * 1e-9
    if duration <= 0:
        raise ValueError("timestamps must be increasing")
    actual_hz = (len(actions) - 1) / duration
    if abs(actual_hz - expected_hz) > 0.1:
        raise ValueError(f"recording is {actual_hz:.2f} Hz, not {expected_hz:.2f} Hz")
    steps = np.abs(np.diff(actions, axis=0))
    arm_steps = steps[:, [*range(6), *range(7, 13)]]
    if arm_steps.max() > 0.15 or steps[:, [6, 13]].max() > 0.35:
        raise ValueError("recording contains an unsafe per-step action jump")
    return actions


def replay(h5_path: str, rate_hz: float) -> None:
    actions = load_actions(h5_path, rate_hz)
    config = get_i2rt_config()
    config.cameras = {}
    env = YAMEnv(config)
    try:
        env.reset()
        env.move_to(actions[0])
        deadline = time.perf_counter()
        for index, action in enumerate(actions):
            env.step(action)
            deadline += 1.0 / rate_hz
            time.sleep(max(0.0, deadline - time.perf_counter()))
            if (index + 1) % 100 == 0:
                print(f"Replayed {index + 1}/{len(actions)} steps")
    finally:
        env.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("h5_path")
    parser.add_argument("--rate-hz", type=float, default=30.0)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    path = str(Path(args.h5_path).resolve())
    actions = load_actions(path, args.rate_hz)
    duration = (len(actions) - 1) / args.rate_hz
    print(
        f"Replay: {path}\nSteps: {len(actions)} @ {args.rate_hz:g} Hz ({duration:.2f}s)"
    )
    if args.dry_run:
        return 0
    if input("Type REPLAY to command the real robot: ").strip() != "REPLAY":
        print("Aborted.")
        return 0

    if args.verbose:
        os.environ["DEPLOY_VERBOSE"] = "1"
    profile = get_i2rt_config()
    specs = follower_specs(profile, quiet=not args.verbose)
    specs.append(
        ProcessSpec(
            "h5_replay",
            "deploy.robot.scripts.replay_h5:replay",
            {"h5_path": path, "rate_hz": args.rate_hz},
        )
    )
    return launch.launch(specs)


if __name__ == "__main__":
    raise SystemExit(main())
