"""Followers only (no GELLO leaders, no cameras): boots both YAM arms, holds, publishes state.

    ROBOT_PROFILE=react_yam_config uv run deploy/robot/scripts/run_followers.py

The follower node torques the motors, runs the small gripper calibration, then waits for a
command and HOLDS; it does not move until something publishes on leader_<side>_actions.
Ctrl-C runs the node's shutdown, which moves the joints to zero over 2 s and powers the
motors off -- keep the workspace clear when stopping.
"""
import os

from deploy.robot import launch
from deploy.robot.config import get_i2rt_config
from deploy.robot.specs import follower_specs


def main() -> None:
    quiet = not os.environ.get("DEPLOY_VERBOSE")
    raise SystemExit(launch.launch(follower_specs(get_i2rt_config(), quiet=quiet)))


if __name__ == "__main__":
    main()
