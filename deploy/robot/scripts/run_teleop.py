"""Bimanual GELLO teleop: leader + follower nodes for both arms."""

import os

from deploy.robot import launch
from deploy.robot.config import get_i2rt_config
from deploy.robot.specs import teleop_specs


def main() -> None:
    # Suppress leader/follower stdout (servo torque/offsets, motor_states)
    # unless verbose.
    quiet = not os.environ.get("DEPLOY_VERBOSE")
    config = get_i2rt_config()
    raise SystemExit(launch.launch(teleop_specs(config, quiet=quiet)))


if __name__ == "__main__":
    main()
