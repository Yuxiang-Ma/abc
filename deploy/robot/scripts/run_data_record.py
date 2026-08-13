"""Teleop data collection: cameras + teleop arms + recorder + pedal listener."""

import argparse
import os

from deploy.robot import launch
from deploy.robot.config import get_i2rt_config
from deploy.robot.key_listeners.key_listener_config import KeyListenerConfig
from deploy.robot.key_listeners.pedal import resolve_foot_pedal_device
from deploy.robot.launch import ProcessSpec
from deploy.robot.specs import camera_specs, teleop_specs
from deploy.robot.tasks import (
    get_data_dir,
    prompt_session_tag,
    select_task,
    task_to_collection_name,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root_directory", type=str, default=None)
    parser.add_argument("--collection_name", type=str, default=None)
    parser.add_argument(
        "--foot_pedal_device",
        type=str,
        default=None,
        help="Pedal evdev path (default: FOOT_PEDAL_INPUT_DEVICE or the "
        "PCsensor device when plugged in; pass '' for keyboard-only)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print all subprocess output (motor states, camera serials, etc.)",
    )
    args = parser.parse_args()

    # Propagate verbosity to all node processes via inherited environment.
    if args.verbose:
        os.environ["DEPLOY_VERBOSE"] = "1"
    else:
        os.environ.pop("DEPLOY_VERBOSE", None)

    # Interactive task selection (unless collection_name already provided)
    if args.collection_name:
        collection_name = args.collection_name
        task_name = collection_name.replace("_", " ")
    else:
        task_name = select_task()
        collection_name = task_to_collection_name(task_name)

    # Optional session tag (e.g. "cage_a" → filenames get "0225_cage_a_" prefix)
    session_tag = prompt_session_tag()

    data_root_directory = get_data_dir("teleop_h5", args.data_root_directory)

    config = get_i2rt_config()
    specs = [
        *camera_specs(config),
        ProcessSpec(
            "recorder",
            "deploy.robot.recorders.recorder",
            {
                "name": "Recorder",
                "control_rate": 1000,
                "data_root_directory": data_root_directory,
                "collection_name": collection_name,
                "task_name": task_name,
                "session_tag": session_tag,
            },
        ),
        *teleop_specs(config, quiet=not args.verbose),
        ProcessSpec(
            "key_listener",
            "deploy.robot.key_listeners.key_listener",
            {
                "cfg": KeyListenerConfig(
                    name="KeyListener",
                    control_rate=60,
                    input_device=resolve_foot_pedal_device(
                        args.foot_pedal_device, tool="record"
                    ),
                )
            },
            terminal_input=True,
        ),
    ]
    raise SystemExit(launch.launch(specs))


if __name__ == "__main__":
    main()
