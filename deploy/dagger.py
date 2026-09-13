"""Launch passive-GELLO, end-effector-relative DAgger collection."""

import tyro

from deploy.dagger_config import DaggerConfig
from deploy.deploy_policy import (
    _configure_process,
    _prepare_task,
    _recorder_spec,
    _server_specs,
)
from deploy.robot import launch
from deploy.robot.config import get_i2rt_config
from deploy.robot.key_listeners.key_listener_config import KeyListenerConfig
from deploy.robot.key_listeners.pedal import resolve_foot_pedal_device
from deploy.robot.launch import ProcessSpec
from deploy.robot.specs import camera_specs, follower_specs


def _gello_spec(cfg: DaggerConfig, profile, side: str) -> ProcessSpec:
    leader = profile.robots[side].leader
    return ProcessSpec(
        f"gello_{side}",
        "deploy.robot.leaders.gello_leader",
        {
            "name": f"dagger_gello_{side}",
            "control_rate": leader.control_rate,
            "device_name": leader.device_name,
            "servo_ids": leader.servo_ids,
            "joint_signs": leader.joint_signs,
            "force_feedback": False,
        },
        quiet=not cfg.verbose,
    )


def _build_specs(cfg: DaggerConfig) -> list[ProcessSpec]:
    task_name, session_tag = _prepare_task(cfg)
    profile = get_i2rt_config()
    specs = _server_specs(cfg)
    if not cfg.debug:
        specs.extend(follower_specs(profile, quiet=not cfg.verbose))
    specs.extend(camera_specs(profile))
    specs.extend(
        [_gello_spec(cfg, profile, "left"), _gello_spec(cfg, profile, "right")]
    )
    specs.extend(
        [
            ProcessSpec(
                "dagger_rollout",
                "deploy.robot.gym.dagger_rollout",
                {"args": cfg.dagger_rollout_config()},
                terminal_input=True,
            ),
            ProcessSpec(
                "key_listener",
                "deploy.robot.key_listeners.key_listener",
                {
                    "cfg": KeyListenerConfig(
                        name="KeyListener",
                        control_rate=120,
                        cooldown_s=0.05,
                        input_device=resolve_foot_pedal_device(
                            cfg.foot_pedal_device, tool="dagger"
                        ),
                        input_keys="a,b,c",
                        grab_device=cfg.grab_foot_pedal,
                    )
                },
                # Attach the terminal so keyboard a/b/c work alongside the
                # pedal (and 'a' can confirm the first-action safety check).
                terminal_input=True,
            ),
            _recorder_spec(
                cfg,
                task_name=task_name,
                session_tag=session_tag,
                dagger=True,
            ),
        ]
    )
    return specs


def main(cfg: DaggerConfig) -> int:
    cfg.record = True
    _configure_process(cfg)
    return launch.launch(_build_specs(cfg))


if __name__ == "__main__":
    raise SystemExit(main(tyro.cli(DaggerConfig)))
