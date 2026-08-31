"""Robot and camera configuration for each deployment station."""

import copy
import json
import os
from dataclasses import dataclass, field


@dataclass
class CameraConfig:
    serial: str
    height: int = 480
    width: int = 640
    fps: int = 30
    socket: str | None = None
    camera_type: str = "realsense"


@dataclass
class LeaderConfig:
    device_name: str
    control_rate: int = 400
    servo_ids: tuple[int, ...] = (40, 41, 42, 43, 44, 45, 46)
    joint_signs: tuple[int, ...] = (1, -1, -1, -1, 1, 1, 1)


@dataclass
class FollowerConfig:
    channel: str
    control_rate: int = 30
    gripper_type: str = "linear_4310"


@dataclass
class RobotConfig:
    leader: LeaderConfig
    follower: FollowerConfig
    root_pos: list[float]
    root_ori: list[float]
    init_q: list[float]


@dataclass
class PolicyConfig:
    # Match the policy and follower 30 Hz control rate.
    dt: float = 0.03333


@dataclass
class CanDevicesConfig:
    can_l_lead: str = ""
    can_l_foll: str = ""
    can_r_lead: str = ""
    can_r_foll: str = ""


@dataclass
class RobotSystemConfig:
    cameras: dict[str, CameraConfig]
    robots: dict[str, RobotConfig]
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    can_devices: CanDevicesConfig = field(default_factory=CanDevicesConfig)


def _profile(
    *,
    camera_serials: tuple[str, str, str],
    leader_devices: tuple[str, str],
    init_q: dict[str, list[float]],
    can_serials: tuple[str, str, str, str],
    gripper_type: str = "linear_4310",
    follower_control_rate: int = 30,
    leader_servo_ids: tuple[tuple[int, ...], tuple[int, ...]] = (
        (40, 41, 42, 43, 44, 45, 46),
        (40, 41, 42, 43, 44, 45, 46),
    ),
    leader_joint_signs: tuple[tuple[int, ...], tuple[int, ...]] = (
        (1, -1, -1, -1, 1, 1, 1),
        (1, -1, -1, -1, 1, 1, 1),
    ),
) -> RobotSystemConfig:
    """Build a two-arm, three-camera station from its hardware differences."""
    camera_names = ("top", "left", "right")
    cameras = {
        name: CameraConfig(serial=serial, socket=f"{name}_rgb")
        for name, serial in zip(camera_names, camera_serials, strict=True)
    }
    robots = {
        side: RobotConfig(
            leader=LeaderConfig(
                device_name=device,
                servo_ids=servo_ids,
                joint_signs=joint_signs,
            ),
            follower=FollowerConfig(
                channel=f"can_{side[0]}_foll",
                control_rate=follower_control_rate,
                gripper_type=gripper_type,
            ),
            root_pos=[0.0, 0.3 if side == "left" else -0.3, 0.0],
            root_ori=[1.0, 0.0, 0.0, 0.0],
            init_q=init_q[side].copy(),
        )
        for side, device, servo_ids, joint_signs in zip(
            ("left", "right"),
            leader_devices,
            leader_servo_ids,
            leader_joint_signs,
            strict=True,
        )
    }
    return RobotSystemConfig(
        cameras=cameras,
        robots=robots,
        policy=PolicyConfig(dt=0.03333),
        can_devices=CanDevicesConfig(*can_serials),
    )


_BBOX_INIT_Q = {
    "left": [
        -0.20656902,
        0.47283894,
        0.99431604,
        -0.7043946,
        -0.30842298,
        -0.32864118,
        0.9987507,
    ],
    "right": [
        0.20160982,
        0.39005876,
        1.1182956,
        -0.8726253,
        0.13332571,
        0.42629892,
        0.9895317,
    ],
}
_DEFAULT_INIT_Q = {
    "left": [0.0, 0.5, 1.0, -1.0, 0.0, 0.0, 1.0],
    "right": [0.0, 0.5, 1.0, -1.0, 0.0, 0.0, 1.0],
}
_BBOX_CAN = (
    "00370039594E501820313332",
    "005300195842500620383152",
    "00550034594E501820313332",
    "004B00555842500820333850",
)
_CBOX_CAN = (
    "0051005C594E501820313332",
    "0045002E594E501820313332",
    "00360066594E501820313332",
    "004100615548501220373234",
)


PROFILES = {
    "bbox_config": _profile(
        camera_serials=("335122272485", "352122272888", "218622274707"),
        leader_devices=("/dev/ttyUSB0", "/dev/ttyUSB1"),
        init_q=_BBOX_INIT_Q,
        can_serials=_BBOX_CAN,
    ),
    "universe_config": _profile(
        camera_serials=("335122272485", "352122272888", "218622274707"),
        leader_devices=("/dev/ttyUSB0", "/dev/ttyUSB1"),
        init_q=_BBOX_INIT_Q,
        can_serials=_BBOX_CAN,
        gripper_type="crank_4310",
    ),
    "cbox_config": _profile(
        camera_serials=("335122270655", "335122270354", "335122270952"),
        leader_devices=("/dev/ttyUSB1", "/dev/ttyUSB0"),
        init_q=_DEFAULT_INIT_Q,
        can_serials=_CBOX_CAN,
    ),
    "rbox_config": _profile(
        camera_serials=("335122270655", "335122270354", "335122270952"),
        leader_devices=("/dev/ttyUSB1", "/dev/ttyUSB0"),
        init_q=_DEFAULT_INIT_Q,
        can_serials=_CBOX_CAN,
        gripper_type="crank_4310",
    ),
    "yambox_config": _profile(
        camera_serials=("335122272966", "352122272888", "335122272540"),
        leader_devices=("/dev/ttyUSB1", "/dev/ttyUSB0"),
        init_q=_DEFAULT_INIT_Q,
        can_serials=(
            "",
            "0019004233354B0332333837",
            "",
            "0018004933354B0332333837",
        ),
    ),
    "gtwy_config": _profile(
        camera_serials=("335122270354", "335122272966", "352122272888"),
        follower_control_rate=60,
        leader_devices=(
            "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTAAMOEB-if00-port0",
            "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTAAMNKG-if00-port0",
        ),
        leader_servo_ids=(
            (20, 21, 22, 23, 24, 25, 26),
            (30, 31, 32, 33, 34, 35, 36),
        ),
        leader_joint_signs=(
            (1, -1, -1, 1, 1, 1, 1),
            (1, -1, -1, 1, 1, 1, 1),
        ),
        init_q=_DEFAULT_INIT_Q,
        can_serials=(
            "",
            "0045002E594E501820313332",
            "",
            "005300195842500620383152",
        ),
    ),
}

DEFAULT_PROFILE = "bbox_config"


def _load_profile() -> RobotSystemConfig:
    profile_name = os.environ.get("ROBOT_PROFILE", DEFAULT_PROFILE)
    try:
        profile = PROFILES[profile_name]
    except KeyError as error:
        available = ", ".join(PROFILES)
        raise ValueError(
            f"Robot profile '{profile_name}' not found. Available: {available}"
        ) from error
    print(f"[config] Using robot profile: {profile_name}")
    return copy.deepcopy(profile)


def get_i2rt_config() -> RobotSystemConfig:
    config = _load_profile()
    if value := os.environ.get("DEPLOY_INIT_Q"):
        init_q = json.loads(value)
        expected = sum(len(robot.init_q) for robot in config.robots.values())
        if len(init_q) != expected:
            raise ValueError(f"DEPLOY_INIT_Q must contain {expected} values")
        offset = 0
        for robot in config.robots.values():
            dof = len(robot.init_q)
            robot.init_q = init_q[offset : offset + dof]
            offset += dof
    return config
