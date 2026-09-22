from dataclasses import dataclass
from typing import Literal

GripperType = Literal["crank_4310", "linear_3507", "linear_4310", "flexible_4310", "no_gripper"]


@dataclass
class YamFollowerConfig:
    name: str = "Follower"
    control_rate: float = 200.0
    channel: str = ""
    leader_name: str = ""
    gripper_type: GripperType = "linear_4310"
