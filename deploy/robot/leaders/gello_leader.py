"""Teleoperation leader backed by seven Dynamixel motors."""

import time
from pathlib import Path

import numpy as np

from deploy.robot.leaders.dynamixel import Dynamixel
from deploy.robot.leaders.leader_robot import Robot
from deploy.robot.node import Node


class GelloLeaderNode(Node):
    CALIBRATION_POSITION = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.357])
    LEADER_GRIPPER_RANGE = (-0.6040225, 0.33736414)
    FOLLOWER_GRIPPER_RANGE = (-0.1, 1.0)

    def __init__(
        self,
        name: str,
        control_rate: float,
        device_name: str,
        servo_ids: tuple[int, ...],
        joint_signs: tuple[int, ...],
        force_feedback: bool = True,
    ) -> None:
        super().__init__(name, control_rate)
        self.force_feedback = force_feedback
        self.device_name = device_name
        self.joint_signs = np.asarray(joint_signs)
        self.leader_topic_name = f"{name}_actions"
        self.create_publisher(self.leader_topic_name)

        connection = Dynamixel.Config(
            baudrate=4_000_000, device_name=device_name
        ).instantiate()
        self.leader = Robot(connection, servo_ids=servo_ids)

        self.joint_min = np.deg2rad([-120.0, 0.0, 0.0, -75.0, -85.0, -115.0])
        self.joint_max = np.deg2rad([120.0, 180.0, 180.0, 75.0, 85.0, 115.0])
        joint_range = self.joint_max - self.joint_min
        self.joint_min += 0.01 * joint_range
        self.joint_max -= 0.01 * joint_range
        self.joint_max[2] -= 0.3 * joint_range[2]

        self._previous_gripper = None
        self._previous_time = None
        self._verify_latency_timer()
        self.joint_offsets = self._calibrate_offsets()

    def _verify_latency_timer(self) -> None:
        # Profiles use stable /dev/serial/by-id paths. Resolve the symlink to
        # the ttyUSB device name expected by sysfs.
        device = Path(self.device_name).resolve().name
        path = Path("/sys/bus/usb-serial/devices") / device / "latency_timer"
        if int(path.read_text().strip()) != 1:
            raise RuntimeError(f"Set {path} to 1 before starting teleoperation")

    def _read_raw_position(self) -> np.ndarray:
        positions, _ = self.leader.read_position()
        return (np.asarray(positions) / 2048 - 1) * np.pi

    def _calibrate_offsets(self) -> np.ndarray:
        """Choose the nearest equivalent calibration pose for every encoder."""
        for _ in range(10):
            self._read_raw_position()
        raw = self._read_raw_position()
        candidates = np.arange(-80, 81) * (np.pi / 4)
        offsets = []
        for index, value in enumerate(raw):
            positions = self.joint_signs[index] * (value - candidates)
            error = np.abs(positions - self.CALIBRATION_POSITION[index])
            offsets.append(candidates[np.argmin(error)])
        offsets = np.asarray(offsets)
        print("Leader offsets:", [f"{value:.3f}" for value in offsets])
        return offsets

    def read_position(self) -> np.ndarray:
        position = self.joint_signs * (self._read_raw_position() - self.joint_offsets)
        leader_min, leader_max = self.LEADER_GRIPPER_RANGE
        follower_min, follower_max = self.FOLLOWER_GRIPPER_RANGE
        position[-1] = np.interp(
            position[-1],
            (leader_min, leader_max),
            (follower_min, follower_max),
        )
        return position

    def _gripper_torque(self, position: float) -> float:
        current = min(0.03, 0.05 * (1.5 - position))
        now = time.perf_counter()
        if self._previous_time is not None:
            velocity = (position - self._previous_gripper) / (now - self._previous_time)
            current -= 0.005 * velocity
        self._previous_gripper = position
        self._previous_time = now
        return current

    def _apply_force_feedback(self, position: np.ndarray) -> None:
        excess = np.maximum(position[:-1] - self.joint_max, 0)
        excess += np.minimum(position[:-1] - self.joint_min, 0)
        torque = np.zeros(7)
        torque[:-1] = -0.3 * excess * self.joint_signs[:-1]
        torque[-1] = self._gripper_torque(position[-1])
        self.leader.set_current((torque * 1158.73).astype(int))

    def initial_bootup(self) -> None:
        self.publish(
            self.leader_topic_name,
            self.read_position(),
            extras={"type": "interp"},
        )

    def tick(self) -> None:
        position = self.read_position()
        self.publish(self.leader_topic_name, position, extras={"type": "servo"})
        if self.force_feedback:
            self._apply_force_feedback(position)

    def on_shutdown(self) -> None:
        try:
            self.leader.close()
        except (ConnectionError, OSError):
            pass


def run(
    name: str,
    control_rate: float,
    device_name: str,
    servo_ids: tuple[int, ...],
    joint_signs: tuple[int, ...],
    force_feedback: bool = True,
) -> None:
    GelloLeaderNode(
        name, control_rate, device_name, servo_ids, joint_signs, force_feedback
    ).run()
