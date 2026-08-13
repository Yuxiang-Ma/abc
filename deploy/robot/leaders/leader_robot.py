"""Synchronous position reader and current writer for a Gello leader arm."""

import time

import numpy as np
from dynamixel_sdk import DXL_HIBYTE, DXL_LOBYTE, GroupSyncRead, GroupSyncWrite


class Robot:
    POSITION_ADDRESS = 132
    GOAL_CURRENT_ADDRESS = 102

    def __init__(self, dynamixel, servo_ids: list[int]) -> None:
        self.dynamixel = dynamixel
        self.servo_ids = servo_ids
        self.position_reader = GroupSyncRead(
            dynamixel.portHandler,
            dynamixel.packetHandler,
            self.POSITION_ADDRESS,
            4,
        )
        for motor_id in servo_ids:
            if not self.position_reader.addParam(motor_id):
                raise RuntimeError(f"Failed to register Dynamixel {motor_id}")

        self.current_writer = GroupSyncWrite(
            dynamixel.portHandler,
            dynamixel.packetHandler,
            self.GOAL_CURRENT_ADDRESS,
            2,
        )
        self._current_mode_enabled = False
        self._disable_torque()

    def read_position(self, tries: int = 2) -> tuple[list[int], float]:
        start = time.perf_counter()
        for attempt in range(tries + 1):
            result = self.position_reader.txRxPacket()
            if result == 0:
                break
            if attempt == tries:
                raise ConnectionError("Failed to read leader joint positions")

        positions = []
        for motor_id in self.servo_ids:
            value = self.position_reader.getData(motor_id, self.POSITION_ADDRESS, 4)
            if value > 2**31:
                value -= 2**32
            positions.append(value)
        return positions, time.perf_counter() - start

    def set_current(self, currents) -> None:
        if not self._current_mode_enabled:
            for motor_id in self.servo_ids:
                self.dynamixel.set_current_mode(motor_id)
            self._current_mode_enabled = True

        for motor_id, current in zip(self.servo_ids, currents, strict=True):
            current = int(np.clip(current, -900, 900))
            payload = [DXL_LOBYTE(current), DXL_HIBYTE(current)]
            if not self.current_writer.addParam(motor_id, payload):
                raise RuntimeError(f"Failed to command Dynamixel {motor_id}")
            self.current_writer.txPacket()
            self.current_writer.clearParam()

    def _disable_torque(self) -> None:
        for motor_id in self.servo_ids:
            self.dynamixel._disable_torque(motor_id)

    def close(self) -> None:
        try:
            self._disable_torque()
        finally:
            self.dynamixel.disconnect()
