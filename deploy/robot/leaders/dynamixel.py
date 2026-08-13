"""Small Dynamixel connection wrapper used by the teleoperation leader."""

import os
from dataclasses import dataclass

from dynamixel_sdk import COMM_SUCCESS, PacketHandler, PortHandler


class Dynamixel:
    TORQUE_ENABLE_ADDRESS = 64
    OPERATING_MODE_ADDRESS = 11

    @dataclass
    class Config:
        baudrate: int = 57_600
        protocol_version: float = 2.0
        device_name: str = ""

        def instantiate(self) -> "Dynamixel":
            return Dynamixel(self)

    def __init__(self, config: Config) -> None:
        if not config.device_name:
            ports = [
                name
                for name in os.listdir("/dev")
                if "ttyUSB" in name or "ttyACM" in name
            ]
            if not ports:
                raise OSError("No Dynamixel serial device found")
            config.device_name = f"/dev/{ports[0]}"

        self.config = config
        self.portHandler = PortHandler(config.device_name)
        self.packetHandler = PacketHandler(config.protocol_version)
        if not self.portHandler.openPort():
            raise ConnectionError(f"Failed to open {config.device_name}")
        if not self.portHandler.setBaudRate(config.baudrate):
            raise ConnectionError(f"Failed to set baudrate to {config.baudrate}")

    def _check(self, comm_result: int, error: int, motor_id: int) -> None:
        if comm_result != COMM_SUCCESS:
            message = self.packetHandler.getTxRxResult(comm_result)
            raise ConnectionError(f"Dynamixel {motor_id}: {message}")
        # Error 128 is the non-fatal voltage warning handled by the old driver.
        if error not in (0, 128):
            raise ConnectionError(f"Dynamixel {motor_id}: error {error}")

    def _set_torque(self, motor_id: int, enabled: bool) -> None:
        result, error = self.packetHandler.write1ByteTxRx(
            self.portHandler,
            motor_id,
            self.TORQUE_ENABLE_ADDRESS,
            int(enabled),
        )
        self._check(result, error, motor_id)

    def _enable_torque(self, motor_id: int) -> None:
        self._set_torque(motor_id, True)

    def _disable_torque(self, motor_id: int) -> None:
        self._set_torque(motor_id, False)

    def set_current_mode(self, motor_id: int) -> None:
        self._disable_torque(motor_id)
        result, error = self.packetHandler.write1ByteTxRx(
            self.portHandler, motor_id, self.OPERATING_MODE_ADDRESS, 0
        )
        self._check(result, error, motor_id)
        self._enable_torque(motor_id)

    def disconnect(self) -> None:
        self.portHandler.closePort()
