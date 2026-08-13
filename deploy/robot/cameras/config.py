from dataclasses import dataclass


@dataclass
class CameraNodeConfig:
    name: str = "Camera"
    camera_serial: str = ""
    control_rate: float = 30.0
    height: int = 480
    width: int = 640
    rgb_socket: str | None = None
    device_id: int | None = None
