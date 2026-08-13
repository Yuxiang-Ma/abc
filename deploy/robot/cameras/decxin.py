import os
import subprocess

import tyro

from deploy.robot.cameras.config import CameraNodeConfig
from deploy.robot.node import Node


def find_device_by_serial(serial: str) -> int | None:
    """Find /dev/videoN index matching a USB camera serial number."""
    for i in range(20):
        dev = f"/dev/video{i}"
        try:
            result = subprocess.run(
                ["udevadm", "info", "--query=property", f"--name={dev}"],
                capture_output=True,
                text=True,
                timeout=2,
            )
            props = dict(
                line.split("=", 1) for line in result.stdout.splitlines() if "=" in line
            )
            if props.get("ID_SERIAL_SHORT") == serial:
                if (
                    props.get("ID_V4L_CAPABILITIES", "").find(":capture:") >= 0
                    or i % 2 == 0
                ):
                    return i
        except (subprocess.TimeoutExpired, FileNotFoundError):
            continue
    return None


class DecxinNode(Node):
    def __init__(
        self,
        name: str,
        control_rate: float,
        device_id: int | None = None,
        camera_serial: str | None = None,
        height: int = 480,
        width: int = 640,
        rgb_socket: str | None = None,
    ):
        super().__init__(name, control_rate)

        self.device_id = device_id
        self.camera_serial = camera_serial
        self.height = height
        self.width = width

        if rgb_socket is not None:
            self.rgb_topic_name = rgb_socket
        else:
            self.rgb_topic_name = f"{self._name}_rgb"
        self.create_publisher(self.rgb_topic_name)

    def init_cam(self):
        import cv2

        dev_id = self.device_id
        if dev_id is None and self.camera_serial:
            dev_id = find_device_by_serial(self.camera_serial)
            if dev_id is None:
                raise RuntimeError(
                    f"Cannot find Decxin camera with serial {self.camera_serial}"
                )
        if dev_id is None:
            raise RuntimeError("Must provide either --device_id or --camera_serial")

        self.cap = cv2.VideoCapture(dev_id, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open /dev/video{dev_id}")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self._control_rate)

        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        if os.environ.get("DEPLOY_VERBOSE"):
            print(
                f"[{self._name}] Decxin camera opened: /dev/video{dev_id} "
                f"{actual_w}x{actual_h} @ {actual_fps}fps"
            )

    def initial_bootup(self) -> None:
        self.init_cam()
        if os.environ.get("DEPLOY_VERBOSE"):
            print(f"[{self._name}] Decxin node initial bootup complete.")

    def tick(self) -> None:
        import cv2

        ret, frame = self.cap.read()
        if not ret:
            return
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.publish(self.rgb_topic_name, rgb_frame)

    def on_shutdown(self) -> None:
        self.cap.release()
        if os.environ.get("DEPLOY_VERBOSE"):
            print(f"[{self._name}] Decxin node shutdown complete.")


def run(cfg: CameraNodeConfig) -> None:
    DecxinNode(
        name=cfg.name,
        control_rate=cfg.control_rate,
        device_id=cfg.device_id,
        camera_serial=cfg.camera_serial,
        height=cfg.height,
        width=cfg.width,
        rgb_socket=cfg.rgb_socket,
    ).run()


if __name__ == "__main__":
    run(tyro.cli(CameraNodeConfig))
