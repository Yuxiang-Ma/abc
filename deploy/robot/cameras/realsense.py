import os

import numpy as np
import tyro

from deploy.robot.cameras.config import CameraNodeConfig
from deploy.robot.node import Node


class RealsenseNode(Node):
    def __init__(
        self,
        name: str,
        control_rate: float,
        camera_serial: str,
        height: int,
        width: int,
        rgb_socket: str = None,
    ):
        super().__init__(name, control_rate)

        self.camera_serial = camera_serial
        self.height = height
        self.width = width
        self.pipeline = None
        self._pipeline_started = False
        if rgb_socket is not None:
            self.rgb_topic_name = rgb_socket
        else:
            self.rgb_topic_name = f"{self._name}_rgb"
        self.create_publisher(self.rgb_topic_name)

    def init_cam(self):
        import pyrealsense2 as rs

        self.rs_context = rs.context()
        devices = self.rs_context.query_devices()
        if not devices:
            raise RuntimeError("No RealSense cameras found")

        self.serial_to_device = {
            device.get_info(rs.camera_info.serial_number): device for device in devices
        }
        if self.camera_serial in self.serial_to_device:
            if os.environ.get("DEPLOY_VERBOSE"):
                print(
                    f"Found real sense camera with serial number {self.camera_serial}"
                )
            self.pipeline = rs.pipeline()
            config = rs.config()
            config.enable_device(self.camera_serial)
            config.enable_stream(
                rs.stream.color,
                self.width,
                self.height,
                rs.format.rgb8,
                int(self._control_rate),
            )
            self.pipeline.start(config)
            self._pipeline_started = True
        else:
            raise RuntimeError(
                f"Cannot find RealSense camera with serial {self.camera_serial}"
            )

    def initial_bootup(self) -> None:
        self.init_cam()
        if os.environ.get("DEPLOY_VERBOSE"):
            print(f"[{self._name}] Realsense node initial bootup complete.")

    def tick(self) -> None:
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            return
        color_image = np.asanyarray(color_frame.get_data())
        self.publish(self.rgb_topic_name, color_image)

    def on_shutdown(self) -> None:
        if self.pipeline is not None and self._pipeline_started:
            self.pipeline.stop()
            self._pipeline_started = False
        if os.environ.get("DEPLOY_VERBOSE"):
            print(f"[{self._name}] Realsense node shutdown complete.")


def run(cfg: CameraNodeConfig) -> None:
    RealsenseNode(
        name=cfg.name,
        control_rate=cfg.control_rate,
        camera_serial=cfg.camera_serial,
        height=cfg.height,
        width=cfg.width,
        rgb_socket=cfg.rgb_socket,
    ).run()


if __name__ == "__main__":
    run(tyro.cli(CameraNodeConfig))
