"""Shared plumbing for recorder nodes.

Hosts the camera watchdog, sensor subscriptions, and H5 writer loop shared by
the teleop and inference recorders.
"""

import os
import queue
import threading
import time
from datetime import datetime, timezone

import numpy as np

from deploy.robot.node import Node
from deploy.robot.tasks import generate_recording_id, recording_filename


class RecorderBase(Node):
    """Common camera and H5-writing behavior for recorder nodes."""

    recording_type = "teleop"

    def __init__(self, name: str, control_rate: float, **node_kwargs):
        super().__init__(name, control_rate, **node_kwargs)
        self._camera_last_seen: dict[str, float] = {}
        self._camera_timeout = 2.0
        self.worker_queue = queue.Queue(maxsize=512)
        self.writer_thread: threading.Thread | None = None
        self.record_data = False
        self.recording_id = None
        self.current_file_name = None

    # ------------------------------------------------------------------
    # Sensor topics (cameras + followers)
    # ------------------------------------------------------------------

    def _init_sensor_topics(self) -> None:
        """Set up camera/follower topic maps and subscribe to them + pedals."""
        self.key_press_topic = "KeyListener_key_presses"
        self.sensor_topics = {
            self.config.cameras[name].socket: ("rgb", name)
            for name in self.camera_names
        }
        self.sensor_topics.update(
            {f"follower_{name}_obs": ("follower", name) for name in self.config.robots}
        )

        self.create_subscriber(self.key_press_topic)
        for topic in self.sensor_topics:
            self.create_subscriber(topic, conflate=1)

    def _poll_sensor_topics(self) -> None:
        """Drain camera/follower topics into the writer queue when recording."""
        for topic, (data_type, name) in self.sensor_topics.items():
            message = self.subscribe(topic, block=False)
            if message[0] is None:
                continue
            timestamp = message[1]["timestamp"] * 1e9
            if data_type == "rgb":
                self._camera_last_seen[name] = time.monotonic()
            if self.record_data:
                self.worker_queue.put((data_type, name, message[0], timestamp))
            else:
                self._handle_unrecorded_sensor_item(
                    data_type, name, message[0], timestamp
                )

    def _handle_unrecorded_sensor_item(
        self, data_type, name, payload, timestamp
    ) -> None:
        """Hook: called for sensor items that arrive while not recording."""

    # ------------------------------------------------------------------
    # H5 writer loop (runs on the writer thread)
    # ------------------------------------------------------------------

    def _write_extra_attrs(self, f) -> None:
        """Hook: write recorder-specific root attributes."""

    def _handle_queue_item(self, f, data_group, timestamps_group, data) -> bool:
        """Hook: handle a recorder-specific queue item. Return True if handled."""
        return False

    @staticmethod
    def _append_stream(
        data_group, timestamps_group, name, value, timestamp, dtype, width=None
    ) -> None:
        if name not in data_group:
            shape = (0,) if width is None else (0, width)
            maxshape = (None,) if width is None else (None, width)
            data_group.create_dataset(
                name, shape, maxshape=maxshape, dtype=dtype, chunks=True
            )
            timestamps_group.create_dataset(
                name, (0,), maxshape=(None,), dtype="uint64", chunks=True
            )
        dataset = data_group[name]
        timestamps = timestamps_group[name]
        dataset.resize(dataset.shape[0] + 1, axis=0)
        timestamps.resize(timestamps.shape[0] + 1, axis=0)
        dataset[-1] = value
        timestamps[-1] = timestamp

    def _write_data(self) -> None:
        import cv2
        import h5py

        _verbose = os.environ.get("DEPLOY_VERBOSE")
        if _verbose:
            print(f"Writer thread started for {self.current_file_name}")
        with h5py.File(self.current_file_name, "w") as f:
            f.attrs["recording_id"] = self.recording_id
            if self.task_name:
                f.attrs["task_name"] = self.task_name
            f.attrs["collection_name"] = self.collection_name
            f.attrs["recording_type"] = self.recording_type
            now = datetime.now(timezone.utc).astimezone()
            f.attrs["start_timestamp"] = now.timestamp()
            f.attrs["start_time"] = now.strftime("%b %d, %H:%M")
            self._write_extra_attrs(f)

            data_group = f.create_group("data")
            timestamps_group = f.create_group("timestamps")
            step = 0

            while True:
                data = self.worker_queue.get()

                step += 1

                if data[0] == "shutdown":
                    if _verbose:
                        print("Writer thread received shutdown signal")
                    break

                if self._handle_queue_item(f, data_group, timestamps_group, data):
                    pass
                elif data[0] == "rgb":
                    _, name, image, timestamp = data
                    # Camera publishes RGB; cv2.imencode expects BGR
                    image_bgr = image[:, :, ::-1]
                    _, jpeg = cv2.imencode(
                        ".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90]
                    )

                    self._append_stream(
                        data_group,
                        timestamps_group,
                        name,
                        jpeg,
                        timestamp,
                        h5py.vlen_dtype(np.uint8),
                    )
                else:
                    # Layout from yam_follower: joint_pos(6) + gripper_pos(1)
                    # + joint_vel(7) + joint_eff(7) + command(7) = 28
                    _, name, robot_obs, timestamp = data
                    topics = {
                        f"q_{name}": (robot_obs[:6], 6, "float32"),
                        f"q_gripper_{name}": (robot_obs[6:7], 1, "float32"),
                        f"q_vel_{name}": (robot_obs[7:14], 7, "float32"),
                        f"q_eff_{name}": (robot_obs[14:21], 7, "float32"),
                        f"q_des_{name}": (robot_obs[21:28], 7, "float32"),
                    }

                    for topic_name, (data_buf, count, dtype) in topics.items():
                        self._append_stream(
                            data_group,
                            timestamps_group,
                            topic_name,
                            data_buf,
                            timestamp,
                            dtype,
                            count,
                        )

                if step % 500 == 0:
                    f.flush()
            if _verbose:
                print(f"Flushing remaining data to {self.current_file_name}...")
            f.flush()

    def _new_recording_file(self) -> None:
        self.recording_id = generate_recording_id()
        self.current_file_name = os.path.join(
            self.data_root_directory,
            recording_filename(
                self.recording_id, self.collection_name, self.session_tag
            ),
        )

    def _start_writer(self, buffered_items=()) -> None:
        if self.writer_thread and self.writer_thread.is_alive():
            if not self._stop_writer(timeout=5):
                raise RuntimeError("previous H5 writer did not stop")
        while True:
            try:
                self.worker_queue.get_nowait()
            except queue.Empty:
                break
        for item in buffered_items:
            self.worker_queue.put(item)
        self.writer_thread = threading.Thread(target=self._write_data, daemon=False)
        self.writer_thread.start()

    @staticmethod
    def _key_press(message) -> str | None:
        """Return the key from a key-topic message if it is a press, else None.

        The KeyListener publishes both press and release edges for evdev
        pedals; only key-down may trigger recorder actions (a pedal release
        must not look like a fresh keypress)."""
        if message[0] is None:
            return None
        extras = message[1]
        if not extras.get("pressed", True):
            return None
        return extras.get("key")

    def _spawn_post_video(self) -> None:
        """Kick off detached MP4 rendering of the file just saved."""
        if not self.current_file_name or not os.path.exists(self.current_file_name):
            return
        from deploy.recording.postprocess import spawn_detached

        spawn_detached(self.current_file_name)

    def _stop_writer(self, timeout: float) -> bool:
        if not self.writer_thread or not self.writer_thread.is_alive():
            return True
        self.worker_queue.put(("shutdown", None, None))
        self.writer_thread.join(timeout=timeout)
        return not self.writer_thread.is_alive()

    def _get_cameras_alive(self) -> set[str]:
        """Return cameras that sent a frame within the timeout window."""
        now = time.monotonic()
        return {
            name
            for name, ts in self._camera_last_seen.items()
            if now - ts < self._camera_timeout
        }

    def _cameras_ready_or_defer(self) -> bool:
        """Refuse to start a recording until every camera is live."""
        cameras_alive = self._get_cameras_alive()
        dead_cameras = set(self.camera_names) - cameras_alive
        if not dead_cameras:
            return True
        print(
            f"Recording not started; cameras have no recent frames: {sorted(dead_cameras)}"
        )
        return False
