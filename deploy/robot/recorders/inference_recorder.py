"""Recorder that captures sensor streams AND policy inference events.

Drop-in replacement for recorder.py that additionally subscribes to
the ``inference_events`` ZMQ topic published by ``policy_rollout.py``
and writes inference metadata into an ``inference/`` group in the H5 file.
"""

from __future__ import annotations

import os
import signal
from collections import deque
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import numpy as np
import tyro

if TYPE_CHECKING:
    import h5py

import deploy.robot.config as config_manager
from deploy.robot.config import RobotSystemConfig
from deploy.robot.recorders.base import RecorderBase
from deploy.robot.recorders.inference_recorder_config import InferenceRecorderConfig


class InferenceRecorderNode(RecorderBase):
    """Records camera frames, follower states, and inference events."""

    def __init__(
        self,
        name: str,
        control_rate: float,
        data_root_directory: str,
        config: RobotSystemConfig,
        collection_name: str,
        *,
        checkpoint_path: str = "",
        model_size: str = "",
        diffusion_steps: int = 10,
        rtc: bool = False,
        rtc_prefix_length: int = 0,
        rtc_inference_lead_steps: int = 0,
        task_name: str = "",
        session_tag: str = "",
        dagger: bool = False,
    ):
        super().__init__(name, control_rate, verbose=False)
        self.data_root_directory = data_root_directory
        self.config = config
        self.collection_name = collection_name
        self.session_tag = session_tag
        self.task_name = task_name
        self.camera_names = list(self.config.cameras.keys())

        # Policy metadata (written as H5 root attributes)
        self.checkpoint_path = checkpoint_path
        self.model_size = model_size
        self.diffusion_steps = diffusion_steps
        self.rtc = rtc
        self.rtc_prefix_length = rtc_prefix_length
        self.rtc_inference_lead_steps = rtc_inference_lead_steps
        self.dagger = dagger
        self.recording_type = "dagger" if dagger else "inference"
        self.control_topic = "inference_control"
        if not dagger:
            self.create_publisher(self.control_topic)

        self._init_sensor_topics()
        # subscribe for inference events (no conflate — every event matters)
        self.inference_topic = "inference_events"
        self.create_subscriber(self.inference_topic)
        if dagger:
            self.dagger_control_topic = "dagger_recorder_control"
            self.dagger_event_topic = "dagger_events"
            self.create_subscriber(self.dagger_control_topic)
            self.create_subscriber(self.dagger_event_topic)
        # Pre-record buffer: camera frames captured before recording starts.
        # Flushed to the writer queue when recording begins so no frames are lost.
        self._pre_record_buffer = deque(maxlen=300)

        self._inference_ready = False
        self._pending_dagger_start = False
        self._finish_pending = False

    def initial_bootup(self) -> None:
        # create data directory
        os.makedirs(self.data_root_directory, exist_ok=True)

        if os.environ.get("DEPLOY_VERBOSE"):
            print(f"[{self._name}] Inference recorder node initial bootup complete.")
        self._waiting_for_inference = True
        if self.dagger:
            print("Waiting for DAgger controller. A toggles | C checkpoints | B ends")
        else:
            print("Waiting for inference. [a/b] start or stop+home | [c/j] shutdown")

    def tick(self) -> None:
        if self.dagger:
            self._poll_dagger_control()
        else:
            message = self.subscribe(self.key_press_topic, block=False)
            key_pressed = self._key_press(message)
            if key_pressed is not None:
                if key_pressed in ["a", "b", "x"]:
                    self._handle_start_stop_key()
                elif key_pressed in ["c", "j"]:
                    self._handle_shutdown_key()

        self._poll_sensor_topics()
        if self._pending_dagger_start and self._cameras_ready_or_defer():
            self._pending_dagger_start = False
            self._start_recording()

        # Poll inference events
        message = self.subscribe(self.inference_topic, block=False)
        if message[0] is not None:
            extras = message[1]
            # Handle ready signal from inference server (not an inference event)
            if extras.get("event") == "ready":
                if not self._inference_ready:
                    self._inference_ready = True
                    print("Inference server ready — key input now accepted.")
            elif (
                self._waiting_for_inference
                and not self.record_data
                and not self._finish_pending
            ):
                self._waiting_for_inference = False
                self._start_recording()
            if self.record_data and extras.get("event") != "ready":
                action_array = message[0]
                extras = message[1]
                timestamp = extras["timestamp"] * 1e9
                self.worker_queue.put(
                    ("inference", None, action_array, timestamp, extras)
                )

        if self.dagger:
            message = self.subscribe(self.dagger_event_topic, block=False)
            if message[0] is not None and self.record_data:
                timestamp = message[1]["timestamp"] * 1e9
                self.worker_queue.put(("dagger", None, None, timestamp, message[1]))

    def on_shutdown(self) -> None:
        if self.record_data and self.writer_thread:
            print("Finishing current recording...")
            if not self._stop_writer(timeout=30):
                print("Warning: Writer thread did not finish within timeout")
            else:
                print("Writer thread finished successfully")
                self._spawn_post_video()
        if os.environ.get("DEPLOY_VERBOSE"):
            print(f"[{self._name}] Inference recorder node shutdown complete.")

    def _start_recording(self) -> None:
        """Start a new recording session."""
        if self.record_data:
            return  # already recording

        # Check camera liveness and restart dead cameras
        if not self._cameras_ready_or_defer():
            return

        self.record_data = True
        self._new_recording_file()

        print(f"\nRECORDING STARTED - Saving to: {self.current_file_name}")

        # Flush pre-record buffer (camera frames captured before recording)
        n_buffered = len(self._pre_record_buffer)
        self._start_writer(self._pre_record_buffer)
        self._pre_record_buffer.clear()
        if n_buffered > 0 and os.environ.get("DEPLOY_VERBOSE"):
            print(
                f"  Flushed {n_buffered} pre-buffered camera frames (captured before first inference)"
            )

    def _stop_recording(self) -> None:
        """Stop the current recording and save."""
        if not self.record_data:
            return  # not recording

        self.record_data = False
        print(f"\nRECORDING STOPPED - File saved: {self.current_file_name}")

        self._stop_writer(timeout=10)
        if not self.dagger:
            # DAgger renders after the save/discard decision instead, so the
            # video reflects any discard segments written to the file.
            self._spawn_post_video()

    def _poll_dagger_control(self) -> None:
        message = self.subscribe(self.dagger_control_topic, block=False)
        if message[0] is None:
            return
        state, extras = message
        command = extras.get("command")
        timestamp_ns = int(extras["timestamp"] * 1e9)
        if command == "start":
            self._finish_pending = False
            self._waiting_for_inference = False
            self._pending_dagger_start = True
            return
        if command == "checkpoint" and self.record_data:
            self.worker_queue.put(
                ("checkpoint", None, state, timestamp_ns, dict(extras))
            )
            return
        if command == "finish_pending":
            self._pending_dagger_start = False
            self._finish_pending = True
            self._waiting_for_inference = False
            self._stop_recording()
            return
        if command == "save":
            if self.record_data:
                self._stop_recording()
            self._finish_pending = False
            print(f"[DAGGER] Recording saved: {self.current_file_name}")
            self._spawn_post_video()
            return
        if command == "discard":
            if self.record_data:
                self._stop_recording()
            self._finish_pending = False
            discard_timestamp_ns = extras.get("discard_timestamp_ns") or timestamp_ns
            self._mark_discard(
                int(discard_timestamp_ns),
                after_checkpoint=bool(extras.get("discard_after_checkpoint")),
            )
            self._spawn_post_video()

    def _last_checkpoint(self, file, end_ns: int) -> tuple[int, int] | None:
        if "dagger_checkpoints" not in file:
            return None
        group = file["dagger_checkpoints"]
        timestamps = np.asarray(group["timestamp_ns"][:], dtype=np.uint64)
        valid = np.flatnonzero(timestamps <= np.uint64(end_ns))
        if not valid.size:
            return None
        row = int(valid[-1])
        return int(group["index"][row]), int(timestamps[row])

    def _mark_discard(self, end_ns: int, *, after_checkpoint: bool) -> None:
        if not self.current_file_name or not os.path.exists(self.current_file_name):
            print("[DAGGER] No recording file is available to discard")
            return
        import h5py

        with h5py.File(self.current_file_name, "r+") as file:
            checkpoint = (
                self._last_checkpoint(file, end_ns) if after_checkpoint else None
            )
            if checkpoint is None:
                file.attrs["usable"] = False
                file.attrs["unusable"] = True
                file.attrs["discarded"] = True
                file.attrs["discard_reason"] = "operator_discard"
                starts = [
                    int(np.min(dataset[:]))
                    for dataset in file.get("timestamps", {}).values()
                    if len(dataset)
                ]
                if starts:
                    file.attrs["discard_start_ns"] = np.uint64(min(starts))
            else:
                checkpoint_index, start_ns = checkpoint
                group = file.require_group("discard_segments")
                values = {
                    "start_ns": ("uint64", np.uint64(start_ns)),
                    "end_ns": ("uint64", np.uint64(max(start_ns, end_ns))),
                    "checkpoint_index": ("int64", checkpoint_index),
                }
                for name, (dtype, _) in values.items():
                    if name not in group:
                        group.create_dataset(
                            name, (0,), maxshape=(None,), dtype=dtype, chunks=True
                        )
                row = len(group["start_ns"])
                for name, (_, value) in values.items():
                    group[name].resize(row + 1, axis=0)
                    group[name][row] = value
                file.attrs["usable"] = True
                file.attrs["unusable"] = False
                file.attrs["discarded"] = False
                file.attrs["partial_discarded"] = True
                file.attrs["discard_reason"] = "operator_discard_after_checkpoint"
                file.attrs["discard_checkpoint_index"] = checkpoint_index
                file.attrs["discard_start_ns"] = np.uint64(start_ns)
            file.attrs["discard_end_ns"] = np.uint64(end_ns)
            file.attrs["discard_timestamp"] = (
                datetime.now(timezone.utc).astimezone().timestamp()
            )
        print(
            "[DAGGER] Recording kept with checkpoint suffix marked discarded: "
            f"{self.current_file_name}"
            if after_checkpoint and checkpoint is not None
            else f"[DAGGER] Recording marked unusable: {self.current_file_name}"
        )

    def _send_inference_command(self, command: str) -> None:
        """Publish a control command to the inference server."""
        self.publish(self.control_topic, np.array([0]), extras={"command": command})

    def _handle_start_stop_key(self) -> None:
        """Keys a/b/x: toggle inference on/off.

        Not running → start inference (recording auto-starts on first event).
        Running → stop recording + go home (ready for next episode).
        """
        if not self._inference_ready:
            print("Inference server not ready yet — ignoring key.")
            return
        if self.record_data:
            # Currently running → stop + go home
            self._stop_recording()
            self._send_inference_command("stop_and_reset")
            self._waiting_for_inference = True
            print(
                "Sent stop_and_reset — robot will go home. Press a/b to start next episode."
            )
        else:
            # Not running → start inference
            self._send_inference_command("start")
            self._waiting_for_inference = True
            print(
                "Sent start — inference will begin. Recording starts on first inference event."
            )

    def _handle_shutdown_key(self) -> None:
        """Keys c/j: stop recording + full shutdown of all processes."""
        if self.record_data:
            self._stop_recording()
        self._send_inference_command("shutdown")
        print("Sent shutdown — exiting.")
        # Signal the parent (deploy_policy.py launcher) to SIGINT all
        # child processes (policy server, cameras, followers, etc.)
        os.kill(os.getppid(), signal.SIGINT)
        raise KeyboardInterrupt

    def _handle_unrecorded_sensor_item(
        self, data_type, name, payload, timestamp
    ) -> None:
        if self._waiting_for_inference and data_type == "rgb":
            # Buffer camera frames before recording starts so the JPEG
            # stream includes frames captured before the first inference.
            self._pre_record_buffer.append((data_type, name, payload, timestamp))

    def _write_extra_attrs(self, f) -> None:
        """Policy metadata + reset pose, written as H5 root attributes."""
        f.attrs["checkpoint_path"] = self.checkpoint_path
        f.attrs["checkpoint_name"] = os.path.basename(self.checkpoint_path)
        f.attrs["model_size"] = self.model_size
        f.attrs["diffusion_steps"] = self.diffusion_steps
        f.attrs["rtc"] = self.rtc
        if self.rtc:
            f.attrs["rtc_prefix_length"] = self.rtc_prefix_length
            f.attrs["rtc_inference_lead_steps"] = self.rtc_inference_lead_steps
        if self.dagger:
            f.attrs["dagger"] = True
            f.attrs["dagger_backend"] = "gello_ee_relative_mink"
            f.attrs["usable"] = True
            f.attrs["unusable"] = False
            f.attrs["discarded"] = False
            f.attrs["partial_discarded"] = False

        # Store concatenated init_q (reset pose) for first-chunk conditioning
        init_q = np.concatenate(
            [np.array(self.config.robots[name].init_q) for name in self.config.robots]
        )
        f.attrs["init_q"] = init_q

    def _handle_queue_item(self, f, data_group, timestamps_group, data) -> bool:
        if data[0] == "inference":
            _, _, action_array, timestamp, extras = data
            self._write_inference_event(
                f, timestamps_group, action_array, timestamp, extras
            )
            return True
        if data[0] == "checkpoint":
            _, _, state, timestamp, extras = data
            self._write_checkpoint(f, state, timestamp, extras)
            return True
        if data[0] == "dagger":
            _, _, _, timestamp, extras = data
            self._write_dagger_event(f, timestamp, extras)
            return True
        return False

    @staticmethod
    def _append(group, name: str, value, dtype) -> None:
        """Append one value to a lazily-created H5 dataset."""
        value = np.asarray(value, dtype=dtype)
        if name not in group:
            group.create_dataset(
                name,
                shape=(0, *value.shape),
                maxshape=(None, *value.shape),
                dtype=dtype,
                chunks=True,
            )
        dataset = group[name]
        dataset.resize(dataset.shape[0] + 1, axis=0)
        dataset[-1] = value

    def _write_inference_event(
        self,
        f: h5py.File,
        timestamps_group: h5py.Group,
        action_array: np.ndarray,
        timestamp: float,
        extras: dict,
    ) -> None:
        """Append the action and observation metadata for one policy request."""
        group = f.require_group("inference")
        if "mode" not in group.attrs:
            group.attrs["mode"] = extras.get("mode", "unknown")

        self._append(group, "actions", action_array, "float32")
        self._append(
            group, "start_ts", extras.get("inference_start_ts", 0.0), "float64"
        )
        self._append(group, "end_ts", extras.get("inference_end_ts", 0.0), "float64")
        self._append(group, "chunk_index", extras.get("chunk_index", 0), "int64")
        self._append(group, "inference_id", extras.get("inference_id", 0), "int64")
        self._append(timestamps_group, "inference", timestamp, "uint64")

        if obs_state := extras.get("obs_state"):
            self._append(group, "obs_state", obs_state, "float32")

        camera_timestamps = extras.get("camera_timestamps", {})
        if camera_timestamps:
            camera_names = sorted(camera_timestamps)
            self._append(
                group,
                "obs_camera_timestamps",
                [camera_timestamps[name] for name in camera_names],
                "float64",
            )
            group["obs_camera_timestamps"].attrs["camera_names"] = camera_names

    def _write_checkpoint(self, f, state, timestamp: int, extras: dict) -> None:
        group = f.require_group("dagger_checkpoints")
        self._append(group, "timestamp_ns", timestamp, "uint64")
        self._append(group, "index", extras.get("checkpoint", 0), "int64")
        self._append(group, "state", state, "float32")

    def _write_dagger_event(self, f, timestamp: int, extras: dict) -> None:
        import h5py

        group = f.require_group("dagger_events")
        string_dtype = h5py.string_dtype(encoding="utf-8")
        self._append(group, "timestamp_ns", timestamp, "uint64")
        self._append(group, "episode", extras.get("episode", 0), "int64")
        self._append(group, "checkpoint", extras.get("checkpoint", 0), "int64")
        self._append(group, "event", extras.get("event", ""), string_dtype)
        self._append(
            group,
            "controller_state",
            extras.get("controller_state", ""),
            string_dtype,
        )


def run(cfg: InferenceRecorderConfig) -> None:
    config = config_manager.get_i2rt_config()
    InferenceRecorderNode(
        name=cfg.name,
        control_rate=cfg.control_rate,
        data_root_directory=cfg.data_root_directory,
        config=config,
        collection_name=cfg.collection_name,
        checkpoint_path=cfg.checkpoint_path,
        model_size=cfg.model_size,
        diffusion_steps=cfg.diffusion_steps,
        rtc=cfg.rtc,
        rtc_prefix_length=cfg.rtc_prefix_length,
        rtc_inference_lead_steps=cfg.rtc_inference_lead_steps,
        task_name=cfg.task_name,
        session_tag=cfg.session_tag,
        dagger=cfg.dagger,
    ).run()


if __name__ == "__main__":
    run(tyro.cli(InferenceRecorderConfig))
