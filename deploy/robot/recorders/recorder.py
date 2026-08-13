import os
import time

import numpy as np

import deploy.robot.config as config_manager
from deploy.robot.config import RobotSystemConfig
from deploy.robot.recorders.base import RecorderBase


class RecorderNode(RecorderBase):
    def __init__(
        self,
        name: str,
        control_rate: float,
        data_root_directory: str,
        config: RobotSystemConfig,
        collection_name: str,
        task_name: str = "",
        session_tag: str = "",
    ):
        super().__init__(name, control_rate, verbose=False)
        self.data_root_directory = data_root_directory
        self.config = config
        self.collection_name = collection_name
        self.session_tag = session_tag
        self.task_name = task_name
        self.camera_names = list(self.config.cameras.keys())

        self._init_sensor_topics()

        self._recording_start_time = None
        self._session_saved_seconds = 0.0
        self._session_saved_count = 0
        # Stage tracking within an active recording. Stages are 1-indexed and
        # contiguous: each middle-pedal press closes the current stage and
        # opens the next using the same timestamp.
        self._stage_index = 0
        self._stage_start_ns = 0
        self._stage_records: list[tuple[int, int, int]] = []

    def initial_bootup(self) -> None:
        # create data directory
        os.makedirs(self.data_root_directory, exist_ok=True)

        if os.environ.get("DEPLOY_VERBOSE"):
            print(f"[{self._name}] Recorder node initial bootup complete.")
        print(
            "Recorder started. Pedals: any=start, left(a)=discard, mid(b)=next stage, right(c)=save. Ctrl+C to quit."
        )
        print("Listening... (not recording)")

    def _recording_duration_seconds(self) -> float:
        if self._recording_start_time is None:
            return 0.0
        return max(0.0, time.monotonic() - self._recording_start_time)

    def _print_session_total(self) -> None:
        total_minutes = self._session_saved_seconds / 60.0
        print(
            "Session total saved: "
            f"{self._session_saved_count} trajs, {total_minutes:.2f} min"
        )

    def tick(self) -> None:
        message = self.subscribe(self.key_press_topic, block=False)

        key_pressed = self._key_press(message)
        if key_pressed is not None:
            if key_pressed in [" ", "a", "b", "c", "x", "j"]:
                if not self.record_data:
                    self._start_recording()
                elif key_pressed == "a":
                    # Left pedal while recording → discard
                    self._stop_recording(discard=True)
                elif key_pressed in ("b", "x"):
                    # Middle pedal while recording → end stage, start next
                    self._advance_stage()
                else:
                    # Right pedal or space while recording → save
                    self._stop_recording(discard=False)

        self._poll_sensor_topics()

    def on_shutdown(self) -> None:
        if self.record_data and self.writer_thread:
            recording_duration_s = self._recording_duration_seconds()
            print("Finishing current recording...")
            self._close_final_stage()
            if self._stage_records:
                self.worker_queue.put(("stages", self._stage_records, None))
                self._stage_records = []
            if not self._stop_writer(timeout=30):
                print("Warning: Writer thread did not finish within timeout")
            else:
                self._session_saved_seconds += recording_duration_s
                self._session_saved_count += 1
                print("Writer thread finished successfully")
                self._print_session_total()
                self._spawn_post_video()
        if os.environ.get("DEPLOY_VERBOSE"):
            print(f"[{self._name}] Recorder node shutdown complete.")

    def _start_recording(self) -> None:
        # Check that all cameras are producing frames
        if not self._cameras_ready_or_defer():
            return

        self.record_data = True
        self._recording_start_time = time.monotonic()
        self._stage_index = 1
        self._stage_start_ns = int(time.perf_counter() * 1e9)
        self._stage_records = []
        self._new_recording_file()

        print(f"\nRECORDING STARTED - Saving to: {self.current_file_name}")
        print(f"  → stage {self._stage_index}")

        self._start_writer()

    def _advance_stage(self) -> None:
        now_ns = int(time.perf_counter() * 1e9)
        self._stage_records.append((self._stage_index, self._stage_start_ns, now_ns))
        self._stage_index += 1
        self._stage_start_ns = now_ns
        print(f"  → stage {self._stage_index}")

    def _close_final_stage(self) -> None:
        if self._stage_index <= 0:
            return
        now_ns = int(time.perf_counter() * 1e9)
        self._stage_records.append((self._stage_index, self._stage_start_ns, now_ns))

    def _stop_recording(self, discard: bool = False) -> None:
        recording_duration_s = self._recording_duration_seconds()
        self.record_data = False
        self._recording_start_time = None
        self._close_final_stage()
        stage_records = self._stage_records
        self._stage_records = []
        self._stage_index = 0

        if self.writer_thread and self.writer_thread.is_alive():
            if not discard and stage_records:
                self.worker_queue.put(("stages", stage_records, None))
            self._stop_writer(timeout=10)

        if discard:
            print(f"\nRECORDING DISCARDED - Deleting: {self.current_file_name}")
            try:
                os.remove(self.current_file_name)
            except OSError as e:
                print(f"  Warning: could not delete file: {e}")
        else:
            self._session_saved_seconds += recording_duration_s
            self._session_saved_count += 1
            print(f"\nRECORDING STOPPED - File saved: {self.current_file_name}")
            self._print_session_total()
            self._spawn_post_video()

    def _handle_queue_item(self, f, data_group, timestamps_group, data) -> bool:
        if data[0] != "stages":
            return False
        records = data[1]
        stages_group = f.require_group("stages")
        n = len(records)
        indices = np.array([r[0] for r in records], dtype=np.int32)
        starts = np.array([r[1] for r in records], dtype=np.uint64)
        ends = np.array([r[2] for r in records], dtype=np.uint64)
        stages_group.create_dataset("index", data=indices)
        stages_group.create_dataset("start_ns", data=starts)
        stages_group.create_dataset("end_ns", data=ends)
        if os.environ.get("DEPLOY_VERBOSE"):
            print(f"Writer thread received {n} stage records")
        return True


def run(
    name: str,
    control_rate: float,
    data_root_directory: str,
    collection_name: str,
    task_name: str = "",
    session_tag: str = "",
):
    config = config_manager.get_i2rt_config()
    RecorderNode(
        name=name,
        control_rate=control_rate,
        data_root_directory=data_root_directory,
        config=config,
        collection_name=collection_name,
        task_name=task_name,
        session_tag=session_tag,
    ).run()
