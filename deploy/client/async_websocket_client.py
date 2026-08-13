"""One-request-ahead inference used by Real-Time Action Chunking."""

import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor

from deploy.client.websocket_client_policy import WebsocketClientPolicy


class AsyncWebsocketClientPolicy(WebsocketClientPolicy):
    """Run blocking websocket inference on one background thread."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._lock = threading.Lock()

    def infer(self, obs: dict) -> dict:
        with self._lock:
            return super().infer(obs)

    def infer_async(self, obs: dict) -> Future:
        return self._executor.submit(self.infer, obs)

    @staticmethod
    def get_result(future: Future, timeout: float | None = None) -> dict:
        return future.result(timeout=timeout)

    @staticmethod
    def is_ready(future: Future) -> bool:
        return future.done()

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
        super().close()


class RTCInferenceManager:
    """Start the next inference while the current action chunk executes."""

    def __init__(
        self,
        client: AsyncWebsocketClientPolicy,
        prefix_length: int = 5,
        inference_lead_steps: int = 10,
    ) -> None:
        self.client = client
        self.prefix_length = prefix_length
        self.inference_lead_steps = inference_lead_steps
        self._pending: Future | None = None
        self._start_ts: float | None = None
        self._end_ts: float | None = None
        self._inference_obs: dict | None = None

    def start_next_inference(
        self, obs: dict, current_actions: dict | None = None
    ) -> None:
        if self._pending is not None:
            raise RuntimeError("RTC inference already pending")
        request = dict(obs)
        self._inference_obs = dict(obs)
        if (
            self.prefix_length
            and current_actions is not None
            and len(current_actions.get("actions", []))
        ):
            actions = current_actions["actions"]
            prefix = actions[-self.prefix_length :]
            request["action_prefix"] = prefix
            request["prefix_length"] = len(prefix)
            request["latency"] = self.inference_lead_steps - self.prefix_length

        self._start_ts = time.perf_counter()
        self._pending = self.client.infer_async(request)

    def get_next_actions(self, timeout: float | None = None) -> dict:
        if self._pending is None:
            raise RuntimeError("No pending RTC inference")
        result = self.client.get_result(self._pending, timeout)
        self._end_ts = time.perf_counter()
        self._pending = None
        return result

    def is_next_ready(self) -> bool:
        return self._pending is not None and self._pending.done()

    def has_pending_inference(self) -> bool:
        return self._pending is not None

    def drain_pending(self, timeout: float | None = 5.0) -> None:
        """Drain an in-flight websocket request so the connection remains usable."""
        if self._pending is None:
            return
        pending = self._pending
        try:
            pending.result(timeout=timeout)
        finally:
            if pending.done():
                self._pending = None
                self._inference_obs = None

    def get_inference_obs(self) -> dict | None:
        return self._inference_obs

    def get_last_timing(self) -> dict:
        return {"start_ts": self._start_ts, "end_ts": self._end_ts}
