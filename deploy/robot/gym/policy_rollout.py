"""Policy inference with optional Real-Time Action Chunking (RTC) support.

Two modes of operation:
1. Standard mode: Sequential inference and execution (original behavior)
2. RTC mode: Overlapped inference and execution for smooth transitions
"""

import os
import sys
import time

import gymnasium as gym
import numpy as np
import tyro
import zmq

import deploy.robot.communication as comms
import deploy.robot.config as config_manager
from deploy.client import websocket_client_policy as _websocket_client_policy
from deploy.robot.gym.policy_rollout_config import PolicyRolloutConfig
from deploy.robot.gym.policy_safety import prompt_first_action_safety_check
from deploy.robot.gym.yam_env import ChunkedYAMEnv


def _extract_obs_state(obs: dict) -> list:
    """Extract the state vector from an observation dict as a JSON-safe list."""
    state = obs.get("state", obs.get("observation/state", None))
    if state is None:
        return []
    if isinstance(state, np.ndarray):
        return state.tolist()
    return list(state)


def _extract_camera_timestamps(obs: dict) -> dict:
    """Extract camera timestamps from an observation dict as a JSON-safe dict."""
    return obs.get("camera_timestamps", {})


def _first_rtc_action_prefix(obs: dict, env, prefix_length: int) -> np.ndarray:
    """Build the first RTC prefix from the live reset state when available.

    Using the observed reset state is safer than the configured ``init_q``,
    especially for grippers where calibration drift or profile mismatches can
    make the config stale by a large margin.
    """
    state = obs.get("state")
    if state is not None:
        state = np.asarray(state).reshape(-1)
        expected_dim = getattr(
            env, "single_timestep_action_dim", getattr(env, "action_dim", state.size)
        )
        if state.shape == (expected_dim,):
            return np.tile(state, (prefix_length, 1))

    init_q = np.asarray(env.get_init_q()).reshape(-1)
    return np.tile(init_q, (prefix_length, 1))


def _publish_inference_event(
    inference_pub,
    *,
    action: np.ndarray,
    obs: dict,
    inference_id: int,
    chunk_index: int,
    start_ts: float,
    end_ts: float,
    mode: str,
) -> None:
    """Publish the action and observation metadata for an inference step."""
    extras = {
        "inference_id": inference_id,
        "inference_start_ts": start_ts,
        "inference_end_ts": end_ts,
        "chunk_index": chunk_index,
        "mode": mode,
        "obs_state": _extract_obs_state(obs),
        "camera_timestamps": _extract_camera_timestamps(obs),
    }
    comms.publish(inference_pub, action, extras=extras)


class InferenceController:
    """Subscribes to ``inference_control`` ZMQ topic and provides operator commands."""

    idle_prompt = "IDLE — press a/b to start inference (c/j to shut down)."

    def __init__(self):
        self._zmq_context = zmq.Context()
        self._sub = comms.create_subscriber(self._zmq_context, "inference_control")
        self._poller = zmq.Poller()
        self._poller.register(self._sub, zmq.POLLIN)

    def poll_command(self) -> str | None:
        """Non-blocking check for a control command. Returns command string or None."""
        socks = dict(self._poller.poll(timeout=0))
        if self._sub in socks:
            _, extras = comms.subscribe(self._sub)
            return extras.get("command")
        return None

    def close(self):
        self._sub.close()
        self._zmq_context.term()


class KeyboardEpisodeController:
    """Consumes KeyListener key presses directly (episode control without a
    recorder). Same key mapping as the recorder's key handling: a/b/x
    toggles start <-> stop-and-home, c/j shuts everything down. Returns
    "toggle" so the rollout can interpret it by phase (IDLE -> start,
    mid-episode -> stop_and_reset)."""

    idle_prompt = "IDLE — press a/b to start inference (c/j to shut down)."
    _TOGGLE_KEYS = frozenset({"a", "b", "x"})
    _SHUTDOWN_KEYS = frozenset({"c", "j"})

    def __init__(self, context: zmq.Context):
        self._sub = comms.create_subscriber(context, "KeyListener_key_presses")

    @classmethod
    def command_for_key(cls, key) -> str | None:
        key = str(key).lower()
        if key in cls._SHUTDOWN_KEYS:
            return "shutdown"
        if key in cls._TOGGLE_KEYS:
            return "toggle"
        return None

    def poll_command(self) -> str | None:
        while self._sub.poll(timeout=0):
            _, extras = comms.subscribe(self._sub)
            if not extras.get("pressed", True):
                continue
            command = self.command_for_key(extras.get("key", ""))
            if command is not None:
                return command
        return None

    def close(self):
        self._sub.close()


def _wait_for_start_key(controller, inference_pub) -> str:
    """Block in IDLE until the controller sends 'start' or 'shutdown'.

    Publishes a ``ready`` heartbeat each tick so the recorder can confirm the
    rollout is alive before forwarding operator commands (ZMQ pub/sub drops early
    messages, so the recorder waits for the first heartbeat).
    """
    while True:
        comms.publish(inference_pub, np.array([0]), extras={"event": "ready"})
        cmd = controller.poll_command()
        if cmd == "toggle":
            return "start"
        if cmd in ("start", "shutdown"):
            return cmd
        time.sleep(0.01)


def _run_episode(
    env: gym.Env,
    policy,
    obs: dict,
    *,
    execute_chunk_dim: int,
    inference_pub,
    controller: "InferenceController | None",
    inference_id: int,
) -> tuple[int, str, dict]:
    """Run one rollout episode from ``obs``. Returns (next_inference_id, reason, last_obs).

    reason is one of: "terminated", "stop_and_reset", "shutdown".
    """
    chunk_index = 0
    while True:
        t_start = time.perf_counter()
        result = policy.infer(obs)
        action = np.array(result["actions"])[:execute_chunk_dim]
        t_end = time.perf_counter()

        # Confirm the first action interactively before the robot moves.
        if chunk_index == 0 and sys.stdin.isatty():
            prompt_first_action_safety_check(obs["state"], action)

        _publish_inference_event(
            inference_pub,
            action=action,
            obs=obs,
            inference_id=inference_id,
            chunk_index=chunk_index,
            start_ts=t_start,
            end_ts=t_end,
            mode="standard",
        )
        inference_id += 1
        chunk_index += 1

        obs, _reward, terminated, truncated, _info = env.step(action)

        if controller is not None:
            cmd = controller.poll_command()
            if cmd in ("stop_and_reset", "toggle"):
                return inference_id, "stop_and_reset", obs
            if cmd == "shutdown":
                return inference_id, "shutdown", obs
        if terminated or truncated:
            return inference_id, "terminated", obs


def run_policy_rollout(
    env,
    execute_chunk_dim: int,
    host: str = "0.0.0.0",
    port: int = 8000,
    recorder_control: bool = False,
    direct_episode_keys: bool = False,
    compress_images: bool = False,
    rtc: bool = False,
    prefix_length: int = 5,
) -> None:
    """Connect to the policy server and run standard or RTC episodes."""
    if rtc:
        from deploy.client.async_websocket_client import (
            AsyncWebsocketClientPolicy,
            RTCInferenceManager,
        )

        client_type = AsyncWebsocketClientPolicy
    else:
        client_type = _websocket_client_policy.WebsocketClientPolicy

    print("Waiting for the policy server to load...")
    policy = client_type(
        host=host,
        port=port,
        api_key=None,
        compress_images=compress_images,
    )
    print("Policy server ready.")
    if os.environ.get("DEPLOY_VERBOSE"):
        print(f"Connected to {host}:{port}; metadata={policy.get_server_metadata()}")

    context = zmq.Context()
    inference_pub = comms.create_publisher(context, "inference_events")
    if recorder_control:
        controller = InferenceController()
    elif direct_episode_keys:
        controller = KeyboardEpisodeController(context)
    else:
        controller = None
    rtc_manager = RTCInferenceManager(policy, prefix_length) if rtc else None

    try:
        obs, _ = env.reset()
        inference_id = 0
        while True:
            if controller is not None:
                print(controller.idle_prompt)
                if _wait_for_start_key(controller, inference_pub) == "shutdown":
                    return

            if rtc:
                inference_id, reason = _run_rtc_episode(
                    env,
                    policy,
                    rtc_manager,
                    execute_chunk_dim,
                    prefix_length,
                    inference_pub,
                    inference_id,
                    obs,
                    controller,
                )
            else:
                inference_id, reason, obs = _run_episode(
                    env,
                    policy,
                    obs,
                    execute_chunk_dim=execute_chunk_dim,
                    inference_pub=inference_pub,
                    controller=controller,
                    inference_id=inference_id,
                )

            if reason == "shutdown":
                return
            print("Going home...")
            obs, _ = env.reset()
    finally:
        policy.close()
        if controller is not None:
            controller.close()


def _run_rtc_episode(
    env,
    policy,
    rtc_manager,
    execute_chunk_dim,
    prefix_length,
    inference_pub,
    inference_id,
    obs,
    controller: InferenceController | None = None,
):
    """Run a single RTC episode. Returns (inference_id, stop_reason).

    stop_reason is one of: "terminated", "stop_and_reset", "shutdown".
    """
    chunk_count = 0
    chunk_index = 0

    # First chunk: condition on the actual reset observation when available.
    # Configured init_q can drift from the live robot state, especially grippers.
    action_prefix = _first_rtc_action_prefix(obs, env, prefix_length)
    obs["action_prefix"] = action_prefix
    obs["prefix_length"] = prefix_length

    print("Starting first chunk (blocking inference, conditioned on reset state)...")
    t_start = time.perf_counter()
    result = policy.infer(obs)
    t_end = time.perf_counter()
    action = np.array(result["actions"])[
        prefix_length : prefix_length + execute_chunk_dim
    ]

    # Confirm the first RTC action interactively before the robot moves.
    if sys.stdin.isatty():
        prompt_first_action_safety_check(
            obs["state"], action, title="FIRST RTC ACTION SAFETY CHECK"
        )

    _publish_inference_event(
        inference_pub,
        action=action,
        obs=obs,
        inference_id=inference_id,
        chunk_index=chunk_index,
        start_ts=t_start,
        end_ts=t_end,
        mode="rtc",
    )
    inference_id += 1
    chunk_index += 1

    while True:
        chunk_count += 1

        # Execute with RTC (starts inference for next chunk during execution)
        obs, _reward, terminated, truncated, info = env.step_rtc(action, rtc_manager)

        # Warn if inference hasn't finished by end of chunk execution
        rtc_info = info.get("rtc", {})
        if rtc_info.get("inference_ready") is False:
            print(
                f"WARNING: Chunk {chunk_count}: inference is still pending; "
                "the robot may pause between chunks."
            )

        # Check for operator commands between chunks
        if controller is not None:
            cmd = controller.poll_command()
            if cmd in ("stop_and_reset", "toggle"):
                print(f"Episode stopped by operator after {chunk_count} chunks")
                rtc_manager.drain_pending()
                return inference_id, "stop_and_reset"
            elif cmd == "shutdown":
                print("Shutdown command received.")
                rtc_manager.drain_pending()
                return inference_id, "shutdown"

        if terminated or truncated:
            print(f"Episode complete after {chunk_count} chunks")
            rtc_manager.drain_pending()
            return inference_id, "terminated"

        # Get the next chunk (should be ready or nearly ready)
        next_result = rtc_manager.get_next_actions(timeout=5.0)

        # Publish inference event with timing from RTCInferenceManager
        timing = rtc_manager.get_last_timing()
        next_action = np.array(next_result["actions"])[
            prefix_length : prefix_length + execute_chunk_dim
        ]

        # Use the obs that was actually sent for inference (captured
        # mid-execution), not the end-of-chunk obs which has diverged.
        inference_obs = rtc_manager.get_inference_obs() or obs

        _publish_inference_event(
            inference_pub,
            action=next_action,
            obs=inference_obs,
            inference_id=inference_id,
            chunk_index=chunk_index,
            start_ts=timing["start_ts"],
            end_ts=timing["end_ts"],
            mode="rtc",
        )
        inference_id += 1
        chunk_index += 1

        action = next_action


def _select_rollout(args: PolicyRolloutConfig, config):
    """Return the environment, runner, and arguments for the selected mode."""
    common = {
        "execute_chunk_dim": args.execute_chunk_dim,
        "host": args.host,
        "port": args.port,
        "compress_images": args.compress_images,
        "recorder_control": args.recorder_control,
        "direct_episode_keys": args.direct_episode_keys,
    }
    if args.rtc:
        from deploy.robot.gym.rtc_gym_env import RTCChunkedYAMEnv

        env = RTCChunkedYAMEnv(
            config,
            chunk_dim=args.execute_chunk_dim,
            prompt=args.prompt,
            prefix_length=args.prefix_length,
            inference_lead_steps=args.inference_lead_steps,
            execute_actions=not args.debug,
        )
        return env, {
            **common,
            "rtc": True,
            "prefix_length": args.prefix_length,
        }

    env = ChunkedYAMEnv(
        config,
        chunk_dim=args.execute_chunk_dim,
        prompt=args.prompt,
        execute_actions=not args.debug,
    )
    return env, {**common, "rtc": False}


def run(args: PolicyRolloutConfig) -> None:
    env, kwargs = _select_rollout(args, config_manager.get_i2rt_config())
    try:
        run_policy_rollout(env, **kwargs)
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


if __name__ == "__main__":
    run(tyro.cli(PolicyRolloutConfig))
