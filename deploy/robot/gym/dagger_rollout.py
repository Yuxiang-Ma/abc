"""Pedal-controlled passive-GELLO DAgger rollout."""

from __future__ import annotations

import os
import select
import sys
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum

import numpy as np
import zmq

import deploy.robot.communication as comms
import deploy.robot.config as config_manager
from deploy.client import websocket_client_policy
from deploy.robot.control.mink_ik import MinkArmController
from deploy.robot.control.relative_pose import apply_relative_pose_delta
from deploy.robot.gym.policy_rollout import _publish_inference_event
from deploy.robot.gym.policy_safety import prompt_first_action_safety_check
from deploy.robot.gym.yam_env import YAMEnv


class DaggerState(str, Enum):
    PRE_RECORDING = "pre_recording"
    POLICY = "policy"
    INTERVENTION = "intervention"
    FINISH_PENDING = "finish_pending"


@dataclass
class DaggerRolloutConfig:
    prompt: str
    debug: bool
    execute_chunk_dim: int
    host: str
    port: int
    compress_images: bool
    rtc: bool
    prefix_length: int
    inference_lead_steps: int
    intervention_rate: float
    position_scale: float
    rotation_scale: float
    gello_stale_timeout: float
    mink_iterations: int
    mink_gain: float
    mink_orientation_cost: float
    mink_max_velocity: float
    mink_max_joint_delta: float
    mink_limit_margin: float
    mink_limit_gain: float
    mink_posture_cost: float


class PedalInput:
    def __init__(self, context: zmq.Context, topic: str = "KeyListener_key_presses"):
        self.socket = comms.create_subscriber(context, topic)

    def poll(self) -> list[str]:
        keys = []
        while self.socket.poll(timeout=0):
            _, extras = comms.subscribe(self.socket)
            key = str(extras.get("key", "")).lower()
            if extras.get("pressed", True) and key in {"a", "b", "c"}:
                keys.append(key)
        return keys

    def close(self) -> None:
        self.socket.close()


class RelativeGelloController:
    """Map anchored GELLO motion to absolute follower joint targets."""

    def __init__(self, config, args: DaggerRolloutConfig):
        kwargs = {
            "dt": 1.0 / args.intervention_rate,
            "iterations": args.mink_iterations,
            "orientation_cost": args.mink_orientation_cost,
            "gain": args.mink_gain,
            "max_velocity": args.mink_max_velocity,
            "max_joint_delta": args.mink_max_joint_delta,
            "limit_margin": args.mink_limit_margin,
            "limit_gain": args.mink_limit_gain,
            "posture_cost": args.mink_posture_cost,
        }
        self.solvers = {
            side: MinkArmController(
                np.asarray(config.robots[side].init_q[:6]), **kwargs
            )
            for side in ("left", "right")
        }
        self.position_scale = args.position_scale
        self.rotation_scale = args.rotation_scale
        self.anchors: dict[str, dict[str, np.ndarray]] = {}
        self.last = np.zeros(14, dtype=np.float32)
        self._gripper_offset = {"left": 0.0, "right": 0.0}
        self._gripper_start = {"left": 0.0, "right": 0.0}
        self._gripper_relative = {"left": True, "right": True}
        self._gripper_open_snap = {"left": False, "right": False}
        self._gripper_open_floor = {"left": 0.0, "right": 0.0}

    @staticmethod
    def _slice(side: str) -> slice:
        return slice(0, 7) if side == "left" else slice(7, 14)

    def anchor(self, gello: dict[str, np.ndarray], follower: np.ndarray) -> None:
        follower = np.asarray(follower, dtype=float).reshape(14)
        self.last = follower.astype(np.float32, copy=True)
        self.anchors.clear()
        for side in ("left", "right"):
            leader_q = np.asarray(gello[side], dtype=float).reshape(7)
            follower_q = follower[self._slice(side)]
            tracked_pos, tracked_wxyz = self.solvers[side].fk(leader_q[:6])
            output_pos, output_wxyz = self.solvers[side].fk(follower_q[:6])
            self.anchors[side] = {
                "tracked_pos": tracked_pos,
                "tracked_wxyz": tracked_wxyz,
                "output_pos": output_pos,
                "output_wxyz": output_wxyz,
            }
            self._gripper_offset[side] = float(follower_q[6] - leader_q[6])
            self._gripper_start[side] = float(leader_q[6])
            self._gripper_relative[side] = True
            self._gripper_open_snap[side] = False

    def _gripper(self, side: str, value: float) -> float:
        value = float(np.clip(value, -0.1, 1.0))
        start = self._gripper_start[side]
        if self._gripper_open_snap[side]:
            if value >= self._gripper_open_floor[side]:
                return 1.0
            self._gripper_open_snap[side] = False
        if self._gripper_relative[side] and start < 0.8 and value - start >= 0.1:
            self._gripper_relative[side] = False
            self._gripper_open_snap[side] = True
            self._gripper_open_floor[side] = min(1.0, start + 0.1)
            return 1.0
        if self._gripper_relative[side]:
            if value <= -0.1:
                self._gripper_relative[side] = False
                return value
            return float(np.clip(value + self._gripper_offset[side], -0.1, 1.0))
        return value

    def step(self, gello: dict[str, np.ndarray]) -> np.ndarray:
        result = self.last.astype(float, copy=True)
        for side in ("left", "right"):
            leader_q = np.asarray(gello[side], dtype=float).reshape(7)
            tracked_pos, tracked_wxyz = self.solvers[side].fk(leader_q[:6])
            anchor = self.anchors[side]
            target_pos, target_wxyz = apply_relative_pose_delta(
                tracked_pos=tracked_pos,
                tracked_wxyz=tracked_wxyz,
                anchor_tracked_pos=anchor["tracked_pos"],
                anchor_tracked_wxyz=anchor["tracked_wxyz"],
                anchor_output_pos=anchor["output_pos"],
                anchor_output_wxyz=anchor["output_wxyz"],
                position_scale=self.position_scale,
                rotation_scale=self.rotation_scale,
            )
            arm_slice = self._slice(side)
            seed = result[arm_slice][:6]
            result[arm_slice][:6] = self.solvers[side].solve(
                target_pos, target_wxyz, seed
            )
            result[arm_slice][6] = self._gripper(side, leader_q[6])
        self.last = result.astype(np.float32)
        return self.last.copy()


class DaggerLoop:
    def __init__(self, env: YAMEnv, config, args: DaggerRolloutConfig):
        self.env = env
        self.config = config
        self.args = args
        if args.rtc and args.inference_lead_steps >= args.execute_chunk_dim:
            raise ValueError(
                "RTC inference lead steps must be smaller than chunk length"
            )
        if args.rtc and args.prefix_length > args.inference_lead_steps:
            raise ValueError("RTC prefix length cannot exceed inference lead steps")
        if args.rtc and args.prefix_length < 1:
            raise ValueError("RTC prefix length must be positive")

        if args.rtc:
            from deploy.client.async_websocket_client import (
                AsyncWebsocketClientPolicy,
                RTCInferenceManager,
            )

            client_type = AsyncWebsocketClientPolicy
            self.policy = client_type(
                host=args.host,
                port=args.port,
                api_key=None,
                compress_images=args.compress_images,
            )
            self.rtc_manager = RTCInferenceManager(self.policy, args.prefix_length)
        else:
            self.policy = websocket_client_policy.WebsocketClientPolicy(
                host=args.host,
                port=args.port,
                api_key=None,
                compress_images=args.compress_images,
            )
            self.rtc_manager = None

        self.context = zmq.Context()
        self.pedals = PedalInput(self.context)
        self.gello_sockets = {
            side: comms.create_subscriber(
                self.context, f"dagger_gello_{side}_actions", conflate=1
            )
            for side in ("left", "right")
        }
        self.inference_pub = comms.create_publisher(self.context, "inference_events")
        self.recorder_pub = comms.create_publisher(
            self.context, "dagger_recorder_control"
        )
        self.event_pub = comms.create_publisher(self.context, "dagger_events")
        self.relative = RelativeGelloController(config, args)
        self.state = DaggerState.PRE_RECORDING
        self.latest_gello: dict[str, np.ndarray] = {}
        self.gello_update_time: dict[str, float] = {}
        self.current_state = np.concatenate(
            [np.asarray(config.robots[side].init_q) for side in ("left", "right")]
        ).astype(np.float32)
        self.current_chunk: np.ndarray | None = None
        self.action_index = 0
        self.chunk_index = 0
        self.inference_id = 0
        self.episode = 0
        self.checkpoint = 0
        self.finish_timestamp_ns: int | None = None
        self.intervention_tail: deque[np.ndarray] = deque(maxlen=args.prefix_length)
        self._next_tail_sample = 0.0
        self._gello_stale_warned = False
        self._safety_confirmed = False

    def _publish_recorder(self, command: str, **extras) -> None:
        comms.publish(
            self.recorder_pub,
            self.current_state,
            extras={
                "command": command,
                "controller_state": self.state.value,
                "episode": self.episode,
                **extras,
            },
        )

    def _publish_event(self, event: str) -> None:
        comms.publish(
            self.event_pub,
            np.array([0], dtype=np.uint8),
            extras={
                "event": event,
                "controller_state": self.state.value,
                "episode": self.episode,
                "checkpoint": self.checkpoint,
            },
        )

    def _update_gellos(self) -> None:
        for side, socket in self.gello_sockets.items():
            while socket.poll(timeout=0):
                position, _ = comms.subscribe(socket)
                self.latest_gello[side] = np.asarray(position, dtype=np.float32)
                self.gello_update_time[side] = time.monotonic()

    def _gellos_ready(self) -> bool:
        self._update_gellos()
        now = time.monotonic()
        return all(
            side in self.latest_gello
            and now - self.gello_update_time.get(side, float("-inf"))
            <= self.args.gello_stale_timeout
            for side in ("left", "right")
        )

    def _rtc_prefix(self) -> np.ndarray:
        if not self.intervention_tail:
            return np.tile(self.current_state, (self.args.prefix_length, 1))
        prefix = list(self.intervention_tail)
        while len(prefix) < self.args.prefix_length:
            prefix.insert(0, prefix[0])
        return np.asarray(prefix[-self.args.prefix_length :], dtype=np.float32)

    def _extract_chunk(self, result: dict) -> np.ndarray:
        actions = np.asarray(result["actions"], dtype=np.float32)
        start = self.args.prefix_length if self.args.rtc else 0
        chunk = actions[start : start + self.args.execute_chunk_dim]
        expected = (self.args.execute_chunk_dim, self.current_state.size)
        if chunk.shape != expected:
            raise ValueError(
                f"Policy returned action chunk {chunk.shape}, expected {expected}"
            )
        return chunk

    def _request_chunk(self, *, first: bool = False) -> None:
        obs = self.env.get_obs(state=self.current_state)
        if self.args.rtc:
            prefix = self._rtc_prefix()
            obs["action_prefix"] = prefix
            obs["prefix_length"] = self.args.prefix_length
        start = time.perf_counter()
        result = self.policy.infer(obs)
        end = time.perf_counter()
        chunk = self._extract_chunk(result)
        if first and sys.stdin.isatty():
            self._confirm_first_action(chunk)
        _publish_inference_event(
            self.inference_pub,
            action=chunk,
            obs=obs,
            inference_id=self.inference_id,
            chunk_index=self.chunk_index,
            start_ts=start,
            end_ts=end,
            mode="rtc" if self.args.rtc else "standard",
        )
        self.inference_id += 1
        self.chunk_index += 1
        self.current_chunk = chunk
        self.action_index = 0

    def _confirm_first_action(self, chunk: np.ndarray) -> None:
        """Blocking safety gate before the session's first policy action.

        Confirm with the A pedal, keyboard 'a' (forwarded by the key
        listener), or ENTER in this terminal; Ctrl-C aborts.
        """
        prompt_first_action_safety_check(
            self.current_state,
            chunk,
            title="FIRST DAGGER POLICY ACTION",
            enter_prompt=None,
        )
        print(
            "Press A (pedal or keyboard) or ENTER to execute, Ctrl-C to abort > ",
            end="",
            flush=True,
        )
        while True:
            if select.select([sys.stdin], [], [], 0)[0]:
                sys.stdin.readline()
                break
            if any(key == "a" for key in self.pedals.poll()):
                break
            time.sleep(0.02)
        print("confirmed.")
        self._safety_confirmed = True

    def _drain_pedals(self, context: str) -> None:
        """Discard pedal presses queued while the loop was blocked."""
        dropped = self.pedals.poll()
        if dropped:
            print(
                f"[DAGGER] Ignoring {len(dropped)} pedal press(es) queued "
                f"during {context}."
            )

    def _start_episode(self) -> None:
        self.episode += 1
        self.checkpoint = 0
        self.intervention_tail.clear()
        self.state = DaggerState.POLICY
        self._publish_recorder("start")
        print(
            f"[DAGGER] Episode {self.episode} started; A intervenes, C checkpoints, B ends."
        )
        # The safety gate only runs before the session's first policy action.
        self._request_chunk(first=not self._safety_confirmed)
        # Announce POLICY only once the first chunk is ready and confirmed:
        # recordings then label the inference/confirmation wait as
        # pre-recording instead of policy time.
        self._publish_event("episode_start")
        # Anything pressed while inference/confirmation blocked the loop
        # (including pedal contact bounce) must not replay as commands.
        self._drain_pedals("episode start")

    def _enter_intervention(self) -> None:
        if not self._gellos_ready():
            print("[DAGGER] GELLO data is missing or stale; intervention refused.")
            return
        if self.rtc_manager is not None:
            self.rtc_manager.drain_pending()
        self.relative.anchor(self.latest_gello, self.current_state)
        self.intervention_tail.clear()
        self._next_tail_sample = time.monotonic()
        self.state = DaggerState.INTERVENTION
        self._publish_event("intervention_start")
        print("[DAGGER] Policy paused; passive GELLO intervention active.")

    def _resume_policy(self) -> None:
        self.state = DaggerState.POLICY
        print(
            "[DAGGER] Intervention ended; requesting a newly conditioned policy chunk."
        )
        self._request_chunk()
        # Recorded state stays INTERVENTION until the handback chunk is
        # ready; the robot holds the intervention pose during that wait.
        self._publish_event("intervention_end")

    def _record_checkpoint(self) -> None:
        self.checkpoint += 1
        self._publish_recorder("checkpoint", checkpoint=self.checkpoint)
        self._publish_event("checkpoint")
        print(f"[DAGGER] Checkpoint {self.checkpoint} recorded.")

    def _begin_finish(self) -> None:
        if self.rtc_manager is not None:
            self.rtc_manager.drain_pending()
        self.finish_timestamp_ns = time.perf_counter_ns()
        self.state = DaggerState.FINISH_PENDING
        self._publish_recorder(
            "finish_pending", finish_timestamp_ns=self.finish_timestamp_ns
        )
        self._publish_event("finish_pending")
        print(
            "[DAGGER] Robot frozen. Press A to discard after the latest checkpoint, or C to save."
        )

    def _finalize(self, discard: bool) -> None:
        command = "discard" if discard else "save"
        self._publish_recorder(
            command,
            discard_after_checkpoint=discard and self.checkpoint > 0,
            checkpoint=self.checkpoint,
            discard_timestamp_ns=self.finish_timestamp_ns,
        )
        self._publish_event(command)
        print("[DAGGER] Returning robot home...")
        obs, _ = self.env.reset()
        self.current_state = np.asarray(obs["state"], dtype=np.float32)
        self.current_chunk = None
        self.finish_timestamp_ns = None
        self.intervention_tail.clear()
        self.state = DaggerState.PRE_RECORDING
        self._drain_pedals("homing")
        print("[DAGGER] Ready. Press A to start the next episode.")

    def _handle_key(self, key: str) -> None:
        if self.state == DaggerState.PRE_RECORDING:
            if key == "a":
                self._start_episode()
            return
        if self.state == DaggerState.FINISH_PENDING:
            if key == "a":
                self._finalize(discard=True)
            elif key == "c":
                self._finalize(discard=False)
            return
        if key == "b":
            self._begin_finish()
        elif key == "c":
            self._record_checkpoint()
        elif key == "a" and self.state == DaggerState.POLICY:
            self._enter_intervention()
        elif key == "a" and self.state == DaggerState.INTERVENTION:
            self._resume_policy()

    def _finish_rtc_chunk(self) -> None:
        result = self.rtc_manager.get_next_actions(timeout=5.0)
        timing = self.rtc_manager.get_last_timing()
        obs = self.rtc_manager.get_inference_obs() or self.env.get_obs(
            state=self.current_state
        )
        chunk = self._extract_chunk(result)
        _publish_inference_event(
            self.inference_pub,
            action=chunk,
            obs=obs,
            inference_id=self.inference_id,
            chunk_index=self.chunk_index,
            start_ts=timing["start_ts"],
            end_ts=timing["end_ts"],
            mode="rtc",
        )
        self.inference_id += 1
        self.chunk_index += 1
        self.current_chunk = chunk
        self.action_index = 0

    def _policy_step(self) -> None:
        if self.current_chunk is None:
            self._request_chunk()
        index = self.action_index
        if (
            self.args.rtc
            and index == self.args.execute_chunk_dim - self.args.inference_lead_steps
        ):
            obs = self.env.get_obs(state=self.current_state)
            self.rtc_manager.start_next_inference(obs, {"actions": self.current_chunk})
        started = time.monotonic()
        command = self.current_chunk[index]
        self.current_state = self.env.step_state(command).astype(np.float32)
        self.action_index += 1
        remaining = self.config.policy.dt - (time.monotonic() - started)
        if remaining > 0:
            time.sleep(remaining)
        if self.action_index < self.args.execute_chunk_dim:
            return
        if self.args.rtc:
            self._finish_rtc_chunk()
        else:
            self._request_chunk()

    def _intervention_step(self) -> None:
        started = time.monotonic()
        if not self._gellos_ready():
            if not self._gello_stale_warned:
                print("[DAGGER] GELLO stream became stale; holding the last target.")
                self._gello_stale_warned = True
            command = self.current_state
        else:
            if self._gello_stale_warned:
                print("[DAGGER] GELLO stream recovered.")
                self._gello_stale_warned = False
            command = self.relative.step(self.latest_gello)
        self.current_state = self.env.step_state(command).astype(np.float32)
        now = time.monotonic()
        if now >= self._next_tail_sample:
            self.intervention_tail.append(np.asarray(command).copy())
            self._next_tail_sample = now + self.config.policy.dt
        remaining = 1.0 / self.args.intervention_rate - (time.monotonic() - started)
        if remaining > 0:
            time.sleep(remaining)

    def run(self) -> None:
        metadata = self.policy.get_server_metadata()
        print(
            f"Policy server ready. metadata={metadata}"
            if os.environ.get("DEPLOY_VERBOSE")
            else "Policy server ready."
        )
        obs, _ = self.env.reset()
        self.current_state = np.asarray(obs["state"], dtype=np.float32)
        print("[DAGGER] Ready. Press A to start an episode.")
        while True:
            self._update_gellos()
            keys = self.pedals.poll()
            for key in keys:
                self._handle_key(key)
            if keys:
                continue
            if self.state == DaggerState.POLICY:
                self._policy_step()
            elif self.state == DaggerState.INTERVENTION:
                self._intervention_step()
            elif self.state == DaggerState.FINISH_PENDING:
                started = time.monotonic()
                self.current_state = self.env.step_state(self.current_state).astype(
                    np.float32
                )
                time.sleep(
                    max(
                        0.0,
                        1.0 / self.args.intervention_rate
                        - (time.monotonic() - started),
                    )
                )
            else:
                time.sleep(0.01)

    def close(self) -> None:
        if self.rtc_manager is not None and self.rtc_manager.has_pending_inference():
            self.rtc_manager.drain_pending()
        self.policy.close()
        self.pedals.close()
        for socket in self.gello_sockets.values():
            socket.close()
        self.inference_pub.close()
        self.recorder_pub.close()
        self.event_pub.close()
        self.context.term()


def run(args: DaggerRolloutConfig) -> None:
    config = config_manager.get_i2rt_config()
    env = YAMEnv(config, prompt=args.prompt, execute_actions=not args.debug)
    print("Waiting for the policy server to load...")
    loop = DaggerLoop(env, config, args)
    try:
        loop.run()
    except KeyboardInterrupt:
        pass
    finally:
        loop.close()
        env.close()
