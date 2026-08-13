import os
import time
from typing import Any

import gymnasium as gym
import numpy as np
import zmq
from gymnasium import spaces

import deploy.robot.communication as comms

# Max time to wait for a camera / follower-state message before raising.
# Picked to be well above any legitimate transient (30Hz follower, 30fps cameras
# give ~33ms nominal; i2rt CAN reads occasionally spike to tens of ms) but low
# enough that a real stall fails loudly in well under a second.
_CAMERA_RECV_TIMEOUT_MS = 500
_FOLLOWER_STATE_RECV_TIMEOUT_MS = 500


class YAMEnv(gym.Env):
    def __init__(
        self,
        config,
        prompt: str = "fold the towel",
        action_resync_max_retries: int = 2,
        execute_actions: bool = True,
    ) -> None:
        super().__init__()
        self.config = config
        self.prompt = prompt
        self.execute_actions = execute_actions
        # PUB/SUB can silently drop the single publish in step(); on
        # follower-state timeout we re-publish and retry get_obs this many
        # times before re-raising (default ~1s worst case at 500ms timeout).
        self._action_resync_max_retries = action_resync_max_retries
        self._black_image_warned: set = set()  # cameras already warned about
        num_robots = len(config.robots)
        self.action_dim = 7 * num_robots

        self.camera_names = list(self.config.cameras.keys())
        self.robot_names = list(self.config.robots.keys())

        self.observation_space = spaces.Dict(
            {
                "images": spaces.Dict(
                    {
                        camera_name: spaces.Box(
                            low=0,
                            high=255,
                            shape=(
                                config.cameras[camera_name].height,
                                config.cameras[camera_name].width,
                                3,
                            ),
                            dtype=np.uint8,
                        )
                        for camera_name in self.camera_names
                    }
                ),
                "state": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.action_dim,),
                    dtype=np.float32,
                ),
            }
        )

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.action_dim,),
            dtype=np.float32,
        )

        self.setup_backend()

    def setup_backend(self):
        self._zmq_context = zmq.Context.instance()
        self.camera_sockets = {
            name: comms.create_subscriber(
                self._zmq_context, self.config.cameras[name].socket, conflate=1
            )
            for name in self.camera_names
        }
        self.robot_state_sockets = {}
        self.robot_action_sockets = {}
        if self.execute_actions:
            self.robot_state_sockets = {
                name: comms.create_subscriber(
                    self._zmq_context, f"follower_{name}_state", conflate=1
                )
                for name in self.robot_names
            }
            self.robot_action_sockets = {
                name: comms.create_publisher(
                    self._zmq_context, f"leader_{name}_actions"
                )
                for name in self.robot_names
            }

    def _publish_action(self, action: np.ndarray) -> None:
        if not self.execute_actions:
            return
        for i, name in enumerate(self.robot_names):
            robot_action = action[i * 7 : (i + 1) * 7]
            comms.publish(self.robot_action_sockets[name], robot_action)

    def _get_state_after_action(self, action: np.ndarray) -> np.ndarray:
        for attempt in range(self._action_resync_max_retries + 1):
            try:
                return self._get_state_obs()
            except comms.SubscribeTimeout as error:
                if attempt >= self._action_resync_max_retries:
                    raise
                print(
                    f"[YAMEnv] follower state recv timed out ({error}); "
                    "re-publishing action "
                    f"(retry {attempt + 1}/{self._action_resync_max_retries})",
                    flush=True,
                )
                self._publish_action(action)

    def step_state(self, action: np.ndarray) -> np.ndarray:
        """Execute one control action without waiting for camera frames."""
        self._publish_action(action)
        return self._get_state_after_action(action)

    def step(self, action: np.ndarray):
        self._publish_action(action)
        # Cameras are read once: a camera stall is not recoverable by
        # re-publishing actions. The follower-state leg is.
        image_obs, mask_obs, camera_timestamps = self._get_camera_obs()

        state_obs = self._get_state_after_action(action)

        obs = {
            "images": image_obs,
            "masks": mask_obs,
            "state": state_obs,
            "prompt": self.prompt,
            "camera_timestamps": camera_timestamps,
        }
        return obs, 0.0, False, False, {}

    def _get_camera_obs(self):
        """Read one frame per camera. Raises SubscribeTimeout on stall."""
        image_obs = {}
        camera_timestamps = {}
        for name, socket in self.camera_sockets.items():
            msg, extras = comms.subscribe(
                socket,
                timeout_ms=_CAMERA_RECV_TIMEOUT_MS,
                topic_label=f"camera:{name}",
            )
            img = np.array(msg).astype(np.uint8)
            if img.max() == 0 and name not in self._black_image_warned:
                self._black_image_warned.add(name)
                print(
                    f"[YAMEnv] WARNING: Camera '{name}' is sending black images. "
                    f"Check camera feed or episode data."
                )
            image_obs[name] = img.transpose(2, 0, 1)
            camera_timestamps[name] = extras.get("timestamp", 0.0)
        # don't mask out any image
        mask_obs = {name: True for name in image_obs}
        return image_obs, mask_obs, camera_timestamps

    def _get_state_obs(self) -> np.ndarray:
        """Read one frame per follower. Raises SubscribeTimeout on stall."""
        if not self.execute_actions:
            return self.get_init_q()
        return np.concatenate(
            [
                comms.subscribe(
                    self.robot_state_sockets[name],
                    timeout_ms=_FOLLOWER_STATE_RECV_TIMEOUT_MS,
                    topic_label=f"follower_state:{name}",
                )[0]
                for name in self.robot_names
            ]
        )

    def get_obs(self, state: np.ndarray | None = None):
        image_obs, mask_obs, camera_timestamps = self._get_camera_obs()
        state_obs = self._get_state_obs() if state is None else np.asarray(state)
        return {
            "images": image_obs,
            "masks": mask_obs,
            "state": state_obs,
            "prompt": self.prompt,
            "camera_timestamps": camera_timestamps,
        }

    def get_init_q(self) -> np.ndarray:
        """Return concatenated init_q across all robots as a flat array."""
        return np.concatenate(
            [np.array(self.config.robots[name].init_q) for name in self.robot_names]
        )

    def move_to(self, action: np.ndarray, settle_time: float = 2.1) -> None:
        """Smoothly interpolate the followers to one concatenated action."""
        if not self.execute_actions:
            return
        for index, name in enumerate(self.robot_names):
            comms.publish(
                self.robot_action_sockets[name],
                action[index * 7 : (index + 1) * 7],
                {"type": "interp"},
            )
        time.sleep(settle_time)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ):
        # On first reset, wait for all followers to be ready before sending
        # commands. Followers publish their joint state while polling for the
        # first command in initial_bootup(), so receiving a state message
        # guarantees the follower is initialised and its ZMQ action subscriber
        # is connected.  We only do this once — on subsequent resets the
        # follower is in tick() and won't publish state until it receives an
        # action, so blocking here would deadlock.
        if self.execute_actions and not hasattr(self, "_followers_ready"):
            if os.environ.get("DEPLOY_VERBOSE"):
                print("[YAMEnv] Waiting for robot followers to connect...")
            # Generous: follower gripper calibration takes several seconds.
            for name, socket in self.robot_state_sockets.items():
                comms.subscribe(
                    socket,
                    timeout_ms=60_000,
                    topic_label=f"follower_state_bootup:{name}",
                )
            self._followers_ready = True
            if os.environ.get("DEPLOY_VERBOSE"):
                print("[YAMEnv] All followers connected.")

        if self.execute_actions:
            if os.environ.get("DEPLOY_VERBOSE"):
                print("[YAMEnv] Moving arms to starting position...")
            for name, socket in self.robot_action_sockets.items():
                command = np.asarray(self.config.robots[name].init_q)
                comms.publish(
                    socket,
                    command,
                    {"type": "interp"},
                )
            time.sleep(2.1)
        return self.get_obs(), {}

    def close(self):
        for sock in self.camera_sockets.values():
            sock.close()
        for sock in self.robot_state_sockets.values():
            sock.close()
        for sock in self.robot_action_sockets.values():
            sock.close()


class ChunkedYAMEnv(YAMEnv):
    def __init__(
        self,
        config,
        chunk_dim: int,
        prompt: str = "fold the towel",
        execute_actions: bool = True,
    ) -> None:
        super().__init__(
            config=config,
            prompt=prompt,
            execute_actions=execute_actions,
        )
        self.chunk_dim = chunk_dim
        self.single_timestep_action_dim = self.action_dim

        self.action_dim = self.single_timestep_action_dim * self.chunk_dim

        # Override the action space to be 2D for chunk semantics
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.chunk_dim, self.single_timestep_action_dim),
            dtype=np.float32,
        )

    def step(self, action: np.ndarray):
        action = action.reshape(self.chunk_dim, self.single_timestep_action_dim)
        for step, command in enumerate(action):
            ret = super().step(command)
            if step + 1 < self.chunk_dim:
                time.sleep(self.config.policy.dt)
        return ret
