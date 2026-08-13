"""Action-chunk execution that overlaps the next policy request."""

import time

import numpy as np

from deploy.robot.gym.yam_env import ChunkedYAMEnv, YAMEnv


class RTCChunkedYAMEnv(ChunkedYAMEnv):
    """Start the next inference near the end of the current action chunk."""

    def __init__(
        self,
        config,
        chunk_dim: int,
        prompt: str,
        prefix_length: int = 5,
        inference_lead_steps: int = 10,
        execute_actions: bool = True,
    ) -> None:
        super().__init__(
            config=config,
            chunk_dim=chunk_dim,
            prompt=prompt,
            execute_actions=execute_actions,
        )
        if inference_lead_steps >= chunk_dim:
            raise ValueError("inference_lead_steps must be smaller than chunk_dim")
        if prefix_length < 1:
            raise ValueError("prefix_length must be positive")
        if prefix_length > inference_lead_steps:
            raise ValueError("prefix_length cannot exceed inference_lead_steps")
        self.prefix_length = prefix_length
        self.inference_lead_steps = inference_lead_steps

    def step_rtc(self, action: np.ndarray, rtc_manager):
        action = action.reshape(self.chunk_dim, self.single_timestep_action_dim)
        inference_start = self.chunk_dim - self.inference_lead_steps
        obs = None

        for step, command in enumerate(action):
            if step == inference_start:
                request_obs = dict(obs or {})
                request_obs["prompt"] = self.prompt
                rtc_manager.start_next_inference(request_obs, {"actions": action})

            ret = YAMEnv.step(self, command)
            obs = ret[0]
            if step + 1 < self.chunk_dim:
                time.sleep(self.config.policy.dt)

        info = {
            "rtc": {
                "inference_ready": rtc_manager.is_next_ready(),
            }
        }
        return ret[0], ret[1], ret[2], ret[3], info
