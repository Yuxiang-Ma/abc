"""Configuration for passive-GELLO DAgger collection."""

from dataclasses import dataclass

from deploy.deploy_config import DEFAULT_PROMPT, DeployConfig


@dataclass
class DaggerConfig(DeployConfig):
    record: bool = True
    collection_name: str = "dagger"
    data_root_directory: str = "./data/dagger_h5"

    foot_pedal_device: str = "/dev/input/by-id/usb-PCsensor_FootSwitch-event-kbd"
    grab_foot_pedal: bool = True

    gello_stale_timeout: float = 0.2

    intervention_rate: float = 60.0
    position_scale: float = 1.0
    rotation_scale: float = 1.0
    mink_iterations: int = 4
    mink_gain: float = 0.6
    mink_orientation_cost: float = 5.0
    mink_max_velocity: float = 3.0
    mink_max_joint_delta: float = 0.12
    mink_limit_margin: float = 0.1
    mink_limit_gain: float = 0.5
    mink_posture_cost: float = 5e-3

    def dagger_rollout_config(self):
        from deploy.robot.gym.dagger_rollout import DaggerRolloutConfig

        return DaggerRolloutConfig(
            prompt=self.prompt or DEFAULT_PROMPT,
            debug=self.debug,
            execute_chunk_dim=self.execute_chunk_dim,
            host=self.remote_host or "0.0.0.0",
            port=self.port,
            compress_images=self.compress_images,
            rtc=self.rtc,
            prefix_length=self.rtc_prefix_length,
            inference_lead_steps=self.rtc_inference_lead_steps,
            intervention_rate=self.intervention_rate,
            position_scale=self.position_scale,
            rotation_scale=self.rotation_scale,
            gello_stale_timeout=self.gello_stale_timeout,
            mink_iterations=self.mink_iterations,
            mink_gain=self.mink_gain,
            mink_orientation_cost=self.mink_orientation_cost,
            mink_max_velocity=self.mink_max_velocity,
            mink_max_joint_delta=self.mink_max_joint_delta,
            mink_limit_margin=self.mink_limit_margin,
            mink_limit_gain=self.mink_limit_gain,
            mink_posture_cost=self.mink_posture_cost,
        )
