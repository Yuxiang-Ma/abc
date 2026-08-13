"""User-facing ABC-DiT deployment configuration."""

from dataclasses import dataclass

from deploy.policy import PolicyConfig
from deploy.robot.gym.policy_rollout_config import PolicyRolloutConfig


@dataclass
class DeployConfig:
    checkpoint_path: str = ""
    prompt: str | None = None
    norm_stats_path: str | None = None
    diffusion_steps: int = 10
    deterministic: bool = False
    fast_inference: bool = False

    debug: bool = False
    rtc: bool = False
    rtc_prefix_length: int = 4
    rtc_inference_lead_steps: int = 7
    execute_chunk_dim: int = 16
    compress_images: bool = False
    init_q: str = ""

    remote_host: str = ""
    port: int = 8000

    record: bool = True
    """Record the robot rollout to H5 (disable with --no-record)."""
    collection_name: str = "policy_rollout"
    data_root_directory: str = "./data/recording"
    session_tag: str = ""
    post_video: bool = True
    """Render saved recordings to review MP4s in a detached process."""
    episode_control: bool = True
    """Keyboard episode control for --no-record runs (a/b start or stop+home,
    c/j shut down). Recording runs always have episode control via the recorder."""
    foot_pedal_device: str | None = None
    """Pedal evdev path. Default: FOOT_PEDAL_INPUT_DEVICE or the PCsensor
    device when plugged in; pass '' for keyboard-only."""
    verbose: bool = False

    def checkpoint(self) -> PolicyConfig:
        return PolicyConfig(
            checkpoint_path=self.checkpoint_path,
            norm_stats_path=self.norm_stats_path,
            prompt=self.prompt or "throw plastic bottles in bin",
            diffusion_steps=self.diffusion_steps,
            deterministic=self.deterministic,
            fast_inference=self.fast_inference,
            rtc_prefix_length=self.rtc_prefix_length if self.rtc else None,
        )

    def rollout_config(
        self, *, pedal_control: bool = False, direct_episode_keys: bool = False
    ) -> PolicyRolloutConfig:
        return PolicyRolloutConfig(
            prompt=self.prompt,
            debug=self.debug,
            direct_episode_keys=direct_episode_keys,
            execute_chunk_dim=self.execute_chunk_dim,
            host=self.remote_host or "0.0.0.0",
            port=self.port,
            rtc=self.rtc,
            prefix_length=self.rtc_prefix_length,
            inference_lead_steps=self.rtc_inference_lead_steps,
            compress_images=self.compress_images,
            pedal_control=pedal_control,
        )
