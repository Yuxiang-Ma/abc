"""User-facing deployment configuration for the DiT and VLA policies."""

from dataclasses import dataclass, field
from typing import Literal

from abc_minimal.config import ClipConfig, DiTConfig, VLAModelConfig
from abc_minimal.policy import InferenceConfig
from deploy.robot.gym.policy_rollout_config import PolicyRolloutConfig
from deploy.serve_policy_config import Args as ServeArgs

DEFAULT_PROMPT = "throw plastic bottles in bin"
MODEL_SIZES = {"dit": "dit_xL", "vla": "vla_4b"}


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
    rtc_inference_lead_steps: int = 4
    execute_chunk_dim: int = 16
    compress_images: bool = False
    init_q: str = ""

    remote_host: str = ""
    port: int = 8000

    policy_type: Literal["auto", "dit", "vla"] = "auto"
    """"auto" classifies the checkpoint (DiT vs VLA); override to force one."""
    model_size: str = ""
    """Model name stamped into recordings and review videos; empty follows policy_type."""
    dit_model: DiTConfig = field(default_factory=DiTConfig)
    clip: ClipConfig = field(default_factory=ClipConfig)
    vla_model: VLAModelConfig = field(default_factory=VLAModelConfig)

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
    verbose: bool = False

    def serve_args(self) -> ServeArgs:
        """Build the policy server's config, both architectures included."""
        return ServeArgs(
            policy=InferenceConfig(
                checkpoint_path=self.checkpoint_path,
                norm_stats_path=self.norm_stats_path,
                prompt=self.prompt or DEFAULT_PROMPT,
                diffusion_steps=self.diffusion_steps,
                deterministic=self.deterministic,
                fast_inference=self.fast_inference,
                rtc_prefix_length=self.rtc_prefix_length if self.rtc else None,
            ),
            policy_type=self.policy_type,
            dit_model=self.dit_model,
            clip=self.clip,
            vla_model=self.vla_model,
            port=self.port,
        )

    def rollout_config(
        self, *, recorder_control: bool = False, direct_episode_keys: bool = False
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
            recorder_control=recorder_control,
        )
