"""Adapt the released ABC-DiT inference engine to the deploy protocol."""

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from abc_minimal.config import ClipConfig, DiTConfig
from abc_minimal.policy import DiTInferencePolicy


@dataclass
class PolicyConfig:
    checkpoint_path: str
    norm_stats_path: str | None = None
    prompt: str = "throw plastic bottles in bin"
    diffusion_steps: int = 10
    device: str = "auto"
    deterministic: bool = False
    fast_inference: bool = False
    fast_compile_mode: str = "max-autotune"
    rtc_prefix_length: int | None = None
    camera_height: int = 480
    camera_width: int = 640
    clip: ClipConfig = field(default_factory=ClipConfig)
    model: DiTConfig = field(default_factory=DiTConfig)


class Policy:
    """Expose ABC-DiT through the websocket server's inference contract."""

    def __init__(self, config: PolicyConfig):
        self.config = config
        if config.deterministic:
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        device = (
            "cuda"
            if config.device == "auto" and torch.cuda.is_available()
            else config.device
        )
        if device == "auto":
            device = "cpu"
        self._policy = DiTInferencePolicy(
            Path(config.checkpoint_path).expanduser().resolve(), config, device
        )
        if config.fast_inference:
            self._policy.enable_fast_inference(
                config.fast_compile_mode,
                rtc_prefix_length=config.rtc_prefix_length,
            )
        self.chunk_len = config.model.chunk_length
        self.action_dim = config.model.action_dim

    def infer(
        self,
        obs: dict,
        *,
        noise: np.ndarray | None = None,
        action_prefix: np.ndarray | None = None,
        prefix_length: int | None = None,
        latency: int | None = None,
    ) -> dict:
        del latency  # Training-time action-future conditioning is not used by ABC-DiT.
        if noise is None:
            noise = np.zeros((self.chunk_len, self.action_dim), dtype=np.float32)
        actions = self._policy.infer(
            obs,
            noise=noise,
            action_prefix=action_prefix,
            prefix_length=prefix_length,
        )
        max_action = float(np.abs(actions).max())
        if max_action > 6.3:
            warnings.warn(
                f"Action bounds exceeded: max |action|={max_action:.3f}",
                RuntimeWarning,
                stacklevel=2,
            )
        return {"actions": actions}
