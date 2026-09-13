"""Pickle-safe policy server configuration."""

from dataclasses import dataclass, field
from typing import Literal

from abc_minimal.config import ClipConfig, DiTConfig, VLAModelConfig
from abc_minimal.policy import (
    InferenceConfig,
    VLAPolicyConfig,
    shared_inference_fields,
)
from deploy.policy import PolicyConfig


@dataclass
class Args:
    policy: InferenceConfig = field(default_factory=InferenceConfig)
    """Checkpoint, prompt, sampler, and runtime knobs shared by both policies."""
    policy_type: Literal["auto", "dit", "vla"] = "auto"
    """"auto" classifies the checkpoint (DiT vs VLA); override to force one."""
    dit_model: DiTConfig = field(default_factory=DiTConfig)
    clip: ClipConfig = field(default_factory=ClipConfig)
    vla_model: VLAModelConfig = field(default_factory=VLAModelConfig)
    port: int = 8000

    def dit_config(self) -> PolicyConfig:
        return PolicyConfig(
            **shared_inference_fields(self.policy),
            clip=self.clip,
            model=self.dit_model,
        )

    def vla_config(self) -> VLAPolicyConfig:
        return VLAPolicyConfig(
            **shared_inference_fields(self.policy), model=self.vla_model
        )
