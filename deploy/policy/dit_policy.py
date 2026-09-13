"""Adapt the released ABC-DiT inference engine to the deploy protocol."""

from dataclasses import dataclass, field

from abc_minimal.config import ClipConfig, DiTConfig
from abc_minimal.policy import DiTInferencePolicy, InferenceConfig
from deploy.policy.base import DeployPolicy


@dataclass
class PolicyConfig(InferenceConfig):
    clip: ClipConfig = field(default_factory=ClipConfig)
    model: DiTConfig = field(default_factory=DiTConfig)


class Policy(DeployPolicy):
    """Expose ABC-DiT through the websocket server's inference contract."""

    engine_cls = DiTInferencePolicy
