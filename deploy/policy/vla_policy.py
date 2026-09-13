"""Adapt ABC-VLA to the deploy protocol.

Mirrors dit_policy.py. Everything specific to the VLA lives in
abc_minimal.policy.VLAInferencePolicy; VLAPolicyConfig is re-exported from
there because sim eval and the policy viewer build it too.
"""

from abc_minimal.policy import VLAInferencePolicy, VLAPolicyConfig
from deploy.policy.base import DeployPolicy

__all__ = ["Policy", "VLAPolicyConfig"]


class Policy(DeployPolicy):
    """Expose the VLA through the websocket server's inference contract."""

    engine_cls = VLAInferencePolicy
