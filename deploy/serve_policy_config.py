"""Pickle-safe policy server configuration."""

from dataclasses import dataclass

from deploy.policy import PolicyConfig


@dataclass
class Args:
    policy: PolicyConfig
    port: int = 8000


def default_args() -> Args:
    """Provide a dataclass instance for tyro without inventing a checkpoint."""
    return Args(policy=PolicyConfig(checkpoint_path=""))
