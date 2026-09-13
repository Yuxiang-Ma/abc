from dataclasses import dataclass


@dataclass
class PolicyRolloutConfig:
    """CLI config for the policy rollout process."""

    prompt: str = "throw plastic bottles in bin"
    debug: bool = False
    """Run inference without connecting to or commanding robot followers."""

    # RTC mode settings
    rtc: bool = False
    """Enable Real-Time Action Chunking mode."""
    prefix_length: int = 5
    """Number of actions from previous chunk to condition next chunk on (RTC only)."""
    inference_lead_steps: int = 10
    """Steps before chunk end to start next inference (RTC only)."""

    execute_chunk_dim: int = 50
    """Number of actions to execute per chunk."""

    host: str = "0.0.0.0"
    port: int = 8000

    recorder_control: bool = False
    """Recorder-driven state machine (start/stop/reset via recorder keys)."""

    direct_episode_keys: bool = False
    """Rollout-owned keyboard episode control for runs without a recorder:
    a/b start or stop+home, c/j shut down."""

    compress_images: bool = False

    """JPEG-compress and resize images before sending to reduce latency."""
