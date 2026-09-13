"""Gemma 3 4B model configuration."""

import dataclasses
import enum
from pathlib import Path

import torch

from .siglip_vision.config import SiglipVisionModelConfig


class AttentionType(enum.Enum):
    GLOBAL = 1
    LOCAL_SLIDING = 2


@dataclasses.dataclass
class GemmaConfig:
    vocab_size: int = 262_144
    max_position_embeddings: int = 8192
    num_hidden_layers: int = 34
    num_attention_heads: int = 8
    num_key_value_heads: int = 4
    hidden_size: int = 2560
    intermediate_size: int = 10240
    head_dim: int = 256
    rms_norm_eps: float = 1e-6
    dtype: str = "bfloat16"
    tokenizer: str = str(
        Path(__file__).resolve().parent
        / "tokenizer"
        / "gemma3_cleaned_262144_v2.spiece.model"
    )
    attn_types: tuple[AttentionType, ...] = (
        AttentionType.LOCAL_SLIDING,
        AttentionType.LOCAL_SLIDING,
        AttentionType.LOCAL_SLIDING,
        AttentionType.LOCAL_SLIDING,
        AttentionType.LOCAL_SLIDING,
        AttentionType.GLOBAL,
    )
    sliding_window_size: int = 1024
    rope_wave_length: dict[AttentionType, int] = dataclasses.field(
        default_factory=lambda: {
            AttentionType.LOCAL_SLIDING: 10_000,
            AttentionType.GLOBAL: 1_000_000,
        }
    )
    use_qk_norm: bool = True
    vision_config: SiglipVisionModelConfig = dataclasses.field(
        default_factory=SiglipVisionModelConfig
    )
    rope_scaling_factor: int = 8

    def get_dtype(self) -> torch.dtype:
        return {
            "float16": torch.float16,
            "float": torch.float32,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }[self.dtype]


def get_config_for_4b(dtype: str = "bfloat16") -> GemmaConfig:
    return GemmaConfig(dtype=dtype)
