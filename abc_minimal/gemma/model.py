# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Gemma 3 decoder layers (the text stack the VLA conditions on)."""

from collections.abc import Mapping

import torch
import torch.nn.functional as F
from torch import nn

from . import config as gemma_config

RotaryFrequencies = torch.Tensor | tuple[torch.Tensor, torch.Tensor]


def precompute_freqs_cis(
    dim: int, end: int, theta: float = 10000.0, rope_scaling_factor: int = 1
) -> torch.Tensor:
    """Precomputes the frequency cis."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    freqs = freqs / rope_scaling_factor
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _rotary_parts_from_complex(
    freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    freqs_real_imag = torch.view_as_real(freqs_cis)
    return freqs_real_imag[..., 0], freqs_real_imag[..., 1]


def refresh_rotary_freqs_buffers(module: nn.Module, name: str) -> None:
    freqs_cis = getattr(module, name)
    cos_half, sin_half = _rotary_parts_from_complex(freqs_cis)
    for suffix, value in (("_cos", cos_half), ("_sin", sin_half)):
        buffer_name = f"{name}{suffix}"
        if hasattr(module, buffer_name):
            getattr(module, buffer_name).copy_(value)
        else:
            module.register_buffer(buffer_name, value, persistent=False)


def register_rotary_freqs_buffers(
    module: nn.Module, name: str, freqs_cis: torch.Tensor
) -> None:
    module.register_buffer(name, freqs_cis)
    refresh_rotary_freqs_buffers(module, name)


def select_rotary_freqs(
    module: nn.Module, name: str, input_positions: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    cos_name = f"{name}_cos"
    sin_name = f"{name}_sin"
    if not hasattr(module, cos_name) or not hasattr(module, sin_name):
        refresh_rotary_freqs_buffers(module, name)
    cos_half = getattr(module, cos_name)
    sin_half = getattr(module, sin_name)
    return cos_half.index_select(0, input_positions), sin_half.index_select(
        0, input_positions
    )


def apply_rotary_emb(x: torch.Tensor, freqs_cis: RotaryFrequencies) -> torch.Tensor:
    """Applies rotary embedding using compile-friendly real-valued fp32 math."""
    if isinstance(freqs_cis, tuple):
        cos_half, sin_half = freqs_cis
    else:
        cos_half, sin_half = _rotary_parts_from_complex(freqs_cis)
    cos = torch.cat((cos_half, cos_half), dim=-1).unsqueeze(0).unsqueeze(2)
    sin = torch.cat((sin_half, sin_half), dim=-1).unsqueeze(0).unsqueeze(2)

    x_fp32 = x.float()
    x_out = (x_fp32 * cos) + (_rotate_half(x_fp32) * sin)
    return x_out.type_as(x)


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty((out_features, in_features)), requires_grad=False
        )

    def forward(self, x):
        return F.linear(x, self.weight)


class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty((num_embeddings, embedding_dim)), requires_grad=False
        )

    def forward(self, x):
        return F.embedding(x, self.weight)


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # Llama does x.to(float16) * w whilst Gemma2 is (x * w).to(float16)
        # See https://github.com/huggingface/transformers/pull/29402
        output = self._norm(x.float()) * (1 + self.weight.float())
        return output.type_as(x)


class GemmaMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = Linear(hidden_size, intermediate_size)
        self.up_proj = Linear(hidden_size, intermediate_size)
        self.down_proj = Linear(intermediate_size, hidden_size)

    def forward(self, x):
        gate = self.gate_proj(x)
        gate = F.gelu(gate, approximate="tanh")
        up = self.up_proj(x)
        fuse = gate * up
        outputs = self.down_proj(fuse)
        return outputs


class GemmaAttention(nn.Module):
    def __init__(
        self,
        config: gemma_config.GemmaConfig,
        attn_type: gemma_config.AttentionType,
    ):
        super().__init__()

        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        self.hidden_size = config.hidden_size
        self.head_dim = config.head_dim

        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        self.scaling = self.head_dim**-0.5

        self.qkv_proj = Linear(
            self.hidden_size, (self.num_heads + 2 * self.num_kv_heads) * self.head_dim
        )
        self.o_proj = Linear(self.num_heads * self.head_dim, self.hidden_size)
        self.query_norm = (
            RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            if config.use_qk_norm
            else None
        )
        self.key_norm = (
            RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            if config.use_qk_norm
            else None
        )

        self.attn_type = attn_type
        self.sliding_window_size = config.sliding_window_size

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: torch.Tensor,
        local_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        hidden_states_shape = hidden_states.shape
        assert len(hidden_states_shape) == 3

        batch_size, input_len, _ = hidden_states_shape

        qkv = self.qkv_proj(hidden_states)
        xq, xk, xv = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        xq = xq.view(batch_size, -1, self.num_heads, self.head_dim)
        xk = xk.view(batch_size, -1, self.num_kv_heads, self.head_dim)
        xv = xv.view(batch_size, -1, self.num_kv_heads, self.head_dim)

        if self.query_norm is not None and self.key_norm is not None:
            xq = self.query_norm(xq)
            xk = self.key_norm(xk)

        # Positional embedding.
        xq = apply_rotary_emb(xq, freqs_cis=freqs_cis)
        xk = apply_rotary_emb(xk, freqs_cis=freqs_cis)

        key = xk
        value = xv
        if self.num_kv_heads != self.num_heads:
            # [batch_size, max_seq_len, n_local_heads, head_dim]
            key = torch.repeat_interleave(key, self.num_queries_per_kv, dim=2)
            value = torch.repeat_interleave(value, self.num_queries_per_kv, dim=2)

        # [batch_size, n_local_heads, input_len, head_dim]
        q = xq.transpose(1, 2)
        # [batch_size, n_local_heads, max_seq_len, head_dim]
        k = key.transpose(1, 2)
        v = value.transpose(1, 2)

        if (
            self.attn_type == gemma_config.AttentionType.LOCAL_SLIDING
            and self.sliding_window_size is not None
            and local_mask is not None
        ):
            mask = local_mask

        output = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=self.scaling)
        # [batch_size, input_len, hidden_dim]
        output = output.transpose(1, 2).contiguous().view(batch_size, input_len, -1)
        return self.o_proj(output)


class Gemma3DecoderLayer(nn.Module):
    """One Gemma 3 block: RMSNorms before and after attention and the MLP, local-sliding or global attention."""

    def __init__(
        self,
        config: gemma_config.GemmaConfig,
        attn_type: gemma_config.AttentionType,
    ):
        super().__init__()
        self.attn_type = attn_type
        self.self_attn = GemmaAttention(
            config=config,
            attn_type=self.attn_type,
        )
        self.mlp = GemmaMLP(config.hidden_size, config.intermediate_size)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: torch.Tensor,
        local_mask: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, freqs_cis, mask, local_mask)
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        return residual + hidden_states


class GemmaModel(nn.Module):
    def __init__(self, config: gemma_config.GemmaConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size

        self.layers = nn.ModuleList()
        for i in range(config.num_hidden_layers):
            attn_type = (
                config.attn_types[i % len(config.attn_types)]
                if config.attn_types is not None
                else gemma_config.AttentionType.GLOBAL
            )
            self.layers.append(Gemma3DecoderLayer(config, attn_type))
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cis: Mapping[gemma_config.AttentionType, torch.Tensor],
        mask: torch.Tensor,
        local_mask: torch.Tensor,
        stop_layer_index: int | None = None,
        apply_final_norm: bool = True,
    ) -> torch.Tensor:
        """Run the layers up to and including ``stop_layer_index`` (all when None)."""
        for i, layer in enumerate(self.layers):
            if self.training and getattr(layer, "gradient_checkpointing", False):
                hidden_states = torch.utils.checkpoint.checkpoint(
                    layer,
                    hidden_states,
                    freqs_cis.get(layer.attn_type),
                    mask,
                    local_mask,
                    use_reentrant=False,
                )
            else:
                hidden_states = layer(
                    hidden_states, freqs_cis.get(layer.attn_type), mask, local_mask
                )
            if stop_layer_index is not None and i >= stop_layer_index:
                break
        if apply_final_norm:
            hidden_states = self.norm(hidden_states)
        return hidden_states
