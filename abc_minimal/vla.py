"""Gemma 3 diffusion VLA: a Gemma/SigLIP context encoder feeding a DiT action head."""

from __future__ import annotations

import contextlib
import copy
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from abc_minimal.config import VLAModelConfig, validate_vla_model_config
from abc_minimal.gemma import config as gemma_config
from abc_minimal.gemma import model as gemma_model
from abc_minimal.gemma.gemma3_model import Gemma3ForMultimodalLM
from abc_minimal.gemma.preprocessor import tokenize_raw_input

STATE_TOKEN_TAG = "<state>"
ACTION_START_TAG = "<action_start>"


@contextlib.contextmanager
def default_dtype(dtype: torch.dtype):
    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


# --- rectified-flow DiT action head ------------------------------------------

def modulate(x, shift, scale):
    if shift.ndim == 2:
        shift = shift.unsqueeze(1)
        scale = scale.unsqueeze(1)
    return x * (1 + scale) + shift


def gate_residual(gate, residual):
    if gate.ndim == 2:
        gate = gate.unsqueeze(1)
    return gate * residual


def get_1d_sincos_pos_embed(embed_dim: int, length: int) -> np.ndarray:
    if embed_dim % 2:
        raise ValueError("DiT hidden size must be even")
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    out = np.einsum("m,d->md", np.arange(length, dtype=np.float64), omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(10000)
            * torch.arange(half, dtype=torch.float32, device=t.device)
            / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        shape = t.shape
        frequency = self.timestep_embedding(
            t.reshape(-1), self.frequency_embedding_size
        )
        embedded = self.mlp(frequency.to(self.mlp[0].weight.dtype))
        return embedded.reshape(*shape, -1)


class Attention(nn.Module):
    """Multi-head self-attention; parameter layout matches timm's Attention."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        qkv_bias: bool = True,
        **_unused,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=qkv_bias)
        self.proj = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, x):
        batch_size, length, hidden_size = x.shape
        qkv = self.qkv(x).reshape(
            batch_size,
            length,
            3,
            self.num_heads,
            hidden_size // self.num_heads,
        )
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        out = F.scaled_dot_product_attention(q, k, v)
        return self.proj(out.transpose(1, 2).reshape(batch_size, length, hidden_size))


class Mlp(nn.Module):
    """Two-layer MLP; parameter layout matches timm's Mlp."""

    def __init__(
        self,
        hidden_size: int | None = None,
        mlp_hidden_size: int | None = None,
        *,
        in_features: int | None = None,
        hidden_features: int | None = None,
        act_layer=nn.GELU,
        **_unused,
    ):
        super().__init__()
        hidden_size = hidden_size if hidden_size is not None else in_features
        mlp_hidden_size = (
            mlp_hidden_size if mlp_hidden_size is not None else hidden_features
        )
        self.fc1 = nn.Linear(hidden_size, mlp_hidden_size)
        self.act = act_layer()
        self.fc2 = nn.Linear(mlp_hidden_size, hidden_size)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class DiTBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = Mlp(
            hidden_size,
            int(hidden_size * mlp_ratio),
            act_layer=lambda: nn.GELU(approximate="tanh"),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_residual(
            gate_msa, self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        )
        x = x + gate_residual(
            gate_mlp, self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        )
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, action_dim: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6
        )
        self.linear = nn.Linear(hidden_size, action_dim)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        return self.linear(modulate(self.norm_final(x), shift, scale))


class DiffusionActionHead(nn.Module):
    """AdaLN-Zero DiT action head with optional direct state conditioning."""

    def __init__(
        self,
        action_dim: int,
        chunk_length: int,
        vla_cond_dim: int,
        hidden_size: int = 512,
        depth: int = 8,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        state_dim: int = 0,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.chunk_length = chunk_length
        self.hidden_size = hidden_size
        self.uses_vla_cond = vla_cond_dim > 0
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.state_dim = state_dim
        self.state_embedder = (
            nn.Linear(state_dim, hidden_size) if state_dim > 0 else None
        )
        cond_input_dim = hidden_size + vla_cond_dim
        if state_dim > 0:
            cond_input_dim += hidden_size
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_input_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
        )
        self.action_embedder = nn.Linear(action_dim, hidden_size)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, chunk_length, hidden_size), requires_grad=False
        )
        self.blocks = nn.ModuleList(
            DiTBlock(hidden_size, num_heads, mlp_ratio) for _ in range(depth)
        )
        self.final_layer = FinalLayer(hidden_size, action_dim)
        self.initialize_weights()

    def initialize_weights(self):
        def basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(basic_init)
        pos_embed = get_1d_sincos_pos_embed(self.hidden_size, self.chunk_length)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        nn.init.xavier_uniform_(self.action_embedder.weight)
        nn.init.zeros_(self.action_embedder.bias)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        for module in self.cond_proj:
            if isinstance(module, nn.Linear):
                nn.init.normal_(
                    module.weight, mean=0.0, std=1.0 / math.sqrt(module.weight.shape[1])
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        for block in self.blocks:
            nn.init.zeros_(block.adaLN_modulation[-1].weight)
            nn.init.zeros_(block.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.final_layer.linear.weight)
        nn.init.zeros_(self.final_layer.linear.bias)

    def _condition(self, vla_cond, timestep, state):
        parts = [vla_cond, timestep] if self.uses_vla_cond else [timestep]
        if self.state_embedder is not None and state is not None:
            state = state.to(self.state_embedder.weight.dtype)
            parts.insert(len(parts) - 1, self.state_embedder(state))
        if timestep.ndim == 3:
            parts = [
                part.unsqueeze(1).expand(-1, timestep.shape[1], -1)
                if part.ndim == 2
                else part
                for part in parts
            ]
        return self.cond_proj(torch.cat(parts, dim=-1))

    def predict_velocity(self, x_t, t, vla_cond, state=None):
        timestep = self.t_embedder(t)
        c = self._condition(vla_cond, timestep, state)
        z_t = self.action_embedder(x_t) + self.pos_embed[:, : x_t.shape[1]]
        for block in self.blocks:
            z_t = block(z_t, c)
        return self.final_layer(z_t, c)

    def forward(
        self,
        vla_cond,
        actions,
        *,
        state=None,
        noise=None,
        t=None,
        max_action_prefix: int = 0,
        prefix_conditioning_prob: float = 1.0,
        prefix_noise_scale: float = 0.0,
        prefix_lengths=None,
        state_is_masked=None,
    ):
        batch_size, length, action_dim = actions.shape
        noise = torch.randn_like(actions) if noise is None else noise
        t = (
            torch.rand(batch_size, device=actions.device, dtype=actions.dtype)
            if t is None
            else t
        )
        t_expanded = t.view(batch_size, 1, 1)
        if max_action_prefix > 0 or prefix_lengths is not None:
            if prefix_lengths is None:
                apply = (
                    torch.rand(batch_size, device=actions.device)
                    < prefix_conditioning_prob
                )
                if state_is_masked is not None:
                    apply = apply & ~state_is_masked.to(actions.device)
                sampled = torch.randint(
                    0, max_action_prefix, (batch_size,), device=actions.device
                )
                prefix_lengths = torch.where(
                    apply, sampled, torch.zeros_like(sampled)
                )
            prefix_mask = (
                torch.arange(length, device=actions.device)[None, :]
                < prefix_lengths[:, None]
            ).unsqueeze(-1)
            t_per_pos = torch.where(
                prefix_mask, torch.zeros_like(t_expanded), t_expanded
            )
        else:
            prefix_mask = None
            t_per_pos = t_expanded
        x_t = (1 - t_per_pos) * actions + t_per_pos * noise
        if prefix_noise_scale > 0.0 and prefix_mask is not None:
            x_t = x_t + prefix_mask.to(x_t.dtype) * torch.randn_like(x_t) * prefix_noise_scale
        t_cond = t_per_pos.squeeze(-1) if prefix_mask is not None else t
        velocity = self.predict_velocity(x_t, t_cond, vla_cond, state=state)
        target = noise - actions
        per_element = (velocity - target) ** 2
        if prefix_mask is None:
            return per_element.mean()
        postfix = ~prefix_mask
        return per_element.mul(postfix).sum() / (postfix.sum() * action_dim + 1e-8)

    @torch.no_grad()
    def sample(
        self,
        vla_cond,
        *,
        state=None,
        num_steps: int = 10,
        noise=None,
        action_prefix=None,
        prefix_length=None,
    ):
        if noise is None:
            reference = vla_cond if vla_cond is not None else state
            noise = torch.randn(
                reference.shape[0],
                self.chunk_length,
                self.action_dim,
                device=reference.device,
                dtype=reference.dtype,
            )
        x_t = noise
        prefix_mask = None
        if action_prefix is not None:
            if prefix_length is None:
                raise ValueError("prefix_length is required with action_prefix")
            if isinstance(prefix_length, int):
                prefix_length = torch.full(
                    (x_t.shape[0],), prefix_length, device=x_t.device
                )
            prefix_mask = (
                torch.arange(self.chunk_length, device=x_t.device)[None, :]
                < prefix_length[:, None]
            ).unsqueeze(-1)
        dt = -1.0 / num_steps
        for i in range(num_steps):
            if prefix_mask is not None:
                x_t = torch.where(prefix_mask, action_prefix, x_t)
                t = torch.full(
                    (x_t.shape[0], self.chunk_length),
                    1.0 + i * dt,
                    device=x_t.device,
                    dtype=x_t.dtype,
                )
                t = torch.where(prefix_mask.squeeze(-1), torch.zeros_like(t), t)
            else:
                t = torch.full(
                    (x_t.shape[0],),
                    1.0 + i * dt,
                    device=x_t.device,
                    dtype=x_t.dtype,
                )
            x_t = x_t + self.predict_velocity(x_t, t, vla_cond, state=state) * dt
        if prefix_mask is not None:
            x_t = torch.where(prefix_mask, action_prefix, x_t)
        return x_t


# --- VLM-to-DiT conditioning -------------------------------------------------

class ObsAttentionPool(nn.Module):
    """Pool a VLM token sequence into fixed-size AdaLN conditioning."""

    def __init__(
        self,
        vla_hidden_size: int,
        diffusion_hidden_size: int,
        num_pool_tokens: int,
        num_heads: int,
        qk_norm: bool = True,
    ):
        super().__init__()
        self.num_pool_tokens = num_pool_tokens
        self.diffusion_hidden_size = diffusion_hidden_size
        self.qk_norm = qk_norm
        self.num_heads = num_heads
        self.obs_input_norm = nn.LayerNorm(vla_hidden_size)
        self.obs_proj = nn.Linear(vla_hidden_size, diffusion_hidden_size)
        self.pool_query = nn.Parameter(
            torch.randn(1, num_pool_tokens, diffusion_hidden_size) * 0.02
        )
        self.pool_attn = nn.MultiheadAttention(
            embed_dim=diffusion_hidden_size,
            num_heads=num_heads,
            batch_first=True,
        )
        self.pool_norm = nn.LayerNorm(diffusion_hidden_size)
        if qk_norm:
            head_dim = diffusion_hidden_size // num_heads
            self.q_norm = nn.LayerNorm(head_dim, elementwise_affine=False)
            self.k_norm = nn.LayerNorm(head_dim, elementwise_affine=False)
        nn.init.xavier_uniform_(self.obs_proj.weight)
        nn.init.zeros_(self.obs_proj.bias)

    def forward(
        self,
        obs_seq: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = obs_seq.shape[0]
        obs_proj = self.obs_proj(self.obs_input_norm(obs_seq))
        query = self.pool_query.expand(batch_size, -1, -1)
        if self.qk_norm:
            pooled = self._forward_qk_norm(query, obs_proj, key_padding_mask)
        else:
            pooled, _ = self.pool_attn(
                query,
                obs_proj,
                obs_proj,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )
        return self.pool_norm(pooled).flatten(1)

    def _forward_qk_norm(self, query, kv, key_padding_mask):
        batch_size, query_len, hidden_size = query.shape
        kv_len = kv.shape[1]
        head_dim = hidden_size // self.num_heads
        attn = self.pool_attn
        w_q, w_k, w_v = attn.in_proj_weight.chunk(3, dim=0)
        b_q, b_k, b_v = attn.in_proj_bias.chunk(3, dim=0)
        q = F.linear(query, w_q, b_q)
        k = F.linear(kv, w_k, b_k)
        v = F.linear(kv, w_v, b_v)
        q = q.view(batch_size, query_len, self.num_heads, head_dim).transpose(1, 2)
        k = k.view(batch_size, kv_len, self.num_heads, head_dim).transpose(1, 2)
        v = v.view(batch_size, kv_len, self.num_heads, head_dim).transpose(1, 2)
        q = self.q_norm(q)
        k = self.k_norm(k)
        logits = torch.matmul(q, k.transpose(-2, -1)) / head_dim**0.5
        if key_padding_mask is not None:
            logits = logits.masked_fill(
                key_padding_mask[:, None, None, :], float("-inf")
            )
        weights = torch.softmax(logits, dim=-1)
        out = torch.matmul(weights, v)
        out = out.transpose(1, 2).contiguous().view(
            batch_size, query_len, hidden_size
        )
        return attn.out_proj(out)


# --- Gemma/SigLIP backbone + policy ------------------------------------------

class GemmaVLABackbone(nn.Module):
    """Gemma/SigLIP context encoder with numeric state-token injection."""

    def __init__(
        self,
        config,
        *,
        state_dim: int,
        model_config: gemma_config.GemmaConfig | None = None,
        parameter_dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.config = config
        self.model_config = copy.deepcopy(
            model_config or gemma_config.get_config_for_4b()
        )
        self.model_config.dtype = str(parameter_dtype).removeprefix("torch.")
        vision = self.model_config.vision_config
        vision.image_size = config.image_size
        vision.encoding_sequence_length = 64
        dtype = self.model_config.get_dtype()
        with default_dtype(dtype):
            self.gemma_model = Gemma3ForMultimodalLM(self.model_config)
        self.tokenizer = self.gemma_model.tokenizer
        self.state_proj_mlp = nn.Sequential(
            nn.Linear(state_dim, 256, dtype=dtype),
            nn.SiLU(),
            nn.Linear(256, self.model_config.hidden_size, dtype=dtype),
        )
        self.state_proj_id = self.tokenizer.get_unused_id()
        self.tokenizer.register_state_token(self.state_proj_id, tag=STATE_TOKEN_TAG)
        self.action_start_token_id = self.tokenizer.get_unused_id()
        self.tokenizer.register_special_token(
            self.action_start_token_id, ACTION_START_TAG
        )
        self.fixed_seq_len = config.fixed_seq_len
        self.register_buffer(
            "siglip_mean", torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "siglip_std", torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
        )
        self._configure_gradients()
        self._configure_activation_checkpointing()
        if config.load_base_checkpoint:
            self.load_base_checkpoint(config.checkpoint)

    def _configure_gradients(self):
        for parameter in self.gemma_model.parameters():
            parameter.requires_grad = self.config.train_backbone
        self.gemma_model.text_token_embedder.weight.requires_grad = (
            self.config.train_backbone and self.config.train_token_embedding
        )
        for parameter in self.gemma_model.siglip_vision_model.parameters():
            parameter.requires_grad = (
                self.config.train_backbone and self.config.train_siglip
            )
        if not self.config.proprio_token:
            for parameter in self.state_proj_mlp.parameters():
                parameter.requires_grad = False

    def _configure_activation_checkpointing(self):
        enabled = self.config.activation_checkpointing
        for layer in self.gemma_model.model.layers:
            layer.gradient_checkpointing = enabled
        for block in self.gemma_model.siglip_vision_model.encoder_blocks:
            block.gradient_checkpointing = enabled

    def load_base_checkpoint(self, checkpoint: str | None):
        if not checkpoint:
            raise ValueError("Gemma base checkpoint path is required")
        payload = torch.load(
            Path(checkpoint).expanduser(), map_location="cpu", weights_only=False
        )
        state = payload.get("model_state_dict", payload.get("model", payload))
        prefix = "vla.gemma_model."
        if any(key.startswith(prefix) for key in state):
            state = {
                key[len(prefix) :]: value
                for key, value in state.items()
                if key.startswith(prefix)
            }
        position_key = "siglip_vision_model.position_embedding.weight"
        if position_key in state:
            old_position = state[position_key]
            target_count = (
                self.gemma_model.siglip_vision_model.position_embedding.num_embeddings
            )
            if old_position.shape[0] != target_count:
                old_side = int(old_position.shape[0] ** 0.5)
                new_side = int(target_count**0.5)
                position_image = old_position.reshape(
                    1, old_side, old_side, old_position.shape[-1]
                ).permute(0, 3, 1, 2)
                state[position_key] = F.interpolate(
                    position_image,
                    size=(new_side, new_side),
                    mode="bicubic",
                    align_corners=True,
                ).permute(0, 2, 3, 1).reshape(target_count, old_position.shape[-1])
        missing, unexpected = self.gemma_model.load_state_dict(state, strict=False)
        allowed_missing = {"local_freqs_cis", "global_freqs_cis"}
        missing = [key for key in missing if key not in allowed_missing]
        if missing or unexpected:
            raise RuntimeError(
                "Gemma checkpoint does not match the retained 4B path: "
                f"missing={missing[:20]} unexpected={unexpected[:20]}"
            )

    def _normalize_images(self, images: torch.Tensor) -> torch.Tensor:
        batch_size, num_images, channels, height, width = images.shape
        target = self.model_config.vision_config.image_size
        if (height, width) != (target, target):
            images = F.interpolate(
                images.reshape(batch_size * num_images, channels, height, width),
                size=(target, target),
                mode="bilinear",
                align_corners=False,
            ).reshape(batch_size, num_images, channels, target, target)
        return (
            images - self.siglip_mean.to(images.device).unsqueeze(1)
        ) / self.siglip_std.to(images.device).unsqueeze(1)

    def _raw_input(self, images, prompts):
        raw_input = []
        for batch_index, prompt in enumerate(prompts):
            text = prompt.replace("_", " ")
            if self.config.proprio_token:
                text = f"{STATE_TOKEN_TAG} {text} {ACTION_START_TAG}"
            else:
                text = f"{text} {ACTION_START_TAG}"
            raw_input.append([*images[batch_index], text])
        return raw_input

    def _embed_text(self, input_ids):
        embeddings = self.gemma_model.text_token_embedder(input_ids)
        normalizer = torch.tensor(
            self.model_config.hidden_size**0.5,
            dtype=embeddings.dtype,
            device=embeddings.device,
        )
        return embeddings * normalizer

    def _inject_state(self, hidden_states, input_ids, state):
        if not self.config.proprio_token:
            return hidden_states
        state_embedding = self.state_proj_mlp(
            state.to(self.state_proj_mlp[0].weight.dtype)
        )
        placeholder_mask = input_ids == self.state_proj_id
        return torch.where(
            placeholder_mask.unsqueeze(-1),
            state_embedding.unsqueeze(1),
            hidden_states,
        )

    def _encode_images(self, hidden_states, image_batch, input_ids):
        batch_size, num_images, channels, height, width = image_batch.shape
        encoded = self.gemma_model.siglip_vision_model(
            image_batch.reshape(batch_size * num_images, channels, height, width)
        )
        encoded = self.gemma_model.mm_soft_embedding_norm(encoded)
        encoded = self.gemma_model.mm_input_projection(encoded)
        return self.gemma_model.populate_image_embeddings(hidden_states, encoded, input_ids)

    def _attention_masks(self, input_ids, dtype):
        boolean, local_boolean = self.gemma_model.create_attention_mask(
            input_ids, input_ids.shape[1]
        )
        min_value = torch.finfo(dtype).min
        attention = torch.where(boolean, 0.0, min_value).to(dtype)
        local_attention = torch.where(local_boolean, 0.0, min_value).to(dtype)
        return attention, local_attention

    def _freqs(self, positions):
        return {
            gemma_config.AttentionType.LOCAL_SLIDING: gemma_model.select_rotary_freqs(
                self.gemma_model, "local_freqs_cis", positions
            ),
            gemma_config.AttentionType.GLOBAL: gemma_model.select_rotary_freqs(
                self.gemma_model, "global_freqs_cis", positions
            ),
        }

    def prepare(self, batch):
        images = batch["images"]
        device = images.device
        normalized = self._normalize_images(images)
        processed = tokenize_raw_input(
            self.tokenizer,
            self._raw_input(normalized, batch["prompt"]),
            self.model_config,
            device,
        )
        context_ids = processed["user_input_token_ids"]
        context_lengths = processed["prompt_lengths"]
        if max(context_lengths) > self.fixed_seq_len:
            raise ValueError(
                f"multimodal sequence needs {max(context_lengths)} tokens but "
                f"fixed_seq_len={self.fixed_seq_len}"
            )
        input_ids = torch.full(
            (images.shape[0], self.fixed_seq_len),
            self.tokenizer.pad_id,
            dtype=context_ids.dtype,
            device=device,
        )
        input_ids[:, : context_ids.shape[1]] = context_ids
        input_embeds = torch.zeros(
            images.shape[0],
            self.fixed_seq_len,
            self.model_config.hidden_size,
            dtype=next(self.parameters()).dtype,
            device=device,
        )
        context_embeds = self._embed_text(context_ids)
        context_embeds = self._inject_state(
            context_embeds, context_ids, batch["state"]
        )
        context_embeds = self._encode_images(
            context_embeds, normalized.to(input_embeds.dtype), context_ids
        )
        positions = torch.arange(self.fixed_seq_len, device=device)
        lengths = torch.tensor(context_lengths, device=device)
        valid = positions[None, :] < lengths[:, None]
        context_length = context_ids.shape[1]
        input_embeds[:, :context_length] = torch.where(
            valid[:, :context_length, None], context_embeds, 0.0
        )
        attention, local_attention = self._attention_masks(
            input_ids, input_embeds.dtype
        )
        return input_ids, input_embeds, attention, local_attention, positions, ~valid

    def set_feature_layer(self, layer_index: int):
        self.feature_layer = layer_index
        for later_layer in self.gemma_model.model.layers[layer_index + 1 :]:
            for parameter in later_layer.parameters():
                parameter.requires_grad = False
        for parameter in self.gemma_model.model.norm.parameters():
            parameter.requires_grad = False

    def forward(self, batch):
        (
            input_ids,
            input_embeds,
            attention,
            local_attention,
            positions,
            padding_mask,
        ) = self.prepare(batch)
        del input_ids
        hidden = self.gemma_model.model(
            hidden_states=input_embeds,
            freqs_cis=self._freqs(positions),
            mask=attention,
            local_mask=local_attention,
            stop_layer_index=self.feature_layer,
            apply_final_norm=False,
        )
        return hidden, padding_mask


class VLAPolicy(nn.Module):
    """Gemma/SigLIP backbone, attention-pooled conditioning, and a diffusion action head."""

    def __init__(
        self,
        config: VLAModelConfig,
        *,
        gemma_model_config: gemma_config.GemmaConfig | None = None,
        backbone_dtype: torch.dtype = torch.float32,
        backbone_autocast: bool = True,
    ):
        super().__init__()
        errors = validate_vla_model_config(config)
        if gemma_model_config is None and errors:
            raise ValueError("Invalid VLA model config:\n  - " + "\n  - ".join(errors))
        self.config = config
        self.backbone_autocast = backbone_autocast
        self.vla = GemmaVLABackbone(
            config.backbone,
            state_dim=config.dit.state_dim,
            model_config=gemma_model_config,
            parameter_dtype=backbone_dtype,
        )
        num_layers = len(self.vla.gemma_model.model.layers)
        feature_layer = config.backbone.feature_layer
        self.feature_layer = (
            feature_layer if feature_layer >= 0 else num_layers + feature_layer
        )
        if not 0 <= self.feature_layer < num_layers:
            raise ValueError(
                f"feature_layer={feature_layer} is invalid for {num_layers} layers"
            )
        self.vla.set_feature_layer(self.feature_layer)
        dit = config.dit
        self.obs_pool = ObsAttentionPool(
            self.vla.model_config.hidden_size,
            dit.hidden_size,
            dit.num_pool_tokens,
            dit.pool_num_heads,
            dit.pool_qk_norm,
        )
        self.diffusion_head = DiffusionActionHead(
            action_dim=dit.action_dim,
            chunk_length=dit.chunk_length,
            vla_cond_dim=dit.num_pool_tokens * dit.hidden_size,
            hidden_size=dit.hidden_size,
            depth=dit.depth,
            num_heads=dit.num_heads,
            mlp_ratio=dit.mlp_ratio,
            state_dim=dit.state_dim if dit.direct_state_conditioning else 0,
        )

    def encode_condition(self, batch):
        device_type = batch["images"].device.type
        # cache_enabled=False: each weight is cast once per forward anyway; the cache
        # would pin bf16 copies of the whole backbone (6.8 GiB) until the context ends.
        with torch.autocast(
            device_type=device_type,
            dtype=torch.bfloat16,
            enabled=self.backbone_autocast,
            cache_enabled=False,
        ):
            hidden, padding_mask = self.vla(batch)
        with torch.autocast(device_type=device_type, enabled=False):
            return self.obs_pool(hidden.float(), key_padding_mask=padding_mask)

    @staticmethod
    def _flatten_draw_tensor(tensor, batch_size, draws):
        if tensor is None:
            return None
        if tensor.shape[0] == batch_size * draws:
            return tensor
        if tensor.shape[:2] == (batch_size, draws):
            return tensor.flatten(0, 1)
        raise ValueError(
            f"draw tensor has shape {tuple(tensor.shape)}; expected leading "
            f"dimensions {(batch_size, draws)} or {batch_size * draws}"
        )

    def forward(
        self,
        batch,
        *,
        num_diffusion_draws: int = 1,
        max_action_prefix: int = 0,
        prefix_conditioning_prob: float = 1.0,
        prefix_noise_scale: float = 0.0,
        noise=None,
        t=None,
        prefix_lengths=None,
    ):
        if num_diffusion_draws <= 0:
            raise ValueError("num_diffusion_draws must be positive")
        condition = self.encode_condition(batch)
        actions = batch["actions"].float()
        state = (
            batch["state"].float()
            if self.config.dit.direct_state_conditioning
            else None
        )
        state_is_masked = batch.get("state_is_masked")
        batch_size = actions.shape[0]
        if num_diffusion_draws > 1:
            condition = condition.repeat_interleave(num_diffusion_draws, dim=0)
            actions = actions.repeat_interleave(num_diffusion_draws, dim=0)
            state = (
                state.repeat_interleave(num_diffusion_draws, dim=0)
                if state is not None
                else None
            )
            noise = self._flatten_draw_tensor(
                noise, batch_size, num_diffusion_draws
            )
            t = self._flatten_draw_tensor(t, batch_size, num_diffusion_draws)
            prefix_lengths = self._flatten_draw_tensor(
                prefix_lengths, batch_size, num_diffusion_draws
            )
            if state_is_masked is not None:
                state_is_masked = state_is_masked.repeat_interleave(
                    num_diffusion_draws, dim=0
                )
        with torch.autocast(device_type=actions.device.type, enabled=False):
            return self.diffusion_head(
                condition,
                actions,
                state=state,
                noise=noise,
                t=t,
                max_action_prefix=max_action_prefix,
                prefix_conditioning_prob=prefix_conditioning_prob,
                prefix_noise_scale=prefix_noise_scale,
                prefix_lengths=prefix_lengths,
                state_is_masked=state_is_masked,
            )

    @torch.no_grad()
    def sample_actions(
        self,
        batch,
        *,
        num_steps: int = 10,
        noise=None,
        action_prefix=None,
        prefix_length=None,
    ):
        condition = self.encode_condition(batch)
        state = (
            batch["state"].float()
            if self.config.dit.direct_state_conditioning
            else None
        )
        with torch.autocast(device_type=condition.device.type, enabled=False):
            return self.diffusion_head.sample(
                condition,
                state=state,
                num_steps=num_steps,
                noise=noise,
                action_prefix=action_prefix,
                prefix_length=prefix_length,
            )


def inference_model_config(config: VLAModelConfig) -> VLAModelConfig:
    """Architecture-only copy: inference loads released weights over the graph,
    so the trainer's Gemma base checkpoint is neither present nor wanted."""
    out = copy.deepcopy(config)
    out.backbone.load_base_checkpoint = False
    out.backbone.checkpoint = None
    out.backbone.activation_checkpointing = False
    return out


def stack_camera_batch(batch: dict, camera_keys: tuple[str, ...]) -> dict:
    """Stack the dataloader's per-camera image dict into a (B, n_cam, C, H, W) tensor."""
    if isinstance(batch["images"], dict):
        images = torch.stack([batch["images"][key] for key in camera_keys], dim=1)
    else:
        images = batch["images"]
    out = dict(batch)
    out["images"] = images
    return out
