"""Training-capable Gemma 3 multimodal backbone (text + SigLIP vision)."""

from __future__ import annotations

import torch
from torch import nn

from . import config as gemma_config
from . import model as gemma_model
from . import tokenizer
from .siglip_vision.siglip_vision_model import SiglipVisionModel


class Gemma3ForMultimodalLM(nn.Module):
    """Gemma text stack plus SigLIP, without language-generation machinery."""

    def __init__(self, config: gemma_config.GemmaConfig):
        super().__init__()
        self.dtype = config.get_dtype()
        self.config = config
        self.tokenizer = tokenizer.Tokenizer(config.tokenizer)
        self.text_token_embedder = gemma_model.Embedding(config.vocab_size, config.hidden_size)
        self.model = gemma_model.GemmaModel(config)
        self.siglip_vision_model = SiglipVisionModel(config.vision_config)
        self.mm_soft_embedding_norm = gemma_model.RMSNorm(
            config.vision_config.embedding_dim, eps=config.rms_norm_eps
        )
        self.mm_input_projection = gemma_model.Linear(
            config.vision_config.embedding_dim, config.hidden_size
        )

        defaults = {
            gemma_config.AttentionType.LOCAL_SLIDING: 10_000,
            gemma_config.AttentionType.GLOBAL: 10_000,
        }
        self._register_freqs_cis(
            "local_freqs_cis",
            config.head_dim,
            config.max_position_embeddings,
            theta=config.rope_wave_length.get(
                gemma_config.AttentionType.LOCAL_SLIDING,
                defaults[gemma_config.AttentionType.LOCAL_SLIDING],
            ),
        )
        self._register_freqs_cis(
            "global_freqs_cis",
            config.head_dim,
            config.max_position_embeddings,
            theta=config.rope_wave_length.get(
                gemma_config.AttentionType.GLOBAL,
                defaults[gemma_config.AttentionType.GLOBAL],
            ),
            rope_scaling_factor=config.rope_scaling_factor,
        )
        self.register_load_state_dict_post_hook(self._refresh_rotary_freqs)

    def _register_freqs_cis(
        self,
        name: str,
        head_dim: int,
        max_seq_len: int,
        theta: int,
        rope_scaling_factor: int = 1,
    ) -> None:
        gemma_model.register_rotary_freqs_buffers(
            self,
            name,
            gemma_model.precompute_freqs_cis(
                head_dim,
                max_seq_len * 2,
                theta=theta,
                rope_scaling_factor=rope_scaling_factor,
            ),
        )

    def _refresh_rotary_freqs(self, module, incompatible_keys) -> None:
        del module, incompatible_keys
        for name in ("local_freqs_cis", "global_freqs_cis"):
            gemma_model.refresh_rotary_freqs_buffers(self, name)

    def populate_image_embeddings(
        self,
        hidden_states: torch.Tensor,
        image_embeddings: torch.Tensor,
        input_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Scatter the image tokens over the placeholder positions, in order, without a host sync."""
        mask = (input_token_ids == self.tokenizer.image_token_placeholder_id).unsqueeze(-1)
        return hidden_states.masked_scatter(mask, image_embeddings.to(hidden_states.dtype))

    def create_attention_mask(
        self, input_ids: torch.Tensor, sequence_length: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = input_ids.shape[0]
        causal_mask = torch.tril(
            torch.ones(
                batch_size,
                1,
                sequence_length,
                sequence_length,
                dtype=torch.bool,
                device=input_ids.device,
            )
        )
        image_token_mask = input_ids == self.tokenizer.image_token_placeholder_id
        padded_mask = nn.functional.pad(image_token_mask, (1, 0), value=0)
        boundary = padded_mask[:, 1:] > padded_mask[:, :-1]
        numbered_boundary = torch.cumsum(boundary, dim=-1)
        block_indices = image_token_mask * numbered_boundary
        bidirectional_mask = torch.logical_and(
            block_indices[:, None, :] == block_indices.unsqueeze(-1),
            block_indices.unsqueeze(-1) > 0,
        )
        attention_mask = torch.logical_or(
            causal_mask, bidirectional_mask.unsqueeze(1)
        )
        local_window = torch.triu(
            torch.ones(
                1,
                1,
                sequence_length,
                sequence_length,
                dtype=torch.bool,
                device=input_ids.device,
            ),
            diagonal=-(self.config.sliding_window_size - 1),
        )
        return attention_mask, torch.logical_and(attention_mask, local_window)
