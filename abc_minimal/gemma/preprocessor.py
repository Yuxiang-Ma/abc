"""Tensor-only multimodal tokenization (interleaved images and text)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from .config import GemmaConfig
from .tokenizer import Tokenizer


def tokenize_raw_input(
    tokenizer: Tokenizer,
    raw_input: Sequence[Sequence[str | torch.Tensor]],
    config: GemmaConfig,
    device: torch.device,
) -> dict[str, Any]:
    """Token ids for interleaved image tensors and text; images only contribute placeholders."""
    all_token_ids: list[list[int]] = []
    prompt_lengths: list[int] = []

    for prompt in raw_input:
        token_ids = [tokenizer.bos_id]
        for element in prompt:
            if isinstance(element, str):
                token_ids.extend(tokenizer.encode(element, bos=False, eos=False))
            elif isinstance(element, torch.Tensor):
                token_ids.extend(tokenizer.encode("\n\n", bos=False, eos=False))
                token_ids.append(tokenizer.boi_id)
                token_ids.extend(
                    [tokenizer.image_token_placeholder_id]
                    * config.vision_config.encoding_sequence_length
                )
                token_ids.append(tokenizer.eoi_id)
                token_ids.extend(tokenizer.encode("\n\n", bos=False, eos=False))
            else:
                raise TypeError(f"unsupported multimodal element {type(element)!r}")
        all_token_ids.append(token_ids)
        prompt_lengths.append(len(token_ids))

    # One padded id matrix built on the host and moved once (no per-sample copies).
    input_ids = torch.full((len(raw_input), max(prompt_lengths)), tokenizer.pad_id, dtype=torch.long)
    for i, token_ids in enumerate(all_token_ids):
        input_ids[i, : len(token_ids)] = torch.tensor(token_ids, dtype=torch.long)
    return {
        "user_input_token_ids": input_ids.to(device, non_blocking=True),
        "prompt_lengths": prompt_lengths,
    }
