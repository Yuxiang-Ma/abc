"""Reusable ABC-DiT inference policy for simulation and real deployment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from abc_minimal.dit import CLIPTextEmbedder, DiTPolicy, load_pretrained
from abc_minimal.preprocess import (
    normalize,
    parse_norm_stats,
    preset_for_backbone,
    resize_pad_normalize,
    unnormalize,
)


def resolve_norm_stats(ckpt: dict[str, Any], override: str | None) -> dict[str, Any]:
    if override:
        raw = json.loads(Path(override).expanduser().read_text())
    elif ckpt.get("norm_stats") is not None:
        raw = ckpt["norm_stats"]
    else:
        raise ValueError("No norm_stats in checkpoint; pass --norm-stats-path")
    return parse_norm_stats(raw)


class DiTInferencePolicy:
    """Checkpoint-backed DiT inference shared by sim and deploy adapters."""

    def __init__(self, checkpoint: Path, config: Any, device: str):
        self.config = config
        self.device = torch.device(device)
        self.diffusion_steps = config.diffusion_steps
        self.model = DiTPolicy(config.model).to(self.device)
        ckpt = load_pretrained(self.model, checkpoint)
        self.model.eval()
        self.norm_preset = preset_for_backbone(config.model.vision_backbone)
        self.norm_stats = resolve_norm_stats(ckpt, config.norm_stats_path)
        self.embedder = CLIPTextEmbedder(config.clip, device=self.device)
        self._prompt = config.prompt
        self.task_vec = self.embedder.encode([self._prompt]).to(self.device)
        self._fast_inference_enabled = False

    def set_prompt(self, prompt: str) -> None:
        if prompt == self._prompt:
            return
        self._prompt = prompt
        self.task_vec = self.embedder.encode([prompt]).to(
            device=self.device, dtype=self.model.x_embedder.weight.dtype
        )

    def enable_fast_inference(
        self,
        compile_mode: str = "max-autotune",
        replay_warmups: int = 24,
        warmup_obs: dict[str, Any] | None = None,
        warmup_noise: np.ndarray | None = None,
        rtc_prefix_length: int | None = None,
    ) -> None:
        """Cast to bf16 and compile the default graph-break-free samplers."""
        if self._fast_inference_enabled:
            return
        if self.device.type != "cuda":
            raise RuntimeError("fast inference requires a CUDA device")

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

        self.model.to(torch.bfloat16)
        self.model.img_backbone.set_bfloat16(True)
        self.task_vec = self.task_vec.to(device=self.device, dtype=torch.bfloat16)

        compile_kwargs: dict[str, Any] = {"dynamic": False, "fullgraph": True}
        if compile_mode:
            compile_kwargs["mode"] = compile_mode
        self.model.sample_actions = torch.compile(
            self.model.sample_actions, **compile_kwargs
        )
        self.model.sample_actions_rtc = torch.compile(
            self.model.sample_actions_rtc, **compile_kwargs
        )
        self._fast_inference_enabled = True

        m = self.config.model
        if warmup_obs is None:
            warmup_obs = {
                "state": np.zeros(m.state_dim, dtype=np.float32),
                "images": {
                    cam: np.zeros(
                        (3, self.config.camera_height, self.config.camera_width),
                        dtype=np.uint8,
                    )
                    for cam in m.camera_keys
                },
                "prompt": self.config.prompt,
            }
        if warmup_noise is None:
            warmup_noise = np.zeros((m.chunk_length, m.action_dim), dtype=np.float32)
        for _ in range(max(1, replay_warmups)):
            self.infer(warmup_obs, noise=warmup_noise)
        torch.cuda.synchronize()
        if rtc_prefix_length is not None:
            self.warmup_rtc(warmup_obs, warmup_noise, rtc_prefix_length)

    def normalized_action_prefix(
        self,
        action_prefix: np.ndarray,
        prefix_length: int,
    ) -> np.ndarray:
        m = self.config.model
        prefix = np.asarray(action_prefix, dtype=np.float32)
        if prefix.shape == (prefix_length, m.action_dim):
            full_prefix = np.zeros((m.chunk_length, m.action_dim), dtype=np.float32)
            full_prefix[:prefix_length] = prefix
            prefix = full_prefix
        if prefix.shape != (m.chunk_length, m.action_dim):
            raise ValueError(
                f"action_prefix must have shape {(m.chunk_length, m.action_dim)} "
                f"or {(prefix_length, m.action_dim)}, got {prefix.shape}"
            )
        return normalize(prefix, self.norm_stats["actions"]).astype(
            np.float32, copy=False
        )

    def warmup_rtc(
        self,
        obs: dict[str, Any],
        noise: np.ndarray | None,
        prefix_length: int,
    ) -> None:
        m = self.config.model
        action_prefix = np.zeros((m.chunk_length, m.action_dim), dtype=np.float32)
        warmup_replays = 8 if self._fast_inference_enabled else 1
        for _ in range(warmup_replays):
            self.infer(
                obs,
                noise=noise,
                action_prefix=action_prefix,
                prefix_length=prefix_length,
            )
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    @torch.no_grad()
    def infer(
        self,
        obs: dict[str, Any],
        noise: np.ndarray | None = None,
        action_prefix: np.ndarray | None = None,
        prefix_length: int | None = 0,
    ) -> np.ndarray:
        self.set_prompt(str(obs.get("prompt", self._prompt)))
        if action_prefix is not None and prefix_length is None:
            prefix_length = len(action_prefix)
        prefix_length = int(prefix_length or 0)
        state = normalize(
            np.asarray(obs["state"], dtype=np.float32), self.norm_stats["state"]
        )
        batch = {
            "state": torch.from_numpy(state[None]).float().to(self.device),
            "actions": torch.zeros(
                1,
                self.config.model.chunk_length,
                self.config.model.action_dim,
                device=self.device,
            ),
            "images": {
                cam: resize_pad_normalize(obs["images"][cam], preset=self.norm_preset)
                .unsqueeze(0)
                .to(self.device)
                for cam in self.config.model.camera_keys
            },
            "task_vec_clip": self.task_vec,
        }
        noise_t = None
        if noise is not None:
            noise_t = torch.from_numpy(noise[None].astype(np.float32)).to(self.device)
        if action_prefix is None:
            actions = self.model.sample_actions(
                batch, num_steps=self.diffusion_steps, noise=noise_t
            )
        else:
            prefix_t = torch.from_numpy(
                self.normalized_action_prefix(action_prefix, prefix_length)[None]
            ).to(device=self.device, dtype=batch["state"].dtype)
            actions = self.model.sample_actions_rtc(
                batch,
                prefix_t,
                prefix_length=prefix_length,
                num_steps=self.diffusion_steps,
                noise=noise_t,
            )
        actions_np = actions[0].float().detach().cpu().numpy()
        return unnormalize(actions_np, self.norm_stats["actions"]).astype(np.float32)
