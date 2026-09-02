"""Reusable ABC-DiT inference policy for simulation and real deployment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from abc_minimal.config import FlowConfig
from abc_minimal.dit import CLIPTextEmbedder, DiTPolicy, load_pretrained
from abc_minimal.preprocess import (
    normalize,
    parse_norm_stats,
    preset_for_backbone,
    resize_pad_normalize,
    resize_pad_normalize_batch,
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


def resolve_trained_max_prefix(ckpt: dict[str, Any]) -> int:
    """The checkpoint's max_action_prefix training bound.

    The bound is EXCLUSIVE: the trainer samples prefix lengths from
    randint(0, max_action_prefix), so the longest prefix actually seen in
    training is max_action_prefix - 1. Production checkpoints store a flat
    train_config dict with max_action_prefix; this repo's trainer saves the
    TrainConfig asdict, which nests it under flow. Fall back to the local
    FlowConfig default when the checkpoint predates either convention.
    """
    train_config = ckpt.get("train_config")
    raw = None
    if isinstance(train_config, dict):
        raw = train_config.get("max_action_prefix")
        if raw is None and isinstance(train_config.get("flow"), dict):
            raw = train_config["flow"].get("max_action_prefix")
    elif train_config is not None:
        # Tolerate non-dict train_configs (dataclass, Namespace, OmegaConf).
        raw = getattr(train_config, "max_action_prefix", None)
        if raw is None:
            flow = getattr(train_config, "flow", None)
            if flow is not None:
                raw = getattr(flow, "max_action_prefix", None)
    if raw is not None:
        return int(raw)
    return FlowConfig().max_action_prefix


class DiTInferencePolicy:
    """Checkpoint-backed DiT inference shared by sim and deploy adapters.

    ``infer`` takes one observation (state ``(S,)``, images ``(3, H, W)``) or a
    batch of worlds (state ``(B, S)``, images ``(B, 3, H, W)`` as numpy arrays
    or CUDA tensors, one prompt or one per world) and returns one action chunk
    per world.
    """

    def __init__(self, checkpoint: Path, config: Any, device: str):
        self.config = config
        self.device = torch.device(device)
        self.diffusion_steps = config.diffusion_steps
        self.model = DiTPolicy(config.model).to(self.device)
        ckpt = load_pretrained(self.model, checkpoint)
        self.model.eval()
        self.norm_preset = preset_for_backbone(config.model.vision_backbone)
        self.norm_stats = resolve_norm_stats(ckpt, config.norm_stats_path)
        self.trained_max_prefix = resolve_trained_max_prefix(ckpt)
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
        if prefix.shape[-2:] == (prefix_length, m.action_dim):
            full_prefix = np.zeros((*prefix.shape[:-2], m.chunk_length, m.action_dim), dtype=np.float32)
            full_prefix[..., :prefix_length, :] = prefix
            prefix = full_prefix
        if prefix.shape[-2:] != (m.chunk_length, m.action_dim):
            raise ValueError(
                f"action_prefix must end in shape {(m.chunk_length, m.action_dim)} "
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
        leading = np.shape(obs["state"])[:-1]
        action_prefix = np.zeros((*leading, m.chunk_length, m.action_dim), dtype=np.float32)
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
        if action_prefix is not None and prefix_length is None:
            prefix_length = np.shape(action_prefix)[-2]
        prefix_length = int(prefix_length or 0)
        m = self.config.model
        state = np.asarray(obs["state"], dtype=np.float32)
        batched = state.ndim == 2
        prompt = obs.get("prompt", self._prompt)
        if batched:
            prompts = [prompt] * len(state) if isinstance(prompt, str) else list(prompt)
            task_vec = self.embedder.encode(prompts).to(
                device=self.device, dtype=self.task_vec.dtype
            )
            images = {
                cam: resize_pad_normalize_batch(
                    torch.as_tensor(obs["images"][cam]).to(self.device), preset=self.norm_preset
                )
                for cam in m.camera_keys
            }
        else:
            self.set_prompt(str(prompt))
            task_vec = self.task_vec
            images = {
                cam: resize_pad_normalize(obs["images"][cam], preset=self.norm_preset)
                .unsqueeze(0)
                .to(self.device)
                for cam in m.camera_keys
            }
        state = normalize(state, self.norm_stats["state"]).reshape(-1, m.state_dim)
        num_worlds = len(state)
        batch = {
            "state": torch.from_numpy(state).to(self.device),
            "actions": torch.zeros(num_worlds, m.chunk_length, m.action_dim, device=self.device),
            "images": images,
            "task_vec_clip": task_vec,
        }
        noise_t = None
        if noise is not None:
            noise_arr = np.asarray(noise, dtype=np.float32).reshape(
                num_worlds, m.chunk_length, m.action_dim
            )
            noise_t = torch.from_numpy(noise_arr).to(self.device)
        if action_prefix is None:
            actions = self.model.sample_actions(
                batch, num_steps=self.diffusion_steps, noise=noise_t
            )
        else:
            prefix = self.normalized_action_prefix(action_prefix, prefix_length).reshape(
                num_worlds, m.chunk_length, m.action_dim
            )
            prefix_t = torch.from_numpy(prefix).to(device=self.device, dtype=batch["state"].dtype)
            actions = self.model.sample_actions_rtc(
                batch,
                prefix_t,
                prefix_length=prefix_length,
                num_steps=self.diffusion_steps,
                noise=noise_t,
            )
        actions_np = actions.float().detach().cpu().numpy()
        actions_np = unnormalize(actions_np, self.norm_stats["actions"]).astype(np.float32)
        return actions_np if batched else actions_np[0]
