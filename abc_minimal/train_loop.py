"""Train ABC-DiT on the bottles-in-bin dataset.

Builds train/val datasets, runs optimization, validation, and checkpointing.
"""

import os
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from abc_minimal.checkpointing import (
    check_topology,
    load_checkpoint,
    restore_training_state,
    save_checkpoint,
    update_last,
)
from abc_minimal.config import TrainConfig, validate_train_config
from abc_minimal.dataloader import (
    build_train_loader,
    build_val_loaders,
    check_shard_consistency,
    data_parallel_scope,
    read_shard_marker,
)
from abc_minimal.dit import (
    CLIPTextEmbedder,
    DiTPolicy,
    load_clip_vision_weights,
    load_pretrained,
    task_name_to_prompt,
)
from abc_minimal.operator import load_operator_label_maps
from abc_minimal.preprocess import load_norm_stats, parse_norm_stats

# Enable TF32-backed fp32 matmul on NVIDIA GPUs.
torch.set_float32_matmul_precision("high")


def batch_to_device(batch, device, embedder):
    out = {
        "state": batch["state"].to(device, non_blocking=True),
        "actions": batch["actions"].to(device, non_blocking=True),
        "images": {
            cam: v.to(device, non_blocking=True) for cam, v in batch["images"].items()
        },
        "state_is_masked": batch["state_is_masked"].to(device, non_blocking=True),
        "task_vec_clip": embedder.encode(batch["prompt"]).to(device, non_blocking=True),
    }
    return out


def main(config: TrainConfig):
    cache_root = Path(config.cache_root)
    checkpoint_path = cache_root / config.pretrained_ckpt_name
    output_dir = cache_root / "finetune_checkpoints"
    components = validate_train_config(config, cache_root, checkpoint_path)

    distributed = "RANK" in os.environ
    if distributed:
        dist.init_process_group(backend="nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
        local_world = int(os.environ.get("LOCAL_WORLD_SIZE", str(world)))
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        rank, world = 0, 1
        local_rank, local_world = 0, 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_scope = data_parallel_scope(cache_root, rank, world, local_rank, local_world)
    if distributed and data_scope.placement == "node_sharded":
        check_shard_consistency(read_shard_marker(cache_root), world)
    resume_ckpt, resume_step = (None, 0)
    if config.resume_from:
        resume_ckpt, resume_step = load_checkpoint(config.resume_from)
        check_topology(resume_ckpt, batch_size=config.batch_size,
                       data_world=data_scope.world, rank=rank)

    torch.manual_seed(config.seed + rank)
    np.random.seed(config.seed + rank)

    backbone = config.model.vision_backbone
    model = DiTPolicy(config.model)
    ckpt_norm_stats = None
    if resume_ckpt is not None:
        # Resume an existing model run. Resume checkpoints embed the run's
        # norm_stats, so inherit_ckpt_norm_stats works without norm_stats.json.
        model.load_state_dict(resume_ckpt["model"])
        ckpt_norm_stats = resume_ckpt.get("norm_stats")
        if rank == 0:
            print(f"resuming from {config.resume_from} at step {resume_step}")
    elif config.load_pretrained:
        # New run fine-tuning the released policy. Fresh optimizer at step 0.
        ckpt = load_pretrained(model, checkpoint_path)
        ckpt_norm_stats = ckpt.get("norm_stats") if isinstance(ckpt, dict) else None
        if rank == 0:
            print(f"loaded pretrained checkpoint {checkpoint_path}")
    elif backbone == "clip":
        # Rank-0-first so a single rank downloads the ViT-B/16 checkpoint; the
        # others then load it from the local cache (same pattern as the text
        # assets below).
        if distributed and rank != 0:
            dist.barrier()
        missing, unexpected = load_clip_vision_weights(model.img_backbone, config.clip)
        if distributed and rank == 0:
            dist.barrier()
        if rank == 0:
            print(f"loaded CLIP ViT-B/16 vision weights "
                  f"(missing={len(missing)} unexpected={len(unexpected)})")
    else:
        # New run from scratch: only the vision tower starts from pretrained (DINOv3) weights.
        dinov3_ckpt = cache_root / "dinov3_vitb16_pretrain_lvd1689m.pth"
        if dinov3_ckpt.exists():
            sd = torch.load(dinov3_ckpt, map_location="cpu", weights_only=False)
            sd = sd.get("model", sd)
            target = model.img_backbone.dinov3_model
            missing, unexpected = target.load_state_dict(sd, strict=False)
            if rank == 0:
                print(f"loaded DINOv3 from {dinov3_ckpt} "
                      f"(missing={len(missing)} unexpected={len(unexpected)})")
        elif rank == 0:
            print(f"no {dinov3_ckpt}, using random DINOv3")
    if config.dino_bf16:
        model.img_backbone.set_bfloat16(True)
        if rank == 0:
            print(f"{backbone} vision bf16 autocast enabled")

    model = model.to(device)

    vision_params = list(model.img_backbone.parameters())
    vision_ids = {id(p) for p in vision_params}
    main_params = [
        p for p in model.parameters() if p.requires_grad and id(p) not in vision_ids
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": main_params, "lr": config.optim.learning_rate},
            {"params": vision_params,
             "lr": config.optim.learning_rate * config.optim.vision_lr_scale},
        ],
        betas=(config.optim.adam_beta1, config.optim.adam_beta2),
        eps=config.optim.adam_epsilon,
        weight_decay=config.optim.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min((step + 1) / config.optim.lr_warmup_steps, 1.0)
    )
    if resume_ckpt is not None:
        restore_training_state(resume_ckpt, optimizer=optimizer, scheduler=scheduler,
                               resume_step=resume_step, rank=rank,
                               source=config.resume_from)

    if config.compile:
        model = torch.compile(model, fullgraph=True)
        if rank == 0:
            print("torch.compile(fullgraph=True) enabled")

    if distributed:
        model = DDP(
            model,
            device_ids=[device.index],
            output_device=device.index,
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
            static_graph=True,
            bucket_cap_mb=256,
        )
    module = model.module if distributed else model
    if hasattr(module, "_orig_mod"):
        module = module._orig_mod

    if config.inherit_ckpt_norm_stats and ckpt_norm_stats is not None:
        norm_stats = parse_norm_stats(ckpt_norm_stats)
        if rank == 0:
            print("using norm_stats embedded in the pretrained checkpoint")
    else:
        stats_path = cache_root / "norm_stats.json"
        if not stats_path.exists():
            raise FileNotFoundError(
                f"{stats_path} not found and the loaded checkpoint embeds no "
                "norm_stats; provide norm_stats.json or finetune from a "
                "checkpoint that embeds its stats."
            )
        norm_stats = load_norm_stats(stats_path)
    if distributed and rank != 0:
        dist.barrier()
    embedder = CLIPTextEmbedder(config.clip, device="cpu")
    if distributed and rank == 0:
        dist.barrier()

    # Operator prompting needs a label-map manifest built a priori with
    # scripts/build_operator_label_map.py (validate_train_config enforces the path).
    operator_label_maps = {}
    if config.prompt.use_operator_id_as_prompt:
        operator_label_maps = load_operator_label_maps(config.prompt.operator_label_map_path)
        if rank == 0:
            if operator_label_maps:
                print(f"[operator] label maps for {len(operator_label_maps)} tasks "
                      f"({config.prompt.operator_label_map_path})")
            else:
                print("\033[93m[operator] WARNING: operator prompting is enabled but "
                      f"{config.prompt.operator_label_map_path} contains no label maps; "
                      "training continues without operator conditioning\033[0m")
    train_loader, train_components = build_train_loader(
        config, components, norm_stats, data_scope, resume_step, operator_label_maps
    )
    val_loaders, val_components = build_val_loaders(
        config, components, norm_stats, data_scope, operator_label_maps
    )

    wandb = None
    if config.log_wandb and rank == 0:
        try:
            import wandb as _wandb

            wandb = _wandb
            wandb.init(project=config.wandb_project, config=asdict(config))
        except Exception as e:
            print(f"wandb disabled: {e}")

    if data_scope.checkpoint_writer:
        output_dir.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        for c, ds in zip(components, train_components):
            prompts = sorted({task_name_to_prompt(t) for *_, t in ds.episodes})
            print(f"train[{c.train_dir}] weight={c.weight:.4f}: "
                  f"{len(ds.episodes)} episodes, {len(ds)} usable frames, prompts={prompts}")
        for name, ds in val_components:
            print(f"val[{name}]: {len(ds.episodes)} episodes")
        print(
            f"world={world} local_world={local_world} "
            f"data_world={data_scope.world} data_placement={data_scope.placement}"
        )

    model.train()
    global_step = resume_step
    t_last = time.monotonic()
    for batch in train_loader:
        if global_step >= config.train_steps:
            break
        batch = batch_to_device(batch, device, embedder)

        loss = model(
            batch,
            max_action_prefix=config.flow.max_action_prefix,
            prefix_conditioning_prob=config.flow.prefix_conditioning_prob,
            prefix_noise_scale=config.flow.prefix_noise_scale,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.optim.max_grad_norm)
        optimizer.step()
        scheduler.step()
        global_step += 1

        if global_step % config.log_every == 0:
            loss_d = loss.detach()
            if distributed:
                dist.all_reduce(loss_d, op=dist.ReduceOp.AVG)
            if rank == 0:
                dt = time.monotonic() - t_last
                t_last = time.monotonic()
                sps = config.log_every / dt
                lr = scheduler.get_last_lr()[0]
                print(f"step {global_step:6d}  loss {loss_d.item():.4f}  "
                      f"lr {lr:.2e}  gnorm {grad_norm:.3f}  {sps:.2f} it/s")
                if wandb:
                    wandb.log({"loss": loss_d.item(), "lr": lr,
                               "grad_norm": grad_norm.item(),
                               "steps_per_s": sps}, step=global_step)

        if global_step % config.val_every == 0:
            model.eval()
            per_component_recon = {}
            per_component_loss = {}
            skipped_val = []
            for name, vl in val_loaders.items():
                err_sum, elem_count, loss_sum, batch_count = 0.0, 0, 0.0, 0
                for vb in vl:
                    vb = batch_to_device(vb, device, embedder)
                    with torch.no_grad():
                        # val_recon_error: pure-generation MSE over the full chunk
                        # (matches the reference reconstruction_error on infer()).
                        pred = module.sample_actions(vb, num_steps=config.flow.num_diffusion_steps)
                        err_sum += F.mse_loss(
                            pred, vb["actions"], reduction="sum"
                        ).item()
                        elem_count += vb["actions"].numel()
                        # val_loss: the training diffusion loss on val data with no
                        # action prefix (matches the reference val_loss: forward with
                        # max_action_prefix=0, prefix_conditioning_prob=0.0).
                        loss_sum += module(
                            vb, max_action_prefix=0, prefix_conditioning_prob=0.0
                        ).item()
                        batch_count += 1
                stats = torch.tensor(
                    [err_sum, float(elem_count), loss_sum, float(batch_count)], device=device
                )
                if distributed:
                    dist.all_reduce(stats, op=dist.ReduceOp.SUM)
                if stats[1].item() > 0:
                    per_component_recon[name] = (stats[0] / stats[1]).item()
                    per_component_loss[name] = (stats[2] / stats[3]).item()
                else:
                    skipped_val.append(name)
            model.train()
            if rank == 0:
                if per_component_recon:
                    parts = "  ".join(f"{n}={v:.4f}" for n, v in per_component_recon.items())
                    avg = sum(per_component_recon.values()) / len(per_component_recon)
                    avg_loss = sum(per_component_loss.values()) / len(per_component_loss)
                    print(f"step {global_step:6d}  val_recon_error {avg:.4f}  "
                          f"val_loss {avg_loss:.4f}  ({parts})")
                    if wandb:
                        log = {"val_recon_error": avg, "val_loss": avg_loss}
                        log.update({f"val_recon_error/{n}": v for n, v in per_component_recon.items()})
                        log.update({f"val_loss/{n}": v for n, v in per_component_loss.items()})
                        wandb.log(log, step=global_step)
                else:
                    print(f"step {global_step:6d}  val skipped (no full validation batches)")
                if skipped_val:
                    print(f"step {global_step:6d}  val skipped components: {', '.join(skipped_val)}")
            t_last = time.monotonic()

        if global_step % config.ckpt_every == 0 and data_scope.checkpoint_writer:
            path = output_dir / f"{global_step}.pt"
            save_checkpoint(path, module=module, optimizer=optimizer, scheduler=scheduler,
                            global_step=global_step, norm_stats=norm_stats,
                            batch_size=config.batch_size, data_world=data_scope.world)
            update_last(output_dir, path)
            print(f"[rank {rank}] saved {path}")

    if distributed:
        dist.destroy_process_group()
