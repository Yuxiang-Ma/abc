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
    load_pretrained,
    task_name_to_prompt,
)
from abc_minimal.preprocess import load_norm_stats

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
    checkpoint_path = cache_root / "abc_dit_xl_200k_model.pt"
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

    model = DiTPolicy(config.model)
    if resume_ckpt is not None:
        # Resume an existing model run
        model.load_state_dict(resume_ckpt["model"])
        if rank == 0:
            print(f"resuming from {config.resume_from} at step {resume_step}")
    elif config.load_pretrained:
        # New run fine-tuning the released policy. Fresh optimizer at step 0.
        load_pretrained(model, checkpoint_path)
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
            print("DINOv3 bf16 autocast enabled")

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

    norm_stats = load_norm_stats(cache_root / "norm_stats.json")
    if distributed and rank != 0:
        dist.barrier()
    embedder = CLIPTextEmbedder(config.clip, device="cpu")
    if distributed and rank == 0:
        dist.barrier()

    train_loader, train_components = build_train_loader(
        config, components, norm_stats, data_scope, resume_step
    )
    val_loaders, val_components = build_val_loaders(config, components, norm_stats, data_scope)

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
            skipped_val = []
            for name, vl in val_loaders.items():
                err_sum, elem_count = 0.0, 0
                for vb in vl:
                    vb = batch_to_device(vb, device, embedder)
                    with torch.no_grad():
                        pred = module.sample_actions(vb, num_steps=config.flow.num_diffusion_steps)
                        err_sum += F.mse_loss(
                            pred, vb["actions"], reduction="sum"
                        ).item()
                        elem_count += vb["actions"].numel()
                stats = torch.tensor([err_sum, float(elem_count)], device=device)
                if distributed:
                    dist.all_reduce(stats, op=dist.ReduceOp.SUM)
                if stats[1].item() > 0:
                    per_component_recon[name] = (stats[0] / stats[1]).item()
                else:
                    skipped_val.append(name)
            model.train()
            if rank == 0:
                if per_component_recon:
                    parts = "  ".join(f"{n}={v:.4f}" for n, v in per_component_recon.items())
                    avg = sum(per_component_recon.values()) / len(per_component_recon)
                    print(f"step {global_step:6d}  val_recon_error {avg:.4f}  ({parts})")
                    if wandb:
                        log = {"val_recon_error": avg}
                        log.update({f"val_recon_error/{n}": v for n, v in per_component_recon.items()})
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
