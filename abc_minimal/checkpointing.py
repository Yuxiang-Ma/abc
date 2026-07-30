"""Training checkpoint format and I/O.

A checkpoint is a dict with keys ``model``, ``optimizer``, ``scheduler``,
``global_step``, ``norm_stats``, ``batch_size``, and ``data_world``. Legacy
checkpoints may lack the optimizer/scheduler/topology keys and may spell the
step ``step`` or ``training_step``; readers tolerate both.
"""

import os
import shutil
from pathlib import Path

import torch


def checkpoint_step(ckpt) -> int:
    for key in ("global_step", "training_step", "step"):
        if key in ckpt:
            return int(ckpt[key])
    return 0


def load_checkpoint(path):
    """Return (checkpoint dict, saved global step)."""
    ckpt = torch.load(Path(path).expanduser(), map_location="cpu", weights_only=False)
    return ckpt, checkpoint_step(ckpt)


def check_topology(ckpt, *, batch_size, data_world, rank) -> None:
    """The step-keyed sampler maps steps to samples via batch size and data
    world, so resuming under a different topology shifts the data stream."""
    if rank != 0:
        return
    for key, current in (("batch_size", batch_size), ("data_world", data_world)):
        saved = ckpt.get(key)
        if saved is not None and int(saved) != int(current):
            print(f"WARNING: resume checkpoint was written with {key}={saved} "
                  f"but this run uses {current}; the resumed data-stream "
                  f"position will not correspond")


def restore_training_state(ckpt, *, optimizer, scheduler, resume_step, rank, source=None) -> None:
    """Restore optimizer/scheduler state, tolerating legacy model-only files."""
    source = source or "resume checkpoint"
    if "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    elif rank == 0:
        print(f"WARNING: {source} has no optimizer state "
              f"(pre-resume-format checkpoint); Adam moments start fresh")
    if "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    else:
        # LambdaLR is a pure function of the step count, so the schedule
        # is recoverable even when the checkpoint predates saved state.
        for _ in range(resume_step):
            scheduler.step()
        if rank == 0 and resume_step:
            print(f"WARNING: no scheduler state in checkpoint; "
                  f"fast-forwarded LR schedule to step {resume_step}")


def save_checkpoint(path, *, module, optimizer, scheduler, global_step, norm_stats,
                    batch_size=None, data_world=None) -> None:
    """Write the checkpoint via tmp file + rename so a crash mid-write can
    never leave a truncated file where resume looks."""
    payload = {
        "model": module.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "global_step": global_step,
        "norm_stats": norm_stats,
    }
    if batch_size is not None:
        payload["batch_size"] = int(batch_size)
    if data_world is not None:
        payload["data_world"] = int(data_world)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def update_last(output_dir: Path, path: Path) -> None:
    """Point last.pt at the newest checkpoint without re-serializing the
    multi-GB payload: hardlink when the filesystem allows it, else copy."""
    tmp = output_dir / "last.pt.tmp"
    tmp.unlink(missing_ok=True)
    try:
        os.link(path, tmp)
    except OSError:
        shutil.copyfile(path, tmp)
    tmp.replace(output_dir / "last.pt")
