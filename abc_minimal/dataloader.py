"""Everything between episodes on disk and training batches.

Sections, in order: data-parallel placement (which ranks read which data),
the on-disk episode format, map-style datasets with the step-keyed sampler,
and the DataLoader builders used by train_loop.
"""

import json
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, Sampler

from abc_minimal.config import DiTConfig
from abc_minimal.dit import task_name_to_prompt
from abc_minimal.preprocess import augment_and_normalize, normalize


# --- data-parallel placement ---------------------------------------------------

@dataclass(frozen=True)
class DataParallelScope:
    rank: int
    world: int
    placement: str
    checkpoint_writer: bool

def read_shard_marker(cache_root: Path):
    marker = cache_root / "hf_status" / "shard.json"
    if not marker.exists():
        return None
    try:
        data = json.loads(marker.read_text())
    except json.JSONDecodeError:
        return None
    return data if data.get("data_placement") == "node_sharded" else None

def cache_is_node_sharded(cache_root: Path) -> bool:
    return read_shard_marker(cache_root) is not None

def check_shard_consistency(marker, world):
    """Every node must have sharded the same dataset snapshot the same way.
    Nodes cannot see each other's disks, so compare markers over the
    process group and fail fast instead of training on overlapping shards."""
    shared_keys = ("repo_id", "revision", "num_nodes", "tasks", "splits",
                   "train_dir", "val_dir", "global_manifest_sha")
    payload = {"shared": {k: marker.get(k) for k in shared_keys},
               "node_rank": marker.get("node_rank")}
    gathered = [None] * world
    dist.all_gather_object(gathered, payload)
    ref = gathered[0]["shared"]
    problems = []
    mismatched = sorted({g["node_rank"] for g in gathered if g["shared"] != ref})
    if mismatched:
        problems.append(f"nodes {mismatched} disagree with node "
                        f"{gathered[0]['node_rank']} on the shard config {ref}")
    num_nodes = ref.get("num_nodes")
    node_ranks = sorted({g["node_rank"] for g in gathered})
    if isinstance(num_nodes, int) and node_ranks != list(range(num_nodes)):
        problems.append(f"markers say num_nodes={num_nodes} but node ranks "
                        f"present are {node_ranks}")
    if problems:
        raise RuntimeError("inconsistent node-sharded caches: " + "; ".join(problems))

def data_parallel_scope(cache_root: Path, rank, world, local_rank, local_world):
    if cache_is_node_sharded(cache_root):
        return DataParallelScope(
            rank=local_rank,
            world=local_world,
            placement="node_sharded",
            checkpoint_writer=local_rank == 0,
        )
    return DataParallelScope(
        rank=rank,
        world=world,
        placement="shared",
        checkpoint_writer=rank == 0,
    )


# --- on-disk episode format ----------------------------------------------------

def scan_episodes(data_dir, default_task_name, model_config: DiTConfig):
    """Return episode metadata needed for frame sampling and video splitting."""
    episodes = []
    row_width = model_config.state_dim + model_config.action_dim
    for ep_dir in sorted(Path(data_dir).iterdir()):
        bin_path = ep_dir / "states_actions.bin"
        if not bin_path.exists():
            continue
        length = bin_path.stat().st_size // (row_width * 8)
        usable = length - (model_config.chunk_length - 1)
        if usable <= 0:
            continue
        meta = {}
        if (ep_dir / "episode_metadata.json").exists():
            meta = json.loads((ep_dir / "episode_metadata.json").read_text())
        cams = meta.get("cameras") or model_config.camera_keys
        task_name = meta.get("task_name") or default_task_name
        episodes.append((ep_dir, length, usable, tuple(cams), task_name))
    return episodes

def read_state_action_rows(ep_dir, start, end, model_config: DiTConfig):
    row_width = model_config.state_dim + model_config.action_dim
    row_bytes = row_width * 8
    with open(ep_dir / "states_actions.bin", "rb") as f:
        f.seek(start * row_bytes)
        raw = f.read((end - start) * row_bytes)
    return np.frombuffer(raw, dtype=np.float64).reshape(-1, row_width)

def decode_frame(ep_dir, idx, episode_length, source_cameras, camera_keys):
    """Decode combined-video frame idx via torchcodec with a synthesized CFR
    frame map (pts = 512*k, 1/15360 timebase), without per-file probing.

    `source_cameras` is the actual stack order in combined mp4. Stereo episodes
    deterministically alias one top eye to `top`, matching production export.
    """
    import hashlib
    from torchcodec.decoders import VideoDecoder

    frames = [
        {"pts": 512 * i, "duration": 512, "key_frame": 1 if i % 30 == 0 else 0}
        for i in range(episode_length)
    ]
    mapping = json.dumps({"frames": frames})
    decoder = VideoDecoder(
        str(ep_dir / "combined_camera-images-rgb.mp4"), custom_frame_mappings=mapping
    )
    frame = decoder[idx]  # (C, n_cams * H, W) uint8
    n_cams = len(source_cameras)
    h = frame.shape[1] // n_cams
    cams_out = {
        name: frame[:, i * h : (i + 1) * h, :].float() / 255.0
        for i, name in enumerate(source_cameras)
    }
    if "top" not in cams_out and "top_left" in cams_out and "top_right" in cams_out:
        digest = hashlib.sha1(ep_dir.name.encode("utf-8")).digest()[0]
        cams_out["top"] = cams_out["top_left" if digest % 2 == 0 else "top_right"]
    return {cam: cams_out[cam] for cam in camera_keys}


# --- datasets and sampling -----------------------------------------------------

class EpisodeDataset(Dataset):
    """Map-style dataset over all usable (episode, frame) pairs."""

    def __init__(
        self,
        data_dir,
        norm_stats,
        train,
        default_task_name,
        mask_state_ratio,
        model_config: DiTConfig,
    ):
        self.episodes = scan_episodes(data_dir, default_task_name, model_config)
        if not self.episodes:
            raise ValueError(f"no episodes found in {data_dir}")
        self.model_config = model_config
        self.camera_keys = tuple(model_config.camera_keys)
        self.norm_stats = norm_stats
        self.train = train
        self.mask_state_ratio = mask_state_ratio
        self.cum = np.cumsum([usable for _, _, usable, _, _ in self.episodes])

    def __len__(self):
        return int(self.cum[-1])

    def sample(self, rng):
        global_idx = int(rng.integers(0, int(self.cum[-1])))
        return self[global_idx]

    def __getitem__(self, global_idx):
        ep_idx = int(np.searchsorted(self.cum, global_idx, side="right"))
        k = int(global_idx - (self.cum[ep_idx - 1] if ep_idx > 0 else 0))
        ep_dir, length, _, source_cameras, task_name = self.episodes[ep_idx]

        rows = read_state_action_rows(
            ep_dir, k, k + self.model_config.chunk_length, self.model_config
        )
        state = normalize(rows[0, : self.model_config.state_dim], self.norm_stats["state"])
        state = state.astype(np.float32)
        actions = normalize(rows[:, self.model_config.state_dim :], self.norm_stats["actions"])
        actions = actions.astype(np.float32)

        state_is_masked = bool(self.train and torch.rand(1).item() < self.mask_state_ratio)
        if state_is_masked:
            state = np.zeros_like(state)

        images = augment_and_normalize(
            decode_frame(ep_dir, k, length, source_cameras, self.camera_keys), self.train
        )
        return {
            "state": torch.from_numpy(state),
            "actions": torch.from_numpy(actions),
            "images": images,
            "state_is_masked": state_is_masked,
            "prompt": task_name_to_prompt(task_name),
        }

class MixtureDataset(Dataset):
    """Train-time mixture: each draw picks a component by `weights`, then a
    uniform usable-frame sample within that component.

    Draws are keyed by (seed, index): different seeds give different data
    streams, while a fixed seed keeps the stream deterministic for resume.
    """

    def __init__(self, components, weights, length, seed=0):
        self.components = list(components)
        self.weights = np.asarray(weights, dtype=np.float64)
        self.length = int(length)
        self.seed = int(seed)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        rng = np.random.default_rng((self.seed, idx))
        comp_idx = int(rng.choice(len(self.components), p=self.weights))
        return self.components[comp_idx].sample(rng)

class GlobalStepSampler(Sampler):
    """Map completed training steps to a deterministic per-rank sample stream."""

    def __init__(self, start_step, stop_step, batch_size, data_rank, data_world):
        self.start = int(start_step) * int(batch_size)
        self.count = max(0, (int(stop_step) - int(start_step)) * int(batch_size))
        self.data_rank = int(data_rank)
        self.data_world = int(data_world)

    def __iter__(self):
        for sample_idx in range(self.start, self.start + self.count):
            yield sample_idx * self.data_world + self.data_rank

    def __len__(self):
        return self.count

def collate(samples, camera_keys):
    return {
        "state": torch.stack([s["state"] for s in samples]),
        "actions": torch.stack([s["actions"] for s in samples]),
        "images": {
            cam: torch.stack([s["images"][cam] for s in samples]) for cam in camera_keys
        },
        "state_is_masked": torch.tensor([s["state_is_masked"] for s in samples]),
        "prompt": [s["prompt"] for s in samples],
    }


# --- loader construction -------------------------------------------------------

def build_train_loader(config, components, norm_stats, data_scope, resume_step):
    """Build the train mixture and its step-keyed loader.

    Returns the loader plus the per-component datasets for logging."""
    train_components = [
        EpisodeDataset(Path(config.cache_root) / c.train_dir, norm_stats, train=True,
                       default_task_name=c.task_name,
                       mask_state_ratio=config.flow.mask_state_ratio,
                       model_config=config.model)
        for c in components
    ]
    component_weights = [c.weight for c in components]
    mixture_length = sum(len(d) for d in train_components)
    train_ds = MixtureDataset(
        train_components, component_weights, mixture_length, seed=config.seed
    )
    train_sampler = GlobalStepSampler(
        start_step=resume_step,
        stop_step=config.train_steps,
        batch_size=config.batch_size,
        data_rank=data_scope.rank,
        data_world=data_scope.world,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        sampler=train_sampler,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=partial(collate, camera_keys=config.model.camera_keys),
        pin_memory=True,
        drop_last=True,
        persistent_workers=config.num_workers > 0,
    )
    return train_loader, train_components

def build_val_loaders(config, components, norm_stats, data_scope):
    """Build per-component val loaders striding this rank's slice.

    Returns the loaders plus (name, dataset) pairs for logging."""
    val_components = [
        (c.val_dir, EpisodeDataset(Path(config.cache_root) / c.val_dir, norm_stats,
                                   train=False,
                                   default_task_name=c.task_name,
                                   mask_state_ratio=config.flow.mask_state_ratio,
                                   model_config=config.model))
        for c in components
    ]
    val_loaders = {}
    for name, val_ds in val_components:
        val_indices = list(
            range(
                data_scope.rank,
                min(len(val_ds), config.val_batches * config.batch_size * data_scope.world),
                data_scope.world,
            )
        )
        val_loaders[name] = DataLoader(
            torch.utils.data.Subset(val_ds, val_indices),
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=2,
            collate_fn=partial(collate, camera_keys=config.model.camera_keys),
            drop_last=True,
        )
    return val_loaders, val_components
