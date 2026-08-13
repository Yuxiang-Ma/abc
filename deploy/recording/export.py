"""Export deploy recordings to the abc_minimal training format.

One CLI for all three recording types (teleop / inference / dagger). Output is
a dataset root whose ``train/`` and ``val/`` directories hold episode dirs in
the exact layout ``abc_minimal.dataloader`` reads:

    <output_dir>/
      train/<episode>/{states_actions.bin, combined_camera-images-rgb.mp4,
                       episode_metadata.json}
      val/            (symlink to train/ in same_as_train mode, or a real split)
      norm_stats.json
      export_summary.json

``states_actions.bin`` is float64 rows of ``[state(14) | action(14)]``; the
combined mp4 stacks cameras along height in ``episode_metadata.json``'s
``cameras`` order at a constant 30 fps.

Modes: dagger recordings default to exporting INTERVENTION segments (the
human corrections); teleop/inference recordings export whole episodes. Both
honor operator discard decisions (``discard_segments`` keeps the checkpoint
sample and drops everything after it; fully-discarded files are skipped).

Norm stats are never silently recomputed: pass ``--norm-stats-from-checkpoint``
(finetuning must reuse the checkpoint's stats) or ``--norm-stats-path``, or
opt in to ``--compute-norm-stats`` for from-scratch experiments.

To use the export for finetuning, point a ``MixtureComponent`` at the
absolute ``train/`` and ``val/`` paths (absolute paths override the cache
root when joined).
"""

from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

import cv2
import h5py
import numpy as np
import tyro

from deploy.recording import io as rio


@dataclass
class ExportConfig:
    input_pattern: str
    """Glob for recording H5 files, e.g. 'data/dagger_h5/*.h5'."""

    output_dir: str
    """Dataset root to write (train/, val/, norm_stats.json)."""

    mode: Literal["auto", "whole_episodes", "interventions"] = "auto"
    """auto: dagger recordings export INTERVENTION segments, others whole episodes."""

    task_name: str = ""
    """Prompt assigned to exported episodes. Default: the H5 task_name attr."""

    modified_after: str | None = None
    """Only include files modified after this local time, e.g. '2026-08-11 14:30'."""

    collection_name: str | None = None
    """Optional filter on the H5 collection_name attr."""

    exclude_files: list[str] = field(default_factory=list)
    """Skip files whose basename contains any of these substrings."""

    state_label: str = "INTERVENTION"
    """Controller state exported in interventions mode."""

    min_segment_frames: int = 30
    """Skip segments shorter than this many synchronized frames
    (must be >= the model chunk length for the episode to be trainable)."""

    context_before_frames: int = 0
    context_after_frames: int = 0
    """Frames to include around each intervention segment (clamped to valid spans)."""

    output_img_height: int = 168
    output_img_width: int = 224
    output_fps: float = 30.0

    cameras: tuple[str, ...] = rio.CAMERA_KEYS
    """Camera stack order in the combined mp4."""

    val_mode: Literal["same_as_train", "split"] = "same_as_train"
    """same_as_train symlinks val/ to train/ (overfit-friendly); split moves episodes."""

    val_ratio: float = 0.05
    seed: int = 123

    norm_stats_from_checkpoint: str | None = None
    """Write the norm_stats embedded in this .pt checkpoint (use for finetuning)."""

    norm_stats_path: str | None = None
    """Copy this existing norm_stats.json."""

    compute_norm_stats: bool = False
    """Opt-in: recompute mean/std from the exported data (from-scratch runs only)."""

    concat_review_mp4: bool = False
    """Also write all_episodes.mp4, an ffmpeg-concat review reel."""

    overwrite: bool = False
    """Remove output_dir first if it already exists."""


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def _parse_modified_after(value: str | None) -> float | None:
    if value is None:
        return None
    value = value.strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%y%m%d-%H%M"):
        try:
            return datetime.strptime(value, fmt).astimezone().timestamp()
        except ValueError:
            pass
    try:
        return datetime.fromisoformat(value).timestamp()
    except ValueError as exc:
        raise ValueError(
            f"Could not parse modified_after={value!r}; use e.g. '2026-08-11 14:30'"
        ) from exc


def select_files(cfg: ExportConfig, summary: dict) -> list[str]:
    paths = sorted(glob.glob(cfg.input_pattern))
    if not paths:
        raise ValueError(f"No H5 files matched {cfg.input_pattern!r}")

    cutoff = _parse_modified_after(cfg.modified_after)
    kept = []
    for path in paths:
        basename = os.path.basename(path)
        if cutoff is not None and os.path.getmtime(path) < cutoff:
            summary["skipped_files"].append(
                {"path": path, "reason": "older than modified_after"}
            )
            continue
        if any(token in basename for token in cfg.exclude_files):
            summary["skipped_files"].append(
                {"path": path, "reason": "matched exclude_files"}
            )
            continue
        if cfg.collection_name is not None:
            with h5py.File(path, "r") as f:
                collection = f.attrs.get("collection_name")
            if collection is None or rio.decode_attr(collection) != cfg.collection_name:
                summary["skipped_files"].append(
                    {"path": path, "reason": "other collection_name"}
                )
                continue
        kept.append(path)
    if not kept:
        raise ValueError("No H5 files left after filtering")
    return kept


# ---------------------------------------------------------------------------
# Norm stats
# ---------------------------------------------------------------------------


def _jsonable_stats(stats) -> dict:
    if isinstance(stats, dict):
        return {key: _jsonable_stats(value) for key, value in stats.items()}
    if isinstance(stats, np.ndarray):
        return stats.tolist()
    if isinstance(stats, (np.floating, np.integer)):
        return stats.item()
    return stats


def resolve_norm_stats(cfg: ExportConfig) -> dict | None:
    """Return the norm-stats payload to write, or None for compute mode."""
    chosen = [
        name
        for name, value in (
            ("norm_stats_from_checkpoint", cfg.norm_stats_from_checkpoint),
            ("norm_stats_path", cfg.norm_stats_path),
            ("compute_norm_stats", cfg.compute_norm_stats),
        )
        if value
    ]
    if len(chosen) != 1:
        raise ValueError(
            "Pick exactly one norm-stats source: --norm-stats-from-checkpoint "
            "<ckpt.pt> (finetuning), --norm-stats-path <json>, or "
            "--compute-norm-stats (from-scratch only)"
        )
    if cfg.norm_stats_path:
        return json.loads(Path(cfg.norm_stats_path).expanduser().read_text())
    if cfg.norm_stats_from_checkpoint:
        import torch

        ckpt = torch.load(
            Path(cfg.norm_stats_from_checkpoint).expanduser(),
            map_location="cpu",
            mmap=True,
            weights_only=False,
        )
        if ckpt.get("norm_stats") is None:
            raise ValueError(
                f"{cfg.norm_stats_from_checkpoint} has no embedded norm_stats"
            )
        return _jsonable_stats(ckpt["norm_stats"])
    return None


class RunningStats:
    """Streaming mean/std/min/max over (N, dim) batches."""

    def __init__(self):
        self.count = 0
        self.mean = None
        self.m2 = None
        self.min = None
        self.max = None

    def update(self, batch: np.ndarray) -> None:
        batch = np.asarray(batch, dtype=np.float64)
        if self.mean is None:
            dim = batch.shape[1]
            self.mean = np.zeros(dim)
            self.m2 = np.zeros(dim)
            self.min = np.full(dim, np.inf)
            self.max = np.full(dim, -np.inf)
        for row in batch:
            self.count += 1
            delta = row - self.mean
            self.mean += delta / self.count
            self.m2 += delta * (row - self.mean)
        self.min = np.minimum(self.min, batch.min(axis=0))
        self.max = np.maximum(self.max, batch.max(axis=0))

    def statistics(self) -> dict:
        std = np.sqrt(self.m2 / max(self.count - 1, 1))
        return {
            "mean": self.mean.tolist(),
            "std": std.tolist(),
            "min": self.min.tolist(),
            "max": self.max.tolist(),
        }


# ---------------------------------------------------------------------------
# Episode writing
# ---------------------------------------------------------------------------


def write_episode(
    f: h5py.File,
    synced: rio.SyncedRecording,
    cfg: ExportConfig,
    episode_dir: Path,
    start: int,
    end: int,
    states: np.ndarray,
    actions: np.ndarray,
    task_name: str,
    provenance: dict,
) -> None:
    episode_dir.mkdir(parents=True, exist_ok=False)
    n_frames = end - start

    rows = np.concatenate([states[start:end], actions[start:end]], axis=1).astype(
        np.float64
    )
    (episode_dir / "states_actions.bin").write_bytes(
        np.ascontiguousarray(rows).tobytes()
    )

    streams = rio.camera_streams(f)
    missing = [cam for cam in cfg.cameras if cam not in streams]
    if missing:
        raise KeyError(
            f"recording is missing cameras {missing}; found {sorted(streams)}"
        )

    size = (cfg.output_img_width, cfg.output_img_height * len(cfg.cameras))
    writer = cv2.VideoWriter(
        str(episode_dir / "combined_camera-images-rgb.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        cfg.output_fps,
        size,
    )
    if not writer.isOpened():
        raise RuntimeError(f"could not open video writer for {episode_dir}")
    try:
        for tick in range(start, end):
            panels = []
            for cam in cfg.cameras:
                key = streams[cam]
                frame = rio.decode_frame(f, key, int(synced.indices[key][tick]))
                panels.append(
                    cv2.resize(
                        frame,
                        (cfg.output_img_width, cfg.output_img_height),
                        interpolation=cv2.INTER_AREA,
                    )
                )
            writer.write(np.concatenate(panels, axis=0))
    finally:
        writer.release()

    metadata = {
        "task_name": task_name,
        "cameras": list(cfg.cameras),
        "usable": True,
        "num_frames": n_frames,
        "fps": cfg.output_fps,
        **provenance,
    }
    (episode_dir / "episode_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def _segment_ranges(
    f: h5py.File,
    synced: rio.SyncedRecording,
    cfg: ExportConfig,
    mode: str,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Return (segment ranges, valid ranges) on the synced timeline."""
    valid = rio.valid_mask(f, synced.base_ts_ns)
    valid_ranges = rio.true_ranges(valid)
    if mode == "whole_episodes":
        return valid_ranges, valid_ranges
    labels = rio.dagger_state_labels(f, synced.base_ts_ns)
    mask = rio.state_mask(labels, cfg.state_label) & valid
    return rio.true_ranges(mask), valid_ranges


def export_recordings(cfg: ExportConfig) -> dict:
    output_root = Path(cfg.output_dir).expanduser().resolve()
    if output_root.exists():
        if not cfg.overwrite:
            raise FileExistsError(f"{output_root} exists; pass --overwrite to replace")
        shutil.rmtree(output_root)
    train_dir = output_root / "train"
    train_dir.mkdir(parents=True)

    norm_stats = resolve_norm_stats(cfg)
    summary: dict = {
        "input_pattern": cfg.input_pattern,
        "mode": cfg.mode,
        "state_label": rio.canonical_state(cfg.state_label),
        "modified_after": cfg.modified_after,
        "collection_name": cfg.collection_name,
        "exported_episodes": [],
        "skipped_segments": [],
        "skipped_files": [],
    }
    state_stats = RunningStats()
    action_stats = RunningStats()

    for h5_path in select_files(cfg, summary):
        with h5py.File(h5_path, "r") as f:
            if rio.is_unusable(f):
                summary["skipped_files"].append(
                    {"path": h5_path, "reason": "recording marked unusable"}
                )
                print(f"[skip] {h5_path}: recording marked unusable")
                continue

            recording_type = rio.detect_recording_type(f)
            mode = cfg.mode
            if mode == "auto":
                mode = (
                    "interventions" if recording_type == "dagger" else "whole_episodes"
                )

            streams = rio.camera_streams(f)
            telemetry = [*rio.STATE_KEYS, *rio.ACTION_KEYS]
            camera_keys = [streams[cam] for cam in cfg.cameras if cam in streams]
            try:
                synced = rio.sync_streams(f, telemetry + camera_keys)
            except (KeyError, ValueError) as exc:
                summary["skipped_files"].append({"path": h5_path, "reason": str(exc)})
                print(f"[skip] {h5_path}: {exc}")
                continue

            states = np.concatenate(
                [synced.gather(f, key) for key in rio.STATE_KEYS], axis=1
            )
            actions = np.concatenate(
                [synced.gather(f, key) for key in rio.ACTION_KEYS], axis=1
            )

            ranges, valid_ranges = _segment_ranges(f, synced, cfg, mode)
            if not ranges:
                summary["skipped_files"].append(
                    {"path": h5_path, "reason": f"no {mode} segments"}
                )
                print(f"[skip] {h5_path}: no {mode} segments")
                continue

            task_name = cfg.task_name or rio.decode_attr(f.attrs.get("task_name", ""))
            basename = Path(h5_path).stem
            for segment_index, (raw_start, raw_end) in enumerate(ranges):
                if mode == "interventions":
                    valid_start, valid_end = rio.containing_range(
                        valid_ranges, raw_start, raw_end
                    )
                    start = max(valid_start, raw_start - cfg.context_before_frames)
                    end = min(valid_end, raw_end + cfg.context_after_frames)
                    episode_id = f"{basename}_intervention_{segment_index:02d}"
                else:
                    start, end = raw_start, raw_end
                    episode_id = (
                        basename
                        if len(ranges) == 1
                        else f"{basename}_valid_{segment_index:02d}"
                    )
                n_frames = end - start
                if n_frames < cfg.min_segment_frames:
                    summary["skipped_segments"].append(
                        {
                            "episode_id": episode_id,
                            "source": h5_path,
                            "frames": n_frames,
                            "reason": f"shorter than min_segment_frames={cfg.min_segment_frames}",
                        }
                    )
                    continue

                provenance = {
                    "source_h5": h5_path,
                    "recording_type": recording_type,
                    "export_mode": mode,
                    "frame_range": [start, end],
                    "avg_sync_hz": synced.avg_hz,
                }
                write_episode(
                    f,
                    synced,
                    cfg,
                    train_dir / episode_id,
                    start,
                    end,
                    states,
                    actions,
                    task_name,
                    provenance,
                )
                state_stats.update(states[start:end])
                action_stats.update(actions[start:end])
                summary["exported_episodes"].append(
                    {
                        "episode_id": episode_id,
                        "source": h5_path,
                        "recording_type": recording_type,
                        "frames": n_frames,
                        "start_time_s": float(
                            (synced.base_ts_ns[start] - synced.base_ts_ns[0]) * 1e-9
                        ),
                    }
                )
                print(f"[export] {episode_id}: {n_frames} frames")

    if not summary["exported_episodes"]:
        (output_root / "export_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )
        raise RuntimeError("No episodes were exported")

    if norm_stats is None:
        print(
            "WARNING: writing norm stats computed from the exported data. "
            "For finetuning, re-export with --norm-stats-from-checkpoint."
        )
        norm_stats = {
            "norm_stats": {
                "state": state_stats.statistics(),
                "actions": action_stats.statistics(),
            }
        }
    (output_root / "norm_stats.json").write_text(
        json.dumps(norm_stats, indent=2) + "\n"
    )

    _write_val(cfg, output_root, train_dir, summary)
    (output_root / "export_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    if cfg.concat_review_mp4:
        _write_review_reel(output_root, summary)

    total = sum(ep["frames"] for ep in summary["exported_episodes"])
    print(
        f"Exported {len(summary['exported_episodes'])} episodes ({total} frames) "
        f"to {output_root}"
    )
    return summary


def _write_val(
    cfg: ExportConfig, output_root: Path, train_dir: Path, summary: dict
) -> None:
    val_dir = output_root / "val"
    if cfg.val_mode == "same_as_train":
        val_dir.symlink_to("train", target_is_directory=True)
        summary["val_mode"] = "same_as_train"
        return
    val_dir.mkdir()
    rng = np.random.default_rng(cfg.seed)
    episode_ids = sorted(ep["episode_id"] for ep in summary["exported_episodes"])
    n_val = max(1, round(len(episode_ids) * cfg.val_ratio))
    val_ids = set(
        rng.choice(np.asarray(episode_ids, dtype=object), size=n_val, replace=False)
    )
    for episode_id in val_ids:
        shutil.move(str(train_dir / episode_id), str(val_dir / episode_id))
    summary["val_mode"] = "split"
    summary["val_episodes"] = sorted(str(v) for v in val_ids)


def _write_review_reel(output_root: Path, summary: dict) -> None:
    import imageio_ffmpeg

    videos = sorted(output_root.glob("train/*/combined_camera-images-rgb.mp4"))
    videos += sorted(output_root.glob("val/*/combined_camera-images-rgb.mp4"))
    if not videos:
        return
    concat_file = output_root / "review_concat.txt"
    concat_file.write_text("".join(f"file {str(path)!r}\n" for path in videos))
    reel = output_root / "all_episodes.mp4"
    subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_file),
            "-c",
            "copy",
            str(reel),
        ],
        check=True,
    )
    summary["review_mp4"] = str(reel)
    print(f"Wrote review reel: {reel}")


def main() -> None:
    export_recordings(tyro.cli(ExportConfig))


if __name__ == "__main__":
    main()
