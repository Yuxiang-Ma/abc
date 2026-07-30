"""Hugging Face node-sharded dataset preparation.

This module is intentionally separate from the training dataloader. It
downloads this node's deterministic shard of ABC-130k MCAPs and converts them
into the local directory layout that ``abc_minimal.train_loop`` already reads.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from multiprocessing import Pool
from pathlib import Path
from typing import Annotated, Literal

import tyro
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import GatedRepoError

from abc_minimal.config import default_cache_root


REPO = "XDOF/ABC-130k"


@dataclass
class HfShardPrepareConfig:
    """Download and convert this node's deterministic Hugging Face shard."""

    tasks: Annotated[
        list[str],
        tyro.conf.arg(help="HF task folder(s), e.g. organize_the_condiment_bottles."),
    ] = field(default_factory=list)
    split: Annotated[
        Literal["train", "val", "all"],
        tyro.conf.arg(help="Dataset split to prepare."),
    ] = "all"
    cache: Annotated[
        Path,
        tyro.conf.arg(help="Local cache root; defaults to ABC_CACHE or ./cache."),
    ] = field(default_factory=default_cache_root)
    repo_id: Annotated[str, tyro.conf.arg(help="HF dataset repo id.")] = REPO
    revision: Annotated[str, tyro.conf.arg(help="HF revision.")] = "main"
    hf_token: Annotated[
        str | None,
        tyro.conf.arg(help="HF token; defaults to HF_TOKEN or the "
                           "huggingface-cli login token."),
    ] = None
    num_nodes: Annotated[
        int,
        tyro.conf.arg(help="Total number of nodes participating in this data shard."),
    ] = int(os.environ.get("NUM_NODES", os.environ.get("NNODES", "1")))
    node_rank: Annotated[
        int,
        tyro.conf.arg(help="This node's rank in [0, num_nodes)."),
    ] = int(os.environ.get("NODE_RANK", os.environ.get("GROUP_RANK", "0")))
    workers: Annotated[
        int,
        tyro.conf.arg(help="Parallel downloads and conversion worker count."),
    ] = 4
    train_dir: Annotated[
        str,
        tyro.conf.arg(help="Converted train output directory under cache."),
    ] = "train_real"
    val_dir: Annotated[
        str,
        tyro.conf.arg(help="Converted val output directory under cache."),
    ] = "val_real"
    max_episodes_per_split: Annotated[
        int | None,
        tyro.conf.arg(help="Optional global per-task/per-split cap for smoke tests."),
    ] = None
    keep_mcaps: Annotated[
        bool,
        tyro.conf.arg(help="Keep raw MCAPs after successful conversion."),
    ] = False
    refresh_manifest: Annotated[
        bool,
        tyro.conf.arg(help="Re-list HF even if a local manifest exists."),
    ] = False
    prune: Annotated[
        bool,
        tyro.conf.arg(help="Remove episodes this tool converted earlier that are no "
                           "longer assigned to this node (e.g. after --num-nodes "
                           "changed). Never touches episodes from other sources."),
    ] = True


def _resolve_revision(cfg: HfShardPrepareConfig, api: HfApi) -> str:
    """Pin a symbolic revision (e.g. "main") to its commit sha so manifests,
    downloads, and the shard marker all reference one immutable snapshot."""
    rev = cfg.revision
    if len(rev) == 40 and all(c in "0123456789abcdef" for c in rev.lower()):
        return rev
    try:
        return api.dataset_info(cfg.repo_id, revision=rev).sha
    except GatedRepoError as e:
        raise RuntimeError(
            "HF access denied. Accept XDOF/ABC-130k access and run "
            "huggingface-cli login, set HF_TOKEN, or pass --hf-token."
        ) from e


def _safe_name(value: str) -> str:
    return value.replace("/", "--")


def _fmt_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def _manifest_path(cfg: HfShardPrepareConfig, task: str, split: str) -> Path:
    return (
        cfg.cache.expanduser()
        / "hf_manifests"
        / _safe_name(cfg.repo_id)
        / _safe_name(cfg.revision)
        / split
        / f"{task}.json"
    )


def _raw_root(cfg: HfShardPrepareConfig) -> Path:
    return (
        cfg.cache.expanduser()
        / "hf_raw"
        / _safe_name(cfg.repo_id)
        / _safe_name(cfg.revision)
    )


def _output_root(cfg: HfShardPrepareConfig, split: str) -> Path:
    return cfg.cache.expanduser() / (cfg.train_dir if split == "train" else cfg.val_dir)


def _status_root(cfg: HfShardPrepareConfig) -> Path:
    return cfg.cache.expanduser() / "hf_status"


def _ledger_path(cfg: HfShardPrepareConfig, split: str, episode_id: str) -> Path:
    return _status_root(cfg) / "episodes" / split / f"{episode_id}.json"


def _read_ledger(cfg: HfShardPrepareConfig, split: str, episode_id: str) -> dict | None:
    path = _ledger_path(cfg, split, episode_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def _write_ledger(cfg: HfShardPrepareConfig, item: dict, state: str, **extra) -> None:
    _write_json(
        _ledger_path(cfg, item["split"], item["episode_id"]),
        {"state": state, "task": item["task"], "split": item["split"],
         "episode_id": item["episode_id"], **extra},
    )


def _shard_owner(path: str, num_nodes: int) -> int:
    digest = hashlib.sha1(path.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % num_nodes


def _local_mcap_path(cfg: HfShardPrepareConfig, item: dict) -> Path:
    # hf_hub_download(local_dir=...) preserves the repo-relative path.
    return _raw_root(cfg) / item["path"]


def _converted_ready(out_root: Path, episode_id: str) -> bool:
    ep = out_root / episode_id
    required = [
        ep / "states_actions.bin",
        ep / "combined_camera-images-rgb.mp4",
        ep / "episode_metadata.json",
    ]
    return all(p.exists() and p.stat().st_size > 0 for p in required)


def _split_names(split: str) -> list[str]:
    return ["train", "val"] if split == "all" else [split]


def list_mcaps(
    cfg: HfShardPrepareConfig, api: HfApi, task: str, split: str
) -> list[dict]:
    manifest = _manifest_path(cfg, task, split)
    if manifest.exists() and not cfg.refresh_manifest:
        data = json.loads(manifest.read_text())
        files = data["files"]
    else:
        files = []
        tree = api.list_repo_tree(
            cfg.repo_id, repo_type="dataset", revision=cfg.revision,
            path_in_repo=f"data/{split}/{task}", recursive=True,
        )
        for entry in tree:
            size = getattr(entry, "size", None)  # RepoFolder entries have no size
            if size is None or not entry.path.endswith("/episode.mcap"):
                continue
            parts = Path(entry.path).parts
            if len(parts) < 5:
                continue
            files.append(
                {
                    "path": entry.path,
                    "size": int(size or 0),
                    "split": split,
                    "task": task,
                    "episode_id": parts[-2],
                    "oid": getattr(entry, "blob_id", None),
                }
            )
        files = sorted(files, key=lambda e: e["path"])
        _write_json(
            manifest,
            {
                "repo_id": cfg.repo_id,
                "revision": cfg.revision,
                "task": task,
                "split": split,
                "count": len(files),
                "bytes": sum(f["size"] for f in files),
                "files": files,
            },
        )

    if cfg.max_episodes_per_split is not None:
        files = files[: cfg.max_episodes_per_split]
    return files


def _download_one(cfg: HfShardPrepareConfig, item: dict) -> Path:
    dst = _local_mcap_path(cfg, item)
    if dst.exists():
        print(f"[skip] {item['episode_id']} raw ({_fmt_bytes(dst.stat().st_size)})")
        return dst
    # The hub client resumes partial downloads, verifies sizes, and retries
    # transient errors internally.
    t0 = time.monotonic()
    hf_hub_download(
        cfg.repo_id,
        item["path"],
        repo_type="dataset",
        revision=cfg.revision,
        local_dir=_raw_root(cfg),
        token=cfg.hf_token,
    )
    dt = max(time.monotonic() - t0, 1e-6)
    got = dst.stat().st_size
    print(f"[get] {item['episode_id']} {_fmt_bytes(got)} ({_fmt_bytes(got / dt)}/s)")
    return dst


def _convert_jobs(jobs: list[tuple[str, str, str]], workers: int) -> list[str | None]:
    try:
        from export_mcap import export_episode
    except ImportError:
        # export_mcap lives at the repo root; make it importable when this
        # module is used from outside a repo-root script.
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from export_mcap import export_episode

    if not jobs:
        return []
    if workers <= 1:
        return [export_episode(job) for job in jobs]
    with Pool(workers) as pool:
        return pool.map(export_episode, jobs)


def _write_shard_marker(
    cfg: HfShardPrepareConfig, splits: list[str], manifest_sha: str
) -> None:
    _write_json(
        _status_root(cfg) / "shard.json",
        {
            "data_placement": "node_sharded",
            "repo_id": cfg.repo_id,
            "revision": cfg.revision,
            "global_manifest_sha": manifest_sha,
            "node_rank": cfg.node_rank,
            "num_nodes": cfg.num_nodes,
            "train_dir": cfg.train_dir,
            "val_dir": cfg.val_dir,
            "tasks": cfg.tasks,
            "splits": splits,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    )


def _prepare_split(
    cfg: HfShardPrepareConfig,
    task: str,
    split: str,
    files: list[dict],
) -> dict:
    assigned = [f for f in files if _shard_owner(f["path"], cfg.num_nodes) == cfg.node_rank]
    out_root = _output_root(cfg, split)
    out_root.mkdir(parents=True, exist_ok=True)

    # Drop ledger-tracked episodes of this task that are no longer ours
    # (reshard, task-list change). Episodes from other sources (e.g.
    # prepare.py previews) have no ledger entry and are never touched.
    pruned = 0
    if cfg.prune:
        assigned_ids = {item["episode_id"] for item in assigned}
        ledger_dir = _status_root(cfg) / "episodes" / split
        for entry_path in sorted(ledger_dir.glob("*.json")) if ledger_dir.exists() else []:
            try:
                entry = json.loads(entry_path.read_text())
            except json.JSONDecodeError:
                continue
            if entry.get("task") != task or entry.get("episode_id") in assigned_ids:
                continue
            ep_dir = out_root / entry["episode_id"]
            if entry.get("state") == "converted" and ep_dir.exists():
                shutil.rmtree(ep_dir)
                print(f"[prune] {split}/{entry['episode_id']} "
                      f"(no longer assigned to node {cfg.node_rank})")
                pruned += 1
            entry_path.unlink()

    skipped_ids = {
        item["episode_id"] for item in assigned
        if (_read_ledger(cfg, split, item["episode_id"]) or {}).get("state") == "skipped"
    }
    already_converted = [
        item for item in assigned
        if item["episode_id"] not in skipped_ids
        and _converted_ready(out_root, item["episode_id"])
    ]
    to_fetch = [
        item for item in assigned
        if item["episode_id"] not in skipped_ids
        and not _converted_ready(out_root, item["episode_id"])
    ]
    print(
        f"[{split}:{task}] global={len(files)} node={len(assigned)} "
        f"converted={len(already_converted)} skipped={len(skipped_ids)} "
        f"pending={len(to_fetch)}"
    )
    for item in already_converted:
        if _read_ledger(cfg, split, item["episode_id"]) is None:
            _write_ledger(cfg, item, "converted")
    if not cfg.keep_mcaps:
        for item in already_converted:
            _local_mcap_path(cfg, item).unlink(missing_ok=True)

    def fetch(item: dict) -> Path | None:
        try:
            return _download_one(cfg, item)
        except Exception as e:
            print(f"[fail] {item['episode_id']} download: {e}")
            return None

    if cfg.workers <= 1:
        raw_paths = [fetch(item) for item in to_fetch]
    else:
        with ThreadPoolExecutor(max_workers=cfg.workers) as pool:
            raw_paths = list(pool.map(fetch, to_fetch))

    job_records = [
        (item, raw_path, (str(raw_path), item["task"], str(out_root)))
        for item, raw_path in zip(to_fetch, raw_paths)
        if raw_path is not None and not _converted_ready(out_root, item["episode_id"])
    ]
    results = _convert_jobs([job for _, _, job in job_records], max(1, cfg.workers))
    converted_now = 0
    skipped_now = 0
    failed = sum(1 for p in raw_paths if p is None)
    for (item, raw_path, _), result in zip(job_records, results):
        if result and _converted_ready(out_root, result):
            converted_now += 1
            _write_ledger(cfg, item, "converted")
            if not cfg.keep_mcaps:
                raw_path.unlink(missing_ok=True)
        elif result is None:
            # The converter rejected the episode (too short, missing cameras);
            # remember that so re-runs stop re-downloading and re-trying it.
            skipped_now += 1
            _write_ledger(cfg, item, "skipped", reason="converter rejected episode")
            if not cfg.keep_mcaps:
                raw_path.unlink(missing_ok=True)
        else:
            failed += 1

    summary = {
        "task": task,
        "split": split,
        "global_episodes": len(files),
        "node_episodes": len(assigned),
        "already_converted": len(already_converted),
        "converted_now": converted_now,
        "skipped": len(skipped_ids) + skipped_now,
        "pruned": pruned,
        "failed": failed,
        "bytes_assigned": sum(f["size"] for f in assigned),
    }
    print(
        f"[done:{split}:{task}] converted_now={converted_now} "
        f"skipped={summary['skipped']} pruned={pruned} failed={failed} "
        f"bytes={_fmt_bytes(summary['bytes_assigned'])}"
    )
    return summary


def prepare_hf_shards(cfg: HfShardPrepareConfig) -> list[dict]:
    if not cfg.tasks:
        raise ValueError("at least one --tasks value is required")
    if cfg.num_nodes <= 0:
        raise ValueError("--num-nodes must be positive")
    if cfg.node_rank < 0 or cfg.node_rank >= cfg.num_nodes:
        raise ValueError("--node-rank must be in [0, num_nodes)")
    if cfg.workers <= 0:
        raise ValueError("--workers must be positive")

    cfg.cache = cfg.cache.expanduser().resolve()
    cfg.cache.mkdir(parents=True, exist_ok=True)
    splits = _split_names(cfg.split)
    api = HfApi(token=cfg.hf_token)
    cfg.revision = _resolve_revision(cfg, api)
    print(f"[revision] {cfg.repo_id} @ {cfg.revision}")

    # List everything up front so the marker records a hash of the exact
    # global file lists this node sharded against (checked across nodes at
    # training startup).
    manifests = {
        (task, split): list_mcaps(cfg, api, task, split)
        for task in cfg.tasks
        for split in splits
    }
    manifest_sha = hashlib.sha256(
        json.dumps(
            {f"{t}/{s}": [f["path"] for f in files] for (t, s), files in sorted(manifests.items())},
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    _write_shard_marker(cfg, splits, manifest_sha)

    summaries = []
    for task in cfg.tasks:
        for split in splits:
            summaries.append(_prepare_split(cfg, task, split, manifests[(task, split)]))

    summary = {
        "repo_id": cfg.repo_id,
        "revision": cfg.revision,
        "node_rank": cfg.node_rank,
        "num_nodes": cfg.num_nodes,
        "cache": str(cfg.cache),
        "summaries": summaries,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _write_json(
        _status_root(cfg)
        / _safe_name(cfg.repo_id)
        / _safe_name(cfg.revision)
        / f"node_{cfg.node_rank}_summary.json",
        summary,
    )
    total_failed = sum(s["failed"] for s in summaries)
    if total_failed:
        raise RuntimeError(
            f"{total_failed} episode(s) failed to download or convert; "
            f"re-run to retry (progress is kept)"
        )
    return summaries
