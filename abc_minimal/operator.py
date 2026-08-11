"""Operator-id -> prompt-label utilities.

Raw ``operator_id`` values are UUIDs, meaningless to a text encoder. To use the
operator as a conditioning signal we rewrite each UUID to a short deterministic
label via a per-task label map: within a task, operators are ranked (e.g. by
total collected hours, descending) so the top operator is rank 0, the next
rank 1, and so on. The label map has shape ``{task_name: {operator_uuid: rank}}``
and stores the bare rank; the rank is rendered to a label at run time per the
active ``--prompt.operator-prompting-mode``, so one manifest serves every mode.

Two rendering modes:
  * ``text_indexed`` (default) -> ``"operator N"``
  * ``text_name``              -> an English first name from a fixed pool;
    overflow (more operators than names) falls back to ``"operator N"``.

An already-rendered label string in place of a rank is also accepted and passed
through verbatim (the mode is then moot for that entry). This is only for
hand-authored or legacy manifests; maps built by ``build_operator_label_map.py``
always store ranks.

With no label map, or an operator/task pair absent from it, the label is the
empty string and the caller leaves the prompt unchanged.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

INDEXED_LABEL_FMT = "operator {idx}"

# Deterministic pool of common English first names for ``text_name`` mode,
# sorted alphabetically. Overflow beyond the pool falls back to "operator N".
ENGLISH_NAME_POOL: tuple[str, ...] = (
    "alice", "bob", "charlie", "diana", "ethan", "fiona", "george", "hannah",
    "ivan", "julia", "kevin", "laura", "mason", "nina", "oliver", "paula",
    "quinn", "rachel", "sam", "tina", "ulysses", "victor", "wendy", "xander",
    "yvonne", "zach",
)


def format_operator_label(rank: int, mode: str) -> str:
    """Render an operator rank as a CLIP-tokenizable label string."""
    if mode == "text_name" and 0 <= rank < len(ENGLISH_NAME_POOL):
        return ENGLISH_NAME_POOL[rank]
    return INDEXED_LABEL_FMT.format(idx=rank)


def load_operator_label_maps(path: str | None) -> dict[str, dict[str, str]]:
    """Load a per-task operator label-map manifest into ``{task: {uuid: rank}}``.

    Accepts either a bare ``{task_name: {uuid: rank}}`` dict or the wrapped
    ``{"label_maps": {task_name: {uuid: rank}}}`` form. Returns ``{}`` for a
    falsy path so "no manifest" == "no operator conditioning".
    """
    if not path:
        return {}
    raw = json.loads(Path(path).expanduser().read_text())
    if isinstance(raw, dict) and isinstance(raw.get("label_maps"), dict):
        raw = raw["label_maps"]
    out: dict[str, dict[str, str]] = {}
    for task, sub in raw.items():
        if isinstance(sub, dict):
            out[str(task)] = {str(k): str(v) for k, v in sub.items()}
    return out


def operator_label_for(
    task_name: str | None,
    operator_id: str | None,
    label_maps: dict[str, dict[str, str]] | None,
    mode: str = "text_indexed",
) -> str:
    """Resolve ``operator_id`` to a rendered label for ``task_name``.

    Returns the empty string when any input is missing or the (task, operator)
    pair is absent from the map — the caller treats that as "no operator signal"
    and leaves the prompt unchanged.
    """
    if not operator_id or not task_name or not label_maps:
        return ""
    task_map = label_maps.get(str(task_name))
    if not task_map:
        return ""
    rank_or_label = task_map.get(str(operator_id), "")
    if rank_or_label == "":
        return ""
    # A rank int (possibly stored as a string) is rendered per mode; an
    # already-rendered label string is passed through unchanged.
    try:
        rank = int(rank_or_label)
    except (TypeError, ValueError):
        return str(rank_or_label)
    return format_operator_label(rank, mode)


# --- label-map construction ----------------------------------------------------
# Used only by build_operator_label_map.py (CLI); training consumes the built
# manifest via load_operator_label_maps and never ranks episodes itself.

DEFAULT_FPS = 30


def _iter_episode_meta(roots: list[Path]):
    """Yield (task_name, operator_id, num_steps) per episode, operator.json fallback."""
    for root in roots:
        if not root.exists():
            continue
        for meta_path in sorted(root.glob("episode_*/episode_metadata.json")):
            try:
                meta = json.loads(meta_path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            op = meta.get("operator_id")
            if not op:
                op_path = meta_path.parent / "operator.json"
                if op_path.exists():
                    try:
                        op = json.loads(op_path.read_text()).get("operator_id")
                    except (OSError, json.JSONDecodeError):
                        op = None
            yield meta.get("task_name"), op, meta.get("num_steps")


def _rank_rows(op_map: dict[str, dict[str, float]]) -> list[dict]:
    """Sort one task's ``{uuid: {hours, count}}`` bucket into ranked rows
    (hours DESC, UUID ASC as the deterministic tie-break)."""
    rows = sorted(
        (
            {"uuid": uuid, "hours": stats["hours"], "episode_count": int(stats["count"])}
            for uuid, stats in op_map.items()
        ),
        key=lambda r: (-r["hours"], r["uuid"]),
    )
    for rank, row in enumerate(rows):
        row["rank"] = rank
    return rows


def rank_operators_per_task(
    roots: list[Path], fps: int = DEFAULT_FPS, verbose: bool = False
) -> dict[str, list[dict]]:
    """Group episodes by task, sum hours per operator, rank desc by hours.

    Returns ``{task_name: [{uuid, hours, episode_count, rank}, ...]}``.
    """
    by_task: dict[str, dict[str, dict[str, float]]] = defaultdict(
        lambda: defaultdict(lambda: {"hours": 0.0, "count": 0})
    )
    skipped = {"no_operator": 0, "no_task": 0, "no_size": 0}
    for task, op_id, num_steps in _iter_episode_meta(roots):
        if not op_id:
            skipped["no_operator"] += 1
            continue
        if not task:
            skipped["no_task"] += 1
            continue
        if not num_steps or num_steps <= 0:
            skipped["no_size"] += 1
            continue
        bucket = by_task[task][str(op_id)]
        bucket["hours"] += float(num_steps) / fps / 3600.0
        bucket["count"] += 1
    if verbose:
        print(
            f"[rank] skipped: no_operator={skipped['no_operator']}, "
            f"no_task={skipped['no_task']}, no_size={skipped['no_size']}"
        )
        print(f"[rank] tasks with operator-annotated episodes: {len(by_task)}")
    return {task: _rank_rows(op_map) for task, op_map in by_task.items()}


def build_label_maps(ranked: dict[str, list[dict]]) -> dict[str, dict[str, int]]:
    """Turn ranked rows into ``{task_name: {uuid: rank}}``.

    Stores the bare rank, not a rendered label -- the mode is applied at run time
    by ``operator_label_for``, so a single manifest works for every mode."""
    return {
        task: {row["uuid"]: row["rank"] for row in rows}
        for task, rows in ranked.items()
    }


def combine_rank_tables(manifests: list[dict]) -> dict[str, list[dict]]:
    """Sum per-uuid hours/counts across manifests' rank_tables, then re-rank globally."""
    agg: dict[str, dict[str, dict[str, float]]] = defaultdict(
        lambda: defaultdict(lambda: {"hours": 0.0, "count": 0})
    )
    for m in manifests:
        for task, rows in m.get("rank_table", {}).items():
            for row in rows:
                bucket = agg[task][str(row["uuid"])]
                bucket["hours"] += float(row.get("hours", 0.0))
                bucket["count"] += int(row.get("episode_count", 0))
    return {task: _rank_rows(op_map) for task, op_map in agg.items()}
