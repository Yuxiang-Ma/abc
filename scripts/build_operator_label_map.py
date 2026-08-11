# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
# No dependencies on purpose: abc_minimal.operator (imported below, resolved via
# the explicit repo-root sys.path insert) is kept stdlib-only so this script runs
# with a bare `uv run` — no torch environment needed.
"""Build a per-task operator label-map manifest from converted episodes.

Within each task, operators are ranked by total collected duration
(num_steps / fps / 3600 hours, descending; ties break by UUID) and each UUID is
mapped to its rank. Reads task_name + num_steps + operator_id from each
episode's episode_metadata.json (operator_id also mirrored in operator.json).
The ranking logic lives in abc_minimal/operator.py; this is the CLI.

The manifest stores bare ranks, not rendered labels: the prompting mode
(``operator N`` vs a name) is applied at train time via
``--prompt.operator-prompting-mode``, so one manifest serves every mode.

Manifest shape (read by --prompt.operator-label-map-path)::

    {
      "fps": 30,
      "sources": [...scanned dirs...],
      "label_maps": {task_name: {operator_uuid: rank}},
      "rank_table": {task_name: [{uuid, hours, episode_count, rank}, ...]}
    }

Usage::

    ABC_CACHE=cache/tshirt uv run build_operator_label_map.py \
        --out cache/tshirt/operator_label_map.json

    uv run build_operator_label_map.py \
        --roots cache/tshirt/train_real cache/tshirt/val_real \
        --out /tmp/operator_label_map.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from abc_minimal.operator import (
    DEFAULT_FPS,
    build_label_maps,
    combine_rank_tables,
    format_operator_label,
    rank_operators_per_task,
)


def _default_roots() -> list[Path]:
    cache = Path(os.environ.get("ABC_CACHE", "cache")).expanduser()
    return [cache / "train_real", cache / "val_real"]


def _canonical(value) -> str:
    """Normalize a label-map value (bare rank or rendered label) to one form so a
    rank-storing manifest and a label-storing one compare equal for the same
    operator: ranks render as "operator N", rendered strings pass through."""
    try:
        return format_operator_label(int(value), "text_indexed")
    except (TypeError, ValueError):
        return str(value)


def compare_label_maps(a: dict[str, dict], b: dict[str, dict]) -> None:
    """Print a per-task agreement report between two {task:{uuid:rank|label}} maps (a vs reference b)."""
    tasks = sorted(set(a) & set(b))
    print(f"[compare] shared tasks: {len(tasks)} (a-only={len(set(a)-set(b))}, b-only={len(set(b)-set(a))})")
    for task in tasks:
        ta = {u: _canonical(v) for u, v in a[task].items()}
        tb = {u: _canonical(v) for u, v in b[task].items()}
        common = set(ta) & set(tb)
        disagree = [u for u in common if ta[u] != tb[u]]
        agree = 100.0 * (len(common) - len(disagree)) / len(common) if common else 100.0
        print(
            f"[compare] {task}: common={len(common)} "
            f"a_only={len(set(ta)-set(tb))} b_only={len(set(tb)-set(ta))} "
            f"disagreements={len(disagree)} agreement={agree:.1f}%"
        )
        for u in disagree[:10]:
            print(f"[compare]     {u}: a={ta[u]!r} b={tb[u]!r}")


def _load_manifest(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def _label_maps_of(m: dict) -> dict[str, dict[str, str]]:
    """Extract ``{task:{uuid:label}}`` from a manifest or a bare map."""
    return m.get("label_maps", m) if isinstance(m, dict) else {}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--roots",
        nargs="+",
        type=Path,
        default=None,
        help="Episode-dir roots to scan (default: $ABC_CACHE/{train_real,val_real}).",
    )
    ap.add_argument("--out", type=Path, required=True, help="Output manifest.json path.")
    ap.add_argument("--fps", type=int, default=DEFAULT_FPS, help="fps for num_steps -> hours.")
    ap.add_argument(
        "--combine",
        nargs="+",
        type=Path,
        default=None,
        help="Sum the rank_tables of these manifests into one global ranking instead of scanning episodes.",
    )
    ap.add_argument(
        "--compare-to",
        type=Path,
        default=None,
        help="Print a per-task agreement report against this reference manifest after building --out.",
    )
    args = ap.parse_args()

    if args.combine:
        print(f"[combine] folding {len(args.combine)} shard manifests into a global ranking")
        ranked = combine_rank_tables([_load_manifest(p) for p in args.combine])
        sources = [str(p) for p in args.combine]
    else:
        roots = args.roots or _default_roots()
        print(f"[sources] scanning {[str(r) for r in roots]}")
        ranked = rank_operators_per_task(roots, args.fps, verbose=True)
        sources = [str(r) for r in roots]
    if not ranked:
        print("ERROR: no operator-annotated episodes found", file=sys.stderr)
        return 2
    label_maps = build_label_maps(ranked)

    manifest = {
        "fps": args.fps,
        "sources": sources,
        "label_maps": label_maps,
        "rank_table": ranked,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2, sort_keys=True))

    n_entries = sum(len(m) for m in label_maps.values())
    print(f"\n[out] wrote {args.out}")
    print(f"[out] tasks covered: {len(label_maps)}; total (task,operator) entries: {n_entries}")
    top = sorted(
        ((task, sum(r["hours"] for r in rows)) for task, rows in ranked.items()),
        key=lambda t: -t[1],
    )[:5]
    print("[out] top tasks by total operator hours:")
    for task, hours in top:
        print(f"  {task}: {hours:.1f}h across {len(ranked[task])} operators")

    if args.compare_to:
        print(f"\n[compare] {args.out} vs reference {args.compare_to}")
        compare_label_maps(label_maps, _label_maps_of(_load_manifest(args.compare_to)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
