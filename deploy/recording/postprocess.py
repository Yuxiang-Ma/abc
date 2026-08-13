"""Post-recording hook: render finished recordings to review MP4s.

Invoked detached by the recorder nodes after each saved episode (opt out with
``--no-post-video`` on deploy/dagger CLIs, or ``DEPLOY_POST_VIDEO=0``), and
manually for catch-up:

    uv run python -m deploy.recording.postprocess <file.h5>
    uv run python -m deploy.recording.postprocess --scan data/

Videos land next to the recordings in a sibling directory named after the
recording type: ``<root>/<type>_video/<recording>.mp4``, where ``<root>`` is
the directory holding the H5. No uploading, no deletion — local only.
"""

from __future__ import annotations

import os
import subprocess
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path

import h5py
import tyro

from deploy.recording import io as rio


@dataclass
class PostprocessArgs:
    recording: str | None = None
    """One recording .h5 to render."""

    scan: str | None = None
    """Directory root: render every H5 under it that has no video yet."""

    force: bool = False
    """Re-render even if the video already exists."""


def video_path_for(h5_path: Path) -> Path:
    with h5py.File(h5_path, "r") as f:
        recording_type = rio.detect_recording_type(f)
    video_dir = h5_path.parent.parent / f"{recording_type}_video"
    return video_dir / f"{h5_path.stem}.mp4"


def process_one(h5_path: Path, *, force: bool = False) -> Path | None:
    video = video_path_for(h5_path)
    if video.exists() and not force:
        return None
    video.parent.mkdir(parents=True, exist_ok=True)
    from deploy.recording.render import render

    render(str(h5_path), str(video))
    return video


def scan(root: Path, *, force: bool = False) -> None:
    recordings = sorted(root.rglob("*.h5"))
    if not recordings:
        print(f"[postprocess] no recordings under {root}")
        return
    done = 0
    for h5_path in recordings:
        try:
            video = process_one(h5_path, force=force)
        except Exception:  # noqa: BLE001 - one bad recording must not stop a scan
            print(f"[postprocess] FAILED {h5_path}")
            traceback.print_exc()
            continue
        if video is not None:
            print(f"[postprocess] {h5_path.name} -> {video}")
            done += 1
    print(f"[postprocess] rendered {done} of {len(recordings)} recordings")


def spawn_detached(h5_path: str) -> None:
    """Fire-and-forget render of a finished recording (called by recorders).

    Output goes to postprocess.log next to the recordings so failures
    (e.g. a missing renderer dependency) are diagnosable, not silent.
    """
    if os.environ.get("DEPLOY_POST_VIDEO", "1") == "0":
        return
    log_path = Path(h5_path).expanduser().resolve().parent / "postprocess.log"
    with open(log_path, "a") as log:
        subprocess.Popen(
            [
                sys.executable,
                "-m",
                "deploy.recording.postprocess",
                "--recording",
                h5_path,
            ],
            stdout=log,
            stderr=log,
            start_new_session=True,
        )
    print(f"[recorder] rendering video in background (log: {log_path})")


def main(args: PostprocessArgs) -> None:
    if (args.recording is None) == (args.scan is None):
        raise SystemExit("pass exactly one of --recording <file.h5> or --scan <dir>")
    if args.scan is not None:
        scan(Path(args.scan).expanduser().resolve(), force=args.force)
        return
    h5_path = Path(args.recording).expanduser().resolve()
    video = process_one(h5_path, force=args.force)
    if video is None:
        print(f"[postprocess] video already exists for {h5_path.name}")
    else:
        print(f"[postprocess] wrote {video}")


if __name__ == "__main__":
    main(tyro.cli(PostprocessArgs))
