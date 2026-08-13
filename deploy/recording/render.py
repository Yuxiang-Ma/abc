"""Render any deploy recording to an annotated MP4."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import tyro

from deploy.recording.video import render_recording


@dataclass
class RenderArgs:
    recording: tyro.conf.Positional[str]
    """Path to a recording .h5 file."""

    video: str | None = None
    """Output MP4 path. Default: the recording path with a .mp4 suffix."""

    type: Literal["auto", "teleop", "inference", "dagger"] = "auto"
    """Recording type; auto-detected from the file by default."""

    fps: int = 30


def render(
    h5_path: str,
    video_path: str | None = None,
    recording_type: str = "auto",
    fps: int = 30,
) -> str:
    if video_path is None:
        video_path = str(Path(h5_path).with_suffix(".mp4"))
    return render_recording(
        h5_path,
        video_path,
        recording_type=recording_type,
        fps=fps,
    )


def main(args: RenderArgs) -> None:
    video = render(args.recording, args.video, args.type, args.fps)
    print(f"[render] wrote {video}")


if __name__ == "__main__":
    main(tyro.cli(RenderArgs))
