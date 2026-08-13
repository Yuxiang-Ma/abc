"""Compatibility entrypoint for inference recording videos."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import tyro

from deploy.recording.video import render_recording


def render(
    h5_path: str,
    video_path: str,
    dof_indices: list[int] | None = None,
    fps: int = 30,
) -> str:
    return render_recording(
        h5_path,
        video_path,
        recording_type="inference",
        fps=fps,
        main_dofs=dof_indices,
    )


@dataclass
class Args:
    recording: tyro.conf.Positional[str]
    video: str | None = None
    dofs: list[int] = field(default_factory=lambda: [0, 4, 7, 11])
    fps: int = 30


def main(args: Args) -> None:
    render(
        args.recording,
        args.video or str(Path(args.recording).with_suffix(".mp4")),
        args.dofs,
        args.fps,
    )


if __name__ == "__main__":
    main(tyro.cli(Args))
