"""Compatibility entrypoint for teleop recording videos."""

from __future__ import annotations

import os

import tyro

from deploy.recording.video import render_recording


def render(h5_path: str, video_path: str, fps: int = 30) -> str:
    return render_recording(h5_path, video_path, recording_type="teleop", fps=fps)


def main(
    h5_path: tyro.conf.Positional[str],
    video: str | None = None,
    fps: int = 30,
) -> None:
    render(h5_path, video or os.path.splitext(h5_path)[0] + ".mp4", fps)


if __name__ == "__main__":
    tyro.cli(main)
