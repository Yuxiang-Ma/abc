"""Compatibility entrypoint for DAgger recording videos."""

from __future__ import annotations

from dataclasses import dataclass

import tyro

from deploy.recording.video import render_recording


def render(h5_path: str, video_path: str, *, fps: int = 30) -> str:
    return render_recording(h5_path, video_path, recording_type="dagger", fps=fps)


@dataclass
class DaggerVideoConfig:
    recording: tyro.conf.Positional[str]
    video: str
    fps: int = 30


def main() -> None:
    config = tyro.cli(DaggerVideoConfig)
    render(config.recording, config.video, fps=config.fps)


if __name__ == "__main__":
    main()
