"""Live camera backend helpers for interactive simulation."""

from __future__ import annotations

from typing import Literal

from abc_sim.rendering.replay.camera_providers import (
    MujocoCameraProvider,
    SimRenderCameraProvider,
)


LiveRenderBackend = Literal["mujoco", "mjwarp", "madrona", "blender"]


def create_live_camera_provider(
    *,
    model,
    data,
    backend: LiveRenderBackend,
    width: int,
    height: int,
    gpu_id: int | None = None,
    camera_names: tuple[str, ...] | None = None,
):
    """Create a live camera renderer for the current MuJoCo state."""
    if backend == "blender":
        from abc_sim.rendering.blender.live import BlenderCameraProvider

        return BlenderCameraProvider(
            model,
            data,
            width=width,
            height=height,
            gpu_id=gpu_id,
            camera_names=camera_names,
        )
    if backend == "mujoco":
        return MujocoCameraProvider(
            model,
            data,
            width=width,
            height=height,
            camera_names=camera_names,
        )
    return SimRenderCameraProvider(
        model,
        data,
        backend=backend,
        width=width,
        height=height,
        gpu_id=gpu_id,
        camera_names=camera_names,
    )
