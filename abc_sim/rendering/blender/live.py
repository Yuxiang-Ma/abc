"""Blender Cycles as a live camera backend: one Blender process per bound model, poses streamed over a unix socket.

ABC_BLENDER_SAMPLES / ABC_BLENDER_DENOISER override the 16 spp + OIDN default (e.g. 4 + OPTIX for the old setting)."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import tempfile
import time

import mujoco
from mujoco.usd import exporter as usd_exporter
import numpy as np
from PIL import Image
from pxr import UsdGeom

from abc_sim.rendering.blender.export_usd import geom_records
from abc_sim.rendering.replay.camera_providers import CameraProvider

RENDER = Path(__file__).with_name("render_usd.py")


class BlenderCameraProvider(CameraProvider):
    def __init__(self, model, data, *, width, height, gpu_id, camera_names):
        super().__init__(camera_names=tuple(camera_names))
        self._data = data
        self._dir = tempfile.mkdtemp(prefix="abc_blender_", dir="/dev/shm")
        self._calls, self._render_s = 0, 0.0
        t0 = time.time()
        mujoco.mj_forward(model, data)   # a freshly bound MjData has no poses yet
        exporter = usd_exporter.USDExporter(model=model, output_directory="usd", output_directory_root=self._dir,
                                            camera_names=list(self.camera_names), verbose=False)
        exporter.update_scene(data)
        UsdGeom.SetStageMetersPerUnit(exporter.stage, 1.0)
        exporter.save_scene(filetype="usdc")
        geoms = geom_records(model, exporter)
        Path(self._dir, "geoms.json").write_text(json.dumps({"geoms": geoms, "cameras": list(self.camera_names)}))
        self._geom_ids = [g["id"] for g in geoms]
        self._cam_ids = [model.cam(name).id for name in self.camera_names]
        env = dict(os.environ)
        if gpu_id is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        self._log = open(f"{self._dir}/blender.log", "w")
        self._proc = subprocess.Popen(
            [os.environ.get("BLENDER", "blender"), "-b", "--python-exit-code", "1", "--python", str(RENDER), "--",
             "--usd", f"{self._dir}/usd/frames/frame_1.usdc", "--meta", f"{self._dir}/geoms.json", "--out", self._dir,
             "--size", f"{width}x{height}", "--samples", os.environ.get("ABC_BLENDER_SAMPLES", "16"), "--device", "OPTIX",
             "--denoiser", os.environ.get("ABC_BLENDER_DENOISER", "OPENIMAGEDENOISE"), "--no-bevel",
             "--serve", f"{self._dir}/sock"],
            env=env, stdout=self._log, stderr=subprocess.STDOUT, start_new_session=True)
        self._sock = socket.socket(socket.AF_UNIX)
        while self._proc.poll() is None:
            try:
                self._sock.connect(f"{self._dir}/sock")
                break
            except (FileNotFoundError, ConnectionRefusedError):
                time.sleep(0.2)
        if self._proc.poll() is not None:
            raise RuntimeError(f"blender exited with {self._proc.returncode}, see {self._log.name}")
        self._io = self._sock.makefile("rwb")
        self.frames_for_step(0, 0.0)     # the exported state: calibrates the server's object transforms, warms the session
        print(f"blender: ready in {time.time() - t0:.1f}s", flush=True)

    def frames_for_step(self, step_idx: int, query_ts: float) -> dict[str, np.ndarray]:
        t0 = time.time()
        d = self._data
        poses = np.concatenate([
            np.concatenate([d.geom_xpos[self._geom_ids], d.geom_xmat[self._geom_ids]], axis=1),
            np.concatenate([d.cam_xpos[self._cam_ids], d.cam_xmat[self._cam_ids]], axis=1),
        ])
        self._io.write(poses.astype(np.float32).tobytes())
        self._io.flush()
        frames = {name: np.asarray(Image.open(self._io.readline().decode().strip())) for name in self.camera_names}
        self._calls, self._render_s = self._calls + 1, self._render_s + time.time() - t0
        return frames

    def close(self) -> None:
        self._io.close()
        self._sock.close()               # EOF ends the server loop
        self._proc.wait()
        self._log.close()
        shutil.rmtree(self._dir)
        print(f"blender: {self._calls} renders, {1000 * self._render_s / self._calls:.0f}ms/call", flush=True)
