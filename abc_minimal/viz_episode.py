"""Viser player for downloaded dataset episodes.

Browses a pool of exported episodes (``prepare.py --sim-data <task>`` unpacks
them into ``cache/{train,val}_sim/``), grouped by task, rebuilds each episode's
own scene, and plays it back two ways:

* ``pose`` (default): every frame is posed straight from the recording, so
  playback is honest, scrubbable and drift-free. Episodes carrying
  ``scene_qpos.npy`` pose the whole scene, objects included; without it only the
  14 arm dofs can be recovered from the states, and free objects hold the
  episode's recorded starting pose.
* ``physics``: the recorded actions are stepped open loop through CPU MuJoCo
  from the recorded initial state, so objects move, at the cost of the
  expected integrator drift (arm RMS ~0.003-0.04 rad; stiff-contact scenes may
  hit solver resets, which are counted in the status line). Scrubbing is
  disabled; playback restarts from frame 0 on wrap.

The recorded combined camera video is decoded alongside (PyAV) as ground
truth. This module never imports torch: there is no policy here.
"""

from __future__ import annotations

import asyncio
import json
import math
import threading
import time
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import mujoco
import numpy as np
import viser
from mjviser import ViserMujocoScene

from abc_minimal.config import VizEpisodeConfig, default_cache_root
from abc_minimal.episode_io import (
    DATA_FPS,
    discover_episodes,
    load_episode,
    load_scene_qpos,
    load_scene_xml,
    seed_initial_state,
)

# The exporter's fixed video clock: pts = VIDEO_PTS_STEP * frame_index, with a
# keyframe every GOP frames (video_io.py writes x264 with -bf 0, GOP 30).
VIDEO_PTS_STEP = 512
VIDEO_GOP = 30

# After a genuine user scrub, the playback thread stops writing the frame
# slider for this long, so the knob is not fought over mid-drag.
SCRUB_BACKOFF_S = 0.3

# A GUI command older than this with no progress means the playback thread is
# hung (not crashed); the watchdog dumps all thread stacks so the terminal
# names the exact line it is stuck on.
WATCHDOG_S = 3.0

# Fixed GUI slots: the frame slider is recreated per episode (viser sliders
# cannot change their bounds in place), and an explicit order pins each
# element's position regardless of creation time.
(_O_STATUS, _O_TASK, _O_EPISODE, _O_PREV, _O_NEXT, _O_PLAY, _O_FRAME,
 _O_SPEED, _O_MODE, _O_VIEW, _O_VIDEO) = range(10, 120, 10)


def playback_period_s(speed: float) -> float:
    """Wall-clock seconds one data frame should occupy at a given speed."""
    return 1.0 / (DATA_FPS * max(float(speed), 1e-3))


def clamp_frame(frame: int, num_steps: int) -> int:
    return max(0, min(int(frame), num_steps - 1))


def episode_status(
    name: str,
    task: str,
    frame: int,
    num_steps: int,
    mode: str,
    resets: int,
    *,
    full_scene: bool,
) -> str:
    """One-line GUI status; solver resets only appear once physics hits one.

    Pose mode names which flavor it is playing, since the two look alike until an
    object is supposed to move: ``full`` poses the whole recorded scene,
    ``arms`` only the 14 robot dofs. Physics moves everything by construction.
    """
    flavor = f"{mode}:{'full' if full_scene else 'arms'}" if mode == "pose" else mode
    text = f"{task} {name} frame {frame + 1}/{num_steps} [{flavor}]"
    return f"{text} solver_resets={resets}" if resets else text


def scan_tasks(episodes: list[Path], workers: int = 16) -> dict[str, list[Path]]:
    """Group episode dirs by their metadata task_name, sorted both ways."""

    def task_of(episode_dir: Path) -> str:
        try:
            meta = json.loads((episode_dir / "episode_metadata.json").read_text())
        except (OSError, json.JSONDecodeError):
            return "(unknown)"
        return str(meta.get("task_name") or meta.get("task") or "(unknown)")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        names = list(pool.map(task_of, episodes))
    by_task: dict[str, list[Path]] = defaultdict(list)
    for episode_dir, name in zip(episodes, names):
        by_task[name].append(episode_dir)
    return dict(sorted(by_task.items()))


class VideoFrames:
    """Random access into the episode's constant-frame-rate combined mp4."""

    def __init__(self, path: Path):
        import av

        self._container = av.open(str(path))
        self._stream = self._container.streams.video[0]
        self._decoder = self._container.decode(self._stream)
        self._last_index = -1

    def frame(self, index: int) -> np.ndarray | None:
        """Decode frame ``index``, seeking only when playback is not sequential."""
        if index <= self._last_index or index > self._last_index + VIDEO_GOP:
            self._container.seek(index * VIDEO_PTS_STEP, stream=self._stream)
            self._decoder = self._container.decode(self._stream)
            self._last_index = -1  # unknown until the next decoded pts
        for frame in self._decoder:
            frame_index = int(frame.pts // VIDEO_PTS_STEP)
            self._last_index = frame_index
            if frame_index >= index:
                return frame.to_ndarray(format="rgb24")
        return None

    def close(self) -> None:
        self._container.close()


def run_episode_viewer(cfg: VizEpisodeConfig) -> None:
    if cfg.mode not in ("pose", "physics"):
        raise SystemExit(f"--mode must be pose or physics, got {cfg.mode!r}")
    if cfg.episode_dir is not None:
        episodes = [cfg.episode_dir.expanduser()]
    else:
        root = (cfg.root or default_cache_root() / "train_sim").expanduser()
        episodes = discover_episodes(root)
        if not episodes:
            raise SystemExit(
                f"No episodes under {root}. Download some first: "
                "uv run prepare.py --sim-data <task>"
            )

    by_task = scan_tasks(episodes)
    if cfg.task and cfg.task not in by_task:
        raise SystemExit(
            f"--task {cfg.task!r} has no episodes here; available: "
            + ", ".join(by_task)
        )
    for name in by_task:
        by_task[name] = by_task[name][: cfg.max_episodes]
    print(
        f"[scan] {len(episodes)} episodes across {len(by_task)} task(s): "
        + ", ".join(f"{k} ({len(v)})" for k, v in by_task.items()),
        flush=True,
    )
    initial_task = cfg.task or next(iter(by_task))

    import abc_sim

    assets_dir = Path(abc_sim.__file__).resolve().parent / "models" / "assets"

    server = viser.ViserServer(host="0.0.0.0", port=cfg.port)
    actual_port = server.get_port()

    default_camera_position = np.array([-0.42, 0.0, 1.66], dtype=np.float64)
    default_camera_look_at = np.array([0.45, 0.0, 0.87], dtype=np.float64)

    def apply_default_view(client: viser.ClientHandle) -> None:
        client.camera.position = default_camera_position
        client.camera.look_at = default_camera_look_at
        client.camera.up_direction = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        client.camera.fov = math.radians(50.0)

    @server.on_client_connect
    async def _(client: viser.ClientHandle) -> None:
        await asyncio.sleep(0.1)
        apply_default_view(client)

    status = server.gui.add_text(
        "status", initial_value="loading", disabled=True, order=_O_STATUS
    )
    task_dropdown = server.gui.add_dropdown(
        "task", options=tuple(by_task), initial_value=initial_task, order=_O_TASK
    )
    episode_dropdown = server.gui.add_dropdown(
        "episode",
        options=tuple(p.name for p in by_task[initial_task]),
        order=_O_EPISODE,
    )
    prev_btn = server.gui.add_button("Prev episode", order=_O_PREV)
    next_btn = server.gui.add_button("Next episode", order=_O_NEXT)
    playing = server.gui.add_checkbox("Play", initial_value=True, order=_O_PLAY)
    speed = server.gui.add_slider(
        "speed", min=0.25, max=4.0, step=0.25, initial_value=float(cfg.speed),
        order=_O_SPEED,
    )
    mode_dropdown = server.gui.add_dropdown(
        "mode", options=("pose", "physics"), initial_value=cfg.mode, order=_O_MODE
    )
    view_btn = server.gui.add_button("Default view", order=_O_VIEW)
    video_image = None  # created on first decoded frame, sized to the video

    # All mutation happens on the playback thread; GUI callbacks only enqueue.
    lock = threading.Lock()
    commands: list[tuple] = []
    queued_since: list[float | None] = [None]

    def enqueue(*command) -> None:
        with lock:
            commands.append(command)
            if queued_since[0] is None:
                queued_since[0] = time.monotonic()

    def watchdog() -> None:
        import faulthandler

        while True:
            time.sleep(1.0)
            stamp = queued_since[0]
            if stamp is not None and time.monotonic() - stamp > WATCHDOG_S:
                print(
                    f"[watchdog] a GUI command has waited >{WATCHDOG_S:.0f}s "
                    "unprocessed; all thread stacks follow",
                    flush=True,
                )
                faulthandler.dump_traceback()
                queued_since[0] = time.monotonic() + 30.0  # re-arm, don't spam

    threading.Thread(target=watchdog, daemon=True).start()

    loader = ThreadPoolExecutor(max_workers=1)

    def build_episode(episode_dir: Path) -> dict:
        """Everything heavy about opening an episode, off the GUI thread.

        Ordered so a failure leaves the current scene untouched: the viser
        scene is only reset after the episode's own env compiled. viser's
        scene/GUI APIs are thread-safe, so the GLB build can happen here
        while the playback thread idles (it stops touching the old scene
        the moment ``loading`` is set).
        """
        metadata, states, actions = load_episode(episode_dir)
        scene_qpos = load_scene_qpos(episode_dir, metadata, len(states))
        env_task = metadata.get("task") or metadata.get("task_name")
        if not env_task:
            raise ValueError(f"{episode_dir.name}: no task in metadata")
        scene_xml, scene_source = load_scene_xml(episode_dir, metadata, assets_dir)
        env = abc_sim.make_env(
            task=env_task,
            render_cameras=False,
            scene_xml_string=scene_xml,
            enable_task_randomizer=scene_xml is None,
        )
        env.reset(seed=0, randomize=scene_xml is None)
        seeded = seed_initial_state(env, episode_dir, metadata, states)
        if scene_qpos is not None and scene_qpos.shape[1] != env.model.nq:
            # A scene_qpos from a differently-compiled model would silently
            # scramble the dofs; arms-only playback is the honest fallback.
            print(
                f"[warn] {episode_dir.name}: scene_qpos has {scene_qpos.shape[1]} "
                f"dofs, model nq={env.model.nq}; posing the arms only",
                flush=True,
            )
            scene_qpos = None
        video = None
        video_path = episode_dir / "combined_camera-images-rgb.mp4"
        if cfg.video_panel and video_path.exists():
            video = VideoFrames(video_path)
        # Each episode compiles a fresh MjModel, and a model may only be
        # wrapped by ViserMujocoScene once; scene.reset() keeps the GUI and
        # every client's camera.
        server.scene.reset()
        scene = ViserMujocoScene(server, env.model, num_envs=1)
        scene.camera_tracking_enabled = False
        # A no-op for scenes without a table plane; mjviser would otherwise
        # render the MuJoCo plane as an infinite grid at table height.
        server.scene.remove_by_name("/fixed_bodies/world/table_plane")
        scene.update_from_mjdata(env.data)
        print(
            f"[load] {episode_dir.name} task={env_task} steps={len(states)} "
            f"scene={scene_source} start={seeded} "
            f"qpos={'full' if scene_qpos is not None else 'arms'}",
            flush=True,
        )
        return {
            "env": env,
            "scene": scene,
            "states": states,
            "actions": actions,
            "scene_qpos": scene_qpos,
            "video": video,
        }

    @view_btn.on_click
    def _(_) -> None:
        for client in server.get_clients().values():
            apply_default_view(client)

    @task_dropdown.on_update
    def _(_) -> None:
        # Printed from viser's event-loop thread on purpose: seeing "[gui]"
        # with no "[load]" following it separates "playback thread stuck"
        # from "GUI events not being delivered at all".
        print(f"[gui] task -> {task_dropdown.value}", flush=True)
        enqueue("task", task_dropdown.value)

    @episode_dropdown.on_update
    def _(_) -> None:
        enqueue("episode", episode_dropdown.value)

    @prev_btn.on_click
    def _(_) -> None:
        enqueue("shift", -1)

    @next_btn.on_click
    def _(_) -> None:
        enqueue("shift", +1)

    @mode_dropdown.on_update
    def _(_) -> None:
        enqueue("mode", mode_dropdown.value)

    class Player:
        env = None
        scene: ViserMujocoScene | None = None
        states: np.ndarray | None = None
        actions: np.ndarray | None = None
        scene_qpos: np.ndarray | None = None  # None: pose the arms alone
        video: VideoFrames | None = None
        frame_slider = None
        name = ""
        task = ""
        num_steps = 0
        frame = 0
        mode = cfg.mode
        resets = 0
        init_qpos: np.ndarray | None = None
        physics_dirty = False  # scene state has diverged from frame-0 snapshot
        generation = 0  # bumps per selection; stale worker builds are dropped
        loading: str | None = None  # episode being built; worker owns the scene
        # Echo suppression: programmatic writes to a GUI element fire its own
        # on_update; remembering what the playback thread last wrote lets those
        # echoes be dropped deterministically (no racy flags).
        _slider_write = -1
        _episode_write: str | None = None
        _scrub_ts = 0.0
        _video_shown = -1

        def task_episodes(self) -> list[Path]:
            return by_task[self.task]

        def rebuild_frame_slider(self) -> None:
            # viser sliders cannot change min/max in place; recreate in the
            # same order slot so the panel layout is stable.
            if self.frame_slider is not None:
                self.frame_slider.remove()
            slider = server.gui.add_slider(
                "frame",
                min=0,
                max=max(self.num_steps - 1, 1),
                step=1,
                initial_value=0,
                order=_O_FRAME,
            )
            self._slider_write = 0

            @slider.on_update
            def _(_) -> None:
                enqueue("scrub", slider.value)

            self.frame_slider = slider

        def begin_load(self, episode_dir: Path) -> None:
            """Hand the heavy episode build to the worker; the GUI stays live.

            Scene construction PNG-encodes every texture into GLB meshes
            (~16 s for the letter-block tasks), so it cannot run on this
            thread: every control funnels through here, and a synchronous
            build reads as a dead panel. The generation counter lets a newer
            selection supersede a build still in flight.
            """
            self.generation += 1
            gen = self.generation
            self.loading = episode_dir.name
            status.value = (
                f"loading {episode_dir.name} … scene build can take ~20 s "
                "for texture-heavy tasks"
            )

            def build() -> None:
                try:
                    payload = build_episode(episode_dir)
                except Exception:  # noqa: BLE001 - reported to GUI, thread lives
                    traceback.print_exc()
                    enqueue("load_failed", gen, episode_dir.name)
                    return
                enqueue("loaded", gen, episode_dir, payload)

            loader.submit(build)

        def install(self, gen: int, episode_dir: Path, payload: dict) -> None:
            if gen != self.generation:
                if payload["video"] is not None:
                    payload["video"].close()
                return
            if self.video is not None:
                self.video.close()
            self.env = payload["env"]
            self.scene = payload["scene"]
            self.states, self.actions = payload["states"], payload["actions"]
            self.scene_qpos = payload["scene_qpos"]
            self.name = episode_dir.name
            self.num_steps = len(payload["states"])
            self.frame = 0
            self.resets = 0
            self.init_qpos = np.array(self.env.data.qpos, dtype=np.float64)
            self.physics_dirty = False
            self._video_shown = -1
            self.video = payload["video"]
            self.loading = None
            self.rebuild_frame_slider()
            self.show_frame(0, force_pose=True)

        def restore_initial_state(self) -> None:
            self.env.data.qpos[:] = self.init_qpos
            self.env.data.qvel[:] = 0.0
            mujoco.mj_forward(self.env.model, self.env.data)
            self.physics_dirty = False
            self.resets = 0

        def show_frame(self, frame: int, *, force_pose: bool = False) -> None:
            """Pose-mode frame display; also used to (re)anchor after loads."""
            frame = clamp_frame(frame, self.num_steps)
            if self.physics_dirty and (self.mode == "pose" or force_pose):
                self.restore_initial_state()
            if self.scene_qpos is not None:
                # Ground truth for every dof, objects included: nothing to
                # reconstruct, and the row overwrites whatever physics left.
                self.env.data.qpos[:] = self.scene_qpos[frame]
                self.env.data.qvel[:] = 0.0
            else:
                self.env._set_qpos_from_state(
                    np.asarray(self.states[frame], dtype=np.float32)
                )
            mujoco.mj_forward(self.env.model, self.env.data)
            self.frame = frame
            self.push_updates()

        def step_physics(self) -> None:
            """Advance one frame open loop; wraps back to the recorded start."""
            next_frame = self.frame + 1
            if next_frame >= self.num_steps:
                self.restore_initial_state()
                self.show_frame(0, force_pose=True)
                return
            # states[t] is the response to actions[t], so stepping
            # actions[next_frame] lands the sim on slot next_frame.
            bad_qacc = mujoco.mjtWarning.mjWARN_BADQACC
            self.env.step(np.asarray(self.actions[next_frame]), render_obs=False)
            if self.env.data.warning[bad_qacc].number:
                self.resets += 1
                self.env.data.warning[bad_qacc].number = 0
            self.physics_dirty = True
            self.frame = next_frame
            self.push_updates()

        def push_updates(self) -> None:
            nonlocal video_image
            self.scene.update_from_mjdata(self.env.data)
            if self.video is not None and self._video_shown != self.frame:
                pixels = self.video.frame(self.frame)
                if pixels is not None:
                    self._video_shown = self.frame
                    if video_image is None:
                        video_image = server.gui.add_image(
                            pixels, label="recorded video", format="jpeg",
                            order=_O_VIDEO,
                        )
                    else:
                        video_image.image = pixels
            if (
                self.frame_slider is not None
                and time.monotonic() - self._scrub_ts > SCRUB_BACKOFF_S
                and self._slider_write != self.frame
            ):
                self._slider_write = self.frame
                self.frame_slider.value = self.frame
            status.value = episode_status(
                self.name,
                self.task,
                self.frame,
                self.num_steps,
                self.mode,
                self.resets,
                full_scene=self.scene_qpos is not None,
            )

        def select_task(self, task: str) -> None:
            self.task = task
            names = tuple(p.name for p in self.task_episodes())
            self._episode_write = names[0]
            episode_dropdown.options = names
            episode_dropdown.value = names[0]
            self.begin_load(self.task_episodes()[0])

        def select_episode(self, name: str) -> None:
            self._episode_write = name
            episode_dropdown.value = name
            self.begin_load(
                next(p for p in self.task_episodes() if p.name == name)
            )

        def handle(self, command: tuple) -> None:
            kind = command[0]
            if kind == "task":
                self.select_task(command[1])
            elif kind == "episode":
                if command[1] == self._episode_write:
                    return  # echo of our own programmatic dropdown write
                self.select_episode(command[1])
            elif kind == "shift":
                paths = self.task_episodes()
                # Relative to the requested episode when a build is in flight,
                # so rapid Prev/Next clicks walk the list instead of piling on
                # the same neighbour.
                current = self.loading or self.name
                index = next(
                    (i for i, p in enumerate(paths) if p.name == current), 0
                )
                self.select_episode(paths[(index + command[1]) % len(paths)].name)
            elif kind == "loaded":
                self.install(command[1], command[2], command[3])
            elif kind == "load_failed":
                if command[1] == self.generation:
                    self.loading = None
                    status.value = (
                        f"failed to load {command[2]} - see terminal; "
                        "still on the previous episode"
                    )
            elif kind == "scrub":
                if (
                    self.env is None
                    or self.loading is not None
                    or int(command[1]) == self._slider_write
                ):
                    return  # echo of the playback thread's own slider write
                self._scrub_ts = time.monotonic()
                if self.mode == "pose":
                    self.show_frame(int(command[1]))
                # physics cannot jump; the slider snaps back after the backoff
            elif kind == "mode":
                self.mode = command[1]
                if self.env is None or self.loading is not None:
                    return
                if self.mode == "physics":
                    self.restore_initial_state()
                    self.show_frame(0, force_pose=True)
                else:
                    self.show_frame(self.frame, force_pose=True)

    player = Player()

    def playback_loop() -> None:
        # Every GUI control funnels through this one thread, so an uncaught
        # exception here presents as a dead panel under a healthy server, and
        # a frame whose work exceeds the playback period must still yield the
        # GIL or the websocket event loop that delivers the GUI callbacks
        # starves (dead controls at 100% CPU).
        player.handle(("task", initial_task))
        while True:
            with lock:
                pending, commands[:] = commands[:], []
                queued_since[0] = None
            for command in pending:
                try:
                    player.handle(command)
                except Exception:  # noqa: BLE001 - the GUI dies with this thread
                    traceback.print_exc()
                    status.value = (
                        f"error handling {command[0]!r} - see terminal; "
                        "still on the previous episode"
                    )
            if player.loading is not None:
                # The worker owns the viser scene during a build; touching the
                # old handles here would race its scene.reset().
                time.sleep(0.05)
                continue
            if not playing.value or player.env is None:
                if player.env is not None:
                    player.push_updates()  # keep slider/status honest while paused
                time.sleep(0.05)
                continue
            frame_start = time.perf_counter()
            wrapped = player.frame + 1 >= player.num_steps
            try:
                if player.mode == "pose":
                    player.show_frame((player.frame + 1) % player.num_steps)
                else:
                    player.step_physics()
            except Exception:  # noqa: BLE001 - pause rather than kill the panel
                traceback.print_exc()
                playing.value = False  # pause instead of spamming the error
                status.value = "error during playback - see terminal (paused)"
                continue
            if wrapped:
                print(
                    f"[loop] {player.name} frames={player.num_steps} "
                    f"mode={player.mode} resets={player.resets}",
                    flush=True,
                )
            sleep_s = playback_period_s(speed.value) - (
                time.perf_counter() - frame_start
            )
            time.sleep(max(sleep_s, 0.001))

    threading.Thread(target=playback_loop, daemon=True).start()
    print(f"Viser ready on port {actual_port}", flush=True)
    while True:
        time.sleep(60)


def main(cfg: VizEpisodeConfig) -> None:
    """Open the episode player."""
    run_episode_viewer(cfg)
