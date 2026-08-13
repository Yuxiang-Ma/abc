"""Shared diagnostic-video renderer for teleop, inference, and DAgger H5s.

The three recording types differ only in their plot data and annotations.  A
single renderer owns camera synchronization, layout, plots, and video output;
small mode-specific builders below supply those differences.
"""

from __future__ import annotations

import math
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from itertools import pairwise

import cv2
import h5py
import numpy as np

from deploy.recording import io as rio

WIDTH, HEIGHT = 1920, 1080
BG = (0xF5, 0xFA, 0xFD)  # BGR for #FDFAF5
PANEL_BORDER = (205, 205, 205)
GRID = (225, 225, 225)
TEXT = (45, 45, 45)
MUTED = (90, 90, 90)

DOF_LABELS = [
    *(f"L-j{i}" for i in range(6)),
    "L-grip",
    *(f"R-j{i}" for i in range(6)),
    "R-grip",
]

STATE_COLORS = {
    "POLICY_RUNNING": (44, 160, 44),
    "INTERVENTION": (14, 127, 255),
    "HOMING": (170, 95, 170),
    "SYNCED_TELEOP": (207, 190, 23),
    "PRE_RECORDING": (135, 135, 135),
    "PRE_RECORDING_UNLOCKED": (195, 135, 45),
    "FINISH_PENDING": (40, 130, 220),
    "UNKNOWN": (105, 105, 105),
}
STATE_LABELS = {
    "POLICY_RUNNING": "POLICY RUNNING",
    "INTERVENTION": "TELEOP / DAGGER",
    "HOMING": "HOMING",
    "SYNCED_TELEOP": "SYNCED TELEOP",
    "PRE_RECORDING": "PRE-RECORDING",
    "PRE_RECORDING_UNLOCKED": "UNLOCKED",
    "FINISH_PENDING": "FINISH PENDING",
    "UNKNOWN": "MODE UNKNOWN",
}


def _relative_seconds(ts_ns: np.ndarray, t0_ns: int) -> np.ndarray:
    return (np.asarray(ts_ns, dtype=np.float64) - float(t0_ns)) / 1e9


def _nearest_indices(source: np.ndarray, query: np.ndarray) -> np.ndarray:
    indices = np.clip(np.searchsorted(source, query), 0, len(source) - 1)
    previous = np.maximum(indices - 1, 0)
    use_previous = np.abs(source[previous] - query) < np.abs(source[indices] - query)
    return np.where(use_previous, previous, indices)


def _hex_bgr(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[i : i + 2], 16) for i in (4, 2, 0))


@dataclass
class Series:
    times: np.ndarray
    values: np.ndarray
    color: tuple[int, int, int]
    thickness: int = 1
    dashed: bool = False


@dataclass
class Panel:
    dof: int
    label: str
    y_min: float
    y_max: float
    series: list[Series] = field(default_factory=list)
    main: bool = True


@dataclass
class InferenceData:
    actions: np.ndarray
    exec_starts: np.ndarray
    action_ends: np.ndarray
    hold_ends: np.ndarray
    dt_control: float
    timeline_t: np.ndarray
    timeline_a: np.ndarray
    windows: list[tuple[float, float]]


@dataclass
class RenderData:
    kind: str
    t0_ns: int
    panels: list[Panel]
    metadata: str
    event_times: np.ndarray = field(default_factory=lambda: np.array([]))
    event_states: list[str] = field(default_factory=list)
    default_state: str = "UNKNOWN"
    stages: list[tuple[int, float]] = field(default_factory=list)
    checkpoints: list[tuple[int, float]] = field(default_factory=list)
    inference_times: np.ndarray = field(default_factory=lambda: np.array([]))
    export_segments: list[tuple[float, float, bool]] = field(default_factory=list)
    inference: InferenceData | None = None
    unusable: bool = False


class _FrameSink:
    """ffmpeg/libx264 output when available, with an OpenCV fallback."""

    def __init__(self, path: str, fps: int):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._proc = None
        self._writer = None
        if shutil.which("ffmpeg"):
            self._proc = subprocess.Popen(
                [
                    "ffmpeg",
                    "-y",
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "bgr24",
                    "-s",
                    f"{WIDTH}x{HEIGHT}",
                    "-r",
                    str(fps),
                    "-i",
                    "-",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-crf",
                    "20",
                    "-preset",
                    "fast",
                    "-movflags",
                    "+faststart",
                    "-an",
                    path,
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        else:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(path, fourcc, fps, (WIDTH, HEIGHT))
            if not self._writer.isOpened():
                raise RuntimeError(f"Could not open video writer for {path}")

    def write(self, frame: np.ndarray) -> None:
        if self._proc is not None:
            assert self._proc.stdin is not None
            self._proc.stdin.write(frame.tobytes())
        else:
            self._writer.write(frame)

    def close(self) -> None:
        if self._proc is not None:
            assert self._proc.stdin is not None
            self._proc.stdin.close()
            if self._proc.wait() != 0:
                raise RuntimeError("ffmpeg failed while rendering recording")
        else:
            self._writer.release()


def _ordered_cameras(f: h5py.File) -> list[tuple[str, str]]:
    streams = rio.camera_streams(f)
    names = [name for name in rio.CAMERA_KEYS if name in streams]
    names += sorted(set(streams) - set(names))
    return [(name, streams[name]) for name in names]


def _camera_regions(
    names: list[str], region: tuple[int, int, int, int], aspect: float, gap: int = 6
) -> dict[str, tuple[int, int, int, int]]:
    x0, y0, x1, y1 = region
    width, height = x1 - x0, y1 - y0
    top = "top" if "top" in names else "front" if "front" in names else None
    if top and "left" in names and "right" in names:
        side_w = (width - gap) // 2
        side_h = int(side_w / aspect)
        main_w, main_h = width, int(width / aspect)
        if side_h + gap + main_h > height:
            scale = height / (side_h + gap + main_h)
            side_w, side_h = int(side_w * scale), int(side_h * scale)
            main_w, main_h = int(main_w * scale), int(main_h * scale)
        regions = {
            "left": (x0, y0, x0 + side_w, y0 + side_h),
            "right": (
                x0 + side_w + gap,
                y0,
                x0 + 2 * side_w + gap,
                y0 + side_h,
            ),
        }
        main_x = x0 + (width - main_w) // 2
        regions[top] = (
            main_x,
            y0 + side_h + gap,
            main_x + main_w,
            y0 + side_h + gap + main_h,
        )
        return regions

    cols = 1 if len(names) <= 2 else 2
    rows = max(1, math.ceil(len(names) / cols))
    cell_w = (width - (cols - 1) * gap) // cols
    cell_h = (height - (rows - 1) * gap) // rows
    return {
        name: (
            x0 + (index % cols) * (cell_w + gap),
            y0 + (index // cols) * (cell_h + gap),
            x0 + (index % cols) * (cell_w + gap) + cell_w,
            y0 + (index // cols) * (cell_h + gap) + cell_h,
        )
        for index, name in enumerate(names)
    }


def _draw_letterboxed(
    frame: np.ndarray, image: np.ndarray, region: tuple[int, int, int, int]
) -> None:
    x0, y0, x1, y1 = region
    width, height = x1 - x0, y1 - y0
    frame[y0:y1, x0:x1] = (24, 24, 24)
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(
        image,
        (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))),
        interpolation=cv2.INTER_AREA,
    )
    y = y0 + (height - resized.shape[0]) // 2
    x = x0 + (width - resized.shape[1]) // 2
    frame[y : y + resized.shape[0], x : x + resized.shape[1]] = resized


def _draw_badge(frame: np.ndarray, text: str, origin: tuple[int, int]) -> None:
    x, y = origin
    (width, height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.42, 1)
    cv2.rectangle(
        frame,
        (x - 5, y - height - 4),
        (x + width + 5, y + baseline + 4),
        (255, 255, 255),
        -1,
    )
    cv2.rectangle(
        frame,
        (x - 5, y - height - 4),
        (x + width + 5, y + baseline + 4),
        (200, 200, 200),
        1,
    )
    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_DUPLEX,
        0.42,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )


def _stacked_regions(
    count: int, region: tuple[int, int, int, int], gap: int = 5
) -> list[tuple[int, int, int, int]]:
    if count == 0:
        return []
    x0, y0, x1, y1 = region
    height = (y1 - y0 - (count - 1) * gap) // count
    return [
        (x0, y0 + i * (height + gap), x1, y0 + i * (height + gap) + height)
        for i in range(count)
    ]


def _plot_regions(
    kind: str, panels: list[Panel]
) -> tuple[list[tuple[int, int, int, int]], tuple[int, int, int, int]]:
    header = 80 if kind == "dagger" else 32
    bottom = HEIGHT - (176 if kind == "dagger" else 12)
    main = [panel for panel in panels if panel.main]
    compact = [panel for panel in panels if not panel.main]
    regions = _stacked_regions(len(main), (650, header, WIDTH - 12, bottom))
    camera_area = (12, header, 630, bottom)
    if compact:
        compact_top = HEIGHT - 260
        camera_area = (12, header, 630, compact_top - 8)
        rows, cols = math.ceil(len(compact) / 5), min(5, len(compact))
        cell_w = (618 - (cols - 1) * 4) // cols
        cell_h = (HEIGHT - compact_top - 12 - (rows - 1) * 4) // rows
        for index in range(len(compact)):
            row, col = divmod(index, cols)
            x0 = 12 + col * (cell_w + 4)
            y0 = compact_top + row * (cell_h + 4)
            regions.append((x0, y0, x0 + cell_w, y0 + cell_h))
    return regions, camera_area


def _points(
    times: np.ndarray,
    values: np.ndarray,
    region: tuple[int, int, int, int],
    t_left: float,
    t_right: float,
    y_min: float,
    y_max: float,
) -> np.ndarray:
    x0, y0, x1, y1 = region
    x = x0 + (times - t_left) / max(t_right - t_left, 1e-9) * (x1 - x0)
    y = y0 + (y_max - values) / max(y_max - y_min, 1e-9) * (y1 - y0)
    points = np.stack([x, y], axis=-1)
    points[:, 0] = np.clip(points[:, 0], x0, x1)
    points[:, 1] = np.clip(points[:, 1], y0, y1)
    return points.astype(np.int32).reshape(-1, 1, 2)


def _visible(
    series: Series, left: float, right: float
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.flatnonzero((series.times >= left) & (series.times <= right))
    if len(indices) == 0:
        return np.array([]), np.array([])
    lo, hi = max(0, indices[0] - 1), min(len(series.times), indices[-1] + 2)
    return series.times[lo:hi], series.values[lo:hi]


def _draw_dashed(
    frame: np.ndarray,
    points: np.ndarray,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    flat = points.reshape(-1, 2)
    for start, end in pairwise(flat):
        delta = end - start
        length = float(np.linalg.norm(delta))
        if length == 0:
            continue
        for offset in np.arange(0, length, 12):
            a = start + delta * (offset / length)
            b = start + delta * (min(offset + 7, length) / length)
            cv2.line(
                frame,
                tuple(a.astype(int)),
                tuple(b.astype(int)),
                color,
                thickness,
                cv2.LINE_AA,
            )


def _draw_panel(
    frame: np.ndarray,
    panel: Panel,
    region: tuple[int, int, int, int],
    series: list[Series],
    t_left: float,
    t_right: float,
    t_now: float,
    *,
    bands: list[tuple[float, float]] = (),
    boundaries: np.ndarray | tuple = (),
    legend: list[tuple[str, tuple[int, int, int]]] = (),
) -> None:
    x0, y0, x1, y1 = region
    compact = not panel.main
    label_h = 16 if compact else 23
    footer_h = 3 if compact else 17
    plot = (x0 + (2 if compact else 48), y0 + label_h, x1 - 6, y1 - footer_h)
    px0, py0, px1, py1 = plot
    cv2.rectangle(frame, (x0, y0), (x1, y1), BG, -1)
    cv2.rectangle(frame, (x0, y0), (x1, y1), PANEL_BORDER, 1)
    for fraction in (0.25, 0.5, 0.75):
        y = round(py0 + fraction * (py1 - py0))
        cv2.line(frame, (px0, y), (px1, y), GRID, 1)
    cv2.putText(
        frame,
        panel.label.upper(),
        (x0 + 5, y0 + (11 if compact else 16)),
        cv2.FONT_HERSHEY_DUPLEX,
        0.28 if compact else 0.42,
        TEXT,
        1,
        cv2.LINE_AA,
    )
    legend_x = x0 + 100
    for label, color in legend:
        cv2.line(frame, (legend_x, y0 + 12), (legend_x + 16, y0 + 12), color, 2)
        cv2.putText(
            frame,
            label,
            (legend_x + 20, y0 + 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            TEXT,
            1,
            cv2.LINE_AA,
        )
        legend_x += 28 + cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.32, 1)[0][0]

    for start, end in bands:
        if end < t_left or start > t_right:
            continue
        bx0 = round(
            px0 + (max(start, t_left) - t_left) / (t_right - t_left) * (px1 - px0)
        )
        bx1 = round(
            px0 + (min(end, t_right) - t_left) / (t_right - t_left) * (px1 - px0)
        )
        if bx1 > bx0:
            roi = frame[py0:py1, bx0:bx1]
            overlay = np.full_like(roi, (44, 160, 44))
            cv2.addWeighted(overlay, 0.12, roi, 0.88, 0, roi)
    for boundary in boundaries:
        if t_left < boundary < t_right:
            x = round(px0 + (boundary - t_left) / (t_right - t_left) * (px1 - px0))
            for y in range(py0, py1, 7):
                cv2.circle(frame, (x, y), 1, (190, 190, 190), -1)

    for item in series:
        times, values = _visible(item, t_left, t_right)
        if len(times) < 2:
            continue
        points = _points(times, values, plot, t_left, t_right, panel.y_min, panel.y_max)
        if item.dashed:
            _draw_dashed(frame, points, item.color, item.thickness)
        else:
            cv2.polylines(
                frame, [points], False, item.color, item.thickness, cv2.LINE_AA
            )

    now_x = round(px0 + (t_now - t_left) / max(t_right - t_left, 1e-9) * (px1 - px0))
    if px0 <= now_x <= px1:
        cv2.line(frame, (now_x, py0), (now_x, py1), (0, 0, 200), 1)
    if not compact:
        cv2.putText(
            frame,
            f"{panel.y_max:.2f}",
            (x0 + 3, py0 + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.27,
            MUTED,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"{panel.y_min:.2f}",
            (x0 + 3, py1),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.27,
            MUTED,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"{t_left:.1f}s",
            (px0, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.27,
            MUTED,
            1,
            cv2.LINE_AA,
        )
        right = f"{t_right:.1f}s"
        width = cv2.getTextSize(right, cv2.FONT_HERSHEY_SIMPLEX, 0.27, 1)[0][0]
        cv2.putText(
            frame,
            right,
            (px1 - width, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.27,
            MUTED,
            1,
            cv2.LINE_AA,
        )


def _limits(values: list[np.ndarray]) -> tuple[float, float]:
    finite = np.concatenate(
        [np.asarray(value)[np.isfinite(value)] for value in values if len(value)]
    )
    if not len(finite):
        return -1.0, 1.0
    low, high = float(finite.min()), float(finite.max())
    margin = max(high - low, 1e-3) * 0.15
    return low - margin, high + margin


def _metadata(f: h5py.File, basename: str) -> str:
    attrs = f.attrs
    parts = [basename]
    task = attrs.get("task_name")
    if task:
        parts.append(rio.decode_attr(task))
    backend = attrs.get("dagger_backend") or attrs.get("backend")
    if backend:
        parts.append(f"backend {rio.decode_attr(backend)}")
    if attrs.get("episode_index") is not None:
        parts.append(f"episode {attrs['episode_index']}")
    checkpoint = attrs.get("checkpoint_path") or attrs.get("checkpoint_name")
    if checkpoint:
        checkpoint = rio.decode_attr(checkpoint).rstrip("/")
        parts.append("/".join(checkpoint.split("/")[-2:]))
    if attrs.get("model_size"):
        parts.append(rio.decode_attr(attrs["model_size"]))
    if attrs.get("diffusion_steps") is not None:
        parts.append(f"{attrs['diffusion_steps']} diffusion steps")
    if "inference" in f:
        rtc = "RTC" if rio.attr_bool(attrs.get("rtc")) else "No RTC"
        if rio.attr_bool(attrs.get("rtc")):
            prefix = attrs.get("rtc_prefix_length")
            lead = attrs.get("rtc_inference_lead_steps")
            if prefix is not None and lead is not None:
                rtc += f" prefix={prefix} lead={lead}"
        parts.append(rtc)
    if attrs.get("start_time"):
        parts.append(rio.decode_attr(attrs["start_time"]))
    return " | ".join(parts)


def _event_data(f: h5py.File, t0_ns: int) -> tuple[np.ndarray, list[str]]:
    if "dagger_events/timestamp_ns" in f:
        group = f["dagger_events"]
        return (
            _relative_seconds(group["timestamp_ns"][:], t0_ns),
            [rio.canonical_state(value) for value in group["controller_state"][:]],
        )
    if "controller_events" in f and "timestamps/controller_events" in f:
        return (
            _relative_seconds(f["timestamps/controller_events"][:], t0_ns),
            [rio.canonical_state(value) for value in f["controller_events/state"][:]],
        )
    return np.array([]), []


def _state_at(data: RenderData, time_s: float) -> str:
    index = int(np.searchsorted(data.event_times, time_s, side="right") - 1)
    return data.event_states[index] if index >= 0 else data.default_state


def _split_by_states(
    times: np.ndarray,
    values: np.ndarray,
    event_times: np.ndarray,
    event_states: list[str],
    allowed: set[str],
    color: tuple[int, int, int],
    default: str,
) -> list[Series]:
    if len(times) < 2:
        return []
    indices = np.searchsorted(event_times, times, side="right") - 1
    states = np.asarray(
        [event_states[index] if index >= 0 else default for index in indices]
    )
    mask = np.isin(states, list(allowed))
    series = []
    for start, end in rio.true_ranges(mask):
        lo, hi = max(0, start - 1), min(len(times), end + 1)
        if hi - lo >= 2:
            series.append(Series(times[lo:hi], values[lo:hi], color, 2))
    return series


def _teleop_data(f: h5py.File, t0_ns: int, basename: str) -> RenderData:
    panels = []
    for side_index, side in enumerate(("left", "right")):
        ts_key = f"timestamps/q_{side}"
        if ts_key not in f:
            continue
        times = _relative_seconds(f[ts_key][:], t0_ns)
        actual = f[f"data/q_{side}"][:]
        desired = f[f"data/q_des_{side}"][:]
        gripper = f[f"data/q_gripper_{side}"][:, 0]
        for local in range(7):
            dof = side_index * 7 + local
            values = [gripper] if local == 6 else [actual[:, local], desired[:, local]]
            y_min, y_max = _limits(values)
            series = [Series(times, values[0], _hex_bgr("#1f77b4"), 2)]
            if local < 6:
                series.append(Series(times, values[1], _hex_bgr("#ff7f0e")))
            panels.append(Panel(dof, DOF_LABELS[dof], y_min, y_max, series))

    stages = []
    if "stages/index" in f and "stages/start_ns" in f:
        starts = _relative_seconds(f["stages/start_ns"][:], t0_ns)
        stages = [
            (int(index), float(start))
            for index, start in zip(f["stages/index"][:], starts, strict=False)
        ]
    return RenderData("teleop", t0_ns, panels, _metadata(f, basename), stages=stages)


def _control_dt(f: h5py.File) -> float:
    for side in ("left", "right"):
        key = f"timestamps/q_{side}"
        if key in f and len(f[key]) > 1:
            dt = float(np.median(np.diff(f[key][:].astype(float))) / 1e9)
            if np.isfinite(dt) and dt > 0:
                return dt
    return 1 / 30


def _inference_timeline(f: h5py.File, t0_ns: int) -> tuple[InferenceData, np.ndarray]:
    actions = np.asarray(f["inference/actions"][:])
    publish = _relative_seconds(f["timestamps/inference"][:], t0_ns)
    chunks, chunk_length, _ = actions.shape
    dt_control = _control_dt(f)
    exec_starts = np.zeros(chunks)
    action_ends = np.zeros(chunks)
    hold_ends = np.zeros(chunks)
    previous_end = 0.0
    typical_duration = (
        np.median(np.diff(publish)) if chunks > 1 else chunk_length * dt_control
    )
    for index in range(chunks):
        start = max(float(publish[index]), previous_end)
        duration = (
            float(publish[index + 1] - publish[index])
            if index + 1 < chunks
            else float(typical_duration)
        )
        end = start + max(duration, 0.0)
        exec_starts[index] = start
        action_ends[index] = min(end, start + chunk_length * dt_control)
        hold_ends[index] = end
        previous_end = end

    times, values = [], []
    for index, chunk in enumerate(actions):
        chunk_times = np.linspace(
            exec_starts[index], action_ends[index], chunk_length, endpoint=False
        )
        chunk_times = np.append(chunk_times, [action_ends[index], hold_ends[index]])
        last = chunk[-1:]
        times.append(chunk_times)
        values.append(np.vstack([chunk, last, last]))

    starts = np.asarray(f["inference/start_ts"][:], dtype=float)
    ends = np.asarray(f["inference/end_ts"][:], dtype=float)
    anchor = ends[0] if len(ends) else 0.0
    windows = list(zip(starts - anchor, ends - anchor, strict=False))
    inference = InferenceData(
        actions,
        exec_starts,
        action_ends,
        hold_ends,
        dt_control,
        np.concatenate(times),
        np.concatenate(values),
        windows,
    )
    return inference, publish


def _inference_data(
    f: h5py.File, t0_ns: int, basename: str, main_dofs: list[int] | None
) -> RenderData:
    inference, _ = _inference_timeline(f, t0_ns)
    action_dim = inference.actions.shape[2]
    requested = [0, 4, 7, 11] if main_dofs is None else main_dofs
    main = [dof for dof in requested if 0 <= dof < action_dim]
    order = main + [dof for dof in range(action_dim) if dof not in main]
    panels = []
    for dof in order:
        y_min, y_max = _limits([inference.timeline_a[:, dof]])
        label = DOF_LABELS[dof] if dof < len(DOF_LABELS) else f"a{dof}"
        panels.append(Panel(dof, label, y_min, y_max, main=dof in main))
    return RenderData(
        "inference", t0_ns, panels, _metadata(f, basename), inference=inference
    )


def _dagger_data(f: h5py.File, t0_ns: int, basename: str) -> RenderData:
    event_times, event_states = _event_data(f, t0_ns)
    default = "PRE_RECORDING" if event_states else "UNKNOWN"
    action_timeline = None
    if "inference/actions" in f and "timestamps/inference" in f:
        actions = np.asarray(f["inference/actions"][:])
        publish = _relative_seconds(f["timestamps/inference"][:], t0_ns)
        times = [
            publish[index] + np.arange(chunk.shape[0]) * _control_dt(f)
            for index, chunk in enumerate(actions)
        ]
        action_timeline = (np.concatenate(times), np.concatenate(actions))

    action_dim = action_timeline[1].shape[1] if action_timeline else 14
    dofs = [0, 4, 7, 11] if action_dim >= 12 else list(range(min(4, action_dim)))
    panels = []
    for dof in dofs:
        side, local = ("left", dof) if dof < 7 else ("right", dof - 7)
        ts_key = f"timestamps/q_{side}"
        if ts_key not in f:
            continue
        times = _relative_seconds(f[ts_key][:], t0_ns)
        actual = (
            f[f"data/q_{side}"][:, local]
            if local < 6
            else f[f"data/q_gripper_{side}"][:, 0]
        )
        values = [actual]
        series = [Series(times, actual, (128, 128, 128))]
        desired_key = f"data/q_des_{side}"
        if desired_key in f and f[desired_key].shape[1] > local:
            desired = np.asarray(f[desired_key][:, local])
            values.append(desired)
            series += _split_by_states(
                times,
                desired,
                event_times,
                event_states,
                {"INTERVENTION"},
                STATE_COLORS["INTERVENTION"],
                default,
            )
        if action_timeline and dof < action_timeline[1].shape[1]:
            policy_t, policy_a = action_timeline
            policy = policy_a[:, dof]
            values.append(policy)
            series += _split_by_states(
                policy_t,
                policy,
                event_times,
                event_states,
                {"POLICY_RUNNING"},
                STATE_COLORS["POLICY_RUNNING"],
                default,
            )
        y_min, y_max = _limits(values)
        panels.append(Panel(dof, DOF_LABELS[dof], y_min, y_max, series))

    checkpoints = []
    if "dagger_checkpoints/timestamp_ns" in f:
        times = _relative_seconds(f["dagger_checkpoints/timestamp_ns"][:], t0_ns)
        indices = (
            f["dagger_checkpoints/index"][:]
            if "dagger_checkpoints/index" in f
            else np.arange(1, len(times) + 1)
        )
        checkpoints = [
            (int(index), float(time_s))
            for index, time_s in zip(indices, times, strict=False)
        ]
    inference_times = (
        _relative_seconds(f["timestamps/inference"][:], t0_ns)
        if "timestamps/inference" in f
        else np.array([])
    )
    return RenderData(
        "dagger",
        t0_ns,
        panels,
        _metadata(f, basename),
        event_times,
        event_states,
        default,
        checkpoints=checkpoints,
        inference_times=inference_times,
        unusable=rio.is_unusable(f),
    )


def _inference_series(data: InferenceData, panel: Panel, now: float) -> list[Series]:
    chunk = int(
        np.clip(
            np.searchsorted(data.exec_starts, now, side="right") - 1,
            0,
            len(data.exec_starts) - 1,
        )
    )
    start, action_end, hold_end = (
        data.exec_starts[chunk],
        data.action_ends[chunk],
        data.hold_ends[chunk],
    )
    chunk_length = data.actions.shape[1]
    step = (
        int(np.clip((now - start) / data.dt_control, 0, chunk_length - 1))
        if now < action_end
        else chunk_length - 1
    )
    step_times = np.linspace(start, action_end, chunk_length, endpoint=False)
    dof = panel.dof
    previous_cutoff = data.exec_starts[chunk - 1] if panel.main and chunk > 0 else start
    past = data.timeline_t < previous_cutoff
    result = [
        Series(data.timeline_t[past], data.timeline_a[past, dof], (170, 170, 170))
    ]
    if panel.main and chunk > 0:
        prev_start, prev_end = data.exec_starts[chunk - 1], data.action_ends[chunk - 1]
        prev_times = np.append(
            np.linspace(prev_start, prev_end, chunk_length, endpoint=False), prev_end
        )
        prev_values = np.append(
            data.actions[chunk - 1, :, dof], data.actions[chunk - 1, -1, dof]
        )
        result.append(Series(prev_times, prev_values, _hex_bgr("#17becf"), 2))
        prev_hold_end = data.hold_ends[chunk - 1]
        if prev_hold_end - prev_end > 1e-3:
            value = data.actions[chunk - 1, -1, dof]
            result.append(
                Series(
                    np.array([prev_end, prev_hold_end]),
                    np.array([value, value]),
                    (100, 100, 220),
                    dashed=True,
                )
            )
    result.append(
        Series(
            step_times[: step + 1],
            data.actions[chunk, : step + 1, dof],
            _hex_bgr("#1f77b4"),
            2 if panel.main else 1,
        )
    )
    result.append(
        Series(
            np.append(step_times[step:], action_end),
            np.append(data.actions[chunk, step:, dof], data.actions[chunk, -1, dof]),
            _hex_bgr("#ff7f0e"),
            2 if panel.main else 1,
            dashed=True,
        )
    )
    if panel.main and hold_end - action_end > 1e-3:
        value = data.actions[chunk, -1, dof]
        result.append(
            Series(
                np.array([action_end, hold_end]),
                np.array([value, value]),
                (0, 0, 220),
                2,
                dashed=True,
            )
        )
    return result


def _state_segments(data: RenderData, duration: float):
    starts = np.concatenate([[0.0], np.maximum(data.event_times, 0.0)])
    states = [data.default_state, *data.event_states]
    ends = np.append(starts[1:], duration)
    return [
        (float(start), min(float(end), duration), state)
        for start, end, state in zip(starts, ends, states, strict=False)
        if end > start and start < duration
    ]


def _export_segments(
    f: h5py.File, frame_times: np.ndarray, frame_ts: np.ndarray
) -> list[tuple[float, float, bool]]:
    duration = float(frame_times[-1]) if len(frame_times) else 0.0
    if rio.is_unusable(f):
        return [(0.0, duration, False)]
    valid = rio.valid_mask(f, frame_ts)
    if valid.all():
        return [(0.0, duration, True)]
    segments = []
    changes = np.flatnonzero(np.diff(valid.astype(np.int8))) + 1
    for start, end in zip(
        np.concatenate([[0], changes]),
        np.concatenate([changes, [len(valid)]]),
        strict=True,
    ):
        segments.append(
            (
                float(frame_times[start]),
                float(frame_times[min(end, len(frame_times) - 1)]),
                bool(valid[start]),
            )
        )
    return segments


def _draw_header(
    frame: np.ndarray, data: RenderData, now: float, duration: float
) -> None:
    if data.kind != "dagger":
        title = f"t = {now:.2f}s / {duration:.1f}s | {data.metadata}"
        if data.kind == "teleop" and data.stages:
            index = max(
                0,
                np.searchsorted([start for _, start in data.stages], now, side="right")
                - 1,
            )
            title += f" | stage {data.stages[index][0]}/{len(data.stages)}"
        elif data.inference and any(
            start <= now <= end for start, end in data.inference.windows
        ):
            title += " | INFERRING"
        cv2.putText(
            frame,
            title,
            (12, 22),
            cv2.FONT_HERSHEY_DUPLEX,
            0.48,
            TEXT,
            1,
            cv2.LINE_AA,
        )
        return

    state = _state_at(data, now)
    color = STATE_COLORS.get(state, STATE_COLORS["UNKNOWN"])
    cv2.rectangle(frame, (12, 10), (330, 66), color, -1)
    cv2.putText(
        frame,
        STATE_LABELS.get(state, state),
        (24, 39),
        cv2.FONT_HERSHEY_DUPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"t = {now:.2f}s",
        (24, 58),
        cv2.FONT_HERSHEY_DUPLEX,
        0.4,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    metadata = data.metadata
    max_chars = 105 if not data.unusable else 72
    if len(metadata) > max_chars:
        metadata = metadata[: max_chars - 3].rstrip() + "..."
    cv2.putText(
        frame,
        metadata,
        (352, 40),
        cv2.FONT_HERSHEY_DUPLEX,
        0.45,
        TEXT,
        1,
        cv2.LINE_AA,
    )
    if data.unusable:
        cv2.rectangle(frame, (WIDTH - 300, 10), (WIDTH - 12, 66), (35, 35, 210), -1)
        cv2.putText(
            frame,
            "UNUSABLE - DO NOT TRAIN",
            (WIDTH - 282, 45),
            cv2.FONT_HERSHEY_DUPLEX,
            0.58,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


def _draw_dagger_timeline(
    frame: np.ndarray, data: RenderData, now: float, duration: float
) -> None:
    x0, y0, x1, y1 = 12, HEIGHT - 164, WIDTH - 12, HEIGHT - 76
    cv2.rectangle(frame, (x0, y0), (x1, y1), BG, -1)
    cv2.rectangle(frame, (x0, y0), (x1, y1), PANEL_BORDER, 1)
    cv2.putText(
        frame,
        "mode",
        (x0 + 8, y0 + 39),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.36,
        MUTED,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "export",
        (x0 + 8, y0 + 69),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.36,
        MUTED,
        1,
        cv2.LINE_AA,
    )
    bar_x0, bar_x1 = x0 + 72, x1 - 8
    scale = (bar_x1 - bar_x0) / max(duration, 1e-9)
    for start, end, state in _state_segments(data, duration):
        cv2.rectangle(
            frame,
            (round(bar_x0 + start * scale), y0 + 20),
            (
                max(round(bar_x0 + start * scale) + 1, round(bar_x0 + end * scale)),
                y0 + 43,
            ),
            STATE_COLORS.get(state, STATE_COLORS["UNKNOWN"]),
            -1,
        )
    for start, end, keep in data.export_segments:
        cv2.rectangle(
            frame,
            (round(bar_x0 + start * scale), y0 + 51),
            (
                max(round(bar_x0 + start * scale) + 1, round(bar_x0 + end * scale)),
                y0 + 73,
            ),
            (76, 175, 80) if keep else (55, 55, 210),
            -1,
        )
    for index, timestamp in data.checkpoints:
        if not 0 <= timestamp <= duration:
            continue
        x = round(bar_x0 + timestamp * scale)
        cv2.line(frame, (x, y0 + 12), (x, y0 + 79), (0, 190, 255), 2)
        cv2.putText(
            frame,
            f"C{index}",
            (x + 3, y0 + 17),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            TEXT,
            1,
            cv2.LINE_AA,
        )
    for timestamp in data.inference_times:
        if 0 <= timestamp <= duration:
            x = round(bar_x0 + timestamp * scale)
            cv2.line(frame, (x, y0 + 16), (x, y0 + 47), (20, 130, 20), 1)
    x = round(bar_x0 + now * scale)
    cv2.line(frame, (x, y0 + 12), (x, y0 + 80), (0, 0, 0), 2)


def _legend(kind: str) -> list[tuple[str, tuple[int, int, int]]]:
    if kind == "teleop":
        return [
            ("actual", _hex_bgr("#1f77b4")),
            ("desired", _hex_bgr("#ff7f0e")),
        ]
    if kind == "inference":
        return [
            ("past", (170, 170, 170)),
            ("previous", _hex_bgr("#17becf")),
            ("executed", _hex_bgr("#1f77b4")),
            ("horizon", _hex_bgr("#ff7f0e")),
            ("hold", (0, 0, 220)),
            ("inference", (44, 160, 44)),
        ]
    return [
        ("actual q", (128, 128, 128)),
        ("dagger command", STATE_COLORS["INTERVENTION"]),
        ("policy action", STATE_COLORS["POLICY_RUNNING"]),
    ]


def render_recording(
    h5_path: str,
    video_path: str,
    *,
    recording_type: str = "auto",
    fps: int = 30,
    main_dofs: list[int] | None = None,
) -> str:
    """Render one recording while preserving each type's diagnostic overlays."""
    basename = os.path.basename(h5_path)
    with h5py.File(h5_path, "r") as f:
        kind = (
            rio.detect_recording_type(f) if recording_type == "auto" else recording_type
        )
        cameras = _ordered_cameras(f)
        if not cameras:
            raise ValueError(f"No camera streams found in {h5_path}")
        ref_name, ref_key = next(
            ((name, key) for name, key in cameras if name == "top"), cameras[0]
        )
        ref_ts = rio.stream_timestamps(f, ref_key)
        if not len(ref_ts):
            raise ValueError(f"Reference camera {ref_name!r} has no frames")
        t0_ns = (
            int(f["timestamps/inference"][0]) if kind == "inference" else int(ref_ts[0])
        )
        frame_times = _relative_seconds(ref_ts, t0_ns)
        duration = float(frame_times[-1]) if len(frame_times) else 0.0

        if kind == "dagger":
            data = _dagger_data(f, t0_ns, basename)
        elif kind == "inference":
            if "inference/actions" not in f:
                raise ValueError(f"{h5_path} has no inference data")
            data = _inference_data(f, t0_ns, basename, main_dofs)
        else:
            data = _teleop_data(f, t0_ns, basename)
        data.export_segments = _export_segments(f, frame_times, ref_ts)

        first = rio.decode_frame(f, ref_key, 0)
        plot_regions, camera_area = _plot_regions(kind, data.panels)
        camera_regions = _camera_regions(
            [name for name, _ in cameras], camera_area, first.shape[1] / first.shape[0]
        )
        indices = {}
        for name, key in cameras:
            if key == ref_key:
                indices[name] = np.arange(len(ref_ts))
            else:
                camera_times = _relative_seconds(rio.stream_timestamps(f, key), t0_ns)
                indices[name] = _nearest_indices(camera_times, frame_times)

        sink = _FrameSink(video_path, fps)
        print(f"Rendering {len(ref_ts)} {kind} frames from {basename}")
        try:
            for frame_index, now in enumerate(frame_times):
                now = float(now)
                frame = np.full((HEIGHT, WIDTH, 3), BG, dtype=np.uint8)
                _draw_header(frame, data, now, duration)
                for name, key in cameras:
                    region = camera_regions.get(name)
                    if region is None:
                        continue
                    image = rio.decode_frame(f, key, int(indices[name][frame_index]))
                    _draw_letterboxed(frame, image, region)
                    _draw_badge(frame, name.upper(), (region[0] + 10, region[1] + 24))

                past = 2.0 if kind == "teleop" else 1.5
                future = 0.5
                left = max(0.0, now - past) if kind != "inference" else now - past
                right = left + past + future
                if kind != "inference" and right > duration:
                    right, left = duration, max(0.0, duration - past - future)
                for panel, region in zip(data.panels, plot_regions, strict=True):
                    series = (
                        _inference_series(data.inference, panel, now)
                        if data.inference
                        else panel.series
                    )
                    _draw_panel(
                        frame,
                        panel,
                        region,
                        series,
                        left,
                        right,
                        now,
                        bands=data.inference.windows if data.inference else (),
                        boundaries=data.inference.exec_starts if data.inference else (),
                        legend=_legend(kind) if panel is data.panels[0] else (),
                    )
                if kind == "dagger":
                    _draw_dagger_timeline(frame, data, now, duration)
                sink.write(frame)
        finally:
            sink.close()
    print(f"Video saved: {video_path}")
    return video_path
