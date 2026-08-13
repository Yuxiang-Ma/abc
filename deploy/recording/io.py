"""Shared H5 reading layer for deploy recordings.

One implementation of the four concerns every post-processing tool needs, so
the renderers and the exporter can never disagree about how a recording is
read:

- recording-type detection (teleop / inference / dagger),
- camera stream discovery with tolerant naming (bare names + legacy aliases),
- synchronization of unsynced streams onto a base camera timeline,
- controller-state labels and discard/usable masks.

Recordings are written by ``deploy/robot/recorders``: every stream lives under
``data/<name>`` with a parallel ``timestamps/<name>`` array of nanosecond
timestamps. Cameras are variable-length JPEG (BGR-encoded); follower telemetry
is ``q_<side>`` (6), ``q_gripper_<side>`` (1), ``q_vel_/q_eff_/q_des_<side>``
(7 each). DAgger recordings add ``dagger_events`` (sparse state transitions),
``dagger_checkpoints``, and ``discard_segments``.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import h5py
import numpy as np

RECORDING_TYPES = ("teleop", "inference", "dagger")

# Logical camera order used across training, eval, and deploy.
CAMERA_KEYS = ("top", "left", "right")

# Legacy dataset-name aliases (older recordings used rgb_-prefixed and
# front-based names). Applied after stripping an optional "rgb_" prefix.
CAMERA_ALIASES = {
    "front": "top",
    "cam_front": "top",
    "cam_top": "top",
    "cam_left": "left",
    "cam_right": "right",
}

# State/action stream layout for export: 14-dim state and action vectors
# matching abc_minimal (left arm 6 + gripper, right arm 6 + gripper).
STATE_KEYS = ("q_left", "q_gripper_left", "q_right", "q_gripper_right")
ACTION_KEYS = ("q_des_left", "q_des_right")

# DAgger controller states, canonicalized. deploy-port records the lowercase
# DaggerState values; upstream recordings used richer/legacy vocabularies.
STATE_ALIASES = {
    "POLICY": "POLICY_RUNNING",
    "GELLO_INTERVENTION": "INTERVENTION",
    "SYNCED_GELLO": "SYNCED_TELEOP",
}


def canonical_state(state) -> str:
    if isinstance(state, bytes):
        state = state.decode("utf-8", errors="replace")
    state = str(state).strip().upper().replace(" ", "_")
    return STATE_ALIASES.get(state, state)


def decode_attr(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def attr_bool(value) -> bool:
    if value is None:
        return False
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


# ---------------------------------------------------------------------------
# Recording type
# ---------------------------------------------------------------------------


def detect_recording_type(f: h5py.File) -> str:
    """Return "teleop", "inference", or "dagger" for an open recording."""
    stamped = f.attrs.get("recording_type")
    if stamped is not None:
        stamped = decode_attr(stamped)
        if stamped in RECORDING_TYPES:
            return stamped
    if attr_bool(f.attrs.get("dagger")) or "dagger_events" in f:
        return "dagger"
    if "inference" in f:
        return "inference"
    return "teleop"


# ---------------------------------------------------------------------------
# Stream discovery and decoding
# ---------------------------------------------------------------------------


def camera_streams(f: h5py.File) -> dict[str, str]:
    """Map logical camera names ("top"/"left"/"right") to data/ dataset keys."""
    mapping: dict[str, str] = {}
    data = f.get("data")
    if data is None:
        return mapping
    for key in data:
        dataset = data[key]
        if h5py.check_vlen_dtype(dataset.dtype) is None:
            continue
        name = key.removeprefix("rgb_")
        name = CAMERA_ALIASES.get(name, name)
        mapping.setdefault(name, key)
    return mapping


def decode_frame(f: h5py.File, stream_key: str, index: int) -> np.ndarray:
    """Decode one camera JPEG to a BGR HWC uint8 image."""
    buf = np.asarray(f["data"][stream_key][index], dtype=np.uint8)
    image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not decode frame {index} of {stream_key!r}")
    return image


def stream_timestamps(f: h5py.File, stream_key: str) -> np.ndarray:
    return np.asarray(f["timestamps"][stream_key][:], dtype=np.uint64)


def stream_values(f: h5py.File, stream_key: str) -> np.ndarray:
    return np.asarray(f["data"][stream_key][:])


# ---------------------------------------------------------------------------
# Timeline synchronization
# ---------------------------------------------------------------------------


@dataclass
class SyncedRecording:
    """Streams joined onto a base camera timeline.

    ``base_ts_ns[i]`` is the timestamp of base tick ``i``; ``indices[key][i]``
    is the index of the latest sample of ``key`` at or before that tick.
    """

    base_key: str
    base_ts_ns: np.ndarray
    indices: dict[str, np.ndarray]
    avg_hz: float

    def gather(self, f: h5py.File, stream_key: str) -> np.ndarray:
        """Return the synchronized samples of a telemetry stream."""
        return np.asarray(f["data"][stream_key][:])[self.indices[stream_key]]


def sync_streams(
    f: h5py.File, stream_keys: list[str], base_key: str | None = None
) -> SyncedRecording:
    """Join streams by latest-sample-at-or-before onto a base timeline.

    The base timeline is the base stream's own timestamps, trimmed at the
    start so every joined stream already has at least one sample.
    """
    cameras = camera_streams(f)
    if base_key is None:
        base_key = cameras.get("top") or next(iter(cameras.values()), None)
    if base_key is None:
        raise ValueError("recording has no camera streams to use as a base timeline")

    base_ts = stream_timestamps(f, base_key)
    keys = list(dict.fromkeys([base_key, *stream_keys]))
    all_ts = {key: stream_timestamps(f, key) for key in keys}
    for key, ts in all_ts.items():
        if len(ts) == 0:
            raise ValueError(f"stream {key!r} has no samples")

    indices = {
        key: np.searchsorted(ts, base_ts, side="right").astype(np.int64) - 1
        for key, ts in all_ts.items()
    }
    first_valid = max(int(np.argmax(idx >= 0)) for idx in indices.values())
    if any(idx[first_valid] < 0 for idx in indices.values()):
        raise ValueError("streams never overlap on a common timeline")

    base_ts = base_ts[first_valid:]
    indices = {key: idx[first_valid:] for key, idx in indices.items()}
    if len(base_ts) > 1:
        avg_hz = float(1e9 * (len(base_ts) - 1) / (base_ts[-1] - base_ts[0]))
    else:
        avg_hz = 0.0
    return SyncedRecording(
        base_key=base_key, base_ts_ns=base_ts, indices=indices, avg_hz=avg_hz
    )


# ---------------------------------------------------------------------------
# Controller-state labels (dagger)
# ---------------------------------------------------------------------------


def dagger_state_labels(f: h5py.File, base_ts_ns: np.ndarray) -> np.ndarray:
    """Per-tick canonical controller state from the sparse dagger event log."""
    labels = np.full(len(base_ts_ns), "UNKNOWN", dtype=object)
    if "dagger_events" not in f:
        return labels
    events = f["dagger_events"]
    event_ts = np.asarray(events["timestamp_ns"][:], dtype=np.uint64)
    states = [canonical_state(s) for s in events["controller_state"][:]]
    if len(event_ts) == 0:
        return labels
    idxs = np.searchsorted(event_ts, base_ts_ns.astype(np.uint64), side="right") - 1
    for i, idx in enumerate(idxs):
        if idx >= 0:
            labels[i] = states[idx]
    return labels


def state_mask(labels: np.ndarray, target_state: str) -> np.ndarray:
    return labels == canonical_state(target_state)


# ---------------------------------------------------------------------------
# Validity masks
# ---------------------------------------------------------------------------


def is_unusable(f: h5py.File) -> bool:
    attrs = f.attrs
    usable = attrs.get("usable")
    return (
        attr_bool(attrs.get("unusable"))
        or attr_bool(attrs.get("discarded"))
        or (usable is not None and not attr_bool(usable))
    )


def discard_segments(f: h5py.File) -> list[tuple[int, int]]:
    if "discard_segments/start_ns" not in f or "discard_segments/end_ns" not in f:
        return []
    starts = np.asarray(f["discard_segments/start_ns"][:], dtype=np.uint64)
    ends = np.asarray(f["discard_segments/end_ns"][:], dtype=np.uint64)
    return [
        (int(start), int(end))
        for start, end in zip(starts, ends, strict=False)
        if int(end) >= int(start)
    ]


def valid_mask(f: h5py.File, base_ts_ns: np.ndarray) -> np.ndarray:
    """True for ticks kept by the operator's discard decisions.

    A discard segment starts at the latest checkpoint: the checkpoint sample
    itself is preserved, samples strictly after it up to end_ns are dropped.
    """
    mask = np.ones(len(base_ts_ns), dtype=bool)
    ts = base_ts_ns.astype(np.uint64)
    for start_ns, end_ns in discard_segments(f):
        mask &= ~((ts > np.uint64(start_ns)) & (ts <= np.uint64(end_ns)))
    return mask


def true_ranges(mask: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous [start, end) index ranges where the mask is True."""
    if mask.size == 0:
        return []
    padded = np.pad(mask.astype(bool), (1, 1), constant_values=False)
    starts = np.flatnonzero(~padded[:-1] & padded[1:])
    ends = np.flatnonzero(padded[:-1] & ~padded[1:])
    return [(int(s), int(e)) for s, e in zip(starts, ends, strict=True)]


def containing_range(
    ranges: list[tuple[int, int]], start: int, end: int
) -> tuple[int, int]:
    """Return the range containing [start, end), or the span itself."""
    for range_start, range_end in ranges:
        if range_start <= start and end <= range_end:
            return range_start, range_end
    return start, end
