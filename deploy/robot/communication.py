import json
import struct
import time

import numpy as np
import zmq


def create_publisher(
    context: zmq.Context,
    topic: str,
    linger: int | None = None,
    send_timeout: int | None = None,
) -> zmq.Socket:
    publisher = context.socket(zmq.PUB)
    if linger is not None:
        publisher.setsockopt(zmq.LINGER, linger)
    if send_timeout is not None:
        publisher.setsockopt(zmq.SNDTIMEO, send_timeout)
    if "://" not in topic:
        publisher.bind(f"ipc:///tmp/{topic}")
    else:
        publisher.bind(topic)
    return publisher


def publish(
    publisher: zmq.Socket,
    message: np.ndarray,
    extras: dict | None = None,
) -> None:
    header = {
        "timestamp": time.perf_counter(),
        "shape": list(message.shape),
        "dtype": message.dtype.str,
        "extras": extras or {},
    }
    header_b = json.dumps(header).encode("utf-8")
    payload = np.ascontiguousarray(message).tobytes()
    buffer = struct.pack("!I", len(header_b)) + header_b + payload
    publisher.send(buffer, copy=False)


def create_subscriber(
    context: zmq.Context, topic: str, conflate: int | None = None
) -> zmq.Socket:
    subscriber = context.socket(zmq.SUB)
    if conflate is not None:
        subscriber.setsockopt(zmq.CONFLATE, conflate)
    subscriber.setsockopt_string(zmq.SUBSCRIBE, "")
    if "://" not in topic:
        subscriber.connect(f"ipc:///tmp/{topic}")
    else:
        subscriber.connect(topic)
    return subscriber


class SubscribeTimeout(RuntimeError):
    """Raised by ``subscribe`` when ``timeout_ms`` elapses with no message.

    Lets callers (e.g. YAMEnv.get_obs) convert a silent follower/camera stall
    into a loud crash instead of an indefinite block.
    """


def subscribe(
    subscriber: zmq.Socket,
    timeout_ms: int | None = None,
    topic_label: str | None = None,
) -> tuple[np.ndarray, dict]:
    if timeout_ms is not None and not subscriber.poll(timeout=int(timeout_ms)):
        label = topic_label or str(subscriber)
        raise SubscribeTimeout(f"no message on {label} within {timeout_ms}ms")

    buffer = subscriber.recv()
    (header_length,) = struct.unpack("!I", buffer[:4])
    header = json.loads(buffer[4 : 4 + header_length].decode("utf-8"))
    message = np.frombuffer(
        buffer[4 + header_length :], dtype=np.dtype(header["dtype"])
    ).reshape(header["shape"])
    extras = header.get("extras", {}) or {}
    extras["timestamp"] = header.get("timestamp")
    return message.copy(), extras
