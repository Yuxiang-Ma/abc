"""Base class for fixed-rate ZMQ deployment nodes."""

import signal
import time
import traceback
from abc import ABC, abstractmethod

import numpy as np
import zmq

import deploy.robot.communication as comms


class Node(ABC):
    def __init__(self, name: str, control_rate: float = -1, verbose: bool = True):
        self._name = name
        self._control_rate = control_rate
        self._period = 1 / control_rate
        self._verbose = verbose
        self._zmq_context = zmq.Context()
        self._publishers = {}
        self._subscribers = {}

    def create_publisher(self, topic: str, linger=None, send_timeout=None) -> None:
        self._publishers[topic] = comms.create_publisher(
            self._zmq_context, topic, linger, send_timeout
        )

    def publish(
        self, topic: str, message: np.ndarray, extras: dict | None = None
    ) -> None:
        try:
            publisher = self._publishers[topic]
        except KeyError as error:
            raise ValueError(f"Publisher for {topic!r} was not created") from error
        comms.publish(publisher, message, extras or {})

    def create_subscriber(self, topic: str, conflate=None) -> None:
        self._subscribers[topic] = comms.create_subscriber(
            self._zmq_context, topic, conflate
        )

    def subscribe(self, topic: str, block: bool = True):
        try:
            subscriber = self._subscribers[topic]
        except KeyError as error:
            raise ValueError(f"Subscriber for {topic!r} was not created") from error
        if block:
            # Polling lets Python deliver KeyboardInterrupt between receives.
            while not subscriber.poll(timeout=100):
                pass
            return comms.subscribe(subscriber)
        if subscriber.poll(timeout=0):
            return comms.subscribe(subscriber)
        return None, {}

    def run(self) -> None:
        def stop_on_sigterm(_signum, _frame):
            raise SystemExit(0)

        signal.signal(signal.SIGTERM, stop_on_sigterm)
        try:
            self.initial_bootup()
            next_tick = time.perf_counter()
            while True:
                self.tick()
                if self._control_rate < 0:
                    continue
                next_tick += self._period
                while (remaining := next_tick - time.perf_counter()) > 0:
                    if remaining > 3e-4:
                        time.sleep(remaining - 1e-4)
                next_tick = time.perf_counter()
        except (KeyboardInterrupt, SystemExit):
            pass
        finally:
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            try:
                self.on_shutdown()
            except Exception:
                traceback.print_exc()
            for socket in (*self._publishers.values(), *self._subscribers.values()):
                socket.setsockopt(zmq.LINGER, 0)
                socket.close()
            self._zmq_context.term()

    @abstractmethod
    def initial_bootup(self) -> None:
        pass

    @abstractmethod
    def tick(self) -> None:
        pass

    @abstractmethod
    def on_shutdown(self) -> None:
        pass
