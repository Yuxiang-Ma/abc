import os
import select
import sys
import termios
import time
import tty

import numpy as np
import tyro

from deploy.robot.key_listeners.key_listener_config import KeyListenerConfig
from deploy.robot.node import Node

# Pedal switches exhibit contact bounce: one physical press can produce
# several electrical press edges within tens of milliseconds.
EVDEV_DEBOUNCE_S = 0.1


class KeyDebouncer:
    """Suppress duplicate press edges of the same key within a cooldown."""

    def __init__(self, cooldown_s: float = EVDEV_DEBOUNCE_S):
        self._cooldown_s = float(cooldown_s)
        self._last: dict[str, float] = {}

    def accept(self, key: str, now: float) -> bool:
        last = self._last.get(key, float("-inf"))
        if now - last < self._cooldown_s:
            return False
        self._last[key] = now
        return True


class KeyListenerNode(Node):
    def __init__(
        self,
        name: str,
        control_rate: float,
        cooldown_s: float = 1.0,
        input_device: str = "",
        input_keys: str = "a,b,c",
        grab_device: bool = True,
    ):
        super().__init__(name, control_rate, verbose=False)

        self.leader_topic_name = f"{self._name}_key_presses"
        self.old = None
        self._stdin_available = False
        self._cooldown_s = float(cooldown_s)
        # Launchers resolve the device (config > FOOT_PEDAL_INPUT_DEVICE env >
        # PCsensor default) via key_listeners.pedal before spawning this node.
        self._input_device_path = input_device
        self._input_keys = self._normalize_key_names(
            os.environ.get("FOOT_PEDAL_INPUT_KEY", input_keys)
        )
        self._grab_device = bool(grab_device)
        self._input_device = None
        self._debouncer = KeyDebouncer()
        self.create_publisher(self.leader_topic_name)

    @staticmethod
    def _normalize_key_name(key: str) -> str:
        key = str(key)
        upper = key.upper()
        if upper.startswith("KEY_") and len(upper) == 5 and upper[-1].isalpha():
            return upper[-1].lower()
        if upper == "KEY_SPACE":
            return " "
        return key.lower() if len(key) == 1 else key

    @classmethod
    def _normalize_key_names(cls, keys: str) -> set[str]:
        return {
            cls._normalize_key_name(key)
            for key in str(keys).replace(",", " ").split()
            if key
        }

    def initial_bootup(self) -> None:
        if os.environ.get("DEPLOY_VERBOSE"):
            print(f"[{self._name}] Key listener node initial bootup complete.")
        self.last_key_time = float("-inf")
        if getattr(self, "_input_device_path", ""):
            try:
                from evdev import InputDevice
            except ImportError as exc:
                raise RuntimeError("evdev is required for foot-pedal input") from exc
            try:
                self._input_device = InputDevice(self._input_device_path)
            except OSError as error:
                print(
                    f"[{self._name}] WARNING: could not open pedal device "
                    f"{self._input_device_path}: {error}. Falling back to "
                    "terminal keys only (check the path and read permissions, "
                    "e.g. a udev rule or 'sudo setfacl -m u:$USER:rw <device>')."
                )
                self._input_device = None
            if self._input_device is not None:
                if self._grab_device:
                    try:
                        self._input_device.grab()
                    except OSError as error:
                        print(
                            f"[{self._name}] WARNING: could not exclusively grab "
                            f"{self._input_device_path}: {error}"
                        )
                print(
                    f"[{self._name}] Reading pedals ({self._input_device.name}) "
                    f"from {self._input_device_path}; keys={sorted(self._input_keys)}"
                )
        if sys.stdin is None or not sys.stdin.isatty():
            print(f"[{self._name}] No terminal available; keyboard input is disabled.")
            return
        self.old = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        self._stdin_available = True

    @classmethod
    def _evdev_key(cls, code: int) -> tuple[str, str]:
        from evdev import ecodes

        raw = ecodes.KEY.get(code, str(code))
        if isinstance(raw, list):
            raw = raw[0]
        return str(raw), cls._normalize_key_name(str(raw))

    def _poll_input_device(self) -> None:
        if getattr(self, "_input_device", None) is None:
            return
        from evdev import ecodes

        if not select.select([self._input_device], [], [], 0)[0]:
            return
        for event in self._input_device.read():
            # value 2 is autorepeat while held; forwarding it would re-fire
            # toggle commands, so only edges (press=1 / release=0) pass.
            if event.type != ecodes.EV_KEY or event.value == 2:
                continue
            raw_key, key = self._evdev_key(event.code)
            if key not in self._input_keys:
                continue
            pressed = event.value == 1
            if pressed and not self._debouncer.accept(key, time.perf_counter()):
                continue
            self.publish(
                self.leader_topic_name,
                np.array([int(pressed)], dtype=np.uint8),
                extras={
                    "key": key,
                    "pressed": pressed,
                    "value": int(event.value),
                    "raw_key": raw_key,
                    "source": "evdev",
                },
            )

    def tick(self) -> None:
        self._poll_input_device()
        if not self._stdin_available:
            return
        key = None
        if select.select([sys.stdin], [], [], 0)[0]:
            key = sys.stdin.read(1)
            if time.perf_counter() - self.last_key_time > self._cooldown_s:
                self.last_key_time = time.perf_counter()
                self.publish(
                    self.leader_topic_name,
                    np.array([1]),
                    extras={"key": key, "pressed": True, "source": "terminal"},
                )

    def on_shutdown(self) -> None:
        if getattr(self, "_input_device", None) is not None:
            if self._grab_device:
                try:
                    self._input_device.ungrab()
                except OSError:
                    pass
            self._input_device.close()
            self._input_device = None
        if self.old is not None:
            try:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old)
            except termios.error:
                pass
        if os.environ.get("DEPLOY_VERBOSE"):
            print(f"[{self._name}] Key listener node shutdown complete.")


def run(cfg: KeyListenerConfig) -> None:
    KeyListenerNode(
        name=cfg.name,
        control_rate=cfg.control_rate,
        cooldown_s=cfg.cooldown_s,
        input_device=cfg.input_device,
        input_keys=cfg.input_keys,
        grab_device=cfg.grab_device,
    ).run()


if __name__ == "__main__":
    run(tyro.cli(KeyListenerConfig))
