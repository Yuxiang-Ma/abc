from dataclasses import dataclass


@dataclass
class KeyListenerConfig:
    name: str = "KeyListener"
    control_rate: float = 60.0
    cooldown_s: float = 1.0
    input_device: str = ""
    """Optional evdev device path (foot pedal). Falls back to the
    FOOT_PEDAL_INPUT_DEVICE environment variable when empty."""
    input_keys: str = "a,b,c,x,j"
    """Comma/space-separated keys to forward, e.g. 'a,b,c' or 'KEY_A KEY_B'.
    Overridable via FOOT_PEDAL_INPUT_KEY."""
    grab_device: bool = True
    """Exclusively grab the evdev device so pedal events do not type into apps."""
