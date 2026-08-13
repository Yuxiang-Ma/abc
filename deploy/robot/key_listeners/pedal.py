"""Foot-pedal device resolution and preflight.

Launchers call :func:`resolve_foot_pedal_device` before spawning the
KeyListener so a misconfigured pedal fails fast with remediation steps
instead of silently degrading to terminal-only keys.

Precedence: explicit config > ``FOOT_PEDAL_INPUT_DEVICE`` > the PCsensor
default path when the device is plugged in. Passing ``""`` explicitly
disables the pedal (keyboard-only).
"""

from __future__ import annotations

import os
import stat
import sys

DEFAULT_FOOT_PEDAL_DEVICE = "/dev/input/by-id/usb-PCsensor_FootSwitch-event-kbd"


def _format_mode(path: str) -> str:
    try:
        st = os.stat(path)
    except OSError:
        return "unknown permissions"
    return f"{stat.filemode(st.st_mode)} {st.st_uid}:{st.st_gid}"


def resolve_foot_pedal_device(
    configured: str | None = None, *, tool: str = "deploy"
) -> str:
    """Return the pedal evdev path to use, or "" for keyboard-only input.

    An explicitly configured (or environment-configured) device that is
    missing or inaccessible raises SystemExit with fixes; the implicit
    PCsensor default falls back to keyboard-only when not plugged in, but
    still fails fast when present with wrong permissions.
    """
    if configured == "":
        return ""
    path = configured or os.environ.get("FOOT_PEDAL_INPUT_DEVICE") or ""
    explicit = bool(path)
    if not path:
        if not os.path.exists(DEFAULT_FOOT_PEDAL_DEVICE):
            return ""
        path = DEFAULT_FOOT_PEDAL_DEVICE

    resolved = os.path.realpath(path)
    if not os.path.exists(path):
        assert explicit  # the implicit default is only used when it exists
        print(
            f"\n[{tool}] Foot pedal device was configured but does not exist:\n"
            f"  {path}\n\n"
            "Fixes:\n"
            "  - Plug in the foot pedal, then retry.\n"
            "  - Check available devices with: ls -l /dev/input/by-id/\n"
            "  - Override with: --foot-pedal-device /dev/input/by-id/<device>\n"
            "  - Use keyboard-only control with: --foot-pedal-device ''\n",
            file=sys.stderr,
        )
        raise SystemExit(2)

    if not os.access(path, os.R_OK | os.W_OK):
        print(
            f"\n[{tool}] Foot pedal device exists but is not readable/writable "
            "by this user:\n"
            f"  configured: {path}\n"
            f"  resolved:   {resolved}\n"
            f"  mode:       {_format_mode(resolved)}\n\n"
            "Fixes:\n"
            f"  - Grant ACL access: sudo setfacl -m u:$USER:rw {resolved}\n"
            "  - Or add the user to the input group and log out/in: "
            "sudo usermod -aG input $USER\n"
            "  - Or use keyboard-only control with: --foot-pedal-device ''\n",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return path
