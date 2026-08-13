"""Spawn deployment nodes as multiprocessing children.

Each node is a module-level ``run`` function started in a fresh ``spawn``
process with its configuration passed as a Python object.

Shutdown mirrors the old semantics: children run in their own process
groups, so terminal Ctrl+C reaches only the supervisor, which forwards
SIGINT to every child (giving nodes a KeyboardInterrupt for graceful
shutdown, e.g. motor-off) and escalates to SIGTERM/SIGKILL if they stall.
A child can end the whole session by signalling its parent
(``os.kill(os.getppid(), signal.SIGINT)``), as the recorder's shutdown pedal
does.

The ``target`` callables below lazy-import their node modules so that the
supervisor process never imports torch or hardware SDKs.
"""

from __future__ import annotations

import importlib
import multiprocessing as mp
import os
import signal
import sys
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ProcessSpec:
    name: str
    target: str
    """Child module, optionally followed by ``:function`` (defaults to ``run``)."""
    kwargs: dict[str, Any] = field(default_factory=dict)
    quiet: bool = False
    """Suppress the child's stdout/stderr (unless DEPLOY_VERBOSE is set)."""
    terminal_input: bool = False
    """Reconnect this child to the supervisor's terminal for interactive input."""


def _terminal_path() -> str | None:
    """Return the supervisor's controlling terminal, if it has one."""
    try:
        if sys.stdin is None or not sys.stdin.isatty():
            return None
        return os.ttyname(sys.stdin.fileno())
    except (OSError, ValueError):
        return None


def _restore_terminal_stdin(path: str) -> None:
    """Undo multiprocessing's replacement of child stdin with ``/dev/null``."""
    flags = os.O_RDONLY | getattr(os, "O_NOCTTY", 0)
    fd = os.open(path, flags)
    encoding = getattr(sys.stdin, "encoding", None) or "utf-8"
    sys.stdin = os.fdopen(fd, "r", encoding=encoding, errors="replace")


def _child_entry(
    target: str, kwargs: dict, quiet: bool, terminal_path: str | None
) -> None:
    # Match subprocess.Popen(start_new_session=True), used by the old launcher:
    # each child gets a process group for targeted shutdown but does not become
    # a background process group of the supervisor's controlling terminal.
    os.setsid()
    if terminal_path is not None:
        _restore_terminal_stdin(terminal_path)
    if quiet and not os.environ.get("DEPLOY_VERBOSE"):
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
    try:
        module_name, _, function_name = target.partition(":")
        function_name = function_name or "run"
        function = getattr(importlib.import_module(module_name), function_name)
        function(**kwargs)
    except KeyboardInterrupt:
        pass


def launch(specs: list[ProcessSpec]) -> int:
    """Run nodes as one unit, stopping the rest when any child exits."""
    # Children write straight to the shared terminal; keep output timely.
    os.environ.setdefault("PYTHONUNBUFFERED", "1")

    ctx = mp.get_context("spawn")
    procs: list[mp.Process] = []
    interrupt = {"count": 0, "time": 0.0}

    def _signal_all(sig: int) -> None:
        for p in procs:
            if p.is_alive() and p.pid:
                try:
                    os.killpg(p.pid, sig)
                except ProcessLookupError:
                    pass

    def _handler(signum, frame) -> None:
        interrupt["count"] += 1
        if interrupt["count"] == 1:
            interrupt["time"] = time.time()
            _signal_all(signal.SIGINT)
        elif interrupt["count"] == 2:
            _signal_all(signal.SIGTERM)
        else:
            _signal_all(signal.SIGKILL)

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)

    for spec in specs:
        terminal_path = _terminal_path() if spec.terminal_input else None
        if spec.terminal_input and terminal_path is None:
            print(
                f"[launch] No terminal available for {spec.name}; "
                "interactive keyboard input is disabled."
            )
        p = ctx.Process(
            target=_child_entry,
            args=(spec.target, spec.kwargs, spec.quiet, terminal_path),
            name=spec.name,
        )
        p.start()
        procs.append(p)

    try:
        while any(p.is_alive() for p in procs):
            # Deployment nodes are long-lived. If one exits (including a camera
            # that returns cleanly after losing its device), stop the session
            # instead of leaving the remaining nodes running indefinitely.
            if not interrupt["count"] and any(p.exitcode is not None for p in procs):
                _handler(signal.SIGINT, None)
            if interrupt["count"] >= 1:
                elapsed = time.time() - interrupt["time"]
                # Auto-escalate: SIGINT → SIGTERM → SIGKILL
                if interrupt["count"] == 1 and elapsed > 5.0:
                    interrupt["count"] = 2
                    _signal_all(signal.SIGTERM)
                elif interrupt["count"] == 2 and elapsed > 8.0:
                    interrupt["count"] = 3
                    _signal_all(signal.SIGKILL)
            time.sleep(0.2)
    finally:
        deadline = time.time() + 3.0
        while any(p.is_alive() for p in procs) and time.time() < deadline:
            time.sleep(0.05)
        _signal_all(signal.SIGKILL)

    for p in procs:
        p.join()

    for p in procs:
        # Deaths by SIGINT/SIGTERM are part of orchestrated shutdown.
        if p.exitcode not in (0, None, -signal.SIGINT, -signal.SIGTERM):
            return int(p.exitcode) if p.exitcode > 0 else 1
    return 0
