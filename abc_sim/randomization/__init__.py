"""Object pose and asset randomization for abc-sim scenes.

This package preserves the historical ``abc_sim.randomization`` import path
while keeping each randomization concern in a focused module.
"""

from __future__ import annotations

from importlib import import_module as _import_module

_MODULE_NAMES = (
    "core",
    "requests",
    "assets",
    "tasks",
    "registry",
)

for _module_name in _MODULE_NAMES:
    _module = _import_module(f"{__name__}.{_module_name}")
    for _name in getattr(_module, "__all__", ()):
        globals()[_name] = getattr(_module, _name)

__all__ = sorted(
    name
    for name in globals()
    if not name.startswith("__")
    and name
    not in {
        "_MODULE_NAMES",
        "_import_module",
        "_module",
        "_module_name",
        "_name",
    }
)

del _import_module, _module, _module_name, _name
