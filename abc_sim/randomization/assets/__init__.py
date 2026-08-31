"""Asset helpers used by task randomizers.

The submodules are split by task family. This compatibility module re-exports
the historical helper names used by older scripts and randomizer modules.
"""

from __future__ import annotations

from importlib import import_module as _import_module

_MODULE_NAMES = (
    "paths",
    "xml_common",
    "inhand",
    "dishrack",
    "chess_tin",
    "mugs",
    "water_bottles",
)

for _module_name in _MODULE_NAMES:
    _module = _import_module(f"{__name__}.{_module_name}")
    for _name in getattr(_module, "__all__", ()):  # compatibility re-export
        globals()[_name] = getattr(_module, _name)

__all__ = sorted(
    name
    for name in globals()
    if (
        (name.startswith("_") and not name.startswith("__"))
        or name == "INHAND_OBJECT_CATEGORIES"
    )
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
