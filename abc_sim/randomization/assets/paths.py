"""Shared paths for randomization asset helpers."""

from __future__ import annotations

from pathlib import Path as _Path

_MODELS_DIR = _Path(__file__).resolve().parents[2] / "models"

__all__ = ["_MODELS_DIR", "_Path"]
