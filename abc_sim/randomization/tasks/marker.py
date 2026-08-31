from __future__ import annotations

from ..core import PerturbRange, SceneRandomizer

class MarkerRandomizer(SceneRandomizer):
    min_clearance_m = 0.03
    perturbations = [
        PerturbRange("marker_joint", delta_x=(-0.10, 0.10), delta_y=(-0.14, 0.14)),
        PerturbRange("cap_joint",    delta_x=(-0.08, 0.08), delta_y=(-0.08, 0.08)),
    ]





__all__ = ["MarkerRandomizer"]
