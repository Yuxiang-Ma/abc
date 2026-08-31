from __future__ import annotations

from ..core import PerturbRange, ScalePerturbRange, SceneRandomizer

class BlocksRandomizer(SceneRandomizer):
    """Per-block perturbation for the 26-letter blocks scene.

    Blocks are arranged in a 4-row grid with 55 mm spacing; perturbations
    scatter blocks across the table (±80 mm x, ±450 mm y, ±0.25 rad yaw).
    """
    min_clearance_m = 0.01
    # Dense 26-block placements need a larger retry budget.
    max_tries = 2000
    perturbations = [
        PerturbRange(f"block_{letter}_jnt",
                     delta_x=(-0.08, 0.08),
                     delta_y=(-0.45, 0.45),
                     delta_yaw=(-0.25, 0.25))
        for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    ]

    def _get_size_perturbations(self) -> list[ScalePerturbRange]:
        return []





__all__ = ["BlocksRandomizer"]
