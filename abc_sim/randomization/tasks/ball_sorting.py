from __future__ import annotations

from ..core import PerturbRange, SceneRandomizer

class BallSortingRandomizer(SceneRandomizer):
    min_clearance_m = 0.06
    perturbations = [
        # Toy box: fixed body, randomized via model.body_pos (no physics).
        # Cylinders are rejection-sampled against it via the MuJoCo contact check.
        PerturbRange("ball-sorting-toy", delta_x=(-0.18, 0.08), delta_y=(-0.15, 0.15),
                     delta_yaw=(-0.3, 0.3), fixed_body=True),
        PerturbRange("cylinder-1", delta_x=(-0.5, 0.5), delta_y=(-0.5, 0.5)),
        PerturbRange("cylinder-2", delta_x=(-0.5, 0.5), delta_y=(-0.5, 0.5)),
        PerturbRange("cylinder-3", delta_x=(-0.5, 0.5), delta_y=(-0.5, 0.5)),
    ]






__all__ = ["BallSortingRandomizer"]
