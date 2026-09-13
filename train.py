"""Launch ABC training.

Trains the ABC-DiT policy by default, or ABC-VLA with
``--policy vla`` (supply the Gemma base checkpoint via
``--vla-model.backbone.checkpoint``). Run ``python train.py --help`` for the
full flag surface."""

import tyro

from abc_minimal.config import TrainConfig
from abc_minimal.train_loop import main as train


def main():
    train(tyro.cli(TrainConfig))


if __name__ == "__main__":
    main()
