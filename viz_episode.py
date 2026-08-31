"""Viser playback of downloaded dataset episodes."""

import tyro

from abc_minimal.config import VizEpisodeConfig
from abc_minimal.viz_episode import main as viz_episode


def main():
    viz_episode(tyro.cli(VizEpisodeConfig))


if __name__ == "__main__":
    main()
