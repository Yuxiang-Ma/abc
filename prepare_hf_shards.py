"""Prepare this node's Hugging Face ABC-130k shard."""

import tyro

from abc_minimal.hf_prepare import HfShardPrepareConfig, prepare_hf_shards


def main():
    prepare_hf_shards(tyro.cli(HfShardPrepareConfig))


if __name__ == "__main__":
    main()
