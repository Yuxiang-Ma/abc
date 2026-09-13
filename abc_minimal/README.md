# ABC Training (`abc_minimal`)

**Contents**

- [Training data](#training-data)
  - [Exporting a task from ABC-130k](#exporting-a-task-from-abc-130k)
  - [Converting local MCAPs & the episode format](#converting-local-mcaps--the-episode-format)
- [Multi-node training](#multi-node-training)
- [Subtask & operator conditioning](#subtask--operator-conditioning)
- [Visualizing episodes and policies](#visualizing-episodes-and-policies)


`abc_minimal` is the training and inference package for ABC-DiT: the model and
its DINOv3/CLIP backbones (`dit.py`), the episode dataloader (`dataloader.py`,
`episode_io.py`, `preprocess.py`), the training loop and checkpointing
(`train_loop.py`, `checkpointing.py`), DiT and VLA policy inference
(`policy.py`), the sim-eval glue (`sim_env.py`, `eval_policy.py`), and
the episode/policy visualizers. The scripts at the repository root —
`train.py`, `eval_policy.py`, `viz_episode.py`, `viz_policy.py` — are thin
wrappers around this package.

Setup, the data download, single-node training, and the evaluation quickstart
are covered in the [top-level README](../README.md). Simulator environments and
the sim-eval task catalogue are covered in the
[abc_sim README](../abc_sim/README.md). This document is the training-side
reference: multi-node jobs, the episode data format and converters, and prompt
conditioning options.

## Training data

While we host a single task in training format, there are many more in the ABC
Dataset. The ABC-130k MCAPs are hosted on Hugging Face at
[`XDOF/ABC-130k`](https://huggingface.co/datasets/XDOF/ABC-130k). The dataset
is gated, so accept access on the dataset page and set `HF_TOKEN` before
downloading.

### Exporting a task from ABC-130k

Download all MCAPs for one task and convert them in place:

```bash
uv run scripts/export_hf_task.py --task organize_the_condiment_bottles
```

By default this downloads both `train` and `val`, stages raw MCAPs under
`$ABC_CACHE/hf_tasks/<task>/`, runs the MCAP converter, writes converted episodes
to `$ABC_CACHE/train_real/` and `$ABC_CACHE/val_real/`, then deletes the staged
raw MCAPs after each successful split conversion. For a quick smoke test:

```bash
uv run scripts/export_hf_task.py --task organize_the_condiment_bottles --split train --max-episodes 1
```

### Converting local MCAPs & the episode format

If you already have local MCAPs, call the lower-level converter directly:

```bash
uv run scripts/export_mcap.py ./train_run_1 ./out
```

The input is expected to look like:

```text
train_run_1/
  <task_name>/
    episode_<uuid>/
      episode.mcap
```

You can also pass the number of worker processes:

```bash
uv run scripts/export_mcap.py ./train_run_1 ./out 8
```

Each output episode is written to `./out/episode_<uuid>/` in the same format
the trainer reads:

```text
episode_<uuid>/
  states_actions.bin               # (num_steps, 28) float64: 14 states + 14 actions
  combined_camera-images-rgb.mp4   # 30 fps vertical stack of 224x224 camera views
  episode_metadata.json            # task name, cameras, resolutions, timing, num_steps
```

The mp4 is encoded in a manner that allows for efficient dataloading. For details, see the ABC paper.

## Multi-node training

For multi-node jobs without a shared filesystem, predownload a deterministic
node-local shard on each node before launching training:

```bash
ABC_CACHE=/local_nvme/abc_cache HF_TOKEN=... \
uv run scripts/prepare_hf_shards.py \
  --tasks organize_the_condiment_bottles \
  --num-nodes 8 --node-rank $NODE_RANK --workers 8

ABC_CACHE=/local_nvme/abc_cache \
uv run torchrun --nnodes 8 --node-rank $NODE_RANK --nproc-per-node 8 train.py
```

The predownload step writes this node's converted episodes into the usual
`train_real/` and `val_real/` directories. Training auto-detects the
`hf_status/shard.json` marker and uses local-rank sampling, so validation
metrics are accumulated across different validation shards on different GPUs.

The revision is pinned to a commit SHA at run time and training verifies at
startup that all nodes sharded the same snapshot. If the dataset may change
while nodes prepare, pass the same explicit `--revision <sha>` to every node.

## Subtask & operator conditioning

In addition to the task prompt, our policies can condition  **subtask** labels and on
the episode's **operator** id. The MCAP converter (`scripts/export_mcap.py`)
extracts both from the release MCAPs when present and writes two optional
extra files next to the episode:

```text
episode_<uuid>/
  subtasks.json      # {"<frame_idx>": "<subtask label>", ...}  — per-frame subtask
  operator.json      # {"operator_id": "<uuid>"}                — the teleoperator id
```

- **Subtasks** Enable at train time with `--prompt.use-subtask-as-prompt`,
  choosing `--prompt.subtask-mode {replace,append}` (`replace` swaps the task prompt for the subtask label;
  `append` formats both via `--prompt.subtask-append-format`).
- **Operators** We map UUIDs to short deterministic labels (`operator 0`, … or names). The labels are computed
  per-task such that operators with more hours (proxy for quality) have lower numbers.
  Enable with `--prompt.use-operator-id-as-prompt` and choose `--prompt.operator-prompting-mode {text_indexed,text_name}`; the label is appended as `"{prompt}. {operator}"`.

  The per-task map must be built prior to training and passed via
  `--prompt.operator-label-map-path`. Build it with:

  ```bash
  ABC_CACHE=cache/tshirt uv run scripts/build_operator_label_map.py \
      --out cache/tshirt/operator_label_map.json
  ```

  then pass `--prompt.operator-label-map-path cache/tshirt/operator_label_map.json`.

  On a node-sharded multi-node cache no single machine holds every episode, and
  a map built from one shard would mis-rank operators and miss those on other
  nodes. Instead, build one manifest per node from its local shard, gather the
  shard manifests on one machine, and fold them into a global ranking —
  hours and episode counts sum exactly across shards, so the result matches a
  full single-machine scan:

  ```bash
  uv run scripts/build_operator_label_map.py --out shard_$NODE.json   # on each node
  uv run scripts/build_operator_label_map.py \
      --combine shard_0.json shard_1.json ... --out operator_label_map.json
  ```

  then copy the combined manifest to every node at the same path and pass it
  via `--prompt.operator-label-map-path`.

Both are off by default. Some episodes do not have eg. subtask annotations and
for these training will drop back to task prompt only.

(We intend to release the global manifest in future but this is TODO.)

## Visualizing episodes and policies

`viz_policy.py` rolls out a checkpoint live in a viser window, and
`viz_episode.py` plays back dataset episodes the same way, no checkpoint
needed — browse a pool (`--root cache/train_sim`) or open one episode
directly; `--mode physics` re-simulates the recorded actions instead of posing
the arms. The default pose mode replays the whole recorded scene, objects
included, for episodes that ship `scene_qpos.npy` (the current sim_224
release), and falls back to posing the arms alone — objects held at their
start pose — for episodes without it:

```bash
uv run viz_policy.py --sim.checkpoint cache/bottles_75k.pt --port 8080
uv run viz_episode.py --root cache/train_sim --port 8080
```
