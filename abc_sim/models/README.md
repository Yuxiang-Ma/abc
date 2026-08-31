# ABC Sim Assets

Large simulator meshes and textures are not stored in git. Download them from
the public asset mirror before running sim environments, rendering, or eval:

```bash
uv run prepare.py --sim
```

Useful commands:

```bash
uv run prepare.py --sim-list                                  # show packages + install state
uv run prepare.py --sim-task put_plastic_bottles_in_bin       # just one task's packages
uv run prepare.py --sim-package dustpan                       # a single package
uv run prepare.py --sim --sim-source /path/to/mirror          # install from local archives
```

Archives are listed in [`assets_manifest.json`](assets_manifest.json) and
install into `abc_sim/models/assets/`. The manifest is split by top-level
task/shared asset folder, so adding a new task usually means publishing one new
`<task>.tar.gz` package, adding one manifest entry, and mapping the task in
`prepare.py`'s `SIM_TASK_PACKAGES`.

For the simulator API, task randomization, rendering, and RL usage, see
[`../README.md`](../README.md).
