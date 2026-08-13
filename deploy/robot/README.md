# deploy/robot — real-robot teleop, data collection, and rollout

Robot-side stack for bimanual YAM arms (i2rt hardware): ZMQ node
infrastructure, camera/leader/follower drivers, teleop, HDF5 recording,
and gym-style rollout environments.

Installed as part of the repo (`uv sync --extra deploy`);
requires the hardware dependencies from the `deploy` optional-dependency
group in `pyproject.toml` (pyrealsense2, dynamixel_sdk, i2rt, ...), which
only install on the robot workstation (Linux x86).

## Robot profile

Before running any scripts, set the `ROBOT_PROFILE` env var to select which
robot station to use:

```bash
export ROBOT_PROFILE=bbox_config   # Original bbox station
export ROBOT_PROFILE=cbox_config   # Cbox station
export ROBOT_PROFILE=gtwy_config   # Gateway station
```

Profiles are defined in the `PROFILES` dict in `deploy/robot/config.py` and
contain all hardware-specific IDs (camera serials, CAN device serials, leader
USB devices). To add a new station, add a new entry there.

## Layout

- `node.py`, `communication.py`, `launch.py` — fixed-rate ZMQ node base,
  pub/sub wire format, multiprocessing node supervisor
- `config.py` — per-station hardware profiles (`ROBOT_PROFILE`)
- `cameras/`, `leaders/`, `followers/` — RealSense/Decxin drivers, GELLO
  leaders, and YAM followers
- `gym/` — rollout environments (`policy_rollout.py` supports standard and RTC)
- `recorders/` — teleop and inference HDF5 recorders
- `scripts/` — entrypoints (see below)

## Scripts

Run from the repo root:

```bash
# Teleoperation
uv run deploy/robot/scripts/run_teleop.py

# Data collection (teleop + cameras + recorder)
uv run deploy/robot/scripts/run_data_record.py

# Policy rollout against a running policy server
uv run deploy/robot/gym/policy_rollout.py
```

The full deployment orchestrator (policy server + followers + cameras +
rollout in one command) is `deploy/deploy_policy.py`; see `deploy/README.md`.

## Foot pedal

Setup, device selection, and troubleshooting are documented in
[`deploy/README.md`](../README.md#foot-pedal). Implementation notes: the
KeyListener forwards pedal key-down/up edges (autorepeat is dropped so held
pedals don't re-fire toggles); `FOOT_PEDAL_INPUT_KEY` overrides the
forwarded key set (default `a,b,c,x,j`); launchers resolve the device via
`deploy/robot/key_listeners/pedal.py`.
