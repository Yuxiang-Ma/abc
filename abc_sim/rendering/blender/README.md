# Blender (Cycles) rendering

Two uses: re-rendering recorded rollouts for presentation, and a live camera backend
(`camera_backend="blender"`) that renders what the policy sees during evaluation.

## Live camera backend

`live.py` exports the bound model to USD once, starts one Blender process per environment
(`render_usd.py --serve`) and streams geom and camera poses to it over a unix socket after every
step; frames come back as BMP files in `/dev/shm`.  Materials and lights are the same look as the
offline path below.  Defaults: 16 samples, OIDN denoise, 4 bounces, ~190 ms per three-camera
224x168 observation on an H100 and ~4 s of startup per reset.  `ABC_BLENDER_SAMPLES` and
`ABC_BLENDER_DENOISER` (`OPTIX`, `OPENIMAGEDENOISE`, `NONE`) override the defaults; 4 samples run
in ~155 ms but boil visibly on the wrist cameras, which cost the VLA about a quarter of its
successes on the bottles task.  Requires `BLENDER` pointing at a Blender 4.2+ binary, `usd-core`
importable as `pxr`, and `ABC_HDRI` for the studio HDRI (a grey world otherwise).

## Offline renders of recorded rollouts

The policy runs against the pinned sim renderer, its MuJoCo states are recorded, and Blender
re-renders those states.

```
rollout .npz ──export_usd──▶ shot/frames/frame_N.usdc + shot/geoms.json ──render_usd (Blender)──▶ PNG frames
```

## Export

```bash
uv pip install usd-core
python -m abc_sim.rendering.blender.export_usd rollout.npz --start 0 --end 93 --out shots/conveyor_01 \
    --active-wrist left --third-camera front_right
```

The `.npz` needs `qpos`, `qvel`, `ctrl`, `mocap_pos`, `mocap_quat`, `time` per state and the scalars
`task`, `prompt`, `seed`.  The scene is rebuilt from the seed and checked against the first recorded
state.  `geoms.json` records every exported prim's MuJoCo geom, body, material and mesh; the renderer
assigns materials from those names, because the sim scenes themselves are flat `rgba` with almost no
textures.

## Render

Blender 4.2+ with Cycles.  `ABC_HDRI` (or `--hdri`) points at an equirectangular studio `.hdr`
for reflections; without one a grey world is used.

```bash
blender -b --python-exit-code 1 --python abc_sim/rendering/blender/render_usd.py -- \
    --usd shots/conveyor_01/frames/frame_93.usdc --meta shots/conveyor_01/geoms.json \
    --out renders/conveyor_01 --cams top,wrist,front --samples 32 --device OPTIX
```

- `top`, `left`, `right`, `wrist`: the recorded robot cameras.  A 4:3 `--size` keeps their 58 degree
  field of view; 16:9 uses 45 degrees, which matches a 16:9 crop of the 4:3 frame.
- `front`, `front_left`, `front_right`, `overhead`, `left_side`, `right_side`, `close`: outside
  cameras that hide the enclosure walls and aim at the visible props.
- `--fstop`, `--motion-blur`, `--glare`, `--look`, `--exposure` for the look; `--engine EEVEE` for
  fast previews.
- The aluminium frame mesh is bisected into short triangles once and cached under `--cache`; its
  metre-long extruded slivers otherwise make Cycles about five times slower.
- Wrist cameras get a black mount bar to the gripper, which the sim does not model.

## Many shots on many GPUs

```bash
BLENDER=/path/to/blender python -m abc_sim.rendering.blender.schedule --shots shots --out renders --gpus 0,1,2,3
```

Renders `top` and `wrist` for every shot, then each shot's `--third-camera` preset.  Frames that
exist are skipped, so it can be re-run.  Assemble with ffmpeg, for example
`ffmpeg -framerate 30 -i renders/conveyor_01/top/frame_%05d.png -c:v libx264 -crf 16 -pix_fmt yuv420p top.mp4`.
