"""Export recorded MuJoCo states of an ABC rollout to an animated USD for Blender.

    python -m abc_sim.rendering.blender.export_usd rollout.npz --start 0 --end 93 --out shots/conveyor_01

The .npz holds qpos, qvel, ctrl, mocap_pos, mocap_quat, time and the scalars task, prompt and seed
of the rollout.  The seeded scene is rebuilt, each recorded state is loaded and forwarded, and the
official mujoco.usd exporter writes <out>/frames/frame_<n>.usdc (metres, one time sample per state)
plus <out>/geoms.json, which maps every USD prim to its MuJoCo geom, body, material and mesh so the
renderer can assign materials by what things are.  Needs `pip install usd-core`.
"""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from mujoco.usd import exporter as usd_exporter
from pxr import UsdGeom

import abc_sim


def load_state(model, data, traj, i):
    data.qpos[:] = traj["qpos"][i]
    data.qvel[:] = traj["qvel"][i]
    data.ctrl[:] = traj["ctrl"][i]
    if model.nmocap:
        data.mocap_pos[:] = traj["mocap_pos"][i]
        data.mocap_quat[:] = traj["mocap_quat"][i]
    data.time = float(traj["time"][i])
    mujoco.mj_forward(model, data)


def geom_records(model, exporter):
    def name(objtype, i):
        return mujoco.mj_id2name(model, objtype, int(i)) if i >= 0 else None

    rgb = mujoco.mjtTextureRole.mjTEXROLE_RGB.value
    records = []
    for i in range(exporter.scene.ngeom):
        g = exporter.scene.geoms[i]
        if g.objtype != mujoco.mjtObj.mjOBJ_GEOM:
            continue
        gid, matid = int(g.objid), int(model.geom_matid[g.objid])
        records.append({
            "usd": exporter._get_geom_name(g),
            "id": gid,
            "geom": name(mujoco.mjtObj.mjOBJ_GEOM, gid),
            "body": name(mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[gid]),
            "material": name(mujoco.mjtObj.mjOBJ_MATERIAL, matid),
            "mesh": name(mujoco.mjtObj.mjOBJ_MESH, model.geom_dataid[gid])
            if g.type == mujoco.mjtGeom.mjGEOM_MESH else None,
            "type": int(g.type),
            "rgba": [round(float(x), 4) for x in g.rgba],
            "textured": bool(matid >= 0 and model.mat_texid[matid][rgb] >= 0),
        })
    return records


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trajectory")
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, help="exclusive; default all recorded states")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--cameras", default="top,left,right")
    ap.add_argument("--active-wrist", default="right", choices=["left", "right"])
    ap.add_argument("--third-camera", default="front",
                    help="outside camera preset for the scheduler: front, front_left, front_right, overhead, left_side, right_side")
    args = ap.parse_args()

    traj = np.load(args.trajectory, allow_pickle=False)
    task, prompt, seed = str(traj["task"].item()), str(traj["prompt"].item()), int(traj["seed"].item())
    kwargs = {"seed": seed} if abc_sim.resolve_task(task).env_task == "inhand_transfer" else {}
    env = abc_sim.make_env(task=task, prompt=prompt, render_cameras=False, **kwargs)
    env.reset(seed=seed, randomize=True)
    err = float(np.max(np.abs(np.asarray(env.data.qpos) - traj["qpos"][0])))
    if err > 2e-5:
        raise SystemExit(f"rebuilt scene does not match the recording (max qpos error {err:.3g})")

    out = Path(args.out).resolve()
    end = args.end if args.end is not None else int(traj["qpos"].shape[0])
    exporter = usd_exporter.USDExporter(
        model=env.model, output_directory=out.name, output_directory_root=str(out.parent),
        camera_names=args.cameras.split(","), verbose=False,
    )
    for i in range(args.start, end):
        load_state(env.model, env.data, traj, i)
        exporter.update_scene(env.data)
    UsdGeom.SetStageMetersPerUnit(exporter.stage, 1.0)
    exporter.stage.SetTimeCodesPerSecond(args.fps)
    exporter.stage.SetFramesPerSecond(args.fps)
    exporter.save_scene(filetype="usdc")

    meta = {
        "name": out.name, "task": task, "prompt": prompt, "seed": seed,
        "start_frame": args.start, "end_frame": end, "frames": end - args.start, "fps": args.fps,
        "active_wrist": args.active_wrist, "third_camera": args.third_camera,
        "usd": f"frames/frame_{exporter.frame_count}.usdc",
        "geoms": geom_records(env.model, exporter),
    }
    (out / "geoms.json").write_text(json.dumps(meta, indent=1))
    env.close()
    print(f"{out}: {end - args.start} states, {len(meta['geoms'])} geoms")


if __name__ == "__main__":
    main()
