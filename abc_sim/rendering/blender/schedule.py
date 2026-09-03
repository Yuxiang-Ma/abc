"""Render exported shots across GPUs, one Blender process per GPU.

    python -m abc_sim.rendering.blender.schedule --shots shots --out renders --gpus 0,1,2,3

Phase "robot" renders the top and active-wrist cameras of every shot, phase "third" the shot's
outside camera preset.  Jobs are ordered longest first; finished frames are skipped on re-runs.
Append GPU ids to <out>/extra_gpus to add workers while running.
"""

import argparse
import json
import os
import subprocess
import threading
import time
from pathlib import Path

RENDER = Path(__file__).with_name("render_usd.py")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", required=True, help="directory with one exported shot per subdirectory")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--phases", default="robot,third")
    ap.add_argument("--samples", default="32")
    ap.add_argument("--size", default="1440x1080")
    ap.add_argument("--wrist-size", default="720x540")
    ap.add_argument("--blender", default=os.environ.get("BLENDER", "blender"))
    ap.add_argument("--render-args", default="", help="extra arguments passed to render_usd.py")
    args = ap.parse_args()

    shots = [json.load(open(p)) for p in sorted(Path(args.shots).glob("*/geoms.json"))]
    phases = args.phases.split(",")
    jobs = []
    for phase in phases:
        for m in shots:
            cams = "top,wrist" if phase == "robot" else m["third_camera"]
            jobs.append((phases.index(phase), -m["frames"], phase, m, cams))
    jobs.sort(key=lambda j: j[:2])
    out = Path(args.out)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()

    def worker(gpu):
        while True:
            with lock:
                if not jobs:
                    return
                _, _, phase, m, cams = jobs.pop(0)
            shot = Path(args.shots) / m["name"]
            cmd = [args.blender, "-b", "--python-exit-code", "1", "--python", str(RENDER), "--",
                   "--usd", str(shot / m["usd"]), "--meta", str(shot / "geoms.json"),
                   "--out", str(out / m["name"]), "--cams", cams, "--size", args.size,
                   "--wrist-size", args.wrist_size, "--samples", args.samples, *args.render_args.split()]
            t0 = time.time()
            with open(out / "logs" / f"{m['name']}_{phase}.log", "w") as log:
                code = subprocess.call(cmd, env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu},
                                       stdout=log, stderr=subprocess.STDOUT)
            print(f"{time.strftime('%H:%M:%S')} gpu{gpu} {m['name']} {phase} exit={code} "
                  f"{time.time() - t0:.0f}s ({m['frames']} frames) queue={len(jobs)}", flush=True)

    threads = {g: threading.Thread(target=worker, args=(g,)) for g in args.gpus.split(",")}
    for t in threads.values():
        t.start()
    while any(t.is_alive() for t in threads.values()):
        time.sleep(20)
        extra = out / "extra_gpus"
        for g in extra.read_text().split() if extra.is_file() else []:
            if g not in threads:
                threads[g] = threading.Thread(target=worker, args=(g,))
                threads[g].start()
    print("done", flush=True)


if __name__ == "__main__":
    main()
