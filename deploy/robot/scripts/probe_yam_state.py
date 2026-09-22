"""Read-only probe: subscribe to the follower state topics and print joints + FK end-effector poses.

    ROBOT_PROFILE=react_yam_config uv run deploy/robot/scripts/probe_yam_state.py [--seconds 5]

Publishes nothing. Use it to confirm which CAN adapter is which arm (move an arm by hand
with motors off, or watch the joint readings) and to record the current EE pose before any
planned motion.
"""
import argparse
import time

import numpy as np
import zmq

from deploy.robot import communication as comms
from deploy.robot.config import get_i2rt_config
from deploy.robot.control.mink_ik import MinkArmController


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=5.0)
    args = ap.parse_args()
    cfg = get_i2rt_config()
    ctx = zmq.Context.instance()
    subs = {n: comms.create_subscriber(ctx, f"follower_{n}_state", conflate=1) for n in cfg.robots}
    ik = {n: MinkArmController(home=np.zeros(6)) for n in cfg.robots}
    t0 = time.time()
    while time.time() - t0 < args.seconds:
        for n, sock in subs.items():
            try:
                q, _ = comms.subscribe(sock, timeout_ms=1000, topic_label=f"follower_state:{n}")
            except comms.SubscribeTimeout:
                print(f"[{n}] no state within 1 s (follower not running / CAN down?)"); continue
            q = np.asarray(q, dtype=float).reshape(-1)
            pos, wxyz = ik[n].fk(q[:6])
            print(f"[{n}] q={np.round(q[:6], 3).tolist()} gripper={q[6]:.3f}  EE pos={np.round(pos, 4).tolist()} wxyz={np.round(wxyz, 3).tolist()}")
        time.sleep(0.5)


if __name__ == "__main__":
    main()
