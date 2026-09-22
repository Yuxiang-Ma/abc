import logging
import os
import time

import numpy as np
import tyro

from deploy.robot.followers.yam_follower_config import YamFollowerConfig
from deploy.robot.node import Node


def _patch_gripper_calibration():
    """Monkey-patch i2rt's detect_gripper_limits to fix a bug where non-gripper
    joints get torqued during calibration.

    The original uses initial effort readings for ALL joints as raw torques
    (``test_torques = init_torque``).  This can send stale/noisy efforts to
    non-gripper joints in open-loop torque mode, causing them to slam to an
    extreme.  The fix: use zeros for non-gripper joints.
    """
    import i2rt.robots.motor_chain_robot as _mcr

    if getattr(_mcr, "_gripper_cal_patched", False):
        return

    def _detect_gripper_limits(
        motor_chain,
        gripper_index=6,
        test_torque=0.2,
        max_duration=2.0,
        position_threshold=0.01,
        check_interval=0.1,
        close_offset=0.05,
    ):
        logger = logging.getLogger("i2rt.robots.utils")
        positions = []
        num_motors = len(motor_chain.motor_list)

        motor_direction = motor_chain.motor_direction[gripper_index]

        initial_states = motor_chain.read_states()
        initial_pos = initial_states[gripper_index].pos
        positions.append(initial_pos)
        logger.info(f"Gripper calibration starting from position: {initial_pos:.4f}")

        for direction in [1, -1]:
            logger.info(f"Testing gripper direction: {direction}")
            # FIX: zero torques for non-gripper joints (original used init efforts)
            test_torques = np.zeros(num_motors)
            test_torques[gripper_index] = direction * test_torque

            start_time = time.time()
            last_pos = None
            position_stable_count = 0

            while time.time() - start_time < max_duration:
                motor_chain.set_commands(torques=test_torques)
                time.sleep(check_interval)

                states = motor_chain.read_states()
                current_pos = states[gripper_index].pos
                positions.append(current_pos)

                if last_pos is not None:
                    pos_change = abs(current_pos - last_pos)
                    if pos_change < position_threshold:
                        position_stable_count += 1
                    else:
                        position_stable_count = 0

                    if position_stable_count >= 6:
                        logger.info(f"Gripper limit detected: pos={current_pos:.4f}")
                        break

                last_pos = current_pos

            time.sleep(0.3)

        motor_chain.set_commands(torques=np.zeros(num_motors))

        min_pos = min(positions)
        max_pos = max(positions)

        if motor_direction > 0:
            detected_limits = [max_pos, min_pos]
        else:
            detected_limits = [min_pos, max_pos]

        detected_limits[0] += close_offset * (max_pos - min_pos) * motor_direction
        logger.info(
            f"Motor direction: {motor_direction}, detected limits: {detected_limits}"
        )
        return detected_limits

    _mcr.detect_gripper_limits = _detect_gripper_limits
    _mcr._gripper_cal_patched = True


class YAMFollowerNode(Node):
    def __init__(
        self,
        name: str,
        control_rate: float,
        channel: str,
        leader_name: str,
        gripper_type: str = "linear_4310",
    ):
        super().__init__(name, control_rate)

        self.channel = channel
        self.leader_name = leader_name
        self.gripper_type = gripper_type

        # Create publisher for joint state
        self.state_topic_name = f"{self._name}_state"
        self.create_publisher(self.state_topic_name)
        self.leader_topic_name = f"{leader_name}_actions"
        self.create_subscriber(self.leader_topic_name, conflate=1)
        self.robot_obs_topic_name = f"{self._name}_obs"
        self.create_publisher(self.robot_obs_topic_name)

    @property
    def _dof(self) -> int:
        return 6 if self.gripper_type == "no_gripper" else 7

    def _to_motor_space(self, command: np.ndarray) -> np.ndarray:
        """Wire format is always 7 per arm; without a gripper motor the 7th value is dropped."""
        return np.asarray(command, dtype=float).reshape(-1)[: self._dof]

    def process_command(self, command: np.ndarray, extras: dict) -> None:
        command = self._to_motor_space(command)
        if extras.get("type", "servo") == "interp":
            current_pos = self.get_joint_pos()
            steps = 50
            time_interval_s = 2.0
            for i in range(steps + 1):
                alpha = i / steps
                target_pos = (1 - alpha) * current_pos + alpha * command
                self.set_joint_pos(target_pos)
                time.sleep(time_interval_s / steps)
        else:
            self.set_joint_pos(command)

    def set_joint_pos(self, joint_pos: np.ndarray) -> None:
        self.robot.command_joint_pos(joint_pos)

    def get_joint_pos(self) -> np.ndarray:
        q = np.asarray(self.robot.get_joint_pos(), dtype=float).reshape(-1)
        if self._dof == 6:            # keep the 7-wide wire format: gripper slot reads 0 (closed)
            q = np.concatenate([q, [0.0]])
        return q

    def get_robot_obs(self) -> np.ndarray:
        obs = self.robot.get_observations()
        joint_pos = np.asarray(obs["joint_pos"]).reshape(-1)
        if self._dof == 6:            # no gripper motor: synthesise the gripper slots so consumers see 7
            obs = dict(obs)
            obs["gripper_pos"] = np.zeros(1)
            obs["joint_vel"] = np.concatenate([np.asarray(obs["joint_vel"]).reshape(-1), [0.0]])
            obs["joint_eff"] = np.concatenate([np.asarray(obs["joint_eff"]).reshape(-1), [0.0]])
        gripper_pos = np.asarray(obs["gripper_pos"]).reshape(-1)

        # Support both i2rt observation layouts.
        if "gripper_vel" in obs and "gripper_eff" in obs:
            joint_vel = np.concatenate(
                [
                    np.asarray(obs["joint_vel"]).reshape(-1),
                    np.asarray(obs["gripper_vel"]).reshape(-1),
                ]
            )
            joint_eff = np.concatenate(
                [
                    np.asarray(obs["joint_eff"]).reshape(-1),
                    np.asarray(obs["gripper_eff"]).reshape(-1),
                ]
            )
        else:
            joint_vel = np.asarray(obs["joint_vel"]).reshape(-1)
            joint_eff = np.asarray(obs["joint_eff"]).reshape(-1)

        if (
            joint_pos.shape != (6,)
            or gripper_pos.shape != (1,)
            or joint_vel.shape != (7,)
            or joint_eff.shape != (7,)
        ):
            raise ValueError(
                "Unexpected robot observation layout from i2rt: "
                f"keys={sorted(obs.keys())}, "
                f"joint_pos={joint_pos.shape}, "
                f"gripper_pos={gripper_pos.shape}, "
                f"joint_vel={joint_vel.shape}, "
                f"joint_eff={joint_eff.shape}"
            )

        return np.concatenate([joint_pos, gripper_pos, joint_vel, joint_eff])

    _GRIPPER_TYPE_MAP = {
        "crank_4310": "CRANK_4310",
        "linear_3507": "LINEAR_3507",
        "linear_4310": "LINEAR_4310",
        "flexible_4310": "FLEXIBLE_4310",
        "no_gripper": "NO_GRIPPER",   # gripper motor physically absent: 6 motors on the bus
    }

    def initial_bootup(self) -> None:
        _patch_gripper_calibration()
        from i2rt.robots.get_robot import get_yam_robot
        from i2rt.robots.utils import GripperType

        gripper_enum = GripperType[self._GRIPPER_TYPE_MAP[self.gripper_type]]
        self.robot = get_yam_robot(
            channel=self.channel,
            gripper_type=gripper_enum,
            zero_gravity_mode=False,
        )
        default_kp = self.robot._kp
        default_kd = self.robot._kd

        # wait for leader to initialize
        time.sleep(0.3)
        message = self.subscribe(self.leader_topic_name, block=False)
        while message[0] is None:
            self.publish(
                self.state_topic_name,
                self.get_joint_pos(),
            )
            message = self.subscribe(self.leader_topic_name, block=False)
        command, extras = message
        extras["type"] = "interp"
        self.process_command(command, extras)
        self.publish(
            self.state_topic_name,
            self.get_joint_pos(),
        )

        self.robot.update_kp_kd(default_kp, default_kd)

    def tick(self) -> None:
        command, extras = self.subscribe(self.leader_topic_name)
        self.process_command(command, extras)
        self.publish(self.state_topic_name, self.get_joint_pos())
        self.publish(
            self.robot_obs_topic_name,
            np.concatenate([self.get_robot_obs(), command]),
        )

    def on_shutdown(self) -> None:
        if not hasattr(self, "robot"):
            return
        try:
            q_des = np.zeros(self._dof)
            if self._dof == 7:
                q_des[6] = self.get_joint_pos()[-1]
            try:
                self.robot.move_joints(q_des, 2.0)
            except Exception:
                pass
        except Exception:
            pass
        try:
            self.robot.close()
        except Exception:
            pass
        time.sleep(0.1)
        for motor_id, _ in self.robot.motor_chain.motor_list:
            try:
                self.robot.motor_chain.motor_interface.motor_off(motor_id)
            except Exception:
                pass
        if os.environ.get("DEPLOY_VERBOSE"):
            print(f"[{self._name}] YAM follower node shutdown complete.")


def run(cfg: YamFollowerConfig) -> None:
    YAMFollowerNode(
        name=cfg.name,
        control_rate=cfg.control_rate,
        channel=cfg.channel,
        leader_name=cfg.leader_name,
        gripper_type=cfg.gripper_type,
    ).run()


if __name__ == "__main__":
    run(tyro.cli(YamFollowerConfig))
