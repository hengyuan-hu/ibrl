import time
import pprint
import numpy as np
from scipy.spatial.transform import Rotation


class Lift:
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.prev_gripper_qpos = 0
        self.hold_count = 0
        self.z_at_hold = 0

    def reset(self):
        self.prev_gripper_qpos = 0
        self.hold_count = 0
        self.z_at_hold = 0

    def reward(self, curr_prop):
        """
        prev_action: action that leads to the curr_prop after step
        curr_prop: current prop reading after applying prev_action and wait
        """

        if self.verbose:
            print("---------------------------")
            pprint.pprint(curr_prop)

        # gripper_action = prev_action[-1]
        desired_gripper_qpos = curr_prop["robot0_desired_gripper_qpos"].item()
        real_gripper_qpos = curr_prop["robot0_gripper_qpos"].item()
        if desired_gripper_qpos <= 0.8:
            # gripper is not even closing
            if self.verbose:
                print(">>> not gripping")

            if self.hold_count > 0:
                print(f"[reset hold count]: giving up gripping: {desired_gripper_qpos:.2f}")

            self.hold_count = 0
            self.prev_gripper_qpos = real_gripper_qpos
            return 0

        diff = abs(self.prev_gripper_qpos - real_gripper_qpos)
        if diff >= 1e-2:
            # gripper is still changing, not stable
            if self.verbose:
                print(f">>> still changing {diff}")

            if self.hold_count > 0:
                print(f"[reset hold count]: still changing {diff}")

            self.hold_count = 0
            self.prev_gripper_qpos = real_gripper_qpos
            return 0

        # the gripper is closing, and it is stable
        gap = desired_gripper_qpos - real_gripper_qpos
        z_loc = curr_prop["robot0_eef_pos"][2]

        if gap > 0.25:
            self.hold_count += 1
            if self.hold_count == 1:
                self.z_at_hold = z_loc
                print("[z_at_hold]:", self.z_at_hold)

        if self.hold_count >= 3 and z_loc - self.z_at_hold > 0.02:
            # hold for at least 0.3 seconds
            # and
            # moving up 2 cm since first holding
            self.prev_gripper_qpos = real_gripper_qpos
            return 1

        if self.verbose:
            print(
                f">>> unhappy {self.hold_count=}, {z_loc=}, delta_z={z_loc - self.z_at_hold:.3f} "
            )
            print("---------------------------")
        self.prev_gripper_qpos = real_gripper_qpos
        return 0


class LiftEEConfig:
    def __init__(self):
        self.init_ee_pos = [0.5, 0, 0.32]
        self.home = np.array(
            [
                np.pi * 0.0,  # 1st joint (from base), horizontal, negative: rotate clockwise
                np.pi * 0.0,  # 2nd joint, vertical, negative: go up, positive: go down
                np.pi * 0.0,  # 3rd joint, horizontal,
                -(3.0 / 4.0) * np.pi,
                0.0,
                0.75 * np.pi,  # 6th, smaller -> inward
                np.pi * 0.0,  # np.pi * 0.5,  # control the rotation of the gripper
            ],
            dtype=np.float32,
        )

        # limits
        self.pos_low = np.array([0.45, -0.15, 0.19])
        self.pos_high = np.array([0.65, 0.15, 0.4])

        self.rot_abs_min = np.pi * np.array([0.75, 0, 0.0]).astype(np.float32)
        self.rot_abs_max = np.pi * np.array([1, 0.25, 1]).astype(np.float32)

        # this is not exactly the same as the range above
        # because 0.75pi -> -0.75pi cannot be expressed as cont. low & high
        self.ee_range_low = self.pos_low.tolist() + [-np.pi, -np.pi, -np.pi]
        self.ee_range_high = self.pos_high.tolist() + [np.pi, np.pi, np.pi]

    def clip(self, pos: np.ndarray, rot: np.ndarray):
        pos = np.clip(pos, self.pos_low, self.pos_high)
        rot = np.sign(rot) * np.clip(np.abs(rot), self.rot_abs_min, self.rot_abs_max)
        return pos, rot

    def ee_in_good_range(self, pos: np.ndarray, quat: np.ndarray, verbose):
        rot = Rotation.from_quat(quat).as_euler("xyz")

        if (pos <= self.pos_low - 0.02).any() or (pos >= self.pos_high + 0.02).any():
            if verbose:
                print(f"bad pos: {pos}")
                print(f"pos min: {self.pos_low - 0.02}")
                print(f"pos max: {self.pos_high + 0.02}")
            return False

        rot_abs = np.abs(rot)
        if (rot_abs <= self.rot_abs_min).any() or (rot_abs >= self.rot_abs_max).any():
            if verbose:
                print(f"bad rot: {rot}")
                print(f"rot abs min: {self.rot_abs_min}")
                print(f"rot abs max: {self.rot_abs_max}")
            return False

        return True

    def reset(self, robot):
        had_no_policy = False

        min_height = 0.4
        ee_pos, ee_quat = robot.get_ee_pose()
        if ee_pos[2].item() < min_height:
            print("fixing position")
            if not robot.is_running_policy():
                robot.start_cartesian_impedance()
                time.sleep(1)
                had_no_policy = True

        while ee_pos[2].item() < min_height:
            ee_pos[2] += 0.02
            robot.update_desired_ee_pose(ee_pos, ee_quat)
            ee_pos, ee_quat = robot.get_ee_pose()

        if had_no_policy:
            robot.terminate_current_policy()
