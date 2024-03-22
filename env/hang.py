import numpy as np
import time
from scipy.spatial.transform import Rotation
from env.franka_utils import ControlFreqGuard


class Hang:
    def __init__(self):
        self.last_min_x = -1
        self.last_min_y = -1
        self.last_max_y = -1

    def reset(self):
        self.last_min_x = -1
        self.last_min_y = -1
        self.last_max_y = -1

    def reward(self, curr_obs) -> float:
        assert "robot0_gripper_qpos" in curr_obs

        if "frontview" in curr_obs:
            # TODO: this is not efficient
            image = curr_obs["frontview"].cpu().numpy()
        else:
            image = curr_obs["frontview_image"]

        mask = get_red_mask(image, 1.8)
        feat = get_mask_features(mask)
        if len(feat) == 0:
            # print("no feat")
            return 0

        last_min_x = self.last_min_x
        last_min_y = self.last_min_y
        last_max_y = self.last_max_y
        self.last_min_x = feat["min_x"]
        self.last_min_y = feat["min_y"]
        self.last_max_y = feat["max_y"]

        qpos = curr_obs["robot0_gripper_qpos"].item()
        # qpos = 0 means the gripper is fully open
        # qpos = 1 means the gripper is fully closed
        if qpos > 0.25:
            # print("bad q_pos:", qpos)
            self.open_count = 0
            return 0

        if abs(last_min_x - feat["min_x"]) > 5:
            # print(" min_x:", feat["min_x"])
            return 0
        if abs(last_min_y - feat["min_y"]) > 1:
            # print("bad min_y:", feat["min_y"])
            return 0
        if abs(last_max_y - feat["max_y"]) > 1:
            # print("bad max_y:", feat["max_y"])
            return 0

        if feat["len_y"] > 20 or feat["len_y"] < 10:
            # print("bad len_y:", feat["len_y"])
            return 0
        if feat["len_x"] > 35 or feat["len_x"] < 20:
            # print("bad len_x:", feat["len_x"])
            return 0

        if feat["min_y"] > 46 or feat["min_y"] < 37:
            # print("bad min_y:", feat["min_y"])
            return 0
        if feat["min_x"] > 20:
            # print("bad min_x:", feat["min_x"])
            return 0

        return 1


def get_red_mask(image, ratio):
    image = image.astype(np.float32)
    if image.shape[2] != 3:
        assert image.shape[0] == 3
        image = image.transpose([1, 2, 0])

    rg = image[:, :, 0] / (image[:, :, 1] + 1)
    rb = image[:, :, 0] / (image[:, :, 2] + 1)
    mask = (rg > ratio).astype(np.float32) * (rb > ratio).astype(np.float32)
    mask = mask * (image[:, :, 0] > 50)
    return mask


def get_mask_features(mask):
    ys, xs = np.where(mask == 1)
    if len(xs) == 0:
        return {}

    feats = {
        "min_x": int(np.min(xs)),
        "max_x": int(np.max(xs)),
        "min_y": int(np.min(ys)),
        "max_y": int(np.max(ys)),
    }
    feats["len_x"] = feats["max_x"] - feats["min_x"]
    feats["len_y"] = feats["max_y"] - feats["min_y"]
    return feats


def calculate_fingertip_pos(ee_pos, ee_quat):
    home_fingertip_offset = np.array([0, 0, -0.17])
    ee_euler = Rotation.from_quat(ee_quat).as_euler("xyz") - np.array([-np.pi, 0, 0])
    fingertip_offset = Rotation.from_euler("xyz", ee_euler).as_matrix() @ home_fingertip_offset
    fingertip_pos = ee_pos + fingertip_offset
    return fingertip_pos


class HangEEConfig:
    def __init__(self):
        # self.init_ee_pos = [0.5582, 0.0849, 0.3215]  # the home ee position
        self.init_ee_pos = [0.5, 0, 0.32]
        self.home = np.pi * np.array([0.0, 0.0, 0.0, -0.75, 0.0, 0.75, 0.0], dtype=np.float32)

        self.finger_pos_low = np.array([0.43, -0.29, 0.00])
        self.finger_pos_high = np.array([0.57, 0.1, 0.24])

        # limits
        self.pos_low = np.array([0.43, -0.29, 0.175])
        self.pos_high = np.array([0.57, 0.13, 0.415])

        self.rot_abs_min = np.pi * np.array([0.9, 0, 0.32]).astype(np.float32)
        self.rot_abs_max = np.pi * np.array([1, 0.15, 0.71]).astype(np.float32)

        # this is not exactly the same as the range above
        # because [-0.45pi, 0.45pi] cannot be expressed as cont. low & high
        self.ee_range_low = self.pos_low.tolist() + [-np.pi, -np.pi, -np.pi]
        self.ee_range_high = self.pos_high.tolist() + [np.pi, np.pi, np.pi]

    def clip(self, pos: np.ndarray, rot: np.ndarray):
        ref_sum = pos.sum() + rot.sum()

        pos = np.clip(pos, self.pos_low, self.pos_high)
        rot = np.sign(rot) * np.clip(np.abs(rot), self.rot_abs_min, self.rot_abs_max)

        new_sum = pos.sum() + rot.sum()
        if abs(ref_sum.item() - new_sum.item()) >= 1e-6:
            print("Clipped!")
        return pos, rot

    def ee_in_good_range(self, pos: np.ndarray, quat: np.ndarray, verbose) -> bool:
        rot = Rotation.from_quat(quat).as_euler("xyz")
        finger_pos = calculate_fingertip_pos(pos, quat)

        if (finger_pos <= self.finger_pos_low - 0.02).any() or (
            finger_pos >= self.finger_pos_high + 0.02
        ).any():
            if verbose:
                print(f"bad finger pos: {finger_pos}")
                print(f"finger pos min: {self.finger_pos_low - 0.02}")
                print(f"finger pos max: {self.finger_pos_high + 0.02}")
            return False

        if finger_pos[1] <= -0.18 and finger_pos[2] <= 0.155:
            print(f"lower than hook: {finger_pos}")
            return False

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

        min_height = 0.32
        min_right = 0.08
        ee_pos, _ = robot._robot.get_ee_pose()
        if ee_pos[1].item() < min_right:
            print("fixing position")
            if not robot._robot.is_running_policy():
                print("[reset]: restart cartesian")
                robot._robot.start_cartesian_impedance()
                time.sleep(1)
                had_no_policy = True

        assert robot.cfg.controller_type == "CARTESIAN_DELTA"
        while ee_pos[2].item() < min_height:
            with ControlFreqGuard(20):
                robot.update([0, 0, 0.02, 0, 0, 0, 0])
            ee_pos, _ = robot._robot.get_ee_pose()

        while ee_pos[1].item() < min_right:
            with ControlFreqGuard(20):
                robot.update([0, 0.02, 0, 0, 0, 0, 0])
            ee_pos, _ = robot._robot.get_ee_pose()

        if had_no_policy:
            robot._robot.terminate_current_policy()
