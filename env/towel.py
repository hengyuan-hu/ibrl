import numpy as np
import time


class Towel:
    def __init__(self):
        self.last_vert_center = -1

    def reset(self):
        self.last_vert_center = -1

    def reward(self, curr_obs) -> float:
        assert "robot0_gripper_qpos" in curr_obs
        qpos = curr_obs["robot0_gripper_qpos"].item()
        # qpos = 0 means the gripper is fully open
        # qpos = 1 means the gripper is fully closed
        if qpos > 0.8:
            # print(f"bad grip {qpos}")
            return 0

        if "frontview" in curr_obs:
            # TODO: this is not efficient
            image = curr_obs["frontview"].cpu().numpy()
        else:
            image = curr_obs["frontview_image"]

        mask = get_red_mask(image, 2.5)
        feat = get_mask_features(mask)

        last_vert_center = self.last_vert_center
        self.last_vert_center = feat["vert_center"]
        if abs(last_vert_center - feat["vert_center"]) > 0.5:
            return 0

        if feat["vert_len"] < 34:
            return 0
        if feat["hori_len"] >= 11:
            return 0
        if feat["vert_max"] >= 94:
            return 0
        if feat["vert_min"] <= 45:
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
    height, width = mask.shape

    hori_lens = []
    min_h = 100
    max_h = 0
    for h in range(height):
        curr_len = 0
        for w in range(width):
            curr_len += mask[h][w]

        if curr_len > 0:
            hori_lens.append(curr_len)
            min_h = min(min_h, h)
            max_h = max(max_h, h)

    vert_len = max_h - min_h
    vert_center = 0.5 * (max_h + min_h)
    return {
        "hori_len": np.mean(hori_lens),
        "vert_len": vert_len,
        "vert_center": vert_center,
        "vert_min": min_h,
        "vert_max": max_h,
    }


class TowelEEConfig:
    def __init__(self):
        # TODO: fix this
        self.init_ee_pos = [0.438, 0, 0.383]  # the home ee position
        self.home = np.pi * np.array([0.0, -0.1, 0.0, -0.8, 0.0, 0.7, 0.0], dtype=np.float32)

        # limits
        self.pos_low = np.array([0.36, -0.15, 0.178])
        self.pos_high = np.array([0.75, 0.15, 0.5])

        self.rot_abs_min = np.pi * np.array([0.8, 0, 0.0]).astype(np.float32)
        self.rot_abs_max = np.pi * np.array([1, 0.2, 0.35]).astype(np.float32)

        # this is not exactly the same as the range above
        # because [-0.45pi, 0.45pi] cannot be expressed as cont. low & high
        self.ee_range_low = self.pos_low.tolist() + [-np.pi, -np.pi, -np.pi]
        self.ee_range_high = self.pos_high.tolist() + [np.pi, np.pi, np.pi]

    def clip(self, pos: np.ndarray, rot: np.ndarray):
        # ref_sum = pos.sum() + rot.sum()
        pos = np.clip(pos, self.pos_low, self.pos_high)
        rot = np.sign(rot) * np.clip(np.abs(rot), self.rot_abs_min, self.rot_abs_max)
        # new_sum = pos.sum() + rot.sum()
        # if abs(ref_sum.item()- new_sum.item()) >= 1e-6:
        #     print("Clipped!")
        return pos, rot

    def ee_in_good_range(self, pos: np.ndarray, rot: np.ndarray, verbose) -> bool:
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

        min_height = 0.3
        ee_pos, ee_quat = robot._robot.get_ee_pose()
        if ee_pos[2].item() < min_height:
            print("fixing position")
            if not robot._robot.is_running_policy():
                robot._robot.start_cartesian_impedance()
                time.sleep(1)
                had_no_policy = True

        assert robot.cfg.controller_type == "CARTESIAN_DELTA"
        while ee_pos[2].item() < min_height:
            # ee_pos[2] += 0.02
            # if not robot._robot.is_running_policy():
            #     robot._robot.start_cartesian_impedance()
            #     time.sleep(1)

            # robot._robot.update_desired_ee_pose(ee_pos, ee_quat)
            robot.update([0, 0, 0.02, 0, 0, 0, 0.9])
            ee_pos, ee_quat = robot._robot.get_ee_pose()

        # open the gripper
        robot.update([0, 0, 0, 0, 0, 0, 0])
        if had_no_policy:
            robot._robot.terminate_current_policy()
