import numpy as np
import cv2
import time
from scipy.spatial.transform import Rotation
from env.franka_utils import ControlFreqGuard


class Drawer:
    def __init__(self):
        pass

    def reset(self):
        pass

    def reward(self, curr_obs):
        if "agentview" in curr_obs:
            # TODO: this is not efficient
            image = curr_obs["agentview"].cpu().numpy()
        else:
            image = curr_obs["agentview_image"]

        # for k, v in curr_obs.items():
        #     print(k, type(v), v.shape)

        assert isinstance(image, np.ndarray)
        if image.shape[0] == 3:
            image = np.transpose(image, [1, 2, 0])
        count = count_red_mask(image, 120, 75)
        return float(count > 20)


class DrawerEEConfig:
    def __init__(self):
        self.init_ee_pos = [0.41, 0, 0.5]  # the home ee position
        self.home = np.array(
            [0.0, -0.15 * np.pi, 0.0, -0.75 * np.pi, 0.0, 0.60 * np.pi, 0], dtype=np.float32
        )

        # limits
        self.pos_low = np.array([0.35, -0.2, 0.19])
        self.pos_high = np.array([0.50, 0.1, 0.65])

        # TODO: update this min rot for x, currently it is too limiting
        self.rot_abs_min = np.pi * np.array([0.45, 0, 0.0]).astype(np.float32)
        self.rot_abs_max = np.pi * np.array([1, 0.20, 0.20]).astype(np.float32)

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

    def ee_in_good_range(self, pos: np.ndarray, quat: np.ndarray, verbose) -> bool:
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

        ee_pos, _ = robot._robot.get_ee_pose()
        target_y = -0.1
        if ee_pos[1].item() > target_y:
            print("fixing position")
            if not robot._robot.is_running_policy():
                robot._robot.start_cartesian_impedance()
                time.sleep(1)
                had_no_policy = True

        assert robot.cfg.controller_type == "CARTESIAN_DELTA"
        while ee_pos[1].item() > target_y:
            with ControlFreqGuard(20):
                robot.update([0, -0.02, 0, 0, 0, 0, 0])
            ee_pos, _ = robot._robot.get_ee_pose()

        if had_no_policy:
            robot._robot.terminate_current_policy()


def show_image(images, wait_key=0):
    image = np.hstack(images)
    # convert the color because capture_rgb performed conversion
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imshow("img", image)
    cv2.waitKey(wait_key)


def get_red_mask(image, red_thres, non_red_thres):
    assert image.shape[2] == 3

    height, width, _ = image.shape
    mask = np.zeros_like(image)
    for h in range(height):
        for w in range(width):
            pixel = image[h][w]
            r, g, b = pixel
            # if r > red_thres:
            if r > red_thres and g < non_red_thres and b < non_red_thres:
                mask[h][w] = 255

    return mask


def count_red_mask(image, red_thres, non_red_thres):
    assert image.shape[2] == 3
    red = (image[:, :, 0] > red_thres).astype(np.int32)
    non_red = (image[:, :, 1:] < non_red_thres).astype(np.int32)
    count = (red * non_red[:, :, 0] * non_red[:, :, 1]).sum()
    return count


if __name__ == "__main__":
    from env.cameras import RealSenseCamera

    left_camera = RealSenseCamera("042222070680", height=96, width=96, depth=False)

    while True:
        images = []

        rgb_image = left_camera.get_frames()[""]
        images.append(rgb_image)

        t = time.time()
        red_mask = get_red_mask(rgb_image, 120, 75)
        print(f"time taken: {time.time() - t:.4f}")

        t = time.time()
        count = count_red_mask(rgb_image, 120, 75)
        print(f"{count=}, time taken for count: {time.time() - t:.4f}")

        images.append(red_mask)
        red_mask = get_red_mask(rgb_image, 127, 70)
        images.append(red_mask)
        show_image(images, 1)
        print("---------")
