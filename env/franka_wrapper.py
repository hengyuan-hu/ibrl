from collections import defaultdict
from dataclasses import dataclass

import torch
import numpy as np
import cv2
from env.polymetis_controller import PolyMetisControllerClient
from env.cameras import RealSenseCamera
from env.robosuite_wrapper import PROP_KEYS
from common_utils import ibrl_utils as utils

from env.franka_utils import ControlFreqGuard
from env.lift import Lift
from env.drawer import Drawer
from env.hang import Hang
from env.towel import Towel


_ROBOT_CAMERAS = {
    "fr2": {
        "agentview": "042222070680",
        "robot0_eye_in_hand": "241222076578",
        "frontview": "838212072814",
    }
}


@dataclass
class FrankaEnvConfig:
    task: str
    episode_length: int = 200
    robot: str = "fr2"  # fr2, fr3
    control_hz: float = 10.0
    image_size: int = 224
    rl_image_size: int = 96
    use_depth: int = 0
    rl_camera: str = "robot0_eye_in_hand"
    randomize: int = 0
    show_camera: int = 0
    drop_after_terminal: int = 1
    record: int = 0
    save_dir: str = ""

    def __post_init__(self):
        self.rl_cameras = self.rl_camera.split("+")

        if self.robot == "fr2":
            self.remote_ip_address = "tcp://172.16.0.1:4242"
        elif self.robot == "local":
            self.remote_ip_address = "tcp://0.0.0.0:4242"
        else:
            assert False, f"unknown robot {self.robot}"


class FrankaEnv:
    """
    A simple Gym Environment for controlling robots.

    gripper: -1: open, 1: close
    """

    def __init__(self, device, cfg: FrankaEnvConfig):
        self.device = device
        self.cfg = cfg

        self.cameras = {}
        assert not self.cfg.use_depth
        for camera in self.cfg.rl_cameras:
            self.cameras[camera] = RealSenseCamera(
                _ROBOT_CAMERAS[self.cfg.robot][camera],
                width=cfg.image_size,
                height=cfg.image_size,
                depth=cfg.use_depth,
            )

        self.reward_camera = {}
        if self.cfg.task == "drawer":
            self.reward_camera["agentview"] = RealSenseCamera(
                _ROBOT_CAMERAS[self.cfg.robot]["agentview"],
                width=cfg.image_size,
                height=cfg.image_size,
                depth=cfg.use_depth,
            )

        self.record_camera = {}
        self.video_frames = defaultdict(list)
        if self.cfg.record:
            record_camera_name = "agentview"
            if record_camera_name in self.cameras:
                self.record_camera[record_camera_name] = self.cameras[record_camera_name]
            elif record_camera_name in self.reward_camera:
                self.record_camera[record_camera_name] = self.reward_camera[record_camera_name]
            else:
                self.record_camera[record_camera_name] = RealSenseCamera(
                    _ROBOT_CAMERAS[self.cfg.robot][record_camera_name],
                    width=cfg.image_size,
                    height=cfg.image_size,
                    depth=cfg.use_depth,
                )

        self.resize_transform = None
        if cfg.rl_image_size != cfg.image_size:
            self.resize_transform = utils.get_rescale_transform(cfg.rl_image_size)

        self.observation_shape: tuple[int, ...] = (3, cfg.rl_image_size, cfg.rl_image_size)
        self.prop_shape: tuple[int] = (8,)
        self.controller = PolyMetisControllerClient(cfg.remote_ip_address, cfg.task)
        self.action_dim = len(self.controller.action_space.low)

        self.time_step = 0
        self.terminal = True

        # for compatibility
        self.rl_cameras = cfg.rl_cameras
        self.state_shape = (-1,)

        if cfg.task == "lift":
            self.task = Lift(verbose=False)
        elif cfg.task == "drawer":
            self.task = Drawer()
        elif cfg.task == "hang":
            self.task = Hang()
        elif cfg.task == "towel":
            self.task = Towel()
        else:
            assert False, f"unknown task {self.task}"

    def get_image_from_camera(self, cameras):
        # TODO: maybe fuse the functions that reads from camera?
        obs = {}
        for name, camera in cameras.items():
            frames = camera.get_frames()
            assert len(frames) == 1
            image = frames[""]
            image = torch.from_numpy(image).permute([2, 0, 1])
            if self.resize_transform is not None:
                image = self.resize_transform(image)
            obs[name] = image
        return obs

    def observe(self):
        props, in_good_range = self.controller.get_state()
        if not in_good_range:
            print("Warning[FrankaEnv]: bad range, should have restarted")

        prop = torch.from_numpy(
            np.concatenate([props[prop_key] for prop_key in PROP_KEYS]).astype(np.float32)
        )
        assert prop.size(0) == self.prop_shape[0], f"{prop.size(0)=}, {self.prop_shape[0]=}"

        rl_obs = {"prop": prop.to(self.device)}
        high_res_images = {}

        for name, camera in self.cameras.items():
            frames = camera.get_frames()
            assert len(frames) == 1
            image = frames[""]
            if name == "frontview":
                image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

            key = f"{name}"
            high_res_images[key] = image

            rl_image_obs = torch.from_numpy(image).permute([2, 0, 1])
            if self.resize_transform is not None:
                # set the device here because transform is 5x faster on GPU
                rl_image_obs = self.resize_transform(rl_image_obs.to(self.device))
            rl_obs[key] = rl_image_obs

        if self.cfg.show_camera:
            images = []
            for _, v in high_res_images.items():
                # np_image = v.cpu().permute([1, 2, 0]).numpy()
                images.append(v)
            image = np.hstack(images)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imshow("img", image)
            cv2.waitKey(1)

        if self.cfg.record:
            for k, camera in self.record_camera.items():
                if k in high_res_images:
                    self.video_frames[k] = high_res_images[k]
                else:
                    frames = camera.get_frames()
                    assert len(frames) == 1
                    self.video_frames[k] = frames[""]

        return rl_obs

    def get_reward_terminal(self):
        if self.cfg.task == "lift":
            props, in_good_range = self.controller.get_state()
            reward: float = self.task.reward(props)
        elif self.cfg.task == "drawer":
            _, in_good_range = self.controller.get_state()
            reward_obs = self.get_image_from_camera(self.reward_camera)
            reward: float = self.task.reward(reward_obs)
        elif self.cfg.task in ["hang", "towel"]:
            props, in_good_range = self.controller.get_state()
            reward_obs = self.observe()
            reward_obs.update(props)
            reward: float = self.task.reward(reward_obs)
        else:
            assert False

        success = reward > 0

        self.terminal = success
        if self.time_step >= self.cfg.episode_length:
            self.terminal = True
        if not in_good_range:
            self.terminal = True

        if success and self.cfg.drop_after_terminal:
            self.release_gripper()

        # print(f"step: {self.time_step}, terminal: {self.terminal}, {reward=}")
        if self.terminal and self.cfg.record:
            pass

        return reward, self.terminal, success

    def apply_action(self, action: torch.Tensor):
        # print(">>>>>>>>>>>>>>>>> apply action", action.size())
        self.controller.update(action.numpy())
        self.time_step += 1
        return

    # ============= gym style api for compatibility with data collection ============= #
    def reset(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """return observation and high resolution observation"""
        print(f"{self.cfg.randomize=}")
        self.controller.reset(randomize=bool(self.cfg.randomize))

        self.time_step = 0
        self.episode_reward = 0
        self.episode_extra_reward = 0
        self.terminal = False
        self.video_frames = []

        self.task.reset()
        return self.observe(), {}

    # ============= gym style api for compatibility with data collection ============= #
    def step(
        self, action: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], float, bool, bool, dict[str, torch.Tensor]]:
        # Immediately update with the action.
        # Note that `action` has been scaled to [-1, 1],
        # `self.controller.update` will perform the unscale

        with ControlFreqGuard(self.cfg.control_hz):
            # print(f"[env] step: {self.time_step}")
            assert action.dim() == 1, "multi-action open loop not supported yet"
            assert action.min() >= -1, action.min()

            self.controller.update(action.numpy())
            self.time_step += 1

        rl_obs = self.observe()
        reward, terminal, success = self.get_reward_terminal()

        if terminal and self.cfg.drop_after_terminal:
            self.release_gripper()
        #     # release the gripper
        #     action[-1] = -1
        #     self.controller.update(action.cpu().numpy())

        return rl_obs, reward, terminal, success, {}

    def release_gripper(self):
        action = np.zeros(self.action_dim)
        action[-1] = -1
        self.controller.update(action)


def test():
    np.set_printoptions(precision=4)

    cfg = FrankaEnvConfig()
    env = FrankaEnv("cuda", cfg)

    env.reset()
    obs = env.observe()
    for k, v in obs.items():
        print(k, v.size())

    with ControlFreqGuard(10.0):
        action = torch.from_numpy(np.random.random(7).astype(np.float32))
        env.apply_action(action)

    obs = env.observe()
    for k, v in obs.items():
        print(k, v.size())


if __name__ == "__main__":
    test()
