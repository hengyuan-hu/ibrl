from typing import Optional
from collections import defaultdict, deque

import torch
import robosuite
from robosuite import load_controller_config
import numpy as np
from common_utils import ibrl_utils as utils
import common_utils


# all avail views:
# 'frontview', 'birdview', --> too far for this task
# 'agentview', 'robot0_robotview', --> same
# 'sideview', 'robot0_eye_in_hand'
GOOD_CAMERAS = {
    "Lift": ["agentview", "sideview", "robot0_eye_in_hand"],
    "PickPlaceCan": ["agentview", "robot0_eye_in_hand"],
    "NutAssemblySquare": ["agentview", "robot0_eye_in_hand"],
}
DEFAULT_CAMERA = "agentview"


DEFAULT_STATE_KEYS = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object"]
STATE_KEYS = {
    "Lift": DEFAULT_STATE_KEYS,
    "PickPlaceCan": DEFAULT_STATE_KEYS,
    "NutAssemblySquare": DEFAULT_STATE_KEYS,
    "TwoArmTransport": [
        "robot0_eef_pos",
        "robot0_eef_quat",
        "robot0_gripper_qpos",
        "robot1_eef_pos",
        "robot1_eef_quat",
        "robot1_gripper_qpos",
        "object",
    ],
    "ToolHang": [
        "object",  # (389, 44)
        "robot0_eef_pos",  # (389, 3)
        "robot0_eef_quat",  # (389, 4)
        "robot0_gripper_qpos",  # (389, 2)
        # "robot0_gripper_qvel",  # (389, 2)
        # "robot0_eef_vel_ang",  # (389, 3)
        # "robot0_eef_vel_lin",  # (389, 3)
        # "robot0_joint_pos", # (389, 7)
        # "robot0_joint_pos_cos",  # (389, 7)
        # "robot0_joint_pos_sin",  # (389, 7)
        # "robot0_joint_vel",  # (389, 7)
    ],
}
STATE_SHAPE = {
    "Lift": (19,),
    "PickPlaceCan": (23,),
    "NutAssemblySquare": (23,),
    "TwoArmTransport": (59,),
    "ToolHang": (53,),
}
PROP_KEYS = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
PROP_DIM = 9


class PixelRobosuite:
    def __init__(
        self,
        env_name,
        robots,
        episode_length,
        *,
        reward_shaping=False,
        image_size=224,
        rl_image_size=96,
        device="cuda",
        camera_names=[DEFAULT_CAMERA],
        rl_cameras=["agentview"],
        env_reward_scale=1.0,
        end_on_success=True,
        use_state=False,
        obs_stack=1,
        state_stack=1,
        prop_stack=1,
        cond_action=0,
        flip_image=True,  # only false if using with eval_with_init_state
        ctrl_delta=True,
        record_sim_state: bool = False,
    ):
        assert isinstance(camera_names, list)
        self.camera_names = camera_names
        self.ctrl_config = load_controller_config(default_controller="OSC_POSE")
        self.ctrl_config["control_delta"] = ctrl_delta
        self.record_sim_state = record_sim_state
        self.env = robosuite.make(
            env_name=env_name,
            robots=robots,
            controller_configs=self.ctrl_config,
            has_offscreen_renderer=True,
            use_camera_obs=True,
            reward_shaping=reward_shaping,
            camera_names=self.camera_names,
            camera_heights=image_size,
            camera_widths=image_size,
            horizon=episode_length,
        )
        self.rl_cameras = rl_cameras if isinstance(rl_cameras, list) else [rl_cameras]
        self.image_size = image_size
        self.rl_image_size = rl_image_size or image_size
        self.env_reward_scale = env_reward_scale
        self.end_on_success = end_on_success
        self.use_state = use_state
        self.state_keys = STATE_KEYS[env_name]
        self.prop_keys = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
        self.flip_image = flip_image

        self.resize_transform = None
        if self.rl_image_size != self.image_size:
            self.resize_transform = utils.get_rescale_transform(self.rl_image_size)

        self.action_dim: int = len(self.env.action_spec[0])
        self._observation_shape: tuple[int, ...] = (3 * obs_stack, rl_image_size, rl_image_size)
        self._state_shape: tuple[int] = (STATE_SHAPE[env_name][0] * state_stack,)
        self.prop_shape: tuple[int] = (PROP_DIM * prop_stack,)
        self.device = device

        self.time_step = 0
        self.episode_reward = 0
        self.episode_extra_reward = 0
        self.terminal = True

        self.obs_stack = obs_stack
        self.state_stack = state_stack
        self.prop_stack = prop_stack
        self.cond_action = cond_action
        self.past_obses = defaultdict(list)
        self.past_actions = deque(maxlen=self.cond_action)

    @property
    def observation_shape(self):
        if self.use_state:
            return self._state_shape
        else:
            return self._observation_shape

    def _extract_images(self, obs):
        # assert self.frame_stack == 1, "frame stack not supported"

        high_res_images = {}
        rl_obs = {}

        if self.use_state:
            states = []
            for key in self.state_keys:
                if key == "object":
                    key = "object-state"
                states.append(obs[key])
            state = torch.from_numpy(np.concatenate(states).astype(np.float32))
            # first append, then concat
            self.past_obses["state"].append(state)
            rl_obs["state"] = utils.concat_obs(
                len(self.past_obses["state"]) - 1, self.past_obses["state"], self.state_stack
            ).to(self.device)

        props = []
        for key in self.prop_keys:
            props.append(obs[key])
        prop = torch.from_numpy(np.concatenate(props).astype(np.float32))
        # first append, then concat
        self.past_obses["prop"].append(prop)
        rl_obs["prop"] = utils.concat_obs(
            len(self.past_obses["prop"]) - 1, self.past_obses["prop"], self.prop_stack
        ).to(self.device)

        for camera_name in self.camera_names:
            image_key = f"{camera_name}_image"
            image_obs = obs[image_key]
            if self.flip_image:
                image_obs = image_obs[::-1]
            image_obs = torch.from_numpy(image_obs.copy()).permute([2, 0, 1])

            # keep the high-res version for rendering
            high_res_images[camera_name] = image_obs
            if camera_name not in self.rl_cameras:
                continue

            rl_image_obs = image_obs
            if self.resize_transform is not None:
                # set the device here because transform is 5x faster on GPU
                rl_image_obs = self.resize_transform(rl_image_obs.to(self.device))
            # first append, then concat
            self.past_obses[camera_name].append(rl_image_obs)
            rl_obs[camera_name] = utils.concat_obs(
                len(self.past_obses[camera_name]) - 1,
                self.past_obses[camera_name],
                self.obs_stack,
            )

        if self.record_sim_state:
            sim_state = self.env.sim.get_state().flatten()
            rl_obs["sim_state"] = torch.from_numpy(sim_state)
            for key in DEFAULT_STATE_KEYS:
                env_key = "object-state" if key == "object" else key
                rl_obs[key] = torch.from_numpy(obs[env_key])

        return rl_obs, high_res_images

    def reset(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        self.time_step = 0
        self.episode_reward = 0
        self.episode_extra_reward = 0
        self.terminal = False
        self.past_obses.clear()
        self.past_actions.clear()
        for _ in range(self.cond_action):
            self.past_actions.append(torch.zeros(self.action_dim))

        obs = self.env.reset()
        rl_obs, high_res_images = self._extract_images(obs)

        if self.cond_action > 0:
            past_action = torch.from_numpy(np.stack(self.past_actions)).to(self.device)
            rl_obs["past_action"] = past_action

        return rl_obs, high_res_images

    def step(self, actions: torch.Tensor) -> tuple[dict, float, bool, bool, dict]:
        """
        all inputs and outputs are tensors
        """
        if actions.dim() == 1:
            actions = actions.unsqueeze(0)
        num_action = actions.size(0)

        rl_obs = {}
        # record the action in original format from model
        if self.cond_action > 0:
            for i in range(actions.size(0)):
                self.past_actions.append(actions[i])
            past_action = torch.stack(list(self.past_actions)).to(self.device)
            rl_obs["past_action"] = past_action

        actions = actions.numpy()

        reward = 0
        success = False
        terminal = False
        high_res_images = {}
        for i in range(num_action):
            self.time_step += 1
            obs, step_reward, terminal, _ = self.env.step(actions[i])
            # NOTE: extract images every step for potential obs stacking
            # this is not efficient
            curr_rl_obs, curr_high_res_images = self._extract_images(obs)

            if i == num_action - 1:
                rl_obs.update(curr_rl_obs)
                high_res_images.update(curr_high_res_images)

            reward += step_reward
            self.episode_reward += step_reward

            if step_reward == 1:
                success = True
                if self.end_on_success:
                    terminal = True

            if terminal:
                break

        reward = reward * self.env_reward_scale
        self.terminal = terminal
        return rl_obs, reward, terminal, success, high_res_images


if __name__ == "__main__":
    from torchvision.utils import save_image

    env = PixelRobosuite("Lift", "Panda", 200, image_size=256, camera_names=GOOD_CAMERAS["Lift"])
    x = env.reset()[0][GOOD_CAMERAS["Lift"][0]].float() / 255
    print(x.dtype)
    print(x.shape)
    save_image(x, "test_env.png")
