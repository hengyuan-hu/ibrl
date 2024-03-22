import os
from dataclasses import dataclass
from functools import cached_property
from collections import namedtuple, defaultdict
import pprint
import json
import h5py
import torch
import numpy as np

import common_utils
from common_utils import ibrl_utils as utils
from env.robosuite_wrapper import PixelRobosuite, STATE_KEYS, PROP_KEYS


Batch = namedtuple("Batch", ["obs", "action"])


@dataclass
class DatasetConfig:
    path: str = ""
    rl_camera: str = "robot0_eye_in_hand"
    num_data: int = -1
    max_len: int = -1
    eval_episode_len: int = 300
    use_state: int = 0
    prop_stack: int = 1
    norm_action: int = 0
    obs_stack: int = 1
    state_stack: int = 1
    real_data: int = 0

    def __post_init__(self):
        self.rl_cameras = self.rl_camera.split("+")

    @cached_property
    def env_config(self) -> dict:
        if self.real_data:
            env_config = {
                "env_name": "real",
                "env_kwargs": {"robots": ["panda"], "controller_configs": {"control_delta": True}},
            }
            return env_config

        with h5py.File(self.path) as f:
            assert "env_args" in f["data"].attrs
            return json.loads(f["data"].attrs["env_args"])  # type: ignore

    @cached_property
    def task_name(self):
        return self.env_config["env_name"]

    @cached_property
    def robot(self):
        return self.env_config["env_kwargs"]["robots"]

    @cached_property
    def ctrl_delta(self):
        return self.env_config["env_kwargs"]["controller_configs"]["control_delta"]


class RobomimicDataset:
    def __init__(self, cfg: DatasetConfig):
        self.cfg = cfg

        self.data = []
        datafile = h5py.File(cfg.path)
        num_episode: int = len(list(datafile["data"].keys()))  # type: ignore
        print(f"Raw Dataset size (#episode): {num_episode}")

        self.idx2entry = []  # store idx -> (episode_idx, timestep_idx)
        episode_lens = []
        all_actions = []  # for # logging purpose
        for episode_id in range(num_episode):
            if cfg.num_data > 0 and len(episode_lens) >= cfg.num_data:
                break

            episode_tag = f"demo_{episode_id}"
            episode = datafile[f"data/{episode_tag}"]

            if self.cfg.real_data:
                rewards = np.array(episode["rewards"]).astype(np.float32)  # type: ignore
                assert rewards.sum() == 1, f"wrong reward for episode {rewards.sum()}"

            actions = np.array(episode["actions"]).astype(np.float32)  # type: ignore
            actions = torch.from_numpy(actions)
            all_actions.append(actions)

            episode_data: dict = {"action": actions}

            if episode_id == 0:
                common_utils.wrap_ruler("all avail obs", 60)
                for k, v in episode["obs"].items():  # type: ignore
                    print(k, np.array(v).shape)
                common_utils.wrap_ruler("", 60)

            if self.cfg.use_state:
                all_states = []
                for key in STATE_KEYS[self.cfg.task_name]:
                    all_states.append(episode["obs"][key])  # type: ignore
                episode_data["state"] = np.concatenate(all_states, axis=1).astype(np.float32)
            else:
                for camera in self.cfg.rl_cameras:
                    obses: np.ndarray = episode[f"obs/{camera}_image"]  # type: ignore
                    assert obses.shape[0] == actions.shape[0]
                    episode_data[camera] = obses

            # proprioception states
            if "obs/prop" in episode:  # type: ignore
                prop = np.array(episode["obs/prop"])  # type: ignore
                episode_data["prop"] = prop.astype(np.float32)
            else:
                robot_locs = []
                for key in PROP_KEYS:
                    robot_locs.append(episode["obs"][key])  # type: ignore
                episode_data["prop"] = np.concatenate(robot_locs, axis=1).astype(np.float32)

            episode_len = actions.shape[0]
            assert (
                episode_len < cfg.eval_episode_len
            ), f"found episode len {episode_len} > {cfg.eval_episode_len} (eval_len)"
            if self.cfg.max_len > 0 and episode_len > self.cfg.max_len:
                print(f"removing {episode_tag} because it is too long {episode_len}")
                continue

            # convert the data to list of dict
            episode_entries = []
            for i in range(episode_len):
                entry = {"action": episode_data["action"][i]}
                if self.cfg.ctrl_delta:
                    assert entry["action"].min() >= -1
                    assert entry["action"].max() <= 1

                entry["prop"] = utils.concat_obs(i, episode_data["prop"], cfg.prop_stack)
                if cfg.use_state:
                    entry["state"] = utils.concat_obs(i, episode_data["state"], cfg.state_stack)
                else:
                    for camera in cfg.rl_cameras:
                        entry[camera] = utils.concat_obs(i, episode_data[camera], cfg.obs_stack)

                self.idx2entry.append((len(self.data), len(episode_entries)))
                episode_entries.append(entry)

            episode_lens.append(len(episode_entries))
            self.data.append(episode_entries)
        datafile.close()

        if cfg.use_state:
            self.obs_shape = self.data[-1][-1]["state"].size()
        else:
            self.obs_shape = self.data[-1][-1][cfg.rl_cameras[0]].size()

        self.prop_shape = self.data[-1][-1]["prop"].size()
        self.action_dim = self.data[-1][-1]["action"].size()[0]
        print(f"Dataset size: {len(self.data)} episodes, {len(self.idx2entry)} steps")
        print(f"average length {np.mean(episode_lens):.1f}")
        print(f"max length: {sorted(episode_lens)[::-1][:5]}")
        print(f"obs shape:", self.obs_shape, "; prop shape:", self.prop_shape)

        all_actions = torch.cat(all_actions, dim=0)
        action_mins = all_actions.min(dim=0)[0]
        action_maxs = all_actions.max(dim=0)[0]
        for i in range(self.action_dim):
            print(f"action dim {i}: [{action_mins[i].item():.2f}, {action_maxs[i].item():.2f}]")

        if cfg.real_data:
            self.env = None
            return

        self.env_params: dict = dict(
            env_name=self.cfg.task_name,
            robots=self.cfg.robot,
            episode_length=cfg.eval_episode_len,
            reward_shaping=False,
            image_size=224,
            rl_image_size=self.obs_shape[-1] if not cfg.use_state else 96,
            camera_names=cfg.rl_cameras,
            rl_cameras=cfg.rl_cameras,
            device="cuda",
            use_state=cfg.use_state,
            obs_stack=self.cfg.obs_stack,
            state_stack=self.cfg.state_stack,
            prop_stack=cfg.prop_stack,
            ctrl_delta=bool(self.cfg.ctrl_delta),
        )
        self.env = PixelRobosuite(**self.env_params)
        self._check_controller_cfg()

    def _check_controller_cfg(self):
        assert self.env is not None
        ref_controller_cfg = self.cfg.env_config["env_kwargs"]["controller_configs"]
        # rename for consistency
        ref_controller_cfg["damping_ratio"] = ref_controller_cfg["damping"]
        ref_controller_cfg["damping_ratio_limits"] = ref_controller_cfg["damping_limits"]
        ref_controller_cfg.pop("damping")
        ref_controller_cfg.pop("damping_limits")
        assert ref_controller_cfg == self.env.ctrl_config
        assert self.env.env.control_freq == self.cfg.env_config["env_kwargs"]["control_freq"]

        print(common_utils.wrap_ruler("config when the data was collected"))
        self.cfg.env_config["env_kwargs"].pop("controller_configs")
        pprint.pprint(self.cfg.env_config["env_kwargs"])
        print(common_utils.wrap_ruler(""))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def _convert_to_batch(self, samples, device):
        batch = {}
        for k, v in samples.items():
            batch[k] = torch.stack(v).to(device)

        action = {"action": batch.pop("action")}
        ret = Batch(obs=batch, action=action)
        return ret

    def sample_bc(self, batchsize, device):
        indices = np.random.choice(len(self.idx2entry), batchsize)
        samples = defaultdict(list)
        for idx in indices:
            episode_idx, step_idx = self.idx2entry[idx]
            entry = self.data[episode_idx][step_idx]
            for k, v in entry.items():
                samples[k].append(v)

        return self._convert_to_batch(samples, device)
