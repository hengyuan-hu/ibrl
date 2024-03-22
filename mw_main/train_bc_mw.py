from dataclasses import dataclass, field
import os
import sys
import pprint
from collections import namedtuple, defaultdict
import json
import copy
import yaml
from env.metaworld_wrapper import PixelMetaWorld

import pyrallis
import numpy as np
import torch
import h5py

import common_utils
from common_utils import ibrl_utils as utils
from mw_main.bc_policy import BcPolicy, BcPolicyConfig
from mw_main.eval_mw import run_eval


Batch = namedtuple("Batch", ["obs", "action"])


root = os.path.dirname(os.path.dirname(__file__))
DATASETS = {
    "Assembly": "data/metaworld/Assembly_frame_stack_1_96x96_end_on_success/dataset.hdf5",
    "BoxClose": "data/metaworld/BoxClose_frame_stack_1_96x96_end_on_success/dataset.hdf5",
    "StickPull": "data/metaworld/StickPull_frame_stack_1_96x96_end_on_success/dataset.hdf5",
    "CoffeePush": "data/metaworld/CoffeePush_frame_stack_1_96x96_end_on_success/dataset.hdf5",
}
for k, v in DATASETS.items():
    DATASETS[k] = os.path.join(root, v)


@dataclass
class DatasetConfig:
    path: str = ""
    rl_camera: str = "corner2"
    num_data: int = 3
    max_len: int = -1
    eval_episode_len: int = 100
    use_state: int = 0
    obs_stack: int = 1
    frame_stack: int = 1
    action_repeat: int = 2

    def __post_init__(self):
        if self.path in DATASETS:
            self.path = DATASETS[self.path]


class MetaWorldDataset:
    def __init__(self, cfg: DatasetConfig):
        config_path = os.path.join(os.path.dirname(cfg.path), "env_cfg.json")
        self.env_config = json.load(open(config_path, "r"))

        self.task_name = self.env_config["env_name"]
        self.robot = self.env_config["env_kwargs"]["robots"]
        self.cfg = cfg
        self.obs_key = "state" if self.cfg.use_state else "obs"

        self.data = []
        f = h5py.File(cfg.path)
        num_episode: int = len(list(f["data"].keys()))  # type: ignore
        self.idx2entry = []
        print(f"Raw Dataset size (#episode): {num_episode}")
        episode_lens = []
        for episode_id in range(num_episode):
            if cfg.num_data > 0 and len(episode_lens) >= cfg.num_data:
                break

            episode_tag = f"demo_{episode_id}"
            episode = f[f"data/{episode_tag}"]
            actions: np.ndarray = episode["actions"]  # type: ignore
            if self.cfg.use_state:
                all_states = []
                all_states.append(episode["obs"]["state"])  # type: ignore
                obses = np.concatenate(all_states, axis=1)
            else:
                obses: np.ndarray = episode[f"obs/{self.cfg.rl_camera}_image"]  # type: ignore
            assert obses.shape[0] == actions.shape[0]
            # proprioception states
            robot_locs = []
            robot_locs.append(episode["obs"]["prop"])  # type: ignore
            props = np.concatenate(robot_locs, axis=1)

            episode_len = actions.shape[0]
            if self.cfg.max_len > 0 and episode_len > self.cfg.max_len:
                print(f"removing {episode_tag} because it is too long {episode_len}")
                continue

            episode_lens.append(episode_len)

            data_episode = []
            for i, (obs, action, prop) in enumerate(zip(obses, actions, props)):
                stack_obs = [torch.from_numpy(obs.astype(np.float32))]
                stack_props = [torch.from_numpy(prop.astype(np.float32))]
                for j in range(1, cfg.obs_stack):
                    if i - j < 0:
                        if cfg.use_state:
                            stack_obs.append(torch.zeros_like(stack_obs[-1]))
                        stack_props.append(torch.zeros_like(stack_props[-1]))
                    else:
                        if cfg.use_state:
                            stack_obs.append(torch.from_numpy(obses[i - j].astype(np.float32)))
                        stack_props.append(torch.from_numpy(props[i - j].astype(np.float32)))

                entry = {
                    self.obs_key: torch.cat(stack_obs, dim=0),
                    "action": torch.from_numpy(action.astype(np.float32)),
                    "prop": torch.cat(stack_props, dim=0),
                }

                self.idx2entry.append((len(self.data), len(data_episode)))
                data_episode.append(entry)
            self.data.append(data_episode)
        f.close()

        self.obs_shape = self.data[-1][-1][self.obs_key].size()
        self.prop_shape = self.data[-1][-1]["prop"].size()
        self.action_dim = self.data[-1][-1]["action"].size()[0]
        print(f"Dataset size: {len(self.data)} episodes")
        print(f"average length {np.mean(episode_lens):.1f}")
        print(f"max length {np.max(episode_lens):.1f}")
        print(f"obs shape:", self.obs_shape)

        self.env_params = dict(
            env_name=self.task_name,
            robots=self.robot,
            episode_length=cfg.eval_episode_len,
            action_repeat=self.cfg.action_repeat,
            frame_stack=self.cfg.frame_stack,
            obs_stack=self.cfg.obs_stack,
            reward_shaping=False,
            rl_image_size=self.obs_shape[-1],
            camera_names=[cfg.rl_camera],
            rl_camera=cfg.rl_camera,
            device="cuda",
            use_state=cfg.use_state,
        )

        # Check that DatasetConfig is consistent with env_cfg
        assert self.cfg.frame_stack == self.env_config["env_kwargs"]["frame_stack"]
        assert self.cfg.obs_stack == self.env_config["env_kwargs"]["obs_stack"]
        assert self.cfg.action_repeat == self.env_config["env_kwargs"]["action_repeat"]

        self.env = PixelMetaWorld(**self.env_params)
        print(common_utils.wrap_ruler("config when the data was collected"))

        pprint.pprint(self.env_config["env_kwargs"])
        print(common_utils.wrap_ruler(""))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def sample(self, batchsize, device):
        samples = self._sample_default(batchsize)

        batch = {}
        for k, v in samples.items():
            batch[k] = torch.stack(v).to(device)

        action = {"action": batch.pop("action")}
        ret = Batch(obs=batch, action=action)
        return ret

    def _sample_default(self, batchsize) -> dict[str, list[torch.Tensor]]:
        indices = np.random.choice(len(self.idx2entry), batchsize)
        batch = defaultdict(list)
        for idx in indices:
            episode_idx, step_idx = self.idx2entry[idx]
            entry = self.data[episode_idx][step_idx]
            for k, v in entry.items():
                batch[k].append(v)

        return batch


@dataclass
class MainConfig(common_utils.RunConfig):
    dataset: DatasetConfig = field(default_factory=lambda: DatasetConfig(""))
    policy: BcPolicyConfig = field(default_factory=lambda: BcPolicyConfig())
    seed: int = 1
    load_model: str = "none"
    # training
    num_epoch: int = 2
    epoch_len: int = 10000
    batchsize: int = 256
    lr: float = 1e-4
    grad_clip: float = 5
    weight_decay: float = 0
    ema: float = -1
    # to be overwritten by run() to facilitate model loading
    task_name: str = ""
    robot: str = ""
    rl_image_size: int = -1
    # log
    use_wb: int = 0
    save_dir: str = "exps/bc/metaworld/run"


def run(cfg: MainConfig, policy):
    dataset = MetaWorldDataset(cfg.dataset)
    cfg.task_name = dataset.task_name
    cfg.robot = dataset.robot[0]  # hack
    cfg.rl_image_size = dataset.env_params["rl_image_size"]

    pyrallis.dump(cfg, open(cfg.cfg_path, "w"))  # type: ignore
    print(common_utils.wrap_ruler("config"))
    with open(cfg.cfg_path, "r") as f:
        print(f.read(), end="")
    cfg_dict = yaml.safe_load(open(cfg.cfg_path, "r"))

    policy = BcPolicy(dataset.obs_shape, dataset.prop_shape, dataset.action_dim, cfg.policy)
    policy = policy.to("cuda")
    print(common_utils.wrap_ruler("policy weights"))
    print(policy)

    ema_policy = None
    if cfg.ema > 0:
        ema_policy = copy.deepcopy(policy)

    common_utils.count_parameters(policy)
    if cfg.weight_decay == 0:
        print("Using Adam optimzer")
        optim = torch.optim.Adam(policy.parameters(), cfg.lr)
    else:
        print("Using AdamW optimzer")
        optim = torch.optim.AdamW(policy.parameters(), cfg.lr, weight_decay=cfg.weight_decay)

    stat = common_utils.MultiCounter(
        cfg.save_dir,
        bool(cfg.use_wb),
        wb_exp_name=cfg.wb_exp,
        wb_run_name=cfg.wb_run,
        wb_group_name=cfg.wb_group,
        config=cfg_dict,
    )

    saver = common_utils.TopkSaver(cfg.save_dir, 3)
    stopwatch = common_utils.Stopwatch()
    best_score = 0
    for epoch in range(cfg.num_epoch):
        stopwatch.reset()

        for _ in range(cfg.epoch_len):
            with stopwatch.time("sample"):
                batch = dataset.sample(cfg.batchsize, "cuda:0")
            with stopwatch.time("train"):
                loss = policy.loss(batch)

                optim.zero_grad()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(  # type: ignore
                    policy.parameters(), max_norm=cfg.grad_clip
                )
                optim.step()
                stat["loss"].append(loss.item())
                stat["grad_norm"].append(grad_norm)
                if ema_policy is not None:
                    utils.soft_update_params(policy, ema_policy, cfg.ema)

        epoch_time = stopwatch.elapsed_time_since_reset
        with stopwatch.time("eval"):
            seed = epoch * 1991991991 % 9997
            scores = run_eval(dataset.env, policy, num_game=50, seed=seed, verbose=False)
            if ema_policy is not None:
                ema_scores = run_eval(
                    dataset.env, ema_policy, num_game=50, seed=seed, verbose=False
                )
                ema_score = float(np.mean(ema_scores))
                cur_score = float(np.mean(scores))
                stat["ema_score"].append(ema_score)
                stat["cur_score"].append(cur_score)
                if ema_score > cur_score:
                    saved = saver.save(ema_policy.state_dict(), ema_score)
                    score = ema_score
                else:
                    saved = saver.save(policy.state_dict(), cur_score)
                    score = cur_score
            else:
                score = float(np.mean(scores))
                saved = saver.save(policy.state_dict(), score)

        best_score = max(best_score, score)
        stat["score"].append(score)
        stat["score(best)"].append(best_score)
        stat["speed"].append(cfg.epoch_len / epoch_time)
        stat.summary(epoch)
        stopwatch.summary()
        if saved:
            print("model saved!")

    # final eval
    best_model = saver.get_best_model()
    policy.load_state_dict(torch.load(best_model))
    scores = run_eval(dataset.env, policy, num_game=50, seed=1, verbose=False)
    stat["final_score"].append(np.mean(scores))
    stat.summary(cfg.num_epoch)

    # quit!
    assert False


# function to load bc models
def load_model(weight_file, device):
    cfg_path = os.path.join(os.path.dirname(weight_file), f"cfg.yaml")
    print(common_utils.wrap_ruler("config of loaded agent"))
    with open(cfg_path, "r") as f:
        print(f.read(), end="")
    print(common_utils.wrap_ruler(""))

    cfg = pyrallis.load(MainConfig, open(cfg_path, "r"))  # type: ignore
    env_params = dict(
        env_name=cfg.task_name,
        robots=cfg.robot,
        episode_length=cfg.dataset.eval_episode_len,
        action_repeat=cfg.dataset.action_repeat,
        frame_stack=cfg.dataset.frame_stack,
        obs_stack=cfg.dataset.obs_stack,
        reward_shaping=False,
        rl_image_size=cfg.rl_image_size,
        camera_names=[cfg.dataset.rl_camera],
        rl_camera=cfg.dataset.rl_camera,
        device=device,
        use_state=cfg.dataset.use_state,
    )

    env = PixelMetaWorld(**env_params)  # type: ignore
    policy = BcPolicy(env.observation_shape, env.prop_shape, env.action_dim, cfg.policy)
    policy.load_state_dict(torch.load(weight_file))
    return policy.to(device), env, env_params


"""
Sample run:
python train_bc_metaworld.py --dataset.path Assembly
"""
if __name__ == "__main__":
    import rich.traceback

    rich.traceback.install()
    cfg = pyrallis.parse(config_class=MainConfig)  # type: ignore

    common_utils.set_all_seeds(cfg.seed)
    log_path = os.path.join(cfg.save_dir, "train.log")
    sys.stdout = common_utils.Logger(log_path, print_to_stdout=True)

    if cfg.load_model is not None and cfg.load_model != "none":
        policy = load_model(cfg.load_model, "cuda")[0]
    else:
        policy = None
    run(cfg, policy=policy)
