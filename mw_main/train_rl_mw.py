import os
import sys
from typing import Any, Optional
from dataclasses import dataclass, field
import yaml
import copy
import pprint
import pyrallis
import torch
import numpy as np

import common_utils
from common_utils import ibrl_utils as utils
from rl.q_agent import QAgent, QAgentConfig
from env.metaworld_wrapper import PixelMetaWorld
import mw_replay
import train_bc_mw
from eval_mw import run_eval


BC_POLICIES = {
    "assembly": "release/model/metaworld/pathAssembly_num_data3_num_epoch2_seed1/model1.pt",
    "boxclose": "release/model/metaworld/pathBoxClose_num_data3_num_epoch2_seed1/model1.pt",
    "coffeepush": "release/model/metaworld/pathCoffeePush_num_data3_num_epoch2_seed1/model1.pt",
    "stickpull": "release/model/metaworld/pathStickPull_num_data3_num_epoch2_seed1/model1.pt",
}

BC_DATASETS = {
    "assembly": "release/data/metaworld/Assembly_frame_stack_1_96x96_end_on_success/dataset.hdf5",
    "boxclose": "release/data/metaworld/BoxClose_frame_stack_1_96x96_end_on_success/dataset.hdf5",
    "coffeepush": "release/data/metaworld/CoffeePush_frame_stack_1_96x96_end_on_success/dataset.hdf5",
    "stickpull": "release/data/metaworld/StickPull_frame_stack_1_96x96_end_on_success/dataset.hdf5",
}


@dataclass
class MainConfig(common_utils.RunConfig):
    seed: int = 1
    # env
    episode_length: int = 200
    # agent
    q_agent: QAgentConfig = field(default_factory=lambda: QAgentConfig())
    stddev_max: float = 1.0
    stddev_min: float = 0.1
    stddev_step: int = 500000
    nstep: int = 3
    discount: float = 0.99
    replay_buffer_size: int = 500
    batch_size: int = 256
    num_critic_update: int = 1
    update_freq: int = 2
    bc_policy: str = ""
    use_bc: int = 1
    # load demo
    mix_rl_rate: float = 1  # 1: only use rl, <1, mix in some bc data
    preload_num_data: int = 0
    preload_datapath: str = ""
    env_reward_scale: int = 1
    # others
    num_train_step: int = 200000
    log_per_step: int = 5000
    num_warm_up_episode: int = 50
    num_eval_episode: int = 10
    # rft
    pretrain_num_epoch: int = 0
    pretrain_epoch_len: int = 10000
    add_bc_loss: int = 0
    # log
    use_wb: int = 0
    save_dir: str = "exps/rl/metaworld/run1"

    def __post_init__(self):
        self.preload_datapath = self.bc_policy
        if self.preload_datapath in BC_DATASETS:
            self.preload_datapath = BC_DATASETS[self.preload_datapath]

    @property
    def stddev_schedule(self):
        return f"linear({self.stddev_max},{self.stddev_min},{self.stddev_step})"


class Workspace:
    def __init__(self, cfg: MainConfig):
        self.work_dir = cfg.save_dir
        print(f"workspace: {self.work_dir}")

        common_utils.set_all_seeds(cfg.seed)
        sys.stdout = common_utils.Logger(cfg.log_path, print_to_stdout=True)

        pyrallis.dump(cfg, open(cfg.cfg_path, "w"))  # type: ignore
        print(common_utils.wrap_ruler("config"))
        with open(cfg.cfg_path, "r") as f:
            print(f.read(), end="")
        print(common_utils.wrap_ruler(""))

        self.cfg = cfg
        self.cfg_dict = yaml.safe_load(open(cfg.cfg_path, "r"))

        # we need bc policy to construct the environment :(, hack!
        assert cfg.bc_policy != "", "bc policy must be set to find the correct env config"
        self.env_params: dict[str, Any]
        if cfg.bc_policy in BC_POLICIES:
            cfg.bc_policy = BC_POLICIES[cfg.bc_policy]
        bc_policy, _, self.env_params = train_bc_mw.load_model(cfg.bc_policy, "cuda")

        self.bc_policy = None
        if cfg.use_bc:
            self.bc_policy = bc_policy

        self.global_step = 0
        self.global_episode = 0
        self.train_step = 0
        self.num_success = 0
        self._setup_env()

        assert not cfg.q_agent.use_prop, "not implemented"
        self.agent = QAgent(
            False,
            self.train_env.observation_shape,
            (4,),  # prop shape, does not matter as we do not use prop in metaworld
            self.train_env.num_action,
            rl_camera="obs",
            cfg=cfg.q_agent,
        )
        if self.bc_policy is not None:
            self.agent.add_bc_policy(bc_policy=copy.deepcopy(self.bc_policy))

        self._setup_replay()
        self.ref_agent: Optional[QAgent] = None

    def _setup_env(self):
        # camera_names = [self.cfg.rl_camera]
        print(common_utils.wrap_ruler("Env Config"))
        pprint.pprint(self.env_params)
        print(common_utils.wrap_ruler(""))

        if "end_on_success" not in self.env_params:
            self.env_params["end_on_success"] = True
        self.env_params["episode_length"] = self.cfg.episode_length
        self.env_params["env_reward_scale"] = self.cfg.env_reward_scale
        self.train_env = PixelMetaWorld(**self.env_params)  # type: ignore

        eval_env_params = self.env_params.copy()
        eval_env_params["env_reward_scale"] = 1.0
        self.eval_env = PixelMetaWorld(**eval_env_params)  # type: ignore

    def _setup_replay(self):
        use_bc = (self.cfg.mix_rl_rate < 1) or self.cfg.add_bc_loss
        assert self.env_params["frame_stack"] == 1
        self.replay = mw_replay.ReplayBuffer(
            self.cfg.nstep,
            self.cfg.discount,
            frame_stack=1,
            max_episode_length=self.cfg.episode_length,  # env_params["episode_length"],
            replay_size=self.cfg.replay_buffer_size,
            use_bc=use_bc,
        )
        if self.cfg.preload_num_data:
            mw_replay.add_demos_to_replay(
                self.replay,
                self.cfg.preload_datapath,
                num_data=self.cfg.preload_num_data,
                rl_camera=self.env_params["rl_camera"],
                use_state=self.env_params["use_state"],
                obs_stack=self.env_params["obs_stack"],
                reward_scale=self.cfg.env_reward_scale,
            )
            self.replay.freeze_bc_replay = True

    def eval(self, seed, policy):
        random_state = np.random.get_state()
        scores = run_eval(
            env=self.eval_env,
            agent=policy,
            num_game=self.cfg.num_eval_episode,
            seed=seed,
            record_dir=None,
            verbose=False,
        )
        np.random.set_state(random_state)
        return float(np.mean(scores))

    def warm_up(self):
        # warm up stage, fill the replay with some episodes
        # it can either be human demos, or generated by the bc, or purely random
        obs, _ = self.train_env.reset()
        for k, v in obs.items():
            print(k, v.size())
        self.replay.new_episode(obs)
        total_reward = 0
        num_episode = 0
        while True:
            if self.bc_policy is not None:
                with torch.no_grad(), utils.eval_mode(self.bc_policy):
                    action = self.bc_policy.act(obs, eval_mode=True)
            else:
                if self.cfg.pretrain_num_epoch > 0:
                    # the policy has been pretrained
                    with torch.no_grad(), utils.eval_mode(self.agent):
                        action = self.agent.act(obs, eval_mode=True)
                else:
                    action = torch.zeros(self.train_env.action_dim)
                    action = action.uniform_(-1.0, 1.0)

            obs, reward, terminal, success, image_obs = self.train_env.step(action.numpy())
            reply = {"action": action}
            self.replay.add(obs, reply, reward, terminal, success, image_obs)

            if terminal:
                num_episode += 1
                total_reward += self.train_env.episode_reward
                if self.replay.size() < self.cfg.num_warm_up_episode:
                    self.replay.new_episode(obs)
                    obs, _ = self.train_env.reset()
                else:
                    break

        print(f"Warm up done. #episode: {self.replay.size()}")
        print(f"#episode from warmup: {num_episode}, #reward: {total_reward}")

    def train(self):
        stat = common_utils.MultiCounter(
            self.work_dir,
            bool(self.cfg.use_wb),
            wb_exp_name=self.cfg.wb_exp,
            wb_run_name=self.cfg.wb_run,
            wb_group_name=self.cfg.wb_group,
            config=self.cfg_dict,
        )
        self.agent.set_stats(stat)
        saver = common_utils.TopkSaver(save_dir=self.work_dir, topk=1)

        self.warm_up()
        self.num_success = self.replay.num_success
        stopwatch = common_utils.Stopwatch()

        obs, _ = self.train_env.reset()
        self.replay.new_episode(obs)
        while self.global_step < self.cfg.num_train_step:
            ### act ###
            with stopwatch.time("act"), torch.no_grad(), utils.eval_mode(self.agent):
                stddev = utils.schedule(self.cfg.stddev_schedule, self.global_step)
                action = self.agent.act(obs, stddev=stddev, eval_mode=False)
                stat["data/stddev"].append(stddev)

            ### env.step ###
            with stopwatch.time("env step"):
                obs, reward, terminal, success, image_obs = self.train_env.step(action.numpy())

            with stopwatch.time("add"):
                assert isinstance(terminal, bool)
                reply = {"action": action}
                self.replay.add(obs, reply, reward, terminal, success, image_obs)
                self.global_step += 1

            if terminal:
                with stopwatch.time("reset"):
                    self.global_episode += 1
                    stat["score/train_score"].append(success)
                    stat["data/episode_len"].append(self.train_env.time_step)
                    if self.replay.bc_replay is not None:
                        stat["data/bc_replay"].append(self.replay.size(bc=True))

                    # reset env
                    obs, _ = self.train_env.reset()
                    self.replay.new_episode(obs)

            ### logging ###
            if self.global_step % self.cfg.log_per_step == 0:
                self.log_and_save(stopwatch, stat, saver)

            ### train ###
            if self.global_step % self.cfg.update_freq == 0:
                with stopwatch.time("train"):
                    self.rl_train(stat)
                    self.train_step += 1

    def log_and_save(
        self,
        stopwatch: common_utils.Stopwatch,
        stat: common_utils.MultiCounter,
        saver: common_utils.TopkSaver,
    ):
        elapsed_time = stopwatch.elapsed_time_since_reset
        stat["other/speed"].append(self.cfg.log_per_step / elapsed_time)
        stat["other/elapsed_time"].append(elapsed_time)
        stat["other/episode"].append(self.global_episode)
        stat["other/step"].append(self.global_step)
        stat["other/train_step"].append(self.train_step)
        stat["other/replay"].append(self.replay.size())
        stat["score/num_success"].append(self.replay.num_success)

        with stopwatch.time("eval"):
            eval_score = self.eval(seed=self.global_step, policy=self.agent)
            stat["score/score"].append(eval_score)

        saved = saver.save(self.agent.state_dict(), eval_score, save_latest=True)
        print(f"saved?: {saved}")

        stat.summary(self.global_step, reset=True)
        stopwatch.summary(reset=True)
        print("total time:", common_utils.sec2str(stopwatch.total_time))
        print(common_utils.get_mem_usage())

    def rl_train(self, stat: common_utils.MultiCounter):
        stddev = utils.schedule(self.cfg.stddev_schedule, self.global_step)
        for i in range(self.cfg.num_critic_update):
            if self.cfg.mix_rl_rate < 1:
                rl_bsize = int(self.cfg.batch_size * self.cfg.mix_rl_rate)
                bc_bsize = self.cfg.batch_size - rl_bsize
                batch = self.replay.sample_rl_bc(rl_bsize, bc_bsize, "cuda:0")
            else:
                batch = self.replay.sample(self.cfg.batch_size, "cuda:0")

            update_actor = i == self.cfg.num_critic_update - 1

            if update_actor and self.cfg.add_bc_loss:
                bc_batch = self.replay.sample_bc(self.cfg.batch_size, "cuda:0")
            else:
                bc_batch = None

            if self.cfg.add_bc_loss:
                metrics = self.agent.update(batch, stddev, update_actor, bc_batch, self.ref_agent)
            else:
                metrics = self.agent.update(batch, stddev, update_actor)

            stat.append(metrics)
            stat["data/discount"].append(batch.bootstrap.mean().item())

    def pretrain_policy(self):
        stat = common_utils.MultiCounter(
            self.work_dir,
            bool(self.cfg.use_wb),
            wb_exp_name=self.cfg.wb_exp,
            wb_run_name=self.cfg.wb_run,
            wb_group_name=self.cfg.wb_group,
            config=self.cfg_dict,
        )
        saver = common_utils.TopkSaver(save_dir=self.work_dir, topk=1)

        for epoch in range(self.cfg.pretrain_num_epoch):
            for _ in range(self.cfg.pretrain_epoch_len):
                batch = self.replay.sample_bc(self.cfg.batch_size, "cuda")
                metrics = self.agent.pretrain_actor_with_bc(batch)
                stat.append(metrics)

            eval_seed = epoch * self.cfg.pretrain_epoch_len
            score = self.eval(eval_seed, policy=self.agent)
            stat["pretrain/score"].append(score)
            saved = saver.save(self.agent.state_dict(), score, save_latest=True)

            stat.summary(epoch, reset=True)
            print(f"saved?: {saved}")
            print(common_utils.get_mem_usage())


def main(cfg: MainConfig):
    workspace = Workspace(cfg)

    if cfg.pretrain_num_epoch > 0:
        print("pretraining:")
        workspace.pretrain_policy()
        workspace.ref_agent = copy.deepcopy(workspace.agent)

    workspace.train()

    if cfg.use_wb:
        wandb.finish()

    assert False


if __name__ == "__main__":
    import wandb
    from rich.traceback import install

    os.environ["MUJOCO_GL"] = "egl"

    install()
    torch.backends.cudnn.allow_tf32 = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore

    cfg = pyrallis.parse(config_class=MainConfig)  # type: ignore
    main(cfg)
