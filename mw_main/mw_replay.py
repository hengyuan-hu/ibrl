from collections import defaultdict
import torch
import h5py
import numpy as np
from common_utils import rela


class Batch:
    def __init__(self, obs, next_obs, action, reward, bootstrap):
        self.obs = obs
        self.next_obs = next_obs
        self.action = action
        self.reward = reward
        self.bootstrap = bootstrap

    @classmethod
    def merge_batches(cls, batch0, batch1):
        obs = {k: torch.cat([v, batch0.obs[k]], dim=0) for k, v in batch1.obs.items()}
        next_obs = {
            k: torch.cat([v, batch0.next_obs[k]], dim=0) for k, v in batch1.next_obs.items()
        }
        action = {k: torch.cat([v, batch0.action[k]], dim=0) for k, v in batch1.action.items()}
        reward = torch.cat([batch1.reward, batch0.reward], dim=0)
        bootstrap = torch.cat([batch1.bootstrap, batch0.bootstrap], dim=0)
        return cls(obs, next_obs, action, reward, bootstrap)


class ReplayBuffer:
    def __init__(
        self,
        nstep,
        gamma,
        frame_stack,
        max_episode_length,
        replay_size,
        use_bc,
        bc_max_len=-1,
        save_per_success=-1,
        save_dir=None,
        rl_camera=None,
    ):
        self.replay_size = replay_size

        self.episode = rela.Episode(nstep, max_episode_length, gamma)
        self.replay = rela.SingleStepTransitionReplay(
            frame_stack=frame_stack,
            n_step=nstep,
            capacity=replay_size,
            seed=1,
            prefetch=3,
            extra=0.1,
        )

        self.bc_replay = None
        if use_bc:
            self.bc_replay = rela.SingleStepTransitionReplay(
                frame_stack=frame_stack,
                n_step=nstep,
                capacity=replay_size,
                seed=1,
                prefetch=3,
                extra=0.1,
            )
        self.bc_max_len = bc_max_len
        self.save_per_success = save_per_success
        self.save_dir = save_dir
        self.rl_camera = rl_camera

        self.visual_reward = None
        self.episode_image_obs = defaultdict(list)
        self.num_success = 0
        self.num_episode = 0
        self.freeze_bc_replay = False

    def set_visual_reward(self, visual_reward):
        self.visual_reward = visual_reward

    def new_episode(self, obs: dict[str, torch.Tensor]):
        self.episode_image_obs = defaultdict(list)
        self.episode.init({})
        self.episode.push_obs(obs)

    def add(
        self,
        obs: dict[str, torch.Tensor],
        reply: dict[str, torch.Tensor],
        reward: float,
        terminal: bool,
        success: bool,
        image_obs: dict[str, torch.Tensor],
    ):
        self.episode.push_action(reply)
        self.episode.push_reward(reward)
        self.episode.push_terminal(float(terminal))

        assert self.visual_reward is None
        if self.visual_reward is not None:
            for key, val in image_obs.items():
                self.episode_image_obs[key].append(val)

        if not terminal:
            self.episode.push_obs(obs)
            return

        transition = self.episode.pop_transition()
        self.replay.add(transition)
        self.num_episode += 1

        if not success:
            return

        self.num_success += 1
        if self.bc_replay is None or self.freeze_bc_replay:
            return

        seq_len = transition.seq_len.item()
        if self.bc_max_len > 0 and seq_len > self.bc_max_len:
            print(f"episode too long {seq_len}, max={self.bc_max_len}, ignore")
            return
        print(f"episode ok {seq_len}, max={self.bc_max_len}, accept")
        self.bc_replay.add(transition)

    def sample(self, batchsize, device):
        return self.replay.sample(batchsize, device)

    def sample_bc(self, batchsize, device):
        assert self.bc_replay is not None
        assert self.num_success > 0
        return self.bc_replay.sample(batchsize, device)

    def sample_rl_bc(self, rl_bsize, bc_bsize, device):
        rl_batch = self.sample(rl_bsize, device)
        bc_batch = self.sample_bc(bc_bsize, device)

        batch = Batch.merge_batches(rl_batch, bc_batch)
        return batch

    def size(self, bc=False):
        if bc:
            assert self.bc_replay is not None
            return self.bc_replay.size()
        else:
            return self.replay.size()


def stack_obs(key, v0, past_obses, obs_stack):
    """past obses should NOT contain the current observation"""
    # TODO: reduce this duplication
    vals = [v0]
    for i in range(1, obs_stack):
        hist_len = len(past_obses[key])
        if hist_len - i >= 0:
            vals.append(past_obses[key][hist_len - i])
        else:
            vals.append(torch.zeros_like(vals[-1]))
    return torch.cat(vals, dim=0)


def add_demos_to_replay(
    replay: ReplayBuffer,
    data_path: str,
    num_data: int,
    rl_camera: str,
    use_state: int,
    obs_stack: int,
    reward_scale: float,
):
    assert not use_state
    assert obs_stack == 1
    f = h5py.File(data_path)
    num_episode: int = len(list(f["data"].keys()))  # type: ignore
    print(f"loading first {num_data} episodes from {data_path}")
    print(f"Raw Dataset size (#episode): {num_episode}")
    for episode_id in range(num_episode):
        if num_data > 0 and episode_id >= num_data:
            break

        episode_tag = f"demo_{episode_id}"
        episode = f[f"data/{episode_tag}"]
        actions: np.ndarray = np.array(episode["actions"])  # type: ignore
        images: np.ndarray = np.array(episode[f"obs/{rl_camera}_image"])  # type: ignore
        rewards: np.ndarray = np.array(f[f"data/{episode_tag}/rewards"])  # type: ignore
        terminals = rewards
        print("rewards:", rewards)

        episode_len = rewards.shape[0]
        for i in range(episode_len + 1):
            if i < episode_len:
                obs = {
                    "obs": torch.from_numpy(images[i]),
                    "prop": torch.zeros(4),
                }

            if i == 0:
                replay.new_episode(obs)
                continue

            action_idx = i - 1
            reply = {"action": torch.from_numpy(actions[action_idx]).float()}
            assert reward_scale == 1
            reward = float(rewards[action_idx]) * reward_scale
            # print("adding reward:", reward)
            terminal = bool(terminals[action_idx])
            success = bool(float(rewards[action_idx]) == 1)
            replay.add(
                obs,
                reply,
                reward,
                terminal,
                success,
                image_obs={},
            )
            if terminal:
                break

    print(f"Size of the replay buffer: {replay.size()}, # success: {replay.num_success}")
    if replay.bc_replay is not None:
        print(f"Size of the bc_replay buffer: {replay.bc_replay.size()}")


if __name__ == "__main__":
    replay = ReplayBuffer(
        nstep=3,
        gamma=0.99,
        frame_stack=1,
        max_episode_length=200,
        replay_size=100,
        use_bc=True,
        save_per_success=-1,
        save_dir=None,
    )
    rl_camera = "robot0_eye_in_hand"
    add_demos_to_replay(
        replay,
        "data/robomimic/can/processed_data96.hdf5",
        num_data=2,
        rl_camera=rl_camera,
        use_state=True,
        obs_stack=1,
        reward_scale=1,
    )
