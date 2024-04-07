import os
from collections import defaultdict
import json
import torch
import h5py
import numpy as np

from env.robosuite_wrapper import DEFAULT_STATE_KEYS, STATE_KEYS, PROP_KEYS
from common_utils import rela
from common_utils import ibrl_utils as utils


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
        self.freeze_bc_replay = False
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

        self.episode_image_obs = defaultdict(list)
        self.num_success = 0
        self.num_episode = 0

    def new_episode(self, obs: dict[str, torch.Tensor]):
        self.episode_image_obs = defaultdict(list)
        self.episode.init({})
        self.episode.push_obs(obs)

    def append_obs(self, obs: dict[str, torch.Tensor]):
        self.episode.push_obs(obs)

    def append_reply(self, reply: dict[str, torch.Tensor]):
        self.episode.push_action(reply)

    def append_reward_terminal(self, reward: float, terminal: bool, success: bool):
        self.episode.push_reward(reward)
        self.episode.push_terminal(float(terminal))

        if terminal:
            self._push_episode(success)

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

        if not terminal:
            self.episode.push_obs(obs)
            return

        self._push_episode(success)

    def _push_episode(self, success):
        transition = self.episode.pop_transition()
        self.replay.add(transition)
        self.num_episode += 1

        if not success:
            return
        self.num_success += 1

        if self.bc_replay is None or self.freeze_bc_replay:
            return
        seq_len = int(transition.seq_len.item())

        if self.bc_max_len > 0 and seq_len > self.bc_max_len:
            print(f"episode too long {seq_len}, max={self.bc_max_len}, ignore")
            return
        self.bc_replay.add(transition)

        # dump the bc dataset for training
        if self.save_per_success <= 0 or self.num_success % self.save_per_success != 0:
            return
        # store the most recent n trajectories
        print(f"Saving bc replay; @{self.num_success} games.")
        size = self.bc_replay.size()
        episodes = self.bc_replay.get_range(size - self.save_per_success, size, "cpu")

        assert self.save_dir is not None
        save_id = self.num_success // self.save_per_success
        filename = os.path.join(self.save_dir, f"data{save_id}.h5")
        self._save_replay(filename, episodes)

    def save_replay(self, filename):
        size = self.replay.size()
        episodes = self.replay.get_range(0, size, "cpu")
        self._save_replay(filename, episodes)

    def _save_replay(self, filename, episodes):
        print(f"writing replay buffer to {filename}")

        size = episodes.seq_len.size(0)
        with h5py.File(filename, "w") as hf:
            data_grp = hf.create_group("data")
            for i in range(size):
                ep_data_grp = data_grp.create_group(f"demo_{i}")
                episode_len = int(episodes.seq_len[i].item())
                # print(f"episode {i}: len: {episode_len}")
                for k, v in episodes.obs.items():
                    if k not in ["prop", "state", "sim_state"] + DEFAULT_STATE_KEYS:
                        k = f"{k}_image"
                    ep_data_grp.create_dataset(f"obs/{k}", data=v[:episode_len, i].numpy())
                action: torch.Tensor = episodes.action["action"][:episode_len, i]
                ep_data_grp.create_dataset(f"actions", data=action.numpy())
                ep_data_grp.create_dataset(f"rewards", data=episodes.reward[:episode_len, i].numpy())

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


def add_demos_to_replay(
    replay: ReplayBuffer,
    data_path: str,
    num_data: int,
    rl_cameras: list[str],
    use_state: int,
    obs_stack: int,
    state_stack: int,
    prop_stack: int,
    reward_scale: float,
    record_sim_state: bool,
    is_demo: bool = True,
):
    f = h5py.File(data_path)
    num_episode: int = len(list(f["data"].keys()))  # type: ignore
    print(f"loading first {num_data} episodes from {data_path}")
    print(f"Raw Dataset size (#episode): {num_episode}")

    all_actions = []
    for episode_id in range(num_episode):
        if num_data > 0 and episode_id >= num_data:
            break

        episode_tag = f"demo_{episode_id}"
        episode = f[f"data/{episode_tag}"]
        actions = np.array(episode["actions"]).astype(np.float32)  # type: ignore
        images = {
            rl_camera: np.array(episode[f"obs/{rl_camera}_image"]) for rl_camera in rl_cameras
        }
        all_actions.append(actions)

        robot_locs = []
        if "prop" in episode["obs"]:
            props = episode["obs"]["prop"]
        else:
            for key in PROP_KEYS:
                robot_locs.append(episode["obs"][key])  # type: ignore
            props = np.concatenate(robot_locs, axis=1).astype(np.float32)

        if use_state:
            all_states = []
            cfg_path = os.path.join(os.path.dirname(data_path), "env_cfg.json")
            env_cfg = json.load(open(cfg_path, "r"))
            task_name = env_cfg["env_name"]
            for key in STATE_KEYS[task_name]:
                state_: np.ndarray = episode["obs"][key]  # type: ignore
                all_states.append(state_)
            states = np.concatenate(all_states, axis=1)
            assert states.shape[0] == actions.shape[0]

        rewards = np.array(f[f"data/{episode_tag}/rewards"])  # type: ignore
        if is_demo:
            assert rewards[-1] == 1
            terminals = rewards
        else:
            terminals = rewards[:]
            terminals[-1] = 1

        episode_len = rewards.shape[0]
        print(f"episode {episode_id} length: {episode_len}")
        past_obses = defaultdict(list)
        for i in range(episode_len + 1):
            if i < episode_len:
                assert obs_stack == 1, "does not support obs stack yet"
                obs = {
                    rl_camera: torch.from_numpy(images[rl_camera][i]) for rl_camera in rl_cameras
                }

                prop = torch.from_numpy(props[i])
                past_obses["prop"].append(prop)
                assert len(past_obses["prop"]) == i + 1, f"{len(past_obses['prop'])} vs {i + 1}"
                obs["prop"] = utils.concat_obs(i, past_obses["prop"], prop_stack)

                if use_state:
                    state = torch.from_numpy(states[i]).float()
                    past_obses["state"].append(state)
                    assert len(past_obses["state"]) == i + 1
                    obs["state"] = utils.concat_obs(i, past_obses["state"], state_stack)

                if record_sim_state:
                    sim_state = torch.from_numpy(np.array(episode["states"][i]))
                    obs["sim_state"] = sim_state
                    for key in DEFAULT_STATE_KEYS:
                        s = np.array(episode["obs"][key][i])
                        obs[key] = torch.from_numpy(s)

            if i == 0:
                replay.new_episode(obs)
                continue

            action_idx = i - 1
            reply = {"action": torch.from_numpy(actions[action_idx])}
            reward = float(rewards[action_idx]) * reward_scale
            success = bool(rewards[action_idx] == 1)
            terminal = bool(terminals[action_idx])

            replay.add(obs, reply, reward, terminal, success, image_obs={})

            if success:
                assert terminal
            if terminal:
                break

    print(f"Size of the replay buffer: {replay.size()}, # success: {replay.num_success}")
    if replay.bc_replay is not None:
        print(f"Size of the bc_replay buffer: {replay.bc_replay.size()}")

    all_actions = np.concatenate(all_actions, axis=0)
    print("demo actions shape:", all_actions.shape)
    actions_norm = np.linalg.norm(all_actions[:, :6], 2, axis=1)
    print(f"demo action norm: mean: {np.mean(actions_norm):.4f}, max: {np.max(actions_norm):.4f}")
    return


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
        rl_cameras=[rl_camera],
        use_state=True,
        obs_stack=1,
        state_stack=1,
        prop_stack=1,
        reward_scale=1,
        record_sim_state=False,
    )
