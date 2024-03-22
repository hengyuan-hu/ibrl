import pickle
import torch
import numpy as np

from env.metaworld_wrapper import PixelMetaWorld
from common_utils import ibrl_utils as utils
from common_utils import Recorder


def run_eval(
    env: PixelMetaWorld,
    agent,
    num_game,
    seed,
    record_dir=None,
    verbose=True,
    eval_mode=True,
):
    recorder = None if record_dir is None else Recorder(record_dir)

    scores = []
    with torch.no_grad(), utils.eval_mode(agent):
        for episode_idx in range(num_game):
            step = 0
            rewards = []
            np.random.seed(seed + episode_idx)
            obs, image_obs = env.reset()

            terminal = False
            while not terminal:
                if recorder is not None:
                    recorder.add(image_obs)

                action = agent.act(obs, eval_mode=eval_mode).numpy()
                obs, reward, terminal, _, image_obs = env.step(action)
                rewards.append(reward)
                step += 1

            if verbose:
                print(
                    f"seed: {seed + episode_idx}, "
                    f"reward: {np.sum(rewards)}, len: {env.time_step}"
                )

            scores.append(np.sum(rewards))

            if recorder is not None:
                save_path = recorder.save(f"episode{episode_idx}")
                reward_path = f"{save_path}.reward.pkl"
                print(f"saving reward to {reward_path}")
                pickle.dump(rewards, open(reward_path, "wb"))

    if verbose:
        print(f"num game: {len(scores)}, seed: {seed}, score: {np.mean(scores)}")

    return scores
