"""
Script to extract observations from low-dimensional simulation states in a robosuite dataset.

Args:
    dataset (str): path to input hdf5 dataset

    output_name (str): name of output hdf5 dataset

    n (int): if provided, stop after n trajectories are processed

    shaped (bool): if flag is set, use dense rewards

    camera_names (str or [str]): camera name(s) to use for image observations.
        Leave out to not use image observations.

    image_size (int): size of image for video.

    obs_size (int): size of image observation

    done_mode (int): how to write done signal. If 0, done is 1 whenever s' is a success state.
        If 1, done is 1 at the end of each trajectory. If 2, both.

    copy_rewards (bool): if provided, copy rewards from source file instead of inferring them

    copy_dones (bool): if provided, copy dones from source file instead of inferring them

Example usage:

    # extract low-dimensional observations
    python dataset_states_to_obs.py --dataset /path/to/demo.hdf5 --output_name low_dim.hdf5 --done_mode 2

    # extract 84x84 image observations
    python dataset_states_to_obs.py --dataset /path/to/demo.hdf5 --output_name image.hdf5 \
        --done_mode 2 --camera_names agentview robot0_eye_in_hand --camera_height 84 --camera_width 84

    # (space saving option) extract 84x84 image observations with compression and without
    # extracting next obs (not needed for pure imitation learning algos)
    python dataset_states_to_obs.py --dataset /path/to/demo.hdf5 --output_name image.hdf5 \
        --done_mode 2 --camera_names agentview robot0_eye_in_hand --camera_height 84 --camera_width 84 \
        --compress --exclude-next-obs

    # use dense rewards, and only annotate the end of trajectories with done signal
    python dataset_states_to_obs.py --dataset /path/to/demo.hdf5 --output_name image_dense_done_1.hdf5 \
        --done_mode 1 --dense --camera_names agentview robot0_eye_in_hand --camera_height 84 --camera_width 84
"""
import os
import json
import h5py
import argparse
import numpy as np
from copy import deepcopy
import imageio
import torch
from common_utils.py.ibrl_utils import get_rescale_transform

import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
from robomimic.envs.env_base import EnvBase


def extract_trajectory(
    env,
    initial_state,
    states,
    actions,
    done_mode,
    cameras,
):
    """
    Helper function to extract observations, rewards, and dones along a trajectory using
    the simulator environment.

    Args:
        env (instance of EnvBase): environment
        initial_state (dict): initial simulation state to load
        states (np.array): array of simulation states to load to extract information
        actions (np.array): array of actions
        done_mode (int): how to write done signal. If 0, done is 1 whenever s' is a
            success state. If 1, done is 1 at the end of each trajectory.
            If 2, do both.
    """
    assert isinstance(env, EnvBase)
    assert states.shape[0] == actions.shape[0]

    # load the initial state
    env.reset()
    obs = env.reset_to(initial_state)

    traj = dict(
        obs=[],
        # next_obs=[],
        rewards=[],
        dones=[],
        actions=np.array(actions),
        states=np.array(states),
        initial_state_dict=initial_state,
    )
    traj_len = states.shape[0]
    # iteration variable @t is over "next obs" indices
    for t in range(1, traj_len + 1):
        # get next observation
        if t == traj_len:
            # play final action to get next observation for last timestep
            next_obs, _, _, _ = env.step(actions[t - 1])
        else:
            # reset to simulator state to get observation
            next_obs = env.reset_to({"states": states[t]})

        # infer reward signal
        # note: our tasks use reward r(s'), reward AFTER transition, so this is
        #       the reward for the current timestep
        r = env.get_reward()

        # infer done signal
        done = False
        if (done_mode == 1) or (done_mode == 2):
            # done = 1 at end of trajectory
            done = done or (t == traj_len)
        if (done_mode == 0) or (done_mode == 2):
            # done = 1 when s' is task success state
            done = done or env.is_success()["task"]
        done = int(done)

        # collect transition
        traj["obs"].append(obs)
        traj["rewards"].append(r)
        traj["dones"].append(done)

        # update for next iter
        obs = deepcopy(next_obs)

    # convert list of dict to dict of list for obs dictionaries (for convenient writes to hdf5 dataset)
    traj["obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["obs"])

    # for k, v in traj["obs"].items():
    #     print(k, len(v), type(v[0]), v[0].shape)

    video = []
    tensors = {camera: [] for camera in cameras}
    for i in range(len(traj["rewards"])):
        frame = []
        for camera in cameras:
            image = traj["obs"][f"{camera}_image"][i]
            frame.append(image)
            tensors[camera].append(torch.from_numpy(image.copy()).permute([2, 0, 1]))
        frame = np.concatenate(frame, axis=1)
        video.append(frame)

    # list to numpy array
    for k in traj:
        if k == "initial_state_dict":
            continue
        if isinstance(traj[k], dict):
            for kp in traj[k]:
                traj[k][kp] = np.array(traj[k][kp])
        else:
            traj[k] = np.array(traj[k])

    return traj, video, tensors


def dataset_states_to_obs(args):
    # create environment to use for data processing
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=args.dataset)
    env = EnvUtils.create_env_for_data_processing(
        env_meta=env_meta,
        camera_names=args.camera_names,
        camera_height=args.image_size,
        camera_width=args.image_size,
        reward_shaping=args.shaped,
    )

    print("==== Using environment with the following metadata ====")
    print(json.dumps(env.serialize(), indent=4))
    print("")
    os.makedirs(args.output_folder, exist_ok=True)
    config_path = os.path.join(args.output_folder, "env_cfg.json")
    json.dump(env.serialize(), open(config_path, "w"), indent=4)

    # some operations for playback are robosuite-specific, so determine if this environment is a robosuite env
    is_robosuite_env = EnvUtils.is_robosuite_env(env_meta)

    # list of all demonstration episodes (sorted in increasing number order)
    f = h5py.File(args.dataset, "r")
    demos = list(f["data"].keys())
    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]

    # maybe reduce the number of demonstrations to playback
    if args.n is not None:
        demos = demos[: args.n]

    os.makedirs(args.output_folder, exist_ok=True)

    if args.save_hdf5:
        output_path = os.path.join(args.output_folder, f"processed_data{args.obs_size}.hdf5")
    else:
        output_path = os.path.join(args.output_folder, f"dummy_processed_data.hdf5")

    f_out = h5py.File(output_path, "w")
    data_grp = f_out.create_group("data")
    data_grp.attrs["env_args"] = json.dumps(env.serialize())

    print("input file: {}".format(args.dataset))
    print("output file: {}".format(output_path))

    total_samples = 0
    for ind in range(len(demos)):
        ep = demos[ind]

        # prepare initial state to reload from
        states = f["data/{}/states".format(ep)][()]
        initial_state = dict(states=states[0])
        if is_robosuite_env:
            initial_state["model"] = f["data/{}".format(ep)].attrs["model_file"]

        # extract obs, rewards, dones
        actions = f["data/{}/actions".format(ep)][()]
        traj, video, obs_tensors = extract_trajectory(
            env=env,
            initial_state=initial_state,
            states=states,
            actions=actions,
            done_mode=args.done_mode,
            cameras=args.camera_names,
        )

        if ind < args.video:
            video_path = os.path.join(args.output_folder, f"video{ind}.mp4")
            print(f"saving video to {video_path}")
            imageio.mimsave(video_path, video, fps=20)

        if ind < args.tensor:
            for k, v in obs_tensors.items():
                tensor_path = os.path.join(args.output_folder, f"episode{ind}-{k}.pt")
                print(f"saving observations to {tensor_path}")
                torch.save(v, tensor_path)

        ref_rewards = f["data/{}/rewards".format(ep)][()]
        assert len(traj["rewards"]) == len(ref_rewards)
        for reward, ref_reward in zip(traj["rewards"], ref_rewards):
            if abs(reward - ref_reward) >= 1e-5:
                print("rew:", traj["rewards"])
                print("ref:", ref_rewards)
                assert False

        # maybe copy reward or done signal from source file
        if args.copy_rewards:
            # print("copy reward")
            traj["rewards"] = f["data/{}/rewards".format(ep)][()]
            assert False, "should not copy reward, for safety"

        if args.copy_dones:
            traj["dones"] = f["data/{}/dones".format(ep)][()]

        if args.image_size != args.obs_size:
            image_transform = get_rescale_transform(args.obs_size)
        else:
            image_transform = lambda x: x
        # store transitions
        # IMPORTANT: keep name of group the same as source file, to make sure that filter keys are
        #            consistent as well
        ep_data_grp = data_grp.create_group(ep)
        ep_data_grp.create_dataset("actions", data=np.array(traj["actions"]))
        ep_data_grp.create_dataset("states", data=np.array(traj["states"]))
        ep_data_grp.create_dataset("rewards", data=np.array(traj["rewards"]))
        ep_data_grp.create_dataset("dones", data=np.array(traj["dones"]))
        ref_reward = f["data/{}/rewards".format(ep)][()]
        reward_match = (traj["rewards"] == ref_reward).all()

        for k in traj["obs"]:
            obs = traj["obs"][k]
            for camera_name in args.camera_names:
                if k.startswith(camera_name):
                    obs = torch.from_numpy(obs)
                    obs = image_transform(obs.permute(0, 3, 1, 2))

            ep_data_grp.create_dataset("obs/{}".format(k), data=np.array(obs))

        for k in f[f"data/{ep}/obs"].keys():
            if k in ep_data_grp["obs"]:
                # print(f"skipping {k}")
                continue
            data = f[f"data/{ep}/obs/{k}"]
            # print(f"copying {k} to the new dataset")
            # print(np.array(data).shape)
            ep_data_grp.create_dataset(f"obs/{k}", data=np.array(data))

        # episode metadata
        if is_robosuite_env:
            # model xml for this episode
            ep_data_grp.attrs["model_file"] = traj["initial_state_dict"]["model"]
        # number of transitions in this episode
        ep_data_grp.attrs["num_samples"] = traj["actions"].shape[0]
        total_samples += traj["actions"].shape[0]
        print(f"ep {ind}: wrote {ep_data_grp.attrs['num_samples']} transitions to group {ep}")
        print("rewards sum:", traj["rewards"].sum(), "reward match?", reward_match)
        if not reward_match:
            print("WARNING: reward does not match")

    # copy over all filter keys that exist in the original hdf5
    if "mask" in f:
        f.copy("mask", f_out)

    # global metadata
    data_grp.attrs["total"] = total_samples
    data_grp.attrs["env_args"] = json.dumps(env.serialize(), indent=4)  # environment info
    print("Wrote {} trajectories to {}".format(len(demos), output_path))

    f.close()
    f_out.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    # name of hdf5 to write - it will be in the same directory as @dataset
    parser.add_argument("--output_folder", type=str, required=True)

    # specify number of demos to process
    # - useful for debugging conversion with a handful of trajectories
    parser.add_argument("--n", type=int, default=None)

    # flag for reward shaping
    parser.add_argument("--shaped", action="store_true", help="(optional) use shaped rewards")
    parser.add_argument("--video", type=int, default=0, help="render the first {video} episode")
    parser.add_argument("--tensor", type=int, default=0, help="dump the first {video} episode")
    parser.add_argument("--save_hdf5", type=int, default=1)

    # camera names to use for observations
    parser.add_argument(
        "--camera_names",
        type=str,
        nargs="+",
        default=[],
        help="(optional) camera name(s) to use for image observations. Leave out to use states",
    )

    parser.add_argument(
        "--image_size",
        type=int,
        default=224,
        help="(optional) height of image observations",
    )
    parser.add_argument(
        "--obs_size",
        type=int,
        default=96,
        help="(optional) height of image observations",
    )

    # specifies how the "done" signal is written. If "0", then the "done" signal is 1 wherever
    # the transition (s, a, s') has s' in a task completion state. If "1", the "done" signal
    # is one at the end of every trajectory. If "2", the "done" signal is 1 at task completion
    # states for successful trajectories and 1 at the end of all trajectories.
    parser.add_argument(
        "--done_mode",
        type=int,
        default=0,
        help="how to write done signal. If 0, done is 1 whenever s' is a success state.\
            If 1, done is 1 at the end of each trajectory. If 2, both.",
    )

    # flag for copying rewards from source file instead of re-writing them
    parser.add_argument(
        "--copy_rewards",
        action="store_true",
        help="(optional) copy rewards from source file instead of inferring them",
    )

    # flag for copying dones from source file instead of re-writing them
    parser.add_argument(
        "--copy_dones",
        action="store_true",
        help="(optional) copy dones from source file instead of inferring them",
    )

    args = parser.parse_args()
    dataset_states_to_obs(args)
    print("===================Success===================")
