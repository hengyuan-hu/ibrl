from dataclasses import dataclass, field
import dataclasses
import pyrallis
from typing import List
import re
import os
import json
import numpy as np
import torch
import random
import h5py
from PIL import Image
from copy import copy
from env.metaworld_wrapper import PixelMetaWorld


"""
Sample run for generating a dataset:
python generate_metaworld_dataset.py \
    --num_episodes 5 \
    --save_gifs 5 \
    --output_path data/metaworld/Assembly_frame_stack_1_96x96_end_on_success \
    --env_cfg.env_name Assembly \
    --env_cfg.frame_stack 1 \
    --env_cfg.rl_image_size 96 \
    --env_cfg.end_on_success true

Sample run for generating a dataset and also saving it in MoDem format:
python generate_metaworld_dataset.py \
    --num_episodes 5 \
    --save_gifs 5 \
    --output_path data/metaworld/Assembly_frame_stack_1_224x224_modem \
    --env_cfg.env_name Assembly \
    --env_cfg.frame_stack 1 \
    --env_cfg.rl_image_size 224 \
    --env_cfg.end_on_success false \
    --add_modem_format true
"""


def run(cfg):
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    # Make the directory if it doesn"t exist
    os.makedirs(cfg.output_path, exist_ok=True)

    # Make an `env_cfg.json` file that looks like the ones made by robomimic
    env_kwargs = dataclasses.asdict(cfg.env_cfg)
    env_cfg_json_dict = dict()
    env_cfg_json_dict["env_name"] = env_kwargs["env_name"]
    env_cfg_json_dict["env_kwargs"] = env_kwargs
    with open(os.path.join(cfg.output_path, "env_cfg.json"), "w") as f:
        json.dump(env_cfg_json_dict, f)

    # Make an instance of the environment
    env = PixelMetaWorld(**env_kwargs)
    output_file = os.path.join(cfg.output_path, "dataset.hdf5")
    f = h5py.File(output_file, "w")

    list_ep_dict = []
    list_ep_dict_np = []
    for ep in range(cfg.num_episodes):
        ep_dict = dict()
        print(f"Generating episode {ep + 1} / {cfg.num_episodes}...")
        rl_obs, _ = env.reset()

        image_key = f"{cfg.env_cfg.rl_camera}_image"
        ep_dict[f"obs/{image_key}"] = [rl_obs["obs"].cpu().numpy()]
        ep_dict["obs/prop"] = [rl_obs["prop"].cpu().numpy()]
        ep_dict["obs/state"] = [rl_obs["state"].cpu().numpy()]
        ep_dict["states"] = [rl_obs["state"].cpu().numpy()]
        ep_dict["actions"] = []
        ep_dict["rewards"] = []
        ep_dict["dones"] = []
        ep_dict["infos"] = []

        for _ in range(cfg.env_cfg.episode_length):
            heuristic_action = env.get_heuristic_action(clip_action=True)
            rl_obs, reward, terminal, _, _ = env.step(heuristic_action)

            ep_dict[f"obs/{image_key}"].append(rl_obs["obs"].cpu().numpy())
            ep_dict["obs/prop"].append(rl_obs["prop"].cpu().numpy())
            ep_dict["obs/state"].append(rl_obs["state"].cpu().numpy())
            ep_dict["states"].append(rl_obs["state"].cpu().numpy())

            ep_dict["actions"].append(np.array(heuristic_action))
            ep_dict["rewards"].append(reward)
            ep_dict["dones"].append(terminal)

            ep_dict["infos"].append(copy(env.most_recent_info))

            if terminal:
                break

        ep_dict_np = dict()
        for key in ep_dict.keys():
            if "image" in key:
                ep_dict_np[key] = np.array(ep_dict[key], dtype=np.uint8)
            elif "dones" in key:
                ep_dict_np[key] = np.array(ep_dict[key], dtype=np.int64)
            elif key != "infos":
                ep_dict_np[key] = np.array(ep_dict[key], dtype=np.float64)

        for key in ep_dict_np.keys():
            if "obs" in key or "state" in key:
                # Chop off the extra observation/state at the end
                f.create_dataset(
                    f"data/demo_{ep}/{key}", ep_dict_np[key][:-1].shape, data=ep_dict_np[key][:-1]
                )
            else:
                f.create_dataset(
                    f"data/demo_{ep}/{key}", ep_dict_np[key].shape, data=ep_dict_np[key]
                )
        # track them for the conversion later
        list_ep_dict.append(ep_dict)
        list_ep_dict_np.append(ep_dict_np)

        if ep < cfg.save_gifs:
            print("Saving gif...")
            gif_output_file = os.path.join(cfg.output_path, f"episode_{ep}.gif")
            camera_name = cfg.env_cfg.rl_camera
            images = ep_dict_np[f"obs/{camera_name}_image"].transpose((0, 2, 3, 1))
            images = [Image.fromarray(img[:, :, -3:]) for img in images]
            images[0].save(
                gif_output_file, save_all=True, append_images=images[1:], duration=50, loop=0
            )
    f.close()
    print(f"Successfully wrote hdf5 file at {output_file}")

    if cfg.add_modem_format == True:
        env_id = re.sub(r"([a-z])([A-Z])", r"\1-\2", cfg.env_cfg.env_name).lower()
        task_name = f"mw-{env_id}"
        modem_path = os.path.join(cfg.output_path, f"demonstrations/{task_name}")
        os.makedirs(os.path.join(modem_path, "frames"), exist_ok=True)

        for ep in range(cfg.num_episodes):
            print(f"Converting episode {ep + 1} / {cfg.num_episodes} to MoDem format...")
            ep_dict_np = list_ep_dict_np[ep]
            ep_dict = list_ep_dict[ep]

            modem_data = dict()
            modem_data["frames"] = []
            modem_data["states"] = []
            modem_data["actions"] = []
            modem_data["rewards"] = []
            modem_data["infos"] = []

            num_steps = len(ep_dict_np["actions"])

            # Do some validation to make sure parameters match MoDem
            if num_steps != 100:
                if cfg.env_cfg.end_on_success == True:
                    print(
                        "WARNING: end_on_success is set to True. "
                        "Note that MoDem does not use early termination in their demos."
                    )
                if cfg.env_cfg.episode_length != 100:
                    print("WARNING: episode_length is not set to 100, as it is in MoDem.")
            if cfg.env_cfg.rl_image_size != 224:
                print("WARNING: rl_image_size is not set to 224, as it is in MoDem")
            if cfg.env_cfg.rl_camera != "corner2":
                print("WARNING: rl_camera is not set to corner2, as it is in MoDem")

            for i in range(num_steps + 1):
                camera_name = cfg.env_cfg.rl_camera
                frame_filename = f"{ep}_" + f"{i}".zfill(3) + ".png"
                frame_path = os.path.join(modem_path, "frames", frame_filename)
                frame_data = ep_dict_np[f"obs/{camera_name}_image"][i].transpose((1, 2, 0))
                frame_img = Image.fromarray(frame_data[:, :, -3:])
                frame_img.save(frame_path)
                modem_data["frames"].append(frame_filename)
                modem_data["states"].append(ep_dict_np["states"][i])

            for i in range(num_steps):
                modem_data["actions"].append(ep_dict_np["actions"][i])
                modem_data["rewards"].append(ep_dict["infos"][i]["original_reward"] / 10)
                modem_data["infos"].append(ep_dict["infos"][i])

            torch.save(modem_data, os.path.join(modem_path, f"{ep}.pt"))

        print(f"Successfully wrote modem format at {modem_path}")


@dataclass
class EnvironmentConfig:
    # Below are all the arguments to PixelMetaWorld
    env_name: str = "Assembly"
    robots: List[str] = field(default_factory=lambda: ["Sawyer"])
    episode_length: int = 100
    action_repeat: int = 2
    frame_stack: int = 2
    obs_stack: int = 1
    reward_shaping: bool = False
    rl_image_size: int = 96  # This is the image size that gets saved in the dataset
    device: str = "cuda"
    camera_names: List[str] = field(default_factory=lambda: ["corner2"])
    rl_camera: str = "corner2"  # This is the camera that gets saved in the dataset
    env_reward_scale: float = 1.0
    end_on_success: bool = False
    use_state: bool = True


@dataclass
class MainConfig:
    output_path: str = "data/metaworld/Assembly"
    num_episodes: int = 1
    save_gifs: int = 0
    add_modem_format: bool = False
    env_cfg: EnvironmentConfig = field(default_factory=lambda: EnvironmentConfig())
    seed: int = 0


if __name__ == "__main__":
    import rich.traceback

    rich.traceback.install()
    cfg = pyrallis.parse(config_class=MainConfig)  # type: ignore

    run(cfg)
