# Imitation Bootstrapped Reinforcement Learning

Implementation of _Imitation Bootstrapped Reinforcement Learning (IBRL)_ and baeslines (RLPD, RFT) on Robomimic and Meta-World Tasks.

[![Paper](https://img.shields.io/badge/Paper-%20%F0%9F%93%84-blue)](https://arxiv.org/abs/2311.02198v4)
[![Website](https://img.shields.io/badge/Website-%F0%9F%8C%90-orange)](https://ibrl.hengyuanhu.com/)

## Clone and compile

### Clone the repo.
We need `--recursive` to get the correct submodule
```shell
git clone --recursive https://github.com/hengyuan-hu/ibrl.git
```

### Install dependencies
First Install MuJoCo

Download the MuJoCo version 2.1 binaries for [Linux](https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz)

Extract the downloaded mujoco210 directory into `~/.mujoco/mujoco210`.

### Create conda env

First create a conda env with name `ibrl`.
```shell
conda create --name ibrl python=3.9
```

Then, source `set_env.sh` to activate `ibrl` conda env. It also setup several important paths such as `MUJOCO_PY_MUJOCO_PATH` and add current project folder to `PYTHONPATH`.
Note that if the conda env has a different name, you will need to manually modify the `set_env.sh`.
You also need to modify the `set_env.sh` if the mujoco is not installed at the default location.

```shell
# NOTE: run this once per shell before running any script from this repo
source set_env.sh
```

Then install python dependencies
```shell
# first install pytorch with correct cuda version, in our case we use torch 2.1 with cu121
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121

# then install extra dependencies from requirement.txt
pip install -r requirements.txt
```
If the command above does not work for your versions.
Please check out `tools/core_packages.txt` for a list of commands to manually install relavent packages.


### Compile C++ code
We have a C++ module in the common utils that requires compilation
```shell
cd common_utils
make
```

### Trouble Shooting
Later when running the training commands, if we encounter the following error
```shell
ImportError: .../libstdc++.so.6: version `GLIBCXX_3.4.30' not found
```
Then we can force the conda to use the system c++ lib.
Use these command to symlink the system c++ lib into conda env. To find `PATH_TO_CONDA_ENV`, run `echo ${CONDA_PREFIX:-"$(dirname $(which conda))/../"}`.

```shell
ln -sf /lib/x86_64-linux-gnu/libstdc++.so.6 PATH_TO_CONDA_ENV/bin/../lib/libstdc++.so
ln -sf /lib/x86_64-linux-gnu/libstdc++.so.6 PATH_TO_CONDA_ENV/bin/../lib/libstdc++.so.6
```

## Reproduce Results

Remember to run `source set_env.sh`  once per shell before running any script from this repo.


### Download data and BC models

Download dataset and models from [Google Drive](https://drive.google.com/file/d/1F2yH84Iqv0qRPmfH8o-kSzgtfaoqMzWE/view?usp=sharing) and put the folders under `release` folder.
The release folder should contain `release/cfgs` (already shipped with the repo), `release/data` and `release/model` (the latter two are from the downloaded zip file).


### Robomimic (pixel)

Train RL policy using the BC policy provided in `release` folder

#### IBRL

```shell
# can
python train_rl.py --config_path release/cfgs/robomimic_rl/can_ibrl.yaml

# square
python train_rl.py --config_path release/cfgs/robomimic_rl/square_ibrl.yaml
```

Use `--save_dir PATH` to specify where to store the logs and models.
Use `--use_wb 0` to disable logging to weight and bias.

Use the following commands to train a BC policy from scratch.
We find that IBRL is not sensitive to the exact performance of the BC policy.
```shell
# can
python train_bc.py --config_path release/cfgs/robomimic_bc/can.yaml

# square
python train_bc.py --config_path release/cfgs/robomimic_bc/square.yaml
```

#### RLPD

```shell
# can
python train_rl.py --config_path release/cfgs/robomimic_rl/can_rlpd.yaml

# square
python train_rl.py --config_path release/cfgs/robomimic_rl/square_rlpd.yaml
```

#### RFT (Regularized Fine-Tuning)

These commands run RFT from pretrained models in `release` folder.
```shell
# can rft
python train_rl.py --config_path release/cfgs/robomimic_rl/can_rft.yaml

# square rft
python train_rl.py --config_path release/cfgs/robomimic_rl/square_rft.yaml
```

To only perform pretraining:
```shell
# can, pretraining for 5 x 10,000 steps
python train_rl.py --config_path release/cfgs/robomimic_rl/can_rft.yaml --pretrain_only 1 --pretrain_num_epoch 5 --load_pretrained_agent None

# square, pretraining for 10 x 10,000 steps
python train_rl.py --config_path release/cfgs/robomimic_rl/square_rft.yaml --pretrain_only 1 --pretrain_num_epoch 10 --load_pretrained_agent None
```
---

### Robomimic (state)

#### IBRL

Train IBRL using the provided state BC policies:
```shell
# can state
python train_rl.py --config_path release/cfgs/robomimic_rl/can_state_ibrl.yaml

# square state
python train_rl.py --config_path release/cfgs/robomimic_rl/square_state_ibrl.yaml
```

To train a state BC policy from scratch:
```shell
# can
python train_bc.py --config_path release/cfgs/robomimic_bc/can_state.yaml

# square
python train_bc.py --config_path release/cfgs/robomimic_bc/square_state.yaml
```

#### RLPD

```shell
# can state
python train_rl.py --config_path release/cfgs/robomimic_rl/can_state_rlpd.yaml

# square state
python train_rl.py --config_path release/cfgs/robomimic_rl/square_state_rlpd.yaml
```

#### RFT

Since state policies are fast to train, we can just run pretrain and RL fine-tuning in one step.
```shell
# can
python train_rl.py --config_path release/cfgs/robomimic_rl/can_state_rft.yaml

# square
python train_rl.py --config_path release/cfgs/robomimic_rl/square_state_rft.yaml
```
---

### Metaworld

#### IBRL

Train RL policy using the BC policy provided in `release` folder
```shell
# assembly
python mw_main/train_rl_mw.py --config_path release/cfgs/metaworld/ibrl_basic.yaml --bc_policy assembly

# boxclose
python mw_main/train_rl_mw.py --config_path release/cfgs/metaworld/ibrl_basic.yaml --bc_policy boxclose

# coffeepush
python mw_main/train_rl_mw.py --config_path release/cfgs/metaworld/ibrl_basic.yaml --bc_policy coffeepush

# stickpull
python mw_main/train_rl_mw.py --config_path release/cfgs/metaworld/ibrl_basic.yaml --bc_policy stickpull
```

If you want to train BC policy from scratch
```shell
python mw_main/train_bc_mw.py --dataset.path Assembly --save_dir SAVE_DIR
```

#### RPLD

Note that we still specify `bc_policy` to specify the task name, but we don't use it in baselines.
This is special to `train_rl_mw.py`.

```shell
python mw_main/train_rl_mw.py --config_path release/cfgs/metaworld/rlpd.yaml --bc_policy assembly --use_wb 0
```

#### RFT

For simplicity, here this one command performs both pretraining and RL training.
```shell
python mw_main/train_rl_mw.py --config_path release/cfgs/metaworld/rft.yaml --bc_policy assembly --use_wb 0
```
---

## Citation

```
@misc{hu2023imitation,
    title={Imitation Bootstrapped Reinforcement Learning},
    author={Hengyuan Hu and Suvir Mirchandani and Dorsa Sadigh},
    year={2023},
    eprint={2311.02198},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```
