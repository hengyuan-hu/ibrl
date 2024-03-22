# add root to python path
export PYTHONPATH=$PWD:$PYTHONPATH
conda activate ibrl

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export MUJOCO_PY_MUJOCO_PATH=~/.mujoco/mujoco210
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin
