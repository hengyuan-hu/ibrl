# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import re

import numpy as np
import torch
import torch.nn as nn
from torch import distributions as pyd
from torch.distributions.utils import _standard_normal
import torchvision.transforms as transforms


def get_rescale_transform(target_size):
    return transforms.Resize(
        target_size,
        interpolation=transforms.InterpolationMode.BICUBIC,
        antialias=True,
    )


def concat_obs(curr_idx, obses, obs_stack) -> torch.Tensor:
    """
    cat obs as [obses[curr_idx], obses[curr_idx-1], ... obs[curr_odx-obs_stack+1]]
    """
    vals = []
    for offset in range(0, obs_stack):
        if curr_idx - offset >= 0:
            val = obses[curr_idx - offset]
            if not isinstance(val, torch.Tensor):
                val = torch.from_numpy(val)
            vals.append(val)
        else:
            vals.append(torch.zeros_like(vals[-1]))
    return torch.cat(vals, dim=0)


class eval_mode:
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


def to_torch(xs, device):
    return tuple(torch.as_tensor(x, device=device) for x in xs)


def orth_weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain("relu")
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)


def clip_action_norm(action, max_norm):
    assert max_norm > 0
    assert action.dim() == 2 and action.size(1) == 7

    ee_action = action[:, :6]
    gripper_action = action[:, 6:]

    ee_action_norm = ee_action.norm(dim=1).unsqueeze(1)
    ee_action = ee_action / ee_action_norm
    assert (ee_action.norm(dim=1).min() - 1).abs() <= 1e-5
    scale = ee_action_norm.clamp(max=max_norm)
    ee_action = ee_action * scale
    action = torch.cat([ee_action, gripper_action], dim=1)
    return action


class TruncatedNormal(pyd.Normal):
    def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6, max_action_norm: float = -1):
        if isinstance(scale, float):
            scale = torch.ones_like(loc) * scale

        super().__init__(loc, scale, validate_args=False)
        self.low = low
        self.high = high
        self.eps = eps
        self.max_action_norm = max_action_norm

    def _clamp(self, x):
        clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
        x = x - x.detach() + clamped_x.detach()
        return x

    def sample(self, clip=None, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
        eps *= self.scale
        if clip is not None:
            eps = torch.clamp(eps, -clip, clip)
        x = self.loc + eps
        x = self._clamp(x)
        if self.max_action_norm > 0:
            x = clip_action_norm(x, self.max_action_norm)
        return x


def schedule(schdl, step):
    try:
        return float(schdl)
    except ValueError:
        match = re.match(r"linear\((.+),(.+),(.+)\)", schdl)
        if match:
            init, final, duration = [float(g) for g in match.groups()]
            mix = np.clip(step / duration, 0.0, 1.0)
            return (1.0 - mix) * init + mix * final
        match = re.match(r"step_linear\((.+),(.+),(.+),(.+),(.+)\)", schdl)
        if match:
            init, final1, duration1, final2, duration2 = [float(g) for g in match.groups()]
            if step <= duration1:
                mix = np.clip(step / duration1, 0.0, 1.0)
                return (1.0 - mix) * init + mix * final1
            else:
                mix = np.clip((step - duration1) / duration2, 0.0, 1.0)
                return (1.0 - mix) * final1 + mix * final2
    raise NotImplementedError(schdl)
