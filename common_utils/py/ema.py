import copy
import torch
from torch.nn.modules.batchnorm import _BatchNorm


# TODO: duplicated code
def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


class SimpleEMA:
    def __init__(self, model, tau):
        self.model: torch.nn.Module = copy.deepcopy(model)
        self.model.train(False)
        self.tau = tau

    def step(self, model: torch.nn.Module, optim_step):
        for param, target_param in zip(model.parameters(), self.model.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        return 1 - self.tau

    @property
    def stable_model(self):
        return self.model


class EMA:
    """
    Exponential Moving Average of models weights, adapted from :
    https://github.com/columbia-ai-robotics/diffusion_policy/tree/main/diffusion_policy/model/diffusion/ema_model.py
    """

    def __init__(
        self,
        base_model,
        update_after_step=0,
        inv_gamma=1.0,
        power=2 / 3,
        min_value=0.0,
        max_value=0.9999,
    ):
        """
        @crowsonkb's notes on EMA Warmup:
            If gamma=1 and power=1, implements a simple average. gamma=1, power=2/3 are good values for models you plan
            to train for a million or more steps (reaches decay factor 0.999 at 31.6K steps, 0.9999 at 1M steps),
            gamma=1, power=3/4 for models you plan to train for less (reaches decay factor 0.999 at 10K steps, 0.9999
            at 215.4k steps).
        Args:
            inv_gamma (float): Inverse multiplicative factor of EMA warmup. Default: 1.
            power (float): Exponential factor of EMA warmup. Default: 2/3.
            min_value (float): The minimum EMA decay rate. Default: 0.
        """

        self._averaged_model = copy.deepcopy(base_model)
        self._averaged_model.eval()
        self._averaged_model.requires_grad_(False)

        self.update_after_step = update_after_step
        self.inv_gamma = inv_gamma
        self.power = power
        self.min_value = min_value
        self.max_value = max_value

        self.decay = 0.0

    def get_decay(self, optim_step):
        """
        Compute the decay factor for the exponential moving average.
        """
        step = max(0, optim_step - self.update_after_step - 1)
        value = 1 - (1 + step / self.inv_gamma) ** -self.power

        if step <= 0:
            return 0.0

        return max(self.min_value, min(value, self.max_value))

    @torch.no_grad()
    def step(self, new_model, optim_step):
        self.decay = self.get_decay(optim_step)

        # all_dataptrs = set()
        for module, ema_module in zip(new_model.modules(), self._averaged_model.modules()):
            for param, ema_param in zip(
                module.parameters(recurse=False), ema_module.parameters(recurse=False)
            ):
                # iterative over immediate parameters only.
                if isinstance(param, dict):
                    raise RuntimeError("Dict parameter not supported")

                assert not isinstance(module, _BatchNorm)
                if not param.requires_grad:
                    # just copy the constant?
                    ema_param.copy_(param.to(dtype=ema_param.dtype).data)
                else:
                    ema_param.mul_(self.decay)
                    ema_param.add_(param.data.to(dtype=ema_param.dtype), alpha=1 - self.decay)
        return self.decay

    @property
    def stable_model(self):
        return self._averaged_model
