"""metaworld uses a separate bc policy file to stay compatible with old model checkpoints"""
from dataclasses import dataclass, field
import torch
import torch.nn as nn

from common_utils import RandomShiftsAug
from common_utils import ibrl_utils as utils
from networks.encoder import ResNetEncoder, ResNetEncoderConfig


@dataclass
class BcPolicyConfig:
    net_type: str = "resnet"
    resnet: ResNetEncoderConfig = field(default_factory=lambda: ResNetEncoderConfig())
    hidden_dim: int = 1024
    dropout: float = 0
    orth_init: int = 1
    use_prop: int = 0
    feature_dim: int = 256
    proj_dim: int = 1024


class BcPolicy(nn.Module):
    def __init__(self, obs_shape, prop_shape, action_dim, cfg: BcPolicyConfig):
        super().__init__()
        self.cfg = cfg
        assert self.cfg.net_type == "resnet"
        self.encoder = ResNetEncoder(obs_shape, cfg.resnet)

        if cfg.use_prop:
            assert len(prop_shape) == 1
            self.compress = nn.Linear(self.encoder.repr_dim, cfg.feature_dim)
            policy_input_dim = cfg.feature_dim + prop_shape[0]
        else:
            policy_input_dim = self.encoder.repr_dim

        self.policy = nn.Sequential(
            nn.Linear(policy_input_dim, self.cfg.hidden_dim),
            nn.Dropout(self.cfg.dropout),
            nn.LayerNorm(self.cfg.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.cfg.hidden_dim, self.cfg.hidden_dim),
            nn.Dropout(self.cfg.dropout),
            nn.LayerNorm(self.cfg.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.cfg.hidden_dim, action_dim),
        )
        self.aug = RandomShiftsAug(pad=4)
        if self.cfg.orth_init:
            self.policy.apply(utils.orth_weight_init)

    def forward(self, obs: dict[str, torch.Tensor]):
        h = self.encoder(obs["obs"])
        assert not self.cfg.use_prop
        mu = self.policy(h)
        mu = torch.tanh(mu)
        return mu

    def act(self, obs: dict[str, torch.Tensor], *, eval_mode=True, cpu=True):
        assert eval_mode
        assert not self.training

        unsqueezed = False
        if obs["obs"].dim() == 3:
            for k, v in obs.items():
                obs[k] = v.unsqueeze(0)
            unsqueezed = True

        greedy_action = self.forward(obs).detach()

        if unsqueezed:
            greedy_action = greedy_action.squeeze()
        if cpu:
            greedy_action = greedy_action.cpu()
        return greedy_action

    def loss(self, batch):
        image: torch.Tensor = batch.obs["obs"]
        action: torch.Tensor = batch.action["action"]

        image = self.aug(image.float())
        pred_a = self.forward({"obs": image})
        loss = nn.functional.mse_loss(pred_a, action, reduction="none")
        loss = loss.sum(1).mean(0)
        return loss
