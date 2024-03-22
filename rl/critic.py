from dataclasses import dataclass
import math
import torch
import torch.nn as nn
import numpy as np
from common_utils import ibrl_utils as utils


class _QNet(nn.Module):
    def __init__(self, repr_dim, prop_dim, action_dim, feature_dim, hidden_dim, orth, drop):
        super().__init__()
        self.feature_dim = feature_dim

        self.obs_proj = nn.Sequential(
            nn.Linear(repr_dim, feature_dim),
            nn.Dropout(drop),
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
        )

        self.prop_dim = prop_dim
        q_in_dim = feature_dim + action_dim
        if prop_dim > 0:
            q_in_dim += prop_dim
        self.q = nn.Sequential(
            nn.Linear(q_in_dim, hidden_dim),
            nn.Dropout(drop),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(drop),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
        if orth:
            self.apply(utils.orth_weight_init)

    def forward(self, feat, prop, action):
        assert feat.dim() == 3, f"should be [batch, patch, dim], got {feat.size()}"
        feat = feat.flatten(1, 2)
        x = self.obs_proj(feat)
        if self.prop_dim > 0:
            x = torch.cat([x, action, prop], dim=-1)
        else:
            x = torch.cat([x, action], dim=-1)
        q = self.q(x).squeeze(-1)
        return q


@dataclass
class CriticConfig:
    feature_dim: int = 128
    hidden_dim: int = 1024
    orth: int = 1
    drop: float = 0
    fuse_patch: int = 1
    norm_weight: int = 0
    spatial_emb: int = 0


class Critic(nn.Module):
    def __init__(self, repr_dim, patch_repr_dim, prop_dim, action_dim, cfg: CriticConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.spatial_emb:
            q_cons = lambda: SpatialEmbQNet(
                fuse_patch=cfg.fuse_patch,
                num_patch=repr_dim // patch_repr_dim,
                patch_dim=patch_repr_dim,
                emb_dim=cfg.spatial_emb,
                prop_dim=prop_dim,
                action_dim=action_dim,
                hidden_dim=self.cfg.hidden_dim,
                orth=self.cfg.orth,
            )
        else:
            q_cons = lambda: _QNet(
                repr_dim=repr_dim,
                prop_dim=prop_dim,
                action_dim=action_dim,
                feature_dim=self.cfg.feature_dim,
                hidden_dim=self.cfg.hidden_dim,
                orth=self.cfg.orth,
                drop=self.cfg.drop,
            )
        self.q1 = q_cons()
        self.q2 = q_cons()

    def forward(self, feat, prop, action) -> tuple[torch.Tensor, torch.Tensor]:
        # assert self.training
        q1 = self.q1(feat, prop, action)
        q2 = self.q2(feat, prop, action)
        return q1, q2


class SpatialEmbQNet(nn.Module):
    def __init__(
        self, num_patch, patch_dim, prop_dim, action_dim, fuse_patch, emb_dim, hidden_dim, orth
    ):
        super().__init__()

        if fuse_patch:
            proj_in_dim = num_patch + action_dim + prop_dim
            num_proj = patch_dim
        else:
            proj_in_dim = patch_dim + action_dim + prop_dim
            num_proj = num_patch

        self.fuse_patch = fuse_patch
        self.patch_dim = patch_dim
        self.prop_dim = prop_dim

        self.input_proj = nn.Sequential(
            nn.Linear(proj_in_dim, emb_dim),
            nn.LayerNorm(emb_dim),
            nn.ReLU(inplace=True),
        )

        self.weight = nn.Parameter(torch.zeros(1, num_proj, emb_dim))
        nn.init.normal_(self.weight)

        self.q = nn.Sequential(
            nn.Linear(emb_dim + action_dim + prop_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
        if orth:
            self.q.apply(utils.orth_weight_init)

    def extra_repr(self) -> str:
        return f"weight: nn.Parameter ({self.weight.size()})"

    def forward(self, feat: torch.Tensor, prop: torch.Tensor, action: torch.Tensor):
        assert feat.size(-1) == self.patch_dim, "are you using CNN, need flatten&transpose"

        if self.fuse_patch:
            feat = feat.transpose(1, 2)

        repeated_action = action.unsqueeze(1).repeat(1, feat.size(1), 1)
        all_feats = [feat, repeated_action]
        if self.prop_dim > 0:
            repeated_prop = prop.unsqueeze(1).repeat(1, feat.size(1), 1)
            all_feats.append(repeated_prop)

        x = torch.cat(all_feats, dim=-1)
        y: torch.Tensor = self.input_proj(x)
        z = (self.weight * y).sum(1)

        if self.prop_dim == 0:
            z = torch.cat((z, action), dim=-1)
        else:
            z = torch.cat((z, prop, action), dim=-1)

        q = self.q(z).squeeze(-1)
        return q


class _MultiLinear(nn.Module):
    def __init__(self, in_dim, out_dim, num_net, orth=False):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_net = num_net
        self.orth = orth

        self.weights = nn.Parameter(torch.zeros(num_net, in_dim, out_dim))
        self.biases = nn.Parameter(torch.zeros(1, num_net, 1))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for i in range(self.num_net):
            if self.orth:
                torch.nn.init.orthogonal_(self.weights.data[i].transpose(0, 1))
            else:
                # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
                # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
                # https://github.com/pytorch/pytorch/issues/57109
                torch.nn.init.kaiming_uniform_(self.weights.data[i].transpose(0, 1), a=math.sqrt(5))

    def __repr__(self):
        return f"_MultiLinear({self.in_dim} x {self.out_dim}, {self.num_net} nets)"

    def forward(self, x: torch.Tensor):
        """
        x: [batch, in_dim] or [batch, num_net, in_dim]
        return: [batch, num_net, out_dim]
        """
        if x.dim() == 2:
            x = x.unsqueeze(1).expand(-1, self.num_net, -1)
        # x: [batch, num_net, in_dim]
        y = torch.einsum("bni,nio->bno", x, self.weights)
        y = y + self.biases
        return y


def _build_multi_fc(
    in_dim, action_dim, hidden_dim, num_q, num_layer, layer_norm, dropout, orth, append_action
):
    dims = [in_dim + action_dim] + [hidden_dim for _ in range(num_layer)]
    layers = []
    for i in range(num_layer):
        in_dim = dims[i]
        if append_action and i > 0:
            in_dim += action_dim
        layers.append(_MultiLinear(in_dim, dims[i + 1], num_q, orth=bool(orth)))
        if layer_norm == 1:
            layers.append(nn.LayerNorm(dims[i + 1]))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.ReLU())

    # layers.append(nn.Linear(dims[-1], 1))
    layers.append(_MultiLinear(dims[-1], 1, num_q, orth=bool(orth)))
    return nn.Sequential(*layers)


@dataclass
class MultiFcQConfig:
    num_q: int = 10
    num_k: int = 2
    num_layer: int = 3
    hidden_dim: int = 512
    dropout: float = 0.0
    layer_norm: int = 0
    orth: int = 0
    append_action: int = 0


class MultiFcQ(nn.Module):
    def __init__(self, obs_shape, action_dim, cfg: MultiFcQConfig):
        super().__init__()
        assert len(obs_shape) == 1
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.cfg = cfg

        self.net = _build_multi_fc(
            in_dim=obs_shape[0],
            action_dim=action_dim,
            hidden_dim=cfg.hidden_dim,
            num_q=cfg.num_q,
            num_layer=cfg.num_layer,
            layer_norm=cfg.layer_norm,
            dropout=cfg.dropout,
            orth=cfg.orth,
            append_action=cfg.append_action,
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if self.cfg.append_action:
            x = obs
            for layer in self.net:
                if isinstance(layer, _MultiLinear):
                    if x.dim() == 3 and action.dim() == 2:
                        action = action.unsqueeze(1).repeat(1, x.size(1), 1)
                    x = torch.cat([x, action], dim=-1)
                x = layer(x)
            y = x.squeeze(-1)
        else:
            x = torch.cat([obs, action], dim=-1)
            y = self.net(x).squeeze(-1)
        return y

    def forward_k(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        y = self.forward(obs, action)
        if self.cfg.num_k == self.cfg.num_q:
            return y

        indices = np.random.choice(self.cfg.num_q, self.cfg.num_k, replace=False)
        # y: [batch, num_q]
        selected_y = y[:, indices]
        return selected_y


def test_spatial_emb_q():
    x = torch.rand(8, 144, 128)
    action = torch.rand(8, 7)

    net = SpatialEmbQNet(144, 128, 0, 7, True, 1024, 1024, False)
    print(net)
    print(net(x, prop=action, action=action).size())


if __name__ == "__main__":
    # test_proj_q()
    test_spatial_emb_q()
