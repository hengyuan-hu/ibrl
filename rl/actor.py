from dataclasses import dataclass
import torch
import torch.nn as nn

from common_utils import ibrl_utils as utils


def build_fc(in_dim, hidden_dim, action_dim, num_layer, layer_norm, dropout):
    dims = [in_dim]
    dims.extend([hidden_dim for _ in range(num_layer)])

    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if layer_norm == 1:
            layers.append(nn.LayerNorm(dims[i + 1]))
        if layer_norm == 2 and (i == num_layer - 1):
            layers.append(nn.LayerNorm(dims[i + 1]))
        layers.append(nn.Dropout(dropout))
        layers.append(nn.ReLU())

    layers.append(nn.Linear(dims[-1], action_dim))
    layers.append(nn.Tanh())
    return nn.Sequential(*layers)


class SpatialEmb(nn.Module):
    def __init__(self, num_patch, patch_dim, prop_dim, proj_dim, dropout):
        super().__init__()

        # if fuse_patch:
        proj_in_dim = num_patch + prop_dim
        num_proj = patch_dim

        self.patch_dim = patch_dim
        self.prop_dim = prop_dim

        self.input_proj = nn.Sequential(
            nn.Linear(proj_in_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.ReLU(inplace=True),
        )
        self.weight = nn.Parameter(torch.zeros(1, num_proj, proj_dim))
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.weight)

    def extra_repr(self) -> str:
        return f"weight: nn.Parameter ({self.weight.size()})"

    def forward(self, feat: torch.Tensor, prop: torch.Tensor):
        feat = feat.transpose(1, 2)

        if self.prop_dim > 0:
            repeated_prop = prop.unsqueeze(1).repeat(1, feat.size(1), 1)
            feat = torch.cat((feat, repeated_prop), dim=-1)

        y = self.input_proj(feat)
        z = (self.weight * y).sum(1)
        z = self.dropout(z)
        return z


@dataclass
class ActorConfig:
    feature_dim: int = 128
    hidden_dim: int = 1024
    dropout: float = 0
    orth: int = 1
    max_action_norm: float = -1
    spatial_emb: int = 0


class Actor(nn.Module):
    def __init__(self, repr_dim, patch_repr_dim, prop_dim, action_dim, cfg: ActorConfig):
        super().__init__()

        if cfg.spatial_emb > 0:
            assert cfg.spatial_emb > 1, "this is the dimension"
            self.compress = SpatialEmb(
                num_patch=repr_dim // patch_repr_dim,
                patch_dim=patch_repr_dim,
                prop_dim=prop_dim,
                proj_dim=cfg.spatial_emb,
                dropout=cfg.dropout,
            )
            policy_in_dim = cfg.spatial_emb
        else:
            self.compress = nn.Sequential(
                nn.Linear(repr_dim, cfg.feature_dim),
                nn.LayerNorm(cfg.feature_dim),
                nn.Dropout(cfg.dropout),
                nn.ReLU(),
            )
            policy_in_dim = cfg.feature_dim

        self.prop_dim = prop_dim
        self.cfg = cfg

        if prop_dim > 0:
            policy_in_dim += prop_dim

        self.policy = build_fc(
            policy_in_dim,
            cfg.hidden_dim,
            action_dim,
            num_layer=2,
            layer_norm=1,
            dropout=cfg.dropout,
        )
        if cfg.orth:
            self.compress.apply(utils.orth_weight_init)
            self.policy.apply(utils.orth_weight_init)

    def forward(self, obs: dict[str, torch.Tensor], std: float):
        if isinstance(self.compress, SpatialEmb):
            feat = self.compress.forward(obs["feat"], obs["prop"])
        else:
            feat = obs["feat"].flatten(1, -1)
            feat = self.compress(feat)

        all_input = [feat]
        if self.prop_dim > 0:
            prop = obs["prop"]
            all_input.append(prop)

        policy_input = torch.cat(all_input, dim=-1)
        mu: torch.Tensor = self.policy(policy_input)

        if self.cfg.max_action_norm > 0:
            mu = utils.clip_action_norm(mu, self.cfg.max_action_norm)

        return utils.TruncatedNormal(mu, std, max_action_norm=self.cfg.max_action_norm)


@dataclass
class FcActorConfig:
    num_layer: int = 3
    hidden_dim: int = 512
    dropout: float = 0.5
    layer_norm: int = 0
    orth: int = 0


class FcActor(nn.Module):
    def __init__(self, obs_shape, action_dim, cfg: FcActorConfig):
        super().__init__()
        assert len(obs_shape) == 1
        self.cfg = cfg
        self.net = build_fc(
            obs_shape[0], cfg.hidden_dim, action_dim, cfg.num_layer, cfg.layer_norm, cfg.dropout
        )

        if cfg.orth:
            self.net.apply(utils.orth_weight_init)

    def forward(self, obs: dict[str, torch.Tensor], std):
        mu = self.net(obs["state"])
        return utils.TruncatedNormal(mu, std)


if __name__ == "__main__":
    cfg = ActorConfig()
    cfg.spatial_emb = 1024
    actor = Actor(128 * 144, 128, 9, 7, cfg)

    print(actor)
    obs = {"feat": torch.rand(8, 144, 128), "prop": torch.rand(8, 9)}
    y = actor.forward(obs, 0)
    print(y.mean.size())
