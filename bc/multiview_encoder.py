from dataclasses import dataclass, field
import torch
import torch.nn as nn
from networks.encoder import ResNetEncoder, ResNetEncoderConfig


@dataclass
class MultiViewEncoderConfig:
    fuse_method: str = "cat"
    resnet: ResNetEncoderConfig = field(default_factory=lambda: ResNetEncoderConfig())
    feat_dim: int = 512
    dropout: float = 0


class MultiViewEncoder(nn.Module):
    def __init__(
        self,
        obs_shape,
        obs_horizon,
        prop_shape,
        rl_cameras,
        use_prop,
        cfg: MultiViewEncoderConfig,
    ):
        super().__init__()
        self.obs_shape = obs_shape
        self.obs_horizon = obs_horizon
        self.rl_cameras = rl_cameras
        self.use_prop = use_prop
        self.cfg = cfg
        self.encoders = nn.ModuleList([ResNetEncoder(obs_shape, cfg.resnet) for _ in rl_cameras])

        enc_repr_dim: int = self.encoders[0].repr_dim  # type: ignore
        compress_layers = [
            nn.Sequential(
                nn.Linear(enc_repr_dim, cfg.feat_dim),
                nn.LayerNorm(cfg.feat_dim),
                nn.Dropout(cfg.dropout),
                nn.ReLU(),
            )
            for _ in range(len(self.rl_cameras))
        ]
        self.compress_streams = nn.ModuleList(compress_layers)
        self.repr_dim = obs_horizon * cfg.feat_dim
        if cfg.fuse_method == "cat":
            self.repr_dim *= len(rl_cameras)

        if self.use_prop:
            assert len(prop_shape) == 1
            self.repr_dim += prop_shape[0]

    def forward(self, obs: dict[str, torch.Tensor]):
        hs = []
        for i, camera in enumerate(self.rl_cameras):
            x = obs[camera]
            if self.obs_horizon > 1:
                x: torch.Tensor = x.unflatten(1, (-1, self.obs_shape[0]))
                x = x.flatten(0, 1)

            h = self.encoders[i](x)
            h = self.compress_streams[i](h)
            if self.obs_horizon > 1:
                h = h.view(-1, self.obs_horizon * self.cfg.feat_dim)
            hs.append(h)

        if self.cfg.fuse_method == "cat":
            h = torch.cat(hs, dim=1)  # dim=0 is the batch dim
        elif self.cfg.fuse_method == "add":
            h = hs[0]
            for i in range(1, len(hs)):
                h = h + hs[i]
        elif self.cfg.fuse_method == "mult":
            h = hs[0]
            for i in range(1, len(hs)):
                h = h * hs[i]
        else:
            assert False, f"invalid fuse method {self.cfg.fuse_method}"

        if self.use_prop:
            prop = obs["prop"]
            h = torch.cat([h, prop], dim=-1)

        return h
