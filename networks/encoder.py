from dataclasses import dataclass, field
import torch
from torch import nn

import common_utils
from networks.resnet import ResNet
from networks.resnet_rl import ResNet96
from networks.min_vit import MinVit


@dataclass
class ResNet96EncoderConfig:
    use_1x1: int = 0
    shallow: int = 0


class ResNet96Encoder(nn.Module):
    def __init__(self, obs_shape: list[int], cfg: ResNet96EncoderConfig):
        super().__init__()

        self.resnet = ResNet96(obs_shape[0], cfg.use_1x1, cfg.shallow)
        self.repr_dim, self.num_patch, self.patch_repr_dim = self._get_repr_dim(obs_shape)

    def _get_repr_dim(self, obs_shape: list[int]):
        x = torch.rand(1, *obs_shape)
        y = self.resnet.forward(x).flatten(2, 3)
        repr_dim = y.flatten().size(0)
        _, patch_repr_dim, num_patch = y.size()
        return repr_dim, num_patch, patch_repr_dim

    def forward(self, obs, flatten=True):
        # assert not flatten
        assert obs.max() > 5

        obs = obs / 255.0 - 0.5
        h: torch.Tensor = self.resnet(obs)

        if flatten:
            # mostly for bc
            h = h.flatten(1, -1)
        else:
            # convert to [batch, num_patch, dim] just to be consistent with RL
            h = h.flatten(2, 3).transpose(1, 2)
        return h


@dataclass
class VitEncoderConfig:
    patch_size: int = 8
    depth: int = 3
    embed_dim: int = 128
    num_heads: int = 4
    act_layer = nn.GELU
    stride: int = -1
    embed_style: str = "embed1"
    embed_norm: int = 0


class VitEncoder(nn.Module):
    def __init__(self, obs_shape: list[int], cfg: VitEncoderConfig):
        super().__init__()
        self.obs_shape = obs_shape
        self.cfg = cfg
        self.vit = MinVit(
            embed_style=cfg.embed_style,
            embed_dim=cfg.embed_dim,
            embed_norm=cfg.embed_norm,
            num_head=cfg.num_heads,
            depth=cfg.depth,
        )

        self.num_patch = self.vit.num_patches
        self.patch_repr_dim = self.cfg.embed_dim
        self.repr_dim = self.cfg.embed_dim * self.vit.num_patches

    def forward(self, obs, flatten=True) -> torch.Tensor:
        assert obs.max() > 5
        obs = obs / 255.0 - 0.5
        feats: torch.Tensor = self.vit.forward(obs)
        if flatten:
            feats = feats.flatten(1, 2)
        return feats


@dataclass
class ResNetEncoderConfig:
    stem: str = "default"
    downsample: str = "default"
    norm_layer: str = "gnn"
    shallow: int = 0


class ResNetEncoder(nn.Module):
    def __init__(self, obs_shape, cfg: ResNetEncoderConfig):
        super().__init__()
        self.obs_shape = obs_shape
        self.cfg = cfg
        layers = [1, 1, 1, 1] if self.cfg.shallow else [2, 2, 2, 2]
        self.nets = ResNet(
            stem=self.cfg.stem,
            downsample=self.cfg.downsample,
            norm_layer=self.cfg.norm_layer,
            layers=layers,
            in_channels=obs_shape[0],
        )
        self.repr_dim, self.num_patch, self.patch_repr_dim = self._get_repr_dim(obs_shape)

    def _get_repr_dim(self, obs_shape: list[int]):
        x = torch.rand(1, *obs_shape)
        y = self.nets.forward(x).flatten(2, 3)
        repr_dim = y.flatten().size(0)
        _, patch_repr_dim, num_patch = y.size()
        return repr_dim, num_patch, patch_repr_dim

    def forward(self, obs, flatten=True):
        obs = obs / 255.0 - 0.5
        h = self.nets(obs)
        if flatten:
            h = h.flatten(1, -1)
        else:
            # convert to [batch, num_patch, dim] just to be consistent with RL
            h = h.flatten(2, 3).transpose(1, 2)
        return h


class DrQEncoder(nn.Module):
    def __init__(self, obs_shape):
        super().__init__()

        assert len(obs_shape) == 3
        self.repr_dim = 32 * 35 * 35
        self.patch_repr_dim = 32
        self.num_patch = 35 * 35
        self.transform = common_utils.ibrl_utils.get_rescale_transform(84)

        self.convnet = nn.Sequential(
            nn.Conv2d(obs_shape[0], 32, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
        )

        self.apply(common_utils.ibrl_utils.orth_weight_init)

    def forward(self, obs, flatten=True):
        if obs.size(-1) != 84:
            obs = self.transform(obs)

        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        if flatten:
            h = h.flatten(1, -1)
        else:
            # convert to [batch, num_patch, dim] just to be consistent with RL
            h = h.flatten(2, 3).transpose(1, 2)
        return h


if __name__ == "__main__":
    import rich.traceback
    import pyrallis

    @dataclass
    class MainConfig:
        net_type: str
        obs_shape: list[int] = field(default_factory=lambda: [3, 96, 96])
        resnet: ResNetEncoderConfig = field(default_factory=lambda: ResNetEncoderConfig())
        vit: VitEncoderConfig = field(default_factory=lambda: VitEncoderConfig())

    rich.traceback.install()
    cfg = pyrallis.parse(config_class=MainConfig)  # type: ignore

    if cfg.net_type == "resnet":
        enc = ResNetEncoder(cfg.obs_shape, cfg.resnet)
    elif cfg.net_type == "vit":
        enc = VitEncoder(cfg.obs_shape, cfg.vit)
    else:
        assert False

    print(enc)
    x = torch.rand(1, *cfg.obs_shape) * 255
    print(common_utils.count_parameters(enc))
    print("output size:", enc(x, flatten=False).size())
    print("repr dim:", enc.repr_dim, ", real dim:", enc(x, flatten=True).size())
