import torch.nn as nn
import torch

from networks.resnet import conv1x1, conv3x3


class Block(nn.Module):
    def __init__(self, in_chan, out_chan, stride, use_1x1, shallow):
        super().__init__()

        self.shallow = shallow
        self.norm1 = nn.GroupNorm(num_groups=in_chan, num_channels=in_chan)

        assert stride in [1, 2], f"invalid stride: {stride}"
        if use_1x1:
            self.conv1 = conv3x3(in_chan, 4 * in_chan, stride=stride)
            self.conv2 = conv1x1(4 * in_chan, out_chan, stride=1)
            self.norm2 = nn.GroupNorm(num_groups=4 * in_chan, num_channels=4 * in_chan)
        else:
            self.conv1 = conv3x3(in_chan, out_chan, stride=stride)
            if self.shallow:
                self.conv2 = nn.Identity()
                self.norm2 = nn.Identity()
            else:
                self.conv2 = conv3x3(out_chan, out_chan, stride=1)
                self.norm2 = nn.GroupNorm(num_groups=out_chan, num_channels=out_chan)

        if stride == 1:
            self.x_path = nn.Identity()
        else:
            self.x_path = conv1x1(in_chan, out_chan, stride)

    def forward(self, x: torch.Tensor):
        y = nn.functional.relu(self.norm1(x))
        y = self.conv1(y)

        if not self.shallow:
            y = nn.functional.relu(self.norm2(y))
            y = self.conv2(y)

        z = self.x_path(x) + y
        return z


class ResNet96(nn.Module):
    def __init__(self, in_chan, use_1x1, shallow):
        super().__init__()

        blocks = [
            conv3x3(in_chan, 32, stride=1),
            Block(32, 64, 2, use_1x1, shallow),  # 96 -> 48
            Block(64, 128, 2, use_1x1, shallow),  # 48 -> 24
            Block(128, 256, 2, use_1x1, shallow),  # 24 -> 12
            nn.GroupNorm(num_groups=256, num_channels=256),
            nn.ReLU(),
        ]

        self.net = nn.Sequential(*blocks)

    def forward(self, x) -> torch.Tensor:
        y = self.net(x)
        return y


if __name__ == "__main__":
    x = torch.rand(256, 3, 96, 96).cuda()

    net = ResNet96(3, False, False).cuda()
    y = net.forward(x)
    print(y.size())
