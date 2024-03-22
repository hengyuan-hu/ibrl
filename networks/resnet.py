from typing import Callable, List, Optional
import torch.nn as nn
from torch import Tensor


def conv3x3(
    in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1
) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def downsample_conv(in_planes: int, out_planes: int, stride: int = 2) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=stride, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        layers: List[int] = [2, 2, 2, 2],
        block=BasicBlock,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: str = "bn",
        stem: str = "default",
        downsample: str = "default",
        in_channels: int = 3,
    ) -> None:
        super().__init__()
        print(f"norm layer:", norm_layer)
        if norm_layer == "bn":
            self._norm_layer = nn.BatchNorm2d
        elif norm_layer.startswith("gn"):
            group = norm_layer[2:]
            if group == "n":
                self._norm_layer = lambda channel: nn.GroupNorm(
                    num_groups=channel, num_channels=channel
                )
            else:
                group = int(group)
                self._norm_layer = lambda channel: nn.GroupNorm(
                    num_groups=channel, num_channels=channel
                )
        else:
            assert False, f"Unknown normalization layer {norm_layer}"

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.downsample = downsample

        if stem == "patch":
            conv1 = nn.Conv2d(in_channels, self.inplanes, kernel_size=4, stride=4, bias=False)
            bn1 = self._norm_layer(self.inplanes)
            relu = nn.ReLU(inplace=True)
            init_downsample = [conv1, bn1, relu]
        elif stem == "convl":
            conv1 = nn.Conv2d(
                in_channels, self.inplanes, kernel_size=6, stride=4, padding=1, bias=False
            )
            bn1 = self._norm_layer(self.inplanes)
            relu = nn.ReLU(inplace=True)
            init_downsample = [conv1, bn1, relu]
        else:
            conv1 = nn.Conv2d(
                in_channels, self.inplanes, kernel_size=4, stride=2, padding=0, bias=False
            )
            bn1 = self._norm_layer(self.inplanes)
            relu = nn.ReLU(inplace=True)
            maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            init_downsample = [conv1, bn1, relu, maxpool]

        self.init_downsample = nn.Sequential(*init_downsample)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block,
            128,
            layers[1],
            stride=2,
            dilate=replace_stride_with_dilation[0],
        )
        self.layer3 = self._make_layer(
            block,
            256,
            layers[2],
            stride=2,
            dilate=replace_stride_with_dilation[1],
        )
        self.layer4 = self._make_layer(
            block,
            512,
            layers[3],
            stride=2,
            dilate=replace_stride_with_dilation[2],
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(
        self,
        block: type[BasicBlock],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1

        if stride != 1 or self.inplanes != planes * block.expansion:
            if self.downsample == "conv":
                downsample = nn.Sequential(
                    downsample_conv(self.inplanes, planes * block.expansion, stride),
                    norm_layer(planes * block.expansion),
                )
            elif self.downsample == "pool":
                downsample = nn.Sequential(
                    nn.MaxPool2d(kernel_size=stride, stride=stride),
                    conv1x1(self.inplanes, planes * block.expansion, stride=1),
                    norm_layer(planes * block.expansion),
                )
            else:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    norm_layer(planes * block.expansion),
                )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.init_downsample(x)
        # print("after init downsample:", x.size())
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x
