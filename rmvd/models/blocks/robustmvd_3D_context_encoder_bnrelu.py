import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv3d(nn.Module):
    """Applies a 3D convolution (optionally with batch normalization and relu activation)
    over an input signal composed of several input planes.
    Attributes:
        conv (nn.Module): convolution module
        bn (nn.Module): batch normalization module
        relu (bool): whether to activate by relu
    Notes:
        Default momentum for batch normalization is set to be 0.01,
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        relu=True,
        bn=True,
        bn_momentum=0.1,
        init_method="xavier",
        **kwargs
    ):
        super(Conv3d, self).__init__()

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        assert stride in [1, 2]
        self.stride = stride

        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            bias=(not bn),
            **kwargs
        )
        self.bn = nn.BatchNorm3d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x


class CostVolume3DContextEncoderBNRelu(nn.Module):
    def __init__(self, in_channels=32) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.layer1 = Conv3d(in_channels, int(in_channels / 2), kernel_size=1, stride=1)
        self.layer2 = Conv3d(
            int(in_channels / 2), int(in_channels / 4), kernel_size=1, stride=1
        )
        self.layer3 = Conv3d(int(in_channels / 4), 1, kernel_size=1, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x
