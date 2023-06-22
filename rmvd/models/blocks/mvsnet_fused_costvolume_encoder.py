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
        kernel_size=3,
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


class MVSNetFusedCostvolumeEncoder(nn.Module):
    def __init__(self, in_channels=32, base_channels=8, batch_norm=True, **kwargs):
        super().__init__()

        self.conv0 = Conv3d(in_channels, base_channels, padding=1, bn=batch_norm)

        self.conv1 = Conv3d(
            base_channels, base_channels * 2, stride=2, padding=1, bn=batch_norm
        )
        self.conv2 = Conv3d(
            base_channels * 2, base_channels * 2, padding=1, bn=batch_norm
        )

        self.conv3 = Conv3d(
            base_channels * 2, base_channels * 4, stride=2, padding=1, bn=batch_norm
        )
        self.conv4 = Conv3d(
            base_channels * 4, base_channels * 4, padding=1, bn=batch_norm
        )

        self.conv5 = Conv3d(
            base_channels * 4, base_channels * 8, stride=2, padding=1, bn=batch_norm
        )
        self.conv6 = Conv3d(
            base_channels * 8, base_channels * 8, padding=1, bn=batch_norm
        )

    def forward(self, fused_corr):
        fused_corr = torch.permute(fused_corr, (0, 2, 1, 3, 4))  # NCSHW

        conv0 = self.conv0(fused_corr)
        conv1 = self.conv1(conv0)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        conv6 = self.conv6(conv5)

        all_enc = {
            "3d_conv0": conv0,
            "3d_conv1": conv1,
            "3d_conv2": conv2,
            "3d_conv3": conv3,
            "3d_conv4": conv4,
            "3d_conv5": conv5,
            "3d_conv6": conv6,
        }

        return all_enc, conv6
