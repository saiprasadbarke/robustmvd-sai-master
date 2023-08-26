import torch
import torch.nn as nn
import torch.nn.functional as F

verbose = False


class Conv2d(nn.Module):
    """Applies a 2D convolution (optionally with batch normalization and relu activation)
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
        padding=1,
        relu=True,
        bn=True,
        bn_momentum=0.1,
        init_method="xavier",
        **kwargs
    ):
        super(Conv2d, self).__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=(not bn),
            **kwargs
        )
        self.bn = nn.BatchNorm2d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x


class FeaturePyramidNet(nn.Module):
    """
    output 3 levels of features using a FPN structure
    """

    def __init__(self):
        super(FeaturePyramidNet, self).__init__()

        self.conv0 = nn.Sequential(
            Conv2d(3, 8, 3, 1, 1),
            Conv2d(8, 8, 3, 1, 1),
        )

        self.conv1 = nn.Sequential(
            Conv2d(8, 16, 5, 2, 2),
            Conv2d(16, 16, 3, 1, 1),
            Conv2d(16, 16, 3, 1, 1),
        )

        self.conv2 = nn.Sequential(
            Conv2d(16, 32, 5, 2, 2),
            Conv2d(32, 32, 3, 1, 1),
            Conv2d(32, 32, 3, 1, 1),
        )

        self.toplayer = nn.Conv2d(32, 32, 1)
        self.lat1 = nn.Conv2d(16, 32, 1)
        self.lat0 = nn.Conv2d(8, 32, 1)

        # to reduce channel size of the outputs from FPN
        self.smooth1 = nn.Conv2d(32, 16, 3, padding=1)
        self.smooth0 = nn.Conv2d(32, 8, 3, padding=1)

    def _upsample_add(self, x, y):
        return (
            F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False) + y
        )

    def forward(self, x):
        # x: (B, 3, H, W)
        conv0 = self.conv0(x)  # (B, 8, H, W)
        print("fpn_conv0.shape: ", conv0.shape) if verbose else None
        conv1 = self.conv1(conv0)  # (B, 16, H//2, W//2)
        print("fpn_conv1.shape: ", conv1.shape) if verbose else None
        conv2 = self.conv2(conv1)  # (B, 32, H//4, W//4)
        print("fpn_conv2.shape: ", conv2.shape) if verbose else None
        feat2 = self.toplayer(conv2)  # (B, 32, H//4, W//4)
        print("fpn_feat2.shape: ", feat2.shape) if verbose else None
        lat_1 = self.lat1(conv1)
        print("fpn_lat1.shape: ", lat_1.shape) if verbose else None
        lat_0 = self.lat0(conv0)
        print("fpn_lat0.shape: ", lat_0.shape) if verbose else None
        feat1 = self._upsample_add(feat2, lat_1)  # (B, 32, H//2, W//2)
        print("fpn_feat1.shape: ", feat1.shape) if verbose else None
        feat0 = self._upsample_add(feat1, lat_0)  # (B, 32, H, W)
        print("fpn_feat0.shape: ", feat0.shape) if verbose else None

        # reduce output channels
        feat1 = self.smooth1(feat1)  # (B, 16, H//2, W//2)
        print("fpn_smooth_feat1.shape: ", feat1.shape) if verbose else None
        feat0 = self.smooth0(feat0)  # (B, 8, H, W)
        print("fpn_smooth_feat0.shape: ", feat0.shape) if verbose else None

        feats = {"level_0": feat2, "level_1": feat1, "level_2": feat0}

        return feats
