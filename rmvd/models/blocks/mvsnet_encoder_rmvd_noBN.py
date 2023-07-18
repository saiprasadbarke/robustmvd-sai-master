import torch.nn as nn
import torch.nn.functional as F


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
        kernel_size,
        stride=1,
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
            bias=(not bn),
            **kwargs
        )
        self.kernel_size = kernel_size
        self.stride = stride
        self.bn = nn.BatchNorm2d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x


class MVSNetEncoderRMVD(nn.Module):
    def __init__(self, base_channels=8, batch_norm=True, **kwargs):
        super().__init__()

        self.base_channels = base_channels

        self.conv1 = nn.Sequential(
            Conv2d(3, base_channels, 3, 2, padding=1, bn=batch_norm),
            Conv2d(base_channels, base_channels, 3, 1, padding=1, bn=False),
        )

        self.conv2 = nn.Sequential(
            Conv2d(
                base_channels, base_channels * 2, 5, stride=2, padding=2, bn=batch_norm
            ),
            Conv2d(
                base_channels * 2, base_channels * 2, 3, 1, padding=1, bn=batch_norm
            ),
            Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1, bn=False),
        )

        self.conv3 = nn.Sequential(
            Conv2d(
                base_channels * 2,
                base_channels * 4,
                5,
                stride=2,
                padding=2,
                bn=batch_norm,
            ),
            Conv2d(
                base_channels * 4, base_channels * 4, 3, 1, padding=1, bn=batch_norm
            ),
            Conv2d(base_channels * 4, base_channels * 4, 3, 1, padding=1, bn=False),
        )

        # self.conv3r = nn.Conv2d(base_channels * 4, base_channels * 4, 1, bias=False)

    def forward(self, x):
        conv1 = self.conv1(x)  # images_key: torch.Size([1, 3, 128, 160])
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        # conv3r = self.conv3r(conv3)

        return {
            "conv1": conv1,
            "conv2": conv2,
            "conv3": conv3,
            #    "conv3r": conv3r,
        }, conv3  # conv3r


# {
# 'all_enc_key[conv1]': torch.Size([1, 8, 128, 160]), 163840
# 'all_enc_key[conv2]': torch.Size([1, 16, 64, 80]), 81920
# 'all_enc_key[conv3]': torch.Size([1, 32, 32, 40]), 40960
# 'all_enc_key[conv3r]': torch.Size([1, 32, 32, 40])
# }
