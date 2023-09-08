# External imports

import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.utils import create_meshgrid


class ConvBnReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=pad,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)


class ConvBnReLU3D(
    nn.Module
):  # TODO: mvsnet_pl does not use the relu in the forward method but the cvp_mvsnet implementation does. Add an optional argument to the init method to configure the use of relu or leaky relu.
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=pad,
            bias=False,
        )
        self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)


class FeatureNet(
    nn.Module
):  # TODO: Modify this code to accept arguments for setting the inchannels, outchannels, kernel size, stride, and padding. This can then be used in cvp_mvsnet implementation. Need an additional optional argument for configuring the use of relu or leaky relu on the convolutions.
    def __init__(self):
        super().__init__()
        self.inplanes = 32

        self.conv0 = ConvBnReLU(3, 8, 3, 1, 1)
        self.conv1 = ConvBnReLU(8, 8, 3, 1, 1)

        self.conv2 = ConvBnReLU(8, 16, 5, 2, 2)
        self.conv3 = ConvBnReLU(16, 16, 3, 1, 1)
        self.conv4 = ConvBnReLU(16, 16, 3, 1, 1)

        self.conv5 = ConvBnReLU(16, 32, 5, 2, 2)
        self.conv6 = ConvBnReLU(32, 32, 3, 1, 1)
        self.feature = nn.Conv2d(32, 32, 3, 1, 1)

    def forward(self, x):
        x = self.conv1(self.conv0(x))
        x = self.conv4(self.conv3(self.conv2(x)))
        x = self.feature(self.conv6(self.conv5(x)))
        return x


class CostRegNet(
    nn.Module
):  # TODO: Modify this code to accept arguments for setting the inchannels, outchannels, kernel size, stride, and padding. This can then be used in cvp_mvsnet implementation. Need an additional optional argument for configuring the use of relu or leaky relu on the convolutions.
    def __init__(self):
        super().__init__()
        self.conv0 = ConvBnReLU3D(32, 8)

        self.conv1 = ConvBnReLU3D(8, 16, stride=2)
        self.conv2 = ConvBnReLU3D(16, 16)

        self.conv3 = ConvBnReLU3D(16, 32, stride=2)
        self.conv4 = ConvBnReLU3D(32, 32)

        self.conv5 = ConvBnReLU3D(32, 64, stride=2)
        self.conv6 = ConvBnReLU3D(64, 64)

        self.conv7 = nn.Sequential(
            nn.ConvTranspose3d(
                64, 32, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False
            ),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )

        self.conv9 = nn.Sequential(
            nn.ConvTranspose3d(
                32, 16, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False
            ),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )

        self.conv11 = nn.Sequential(
            nn.ConvTranspose3d(
                16, 8, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False
            ),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True),
        )

        self.prob = nn.Conv3d(8, 1, 3, stride=1, padding=1)

    def forward(self, x):
        conv0 = self.conv0(x)
        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))
        x = self.conv6(self.conv5(conv4))
        x = conv4 + self.conv7(x)
        del conv4
        x = conv2 + self.conv9(x)
        del conv2
        x = conv0 + self.conv11(x)
        del conv0
        x = self.prob(x)
        return x


#################### Utils for mvsnet ####################
def homo_warp(
    src_feat, src_proj, ref_proj_inv, depth_values
):  # TODO: How to unite all homography warping fucntions.
    # src_feat: (B, C, H, W)
    # src_proj: (B, 4, 4)
    # ref_proj_inv: (B, 4, 4)
    # depth_values: (B, D)
    # out: (B, C, D, H, W)
    """This is a Python function that performs a homography warp on a given source feature map (src_feat) using the provided source and reference camera projection matrices (src_proj and ref_proj_inv), and a set of depth values (depth_values). The function returns a warped source feature map."""
    B, C, H, W = src_feat.shape
    D = depth_values.shape[1]
    device = src_feat.device
    dtype = src_feat.dtype
    # Compute the transformation matrix by multiplying the source and inverse reference camera projection matrices.
    transform = src_proj @ ref_proj_inv
    # Extract the rotation (R) and translation (T) matrices from the transformation matrix.
    R = transform[:, :3, :3]  # (B, 3, 3)
    T = transform[:, :3, 3:]  # (B, 3, 1)
    # create grid from the ref frame
    # Create a grid of points in the reference frame using create_meshgrid function.
    ref_grid = create_meshgrid(H, W, normalized_coordinates=False)  # (1, H, W, 2)
    ref_grid = ref_grid.to(device).to(dtype)
    # Reshape and expand the reference grid to match the input batch size.
    ref_grid = ref_grid.permute(0, 3, 1, 2)  # (1, 2, H, W)
    ref_grid = ref_grid.reshape(1, 2, H * W)  # (1, 2, H*W)
    ref_grid = ref_grid.expand(B, -1, -1)  # (B, 2, H*W)
    ref_grid = torch.cat((ref_grid, torch.ones_like(ref_grid[:, :1])), 1)  # (B, 3, H*W)
    # Compute the reference grid points in 3D by multiplying with the depth values.
    ref_grid_d = ref_grid.unsqueeze(2) * depth_values.view(B, 1, D, 1)  # (B, 3, D, H*W)
    ref_grid_d = ref_grid_d.view(B, 3, D * H * W)
    # Transform the 3D reference grid points to the source frame using the rotation and translation matrices.
    src_grid_d = R @ ref_grid_d + T  # (B, 3, D*H*W)
    del ref_grid_d, ref_grid, transform, R, T  # release (GPU) memory
    src_grid = src_grid_d[:, :2] / src_grid_d[:, -1:]  # divide by depth (B, 2, D*H*W)
    del src_grid_d
    # Normalize the 2D source grid points to the range of -1 to 1.
    src_grid[:, 0] = src_grid[:, 0] / ((W - 1) / 2) - 1  # scale to -1~1
    src_grid[:, 1] = src_grid[:, 1] / ((H - 1) / 2) - 1  # scale to -1~1
    src_grid = src_grid.permute(0, 2, 1)  # (B, D*H*W, 2)
    src_grid = src_grid.view(B, D, H * W, 2)

    # Perform a bilinear grid sampling on the source feature map using the transformed source grid points.
    warped_src_feat = F.grid_sample(
        src_feat, src_grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )  # (B, C, D, H*W)
    # Reshape the result to match the input dimensions.
    warped_src_feat = warped_src_feat.view(B, C, D, H, W)

    return warped_src_feat


def depth_regression(
    p, depth_values
):  # This is reused inthe implementation for cvp_mvsnet
    depth_values = depth_values.view(*depth_values.shape, 1, 1)
    depth = torch.sum(p * depth_values, 1)
    return depth
