import torch
import torch.nn as nn
import torch.nn.functional as F

verbose = False


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


class Deconv3d(nn.Module):
    """Applies a 3D deconvolution (optionally with batch normalization and relu activation)
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
        super(Deconv3d, self).__init__()
        self.out_channels = out_channels
        assert stride in [1, 2]
        self.stride = stride

        self.conv = nn.ConvTranspose3d(
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


class MVSNetDecoder(nn.Module):
    def __init__(self, in_channels=64, batch_norm=False, **kwargs):
        super().__init__()

        C_last = in_channels
        C_curr = int(C_last / 2)  # 4 * C
        self.conv7 = Deconv3d(
            C_last, C_curr, stride=2, padding=1, output_padding=1, bn=batch_norm
        )

        C_last = C_curr  # 4 * C
        C_curr = int(C_curr / 2)  # 2 * C
        self.conv9 = Deconv3d(
            C_last, C_curr, stride=2, padding=1, output_padding=1, bn=batch_norm
        )

        C_last = C_curr  # 2 * C
        C_curr = int(C_curr / 2)  # 1 * C
        self.conv11 = Deconv3d(
            C_last, C_curr, stride=2, padding=1, output_padding=1, bn=batch_norm
        )

        self.prob = nn.Conv3d(C_curr, 1, 3, stride=1, padding=1, bias=False)

    def forward(self, enc_fused, sampling_invdepths, all_enc):
        # sampling_invdepths has shape (N, S, H, W) or (N, S, 1, 1)
        steps = sampling_invdepths.shape[1]

        conv7_a = self.conv7(enc_fused)
        print("conv7_a.shape", conv7_a.shape) if verbose else None
        conv7 = all_enc["3d_conv4"] + conv7_a
        print("conv7.shape", conv7.shape) if verbose else None
        conv9_a = self.conv9(conv7)
        print("conv9_a.shape", conv9_a.shape) if verbose else None
        conv9 = all_enc["3d_conv2"] + conv9_a
        print("conv9.shape", conv9.shape) if verbose else None
        conv11_a = self.conv11(conv9)
        print("conv11_a.shape", conv11_a.shape) if verbose else None
        conv11 = all_enc["3d_conv0"] + conv11_a
        print("conv11.shape", conv11.shape) if verbose else None
        prob = self.prob(conv11)  # N1SHW
        prob = prob.squeeze(1)  # NSHW
        prob = F.softmax(prob, dim=1)  # NSHW
        pred_invdepth = torch.sum(
            prob * sampling_invdepths, dim=1, keepdim=True
        )  # N1HW
        pred_depth = 1 / (pred_invdepth + 1e-9)

        with torch.no_grad():
            # photometric confidence; not used in training, therefore no_grad is used
            # sum probability of 4 consecutive depth indices:
            prob_sum4 = 4 * F.avg_pool3d(
                F.pad(prob.unsqueeze(1), pad=(0, 0, 0, 0, 1, 2)),
                (4, 1, 1),
                stride=1,
                padding=0,
            ).squeeze(1)
            # find the (rounded) index that is the final prediction:
            d_indices = torch.arange(steps, device=prob.device, dtype=torch.float)
            d_indices = d_indices.view(1, -1, 1, 1)
            pred_idx = torch.sum(prob * d_indices, dim=1, keepdim=True).long()  # N1HW
            pred_idx = pred_idx.clamp(min=0, max=steps - 1)
            # # the confidence is the 4-sum probability at this index:
            pred_depth_confidence = torch.gather(prob_sum4, 1, pred_idx)

        all_dec = {
            "3d_conv7": conv7,
            "3d_conv9": conv9,
            "3d_conv11": conv11,
            "depth_values_prob_volume": prob,
            "depth": pred_depth,
            "invdepth": pred_invdepth,
            "uncertainty": 1 - pred_depth_confidence,
            "sampling_invdepths": sampling_invdepths,
        }

        return all_dec
