import torch
import torch.nn as nn

from .utils import conv


class DispnetCostvolumeEncoder(nn.Module):
    def __init__(self, corr_type="full", in_channels=256):
        super().__init__()

        C_curr = in_channels
        self.conv3_1 = conv(C_curr + 32, C_curr)

        C_last = C_curr  # 256
        C_curr *= 2  # 512
        self.conv4 = conv(C_last, C_curr, stride=2)
        self.conv4_1 = conv(C_curr, C_curr)
        self.conv5 = conv(C_curr, C_curr, stride=2)
        self.conv5_1 = conv(C_curr, C_curr)

        C_last = C_curr  # 512
        C_curr *= 2  # 1024
        self.conv6 = conv(C_last, C_curr, stride=2)
        self.conv6_1 = conv(C_curr, C_curr)

    def forward(self, corr, ctx):
        merged = torch.cat(
            [ctx, corr], 1
        )  # ctx : torch.Size([1, 32, 48, 96]) corr: torch.Size([1, 256, 48, 96])
        conv3_1 = self.conv3_1(
            merged
        )  # merged: torch.Size([1, 288, 48, 96]) ; conv3_1: torch.Size([1, 256, 48, 96])

        conv4 = self.conv4(
            conv3_1
        )  # conv3_1: torch.Size([1, 256, 48, 96]) ; conv4: torch.Size([1, 512, 24, 48]) ; 2C, H/2, W/2
        conv4_1 = self.conv4_1(
            conv4
        )  # conv4: torch.Size([1, 512, 24, 48]) ; conv4_1: torch.Size([1, 512, 24, 48]) ;

        conv5 = self.conv5(
            conv4_1
        )  # conv4_1: torch.Size([1, 512, 24, 48]) ; conv5: torch.Size([1, 512, 12, 24])
        conv5_1 = self.conv5_1(
            conv5
        )  # conv5: torch.Size([1, 512, 12, 24]) ; conv5_1: torch.Size([1, 512, 12, 24])

        conv6 = self.conv6(
            conv5_1
        )  # conv5_1: torch.Size([1, 512, 12, 24]) ; conv6: torch.Size([1, 1024, 6, 12])
        conv6_1 = self.conv6_1(
            conv6
        )  # conv6: torch.Size([1, 1024, 6, 12]) ; conv6_1: torch.Size([1, 1024, 6, 12])

        all_enc = {
            "merged": merged,
            "conv3_1": conv3_1,
            "conv4": conv4,
            "conv4_1": conv4_1,
            "conv5": conv5,
            "conv5_1": conv5_1,
            "conv6": conv6,
            "conv6_1": conv6_1,
        }

        return all_enc, conv6_1
