import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import ReLUAndSigmoid
from math import ceil


def iconv_block(in_planes, out_planes):
    return nn.Sequential(
        nn.Conv2d(
            in_planes + 2, out_planes, kernel_size=3, stride=1, padding=1, bias=True
        ),
        nn.LeakyReLU(0.2, inplace=True),
    )


def pred_block(in_planes, first):
    inc = 0 if first else 2
    return nn.Sequential(
        nn.Conv2d(in_planes + inc, 2, kernel_size=3, stride=1, padding=1, bias=True),
        ReLUAndSigmoid(inplace=True, min=-10, max=10),
    )


def deconv(in_planes, out_planes, first):
    inc = 0 if first else 2

    return nn.Sequential(
        nn.ConvTranspose2d(
            in_planes + inc, out_planes, kernel_size=4, stride=2, padding=1, bias=True
        ),
        nn.LeakyReLU(0.2, inplace=True),
    )


def threeD_conv(
    in_planes,
    out_planes,
    kernel_size=3,
    stride=1,
    padding=1,
    relu=True,
    bn=True,
    bn_momentum=0.1,
    init_method="xavier",
    **kwargs
):
    layers = []
    conv = nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size,
        stride=stride,
        padding=padding,
        bias=not bn,
    )
    layers.append(conv)

    if bn:
        batch_norm = nn.BatchNorm3d(out_planes, momentum=bn_momentum)
        layers.append(batch_norm)

    if relu:
        relu_layer = nn.LeakyReLU(0.2, inplace=True)
        layers.append(relu_layer)

    return nn.Sequential(*layers)


class DispnetDecoder3DDownSample(nn.Module):
    def __init__(self, arch: str = "dispnet"):
        super().__init__()
        self.arch = arch
        C_curr = 16
        self.threeDconv_0 = (
            threeD_conv(64, 1, relu=False, bn=False) if not arch == "dispnet" else None
        )
        # self.threeDconv_0 = (
        #     None
        #     if arch == "dispnet"
        #     else nn.Sequential(
        #         threeD_conv(64, 32, relu=False, bn=False),
        #         threeD_conv(32, 16, relu=False, bn=False),
        #         threeD_conv(16, 8, relu=False, bn=False),
        #         threeD_conv(8, 1, relu=False, bn=False),
        #     )
        # )
        self.pred_0 = pred_block(C_curr, first=True)

        C_last = C_curr
        C_curr = int(C_curr / 2)  # 512

        self.deconv_1 = deconv(C_last, C_curr, first=True)
        self.threeDconv_1 = (
            threeD_conv(32, 1, relu=False, bn=False) if not arch == "dispnet" else None
        )
        # self.threeDconv_1 = (
        #     None
        #     if arch == "dispnet"
        #     else nn.Sequential(
        #         threeD_conv(32, 16, relu=False, bn=False),
        #         threeD_conv(16, 8, relu=False, bn=False),
        #         threeD_conv(8, 1, relu=False, bn=False),
        #     )
        # )
        self.rfeat1 = iconv_block(C_curr + (512 if arch == "dispnet" else 32), C_curr)

        self.pred_1 = pred_block(C_curr, first=True)

        C_last = C_curr  # 512
        C_curr = ceil(C_curr / 2)  # 256

        self.deconv_2 = deconv(C_last, C_curr, first=True)
        self.threeDconv_2 = (
            threeD_conv(16, 1, relu=False, bn=False) if not arch == "dispnet" else None
        )
        # self.threeDconv_2 = (
        #     None
        #     if arch == "dispnet"
        #     else nn.Sequential(
        #         threeD_conv(16, 8, relu=False, bn=False),
        #         threeD_conv(8, 1, relu=False, bn=False),
        #     )
        # )
        self.rfeat2 = iconv_block(C_curr + (512 if arch == "dispnet" else 64), C_curr)
        self.pred_2 = pred_block(C_curr, first=True)

        C_last = C_curr  # 256
        C_curr = ceil(C_curr / 2)  # 128

        self.deconv_3 = deconv(C_last, C_curr, first=True)
        self.threeDconv_3 = (
            None if arch == "dispnet" else threeD_conv(8, 1, relu=False, bn=False)
        )
        self.rfeat3 = iconv_block(C_curr + (256 if arch == "dispnet" else 128), C_curr)
        self.pred_3 = pred_block(C_curr, first=True)

        C_last = C_curr  # 128
        C_curr = ceil(C_curr / 2)  # 64

        self.deconv_4 = deconv(C_last, C_curr, first=True)
        self.rfeat4 = iconv_block(C_curr + (128 if arch == "dispnet" else 16), C_curr)
        self.pred_4 = pred_block(C_curr, first=True)

        C_last = C_curr  # 64
        C_curr = ceil(C_curr / 2)  # 32

        self.deconv_5 = deconv(C_last, C_curr, first=True)
        self.rfeat5 = iconv_block(C_curr + (64 if arch == "dispnet" else 8), C_curr)
        self.pred_5 = pred_block(C_curr, first=True)

    def forward(self, enc_fused, sampling_invdepths, all_enc):
        preds = {}

        enc_fused = (
            self.threeDconv_0(enc_fused) if hasattr(self, "threeDconv_0") else enc_fused
        )
        # print("enc_fused_post_downsample", enc_fused.shape)
        enc_fused = enc_fused.squeeze(1)
        # enc_fused = (
        #     enc_fused
        #     if len(enc_fused.shape) == 4
        #     else enc_fused.reshape(
        #         enc_fused.shape[0],
        #         enc_fused.shape[1] * enc_fused.shape[2],
        #         enc_fused.shape[3],
        #         enc_fused.shape[4],
        #     )
        # )
        # print("enc_fused_post_reshape", enc_fused.shape)
        pred_0 = self.pred_0(enc_fused)  # >= 0
        # print("pred_0", pred_0.shape)
        self.add_outputs(pred=pred_0, preds=preds)

        deconv_1 = self.deconv_1(enc_fused)
        # print("deconv_1", deconv_1.shape)
        pred_0_up = F.interpolate(
            pred_0, size=deconv_1.shape[-2:], mode="bilinear", align_corners=False
        ).detach()
        # print("pred_0_up", pred_0_up.shape)
        skip_5 = (
            all_enc.get("conv5_1")
            if all_enc.get("conv5_1") is not None
            else all_enc.get("3d_conv4")
        )
        # print("skip_5", skip_5.shape)
        skip_5 = self.threeDconv_1(skip_5) if hasattr(self, "threeDconv_1") else skip_5
        # print("skip_5_post_downsample", skip_5.shape)
        skip_5 = skip_5.squeeze(1)
        # print("skip_5_post_reshape", skip_5.shape)
        # skip_5 = (
        #     skip_5
        #     if len(skip_5.shape) == 4
        #     else skip_5.reshape(
        #         skip_5.shape[0],
        #         skip_5.shape[1] * skip_5.shape[2],
        #         skip_5.shape[3],
        #         skip_5.shape[4],
        #     )
        # )
        skip_5_up = F.interpolate(
            skip_5, size=deconv_1.shape[-2:], mode="bilinear", align_corners=False
        )
        # print("skip_5_up", skip_5_up.shape)
        rfeat1 = self.rfeat1(torch.cat((skip_5_up, deconv_1, pred_0_up), 1))
        # print("rfeat1", rfeat1.shape)
        pred_1 = self.pred_1(rfeat1)
        # print("pred_1", pred_1.shape)
        self.add_outputs(pred=pred_1, preds=preds)

        deconv_2 = self.deconv_2(rfeat1)
        # print("deconv_2", deconv_2.shape)
        pred_1_up = F.interpolate(
            pred_1, size=deconv_2.shape[-2:], mode="bilinear", align_corners=False
        ).detach()
        # print("pred_1_up", pred_1_up.shape)
        skip_4 = (
            all_enc.get("conv4_1")
            if all_enc.get("conv4_1") is not None
            else all_enc.get("3d_conv2")
        )
        # print("skip_4", skip_4.shape)
        skip_4 = self.threeDconv_2(skip_4) if hasattr(self, "threeDconv_2") else skip_4
        # print("skip_4_post_downsample", skip_4.shape)
        skip_4 = skip_4.squeeze(1)
        # print("skip_4_post_reshape", skip_4.shape)
        # skip_4 = (
        #     skip_4
        #     if len(skip_4.shape) == 4
        #     else skip_4.reshape(
        #         skip_4.shape[0],
        #         skip_4.shape[1] * skip_4.shape[2],
        #         skip_4.shape[3],
        #         skip_4.shape[4],
        #     )
        # )
        skip_4_up = F.interpolate(
            skip_4, size=deconv_2.shape[-2:], mode="bilinear", align_corners=False
        )
        # print("skip_4_up", skip_4_up.shape)
        rfeat2 = self.rfeat2(torch.cat((skip_4_up, deconv_2, pred_1_up), 1))
        # print("rfeat2", rfeat2.shape)
        pred_2 = self.pred_2(rfeat2)
        # print("pred_2", pred_2.shape)
        self.add_outputs(pred=pred_2, preds=preds)

        deconv_3 = self.deconv_3(rfeat2)
        # print("deconv_3", deconv_3.shape)
        pred_2_up = F.interpolate(
            pred_2, size=deconv_3.shape[-2:], mode="bilinear", align_corners=False
        ).detach()
        # print("pred_2_up", pred_2_up.shape)
        skip_3 = (
            all_enc.get("conv3_1")
            if all_enc.get("conv3_1") is not None
            else all_enc.get("3d_conv0")
        )
        # print("skip_3", skip_3.shape)
        skip_3 = self.threeDconv_3(skip_3) if hasattr(self, "threeDconv_3") else skip_3
        # print("skip_3_post_downsample", skip_3.shape)
        skip_3 = skip_3.squeeze(1)
        # print("skip_3_post_reshape", skip_3.shape)
        # skip_3 = (
        #     skip_3
        #     if len(skip_3.shape) == 4
        #     else skip_3.reshape(
        #         skip_3.shape[0],
        #         skip_3.shape[1] * skip_3.shape[2],
        #         skip_3.shape[3],
        #         skip_3.shape[4],
        #     )
        # )
        skip_3_up = F.interpolate(
            skip_3, size=deconv_3.shape[-2:], mode="bilinear", align_corners=False
        )
        # print("skip_3_up", skip_3_up.shape)
        rfeat3 = self.rfeat3(torch.cat((skip_3_up, deconv_3, pred_2_up), 1))
        # print("rfeat3", rfeat3.shape)
        pred_3 = self.pred_3(rfeat3)
        # print("pred_3", pred_3.shape)
        self.add_outputs(pred=pred_3, preds=preds)

        deconv_4 = self.deconv_4(rfeat3)
        # print("deconv_4", deconv_4.shape)
        pred_3_up = F.interpolate(
            pred_3, size=deconv_4.shape[-2:], mode="bilinear", align_corners=False
        ).detach()
        # print("pred_3_up", pred_3_up.shape)
        skip_2 = all_enc["conv2"]
        # print("skip_2", skip_2.shape)
        rfeat4 = self.rfeat4(torch.cat((skip_2, deconv_4, pred_3_up), 1))
        # print("rfeat4", rfeat4.shape)
        pred_4 = self.pred_4(rfeat4)
        # print("pred_4", pred_4.shape)
        self.add_outputs(pred=pred_4, preds=preds)

        deconv_5 = self.deconv_5(rfeat4)
        # print("deconv_5", deconv_5.shape)
        pred_4_up = F.interpolate(
            pred_4, size=deconv_5.shape[-2:], mode="bilinear", align_corners=False
        ).detach()
        # print("pred_4_up", pred_4_up.shape)
        skip_1 = all_enc["conv1"]
        # print("skip_1", skip_1.shape)
        rfeat5 = self.rfeat5(torch.cat((skip_1, deconv_5, pred_4_up), 1))
        # print("rfeat5", rfeat5.shape)
        pred_5 = self.pred_5(rfeat5)
        # print("pred_5", pred_5.shape)
        self.add_outputs(pred=pred_5, preds=preds)

        # if self.arch != "dispnet":
        #     prob = preds["invdepth"]
        #     prob = F.softmax(prob, dim=1)  # NSHW
        #     pred_invdepth = torch.sum(
        #         prob * sampling_invdepths, dim=1, keepdim=True
        #     )  # N1HW
        #     preds["invdepth"] = pred_invdepth

        return preds

    def add_outputs(self, pred, preds):
        pred_mean = pred[:, 0:1, :, :]
        pred_invdepth_log_b = pred[:, 1:2, :, :]
        pred_invdepth_b = torch.exp(pred_invdepth_log_b)
        pred_invdepth_ent = torch.log(2 * pred_invdepth_b + 1e-4) + 1

        preds.setdefault("invdepth_uncertainties_all", []).append(pred_invdepth_ent)
        preds.setdefault("invdepth_log_bs_all", []).append(pred_invdepth_log_b)
        preds.setdefault("invdepths_all", []).append(pred_mean)

        preds["invdepth_uncertainty"] = pred_invdepth_ent
        preds["invdepth_log_b"] = pred_invdepth_log_b
        preds["invdepth"] = pred_mean
