import math

import torch
import torch.nn as nn
import numpy as np

from .registry import register_model
from .helpers import build_model_with_cfg
from .blocks.mvsnet_encoder import MVSNetEncoder
from .blocks.planesweep_corr import PlanesweepCorrelation
from .blocks.variance_costvolume_fusion import VarianceCostvolumeFusion
from .blocks.mvsnet_fused_costvolume_encoder import MVSNetFusedCostvolumeEncoder
from .blocks.mvsnet_decoder import MVSNetDecoder
from rmvd.utils import (
    get_torch_model_device,
    to_numpy,
    to_torch,
    select_by_index,
    exclude_index,
)
from rmvd.data.transforms import (
    UpscaleInputsToNextMultipleOf,
    NormalizeIntrinsics,
    NormalizeImagesToMinMax,
    NormalizeImagesByShiftAndScale,
)

verbose = False


class MVSNet_stackedhourglass(nn.Module):
    def __init__(self, num_sampling_points=128, sampling_type="linear_depth"):
        super().__init__()

        base_channels = 8
        self.num_sampling_points = num_sampling_points
        self.sampling_type = sampling_type
        self.encoder = MVSNetEncoder(base_channels=base_channels)
        self.corr_block = PlanesweepCorrelation(normalize=False, corr_type="warponly")
        self.fusion_block = VarianceCostvolumeFusion()
        self.hourglass1 = nn.ModuleList(
            [
                MVSNetFusedCostvolumeEncoder(
                    in_channels=32, base_channels=8, batch_norm=True
                ),
                MVSNetDecoder(in_channels=64),
            ]
        )
        self.hourglass2 = nn.ModuleList(
            [
                MVSNetFusedCostvolumeEncoder(
                    in_channels=8, base_channels=8, batch_norm=True
                ),
                MVSNetDecoder(in_channels=64),
            ]
        )
        self.hourglass3 = nn.ModuleList(
            [
                MVSNetFusedCostvolumeEncoder(
                    in_channels=8, base_channels=8, batch_norm=True
                ),
                MVSNetDecoder(in_channels=64),
            ]
        )
        self.init_weights()

    def init_weights(self):  # TODO
        for m in self.modules():
            if (
                isinstance(m, nn.Conv2d)
                or isinstance(m, nn.ConvTranspose2d)
                or isinstance(m, nn.Conv3d)
                or isinstance(m, nn.ConvTranspose3d)
            ):
                if m.weight is not None:
                    nn.init.kaiming_normal_(m.weight, a=0.2, nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, images, poses, intrinsics, keyview_idx, depth_range=None, **_):
        image_key = select_by_index(images, keyview_idx)
        images_source = exclude_index(images, keyview_idx)

        intrinsics_key = select_by_index(intrinsics, keyview_idx)
        intrinsics_source = exclude_index(intrinsics, keyview_idx)

        source_to_key_transforms = exclude_index(poses, keyview_idx)

        if depth_range is None:
            device = get_torch_model_device(self)
            N = images[0].shape[0]
            min_depth = torch.tensor([0.2] * N, dtype=torch.float32, device=device)
            max_depth = torch.tensor([100.0] * N, dtype=torch.float32, device=device)
            depth_range = [min_depth, max_depth]
        else:
            min_depth, max_depth = depth_range

        print(f"images_key: {image_key.shape}") if verbose else None
        all_enc_key, enc_key = self.encoder(image_key)
        print(f"enc_key: {enc_key.shape}") if verbose else None
        print(
            {f"all_enc_key[{k}]": v.shape for k, v in all_enc_key.items()}
        ) if verbose else None
        print(f"images_source: {images_source[0].shape}") if verbose else None
        enc_sources = [self.encoder(image_source)[1] for image_source in images_source]
        print(f"enc_sources: {enc_sources[0].shape}") if verbose else None
        corrs, masks, sampling_invdepths = self.corr_block(
            feat_key=enc_key,
            intrinsics_key=intrinsics_key,
            feat_sources=enc_sources,
            source_to_key_transforms=source_to_key_transforms,
            intrinsics_sources=intrinsics_source,
            num_sampling_points=self.num_sampling_points,
            sampling_type=self.sampling_type,
            min_depth=min_depth,
            max_depth=max_depth,
        )

        fused_corr, _ = self.fusion_block(feat_key=enc_key, corrs=corrs, masks=masks)
        print(f"fused_corr: {fused_corr.shape}") if verbose else None

        ############################Stacked Hourglass############################
        all_enc_fused_1, enc_fused_1 = self.hourglass1[0](fused_corr=fused_corr)
        dec_1 = self.hourglass1[1](
            enc_fused=enc_fused_1,
            sampling_invdepths=sampling_invdepths,
            all_enc=all_enc_fused_1,
        )
        out_1 = dec_1["3d_conv11"].permute(0, 2, 1, 3, 4)
        all_enc_fused_2, enc_fused_2 = self.hourglass2[0](fused_corr=out_1)
        dec_2 = self.hourglass2[1](
            enc_fused=enc_fused_2,
            sampling_invdepths=sampling_invdepths,
            all_enc=all_enc_fused_2,
        )
        out_2 = dec_2["3d_conv11"].permute(0, 2, 1, 3, 4)
        all_enc_fused_3, enc_fused_3 = self.hourglass2[0](fused_corr=out_2)
        dec_3 = self.hourglass2[1](
            enc_fused=enc_fused_3,
            sampling_invdepths=sampling_invdepths,
            all_enc=all_enc_fused_3,
        )

        ##############################Output####################################
        pred = {
            "depth": dec_3["depth"],
            "depth_uncertainty": dec_3["uncertainty"],
            "invdepth": dec_3["invdepth"],
        }

        depths_all = {
            "depths_all": [
                dec_1["depth"],
                dec_2["depth"],
                dec_3["depth"],
            ]
        }
        depth_uncertainties_all = {
            "depth_uncertainties_all": [
                dec_1["uncertainty"],
                dec_2["uncertainty"],
                dec_3["uncertainty"],
            ]
        }
        invdepths_all = {
            "invdepths_all": [
                dec_1["invdepth"],
                dec_2["invdepth"],
                dec_3["invdepth"],
            ]
        }
        aux = {**dec_3, **depths_all, **depth_uncertainties_all, **invdepths_all}

        return pred, aux

    def input_adapter(self, images, keyview_idx, poses, intrinsics, depth_range=None):
        device = get_torch_model_device(self)

        resized = UpscaleInputsToNextMultipleOf(32)(
            {"images": images, "intrinsics": intrinsics}
        )
        resized = NormalizeIntrinsics()(resized)
        resized = NormalizeImagesToMinMax(min_val=0.0, max_val=1.0)(resized)
        resized = NormalizeImagesByShiftAndScale(
            shift=[0.485, 0.456, 0.406], scale=[0.229, 0.224, 0.225]
        )(resized)
        images = resized["images"]
        intrinsics = resized["intrinsics"]

        images, keyview_idx, poses, intrinsics, depth_range = to_torch(
            (images, keyview_idx, poses, intrinsics, depth_range), device=device
        )

        sample = {
            "images": images,
            "keyview_idx": keyview_idx,
            "poses": poses,
            "intrinsics": intrinsics,
            "depth_range": depth_range,
        }
        return sample

    def output_adapter(self, model_output):
        pred, aux = model_output
        return to_numpy(pred), to_numpy(aux)


@register_model
def mvsnet_stackedhourglass(
    pretrained=True, weights=None, train=False, num_gpus=1, **kwargs
):
    assert not (
        pretrained and weights is None
    ), "Pretrained weights are not available for this model."
    # weights = pretrained_weights if (pretrained and weights is None) else weights
    model = build_model_with_cfg(
        model_cls=MVSNet_stackedhourglass,
        weights=weights,
        train=train,
        num_gpus=num_gpus,
    )
    return model


def convbn_3d(in_channels, out_channels, kernel_size, stride, pad):
    return nn.Sequential(
        nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=pad,
            bias=False,
        ),
        nn.BatchNorm3d(out_channels),
    )


def disparity_regression(x, maxdisp):
    assert len(x.shape) == 4
    disp_values = torch.arange(0, maxdisp, dtype=x.dtype, device=x.device)
    disp_values = disp_values.view(1, maxdisp, 1, 1)
    return torch.sum(x * disp_values, 1, keepdim=False)
