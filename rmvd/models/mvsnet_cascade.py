import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pprint import pprint
from .registry import register_model
from .helpers import build_model_with_cfg

from .blocks.fpn import FeaturePyramidNet
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


class MVSNet_Cascade(nn.Module):
    def __init__(self):
        super().__init__()

        self.num_sampling_points = 128
        self.sampling_type = "linear_depth"
        self.levels = 3
        self.n_depths = [32, 16, 8]
        self.interval_ratios = [4, 2, 1]
        self.encoder = FeaturePyramidNet()
        self.corr_block = PlanesweepCorrelation(normalize=False, corr_type="warponly")
        self.fusion_block = VarianceCostvolumeFusion()
        fusion_enc_block_dict = {}
        fusion_dec_block_dict = {}
        for level in range(self.levels):
            fusion_enc_block_dict[
                f"fusion_enc_block_{level}"
            ] = MVSNetFusedCostvolumeEncoder(
                in_channels=8 * self.interval_ratios[level],
                base_channels=8,
                batch_norm=True,
            )
            fusion_dec_block_dict[f"fusion_dec_block_{level}"] = MVSNetDecoder(
                in_channels=64, batch_norm=True
            )

        for level in range(self.levels):
            cost_reg_network = nn.ModuleList(
                [
                    fusion_enc_block_dict[f"fusion_enc_block_{level}"],
                    fusion_dec_block_dict[f"fusion_dec_block_{level}"],
                ]
            )
            setattr(self, f"cost_reg_{level}", cost_reg_network)
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
        intrinsics_sources = exclude_index(intrinsics, keyview_idx)

        source_to_key_transforms = exclude_index(poses, keyview_idx)

        if depth_range is None:
            device = get_torch_model_device(self)
            N = images[0].shape[0]
            min_depth = torch.tensor([0.2] * N, dtype=torch.float32, device=device)
            max_depth = torch.tensor([100.0] * N, dtype=torch.float32, device=device)
            depth_range = [min_depth, max_depth]
        else:
            min_depth, max_depth = depth_range

        depth_interval = (max_depth - min_depth) / self.num_sampling_points

        print(f"image_key: {image_key.shape}") if verbose else None
        all_enc_key = self.encoder(image_key)
        pprint(
            {f"all_enc_key[{k}]": v.shape for k, v in all_enc_key.items()}
        ) if verbose else None
        del image_key
        print(f"images_source: {images_source[0].shape}") if verbose else None
        all_enc_sources = [self.encoder(image_source) for image_source in images_source]
        del images_source
        results = {}
        for level in range(self.levels):
            D = self.n_depths[level]
            feat_key = all_enc_key[f"level_{level}"]
            feat_sources = [
                all_enc_source[f"level_{level}"] for all_enc_source in all_enc_sources
            ]
            delta = D / 2 * self.interval_ratios[level] * depth_interval
            prev_level_depth = (
                results.get(f"depth_{level-1}").detach() if level > 0 else None
            )
            upscaled_prev_level_depth = (
                F.interpolate(
                    prev_level_depth,
                    scale_factor=2,
                    mode="bilinear",
                    align_corners=False,
                )
                if prev_level_depth is not None
                else None
            )
            current_level_intrinsics_key = intrinsics_key.clone()
            current_level_intrinsics_key[:, 0, 0] *= 2**level
            current_level_intrinsics_key[:, 1, 1] *= 2**level
            current_level_intrinsics_key[:, 0, 2] *= 2**level
            current_level_intrinsics_key[:, 1, 2] *= 2**level

            current_level_intrinsics_sources = []
            for intrinsics_source in intrinsics_sources:
                current_level_intrinscs_source = intrinsics_source.clone()
                current_level_intrinscs_source[:, 0, 0] *= 2**level
                current_level_intrinscs_source[:, 1, 1] *= 2**level
                current_level_intrinscs_source[:, 0, 2] *= 2**level
                current_level_intrinscs_source[:, 1, 2] *= 2**level
                current_level_intrinsics_sources.append(current_level_intrinscs_source)
            min_depth = min_depth if level == 0 else upscaled_prev_level_depth - delta
            max_depth = max_depth if level == 0 else upscaled_prev_level_depth + delta
            corrs, masks, sampling_invdepths = self.corr_block(
                feat_key=feat_key,
                intrinsics_key=current_level_intrinsics_key,
                feat_sources=feat_sources,
                source_to_key_transforms=source_to_key_transforms,
                intrinsics_sources=current_level_intrinsics_sources,
                num_sampling_points=D,
                sampling_type=self.sampling_type,
                min_depth=min_depth,
                max_depth=max_depth,
                interval_ratio=self.interval_ratios[level] if level == 0 else 1,
            )
            print(f"corrs_{level}: {corrs[0].shape}") if verbose else None
            print(f"masks_{level}: {masks[0].shape}") if verbose else None
            fused_corr, _ = self.fusion_block(
                feat_key=feat_key, corrs=corrs, masks=masks
            )
            del corrs, masks
            print(f"fused_corr: {fused_corr.shape}") if verbose else None
            all_enc_fused, enc_fused = getattr(self, f"cost_reg_{level}")[0](
                fused_corr=fused_corr
            )
            del fused_corr
            print(f"enc_fused: {enc_fused.shape}") if verbose else None
            pprint(
                {f"all_enc_fused[{k}]": v.shape for k, v in all_enc_fused.items()}
            ) if verbose else None

            dec = getattr(self, f"cost_reg_{level}")[1](
                enc_fused=enc_fused,
                sampling_invdepths=sampling_invdepths,
                all_enc=all_enc_fused,
            )
            del enc_fused, all_enc_fused, sampling_invdepths

            results[f"depth_{level}"] = dec["depth"]
            results[f"depth_uncertainty_{level}"] = dec["uncertainty"]
            results[f"aux_{level}"] = dec
            del dec

        aux = {}
        aux["depths_all"] = [
            results.get(f"depth_{level}") for level in range(self.levels)
        ]  # sorted from lowest to highest resolution
        pred = {
            "depth": results.get(f"depth_{2}"),
            "depth_uncertainty": results.get(f"depth_uncertainty_{2}"),
        }

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
def mvsnet_cascade(pretrained=True, weights=None, train=False, num_gpus=1, **kwargs):
    assert not (
        pretrained and weights is None
    ), "Pretrained weights are not available for this model."
    # weights = pretrained_weights if (pretrained and weights is None) else weights
    model = build_model_with_cfg(
        model_cls=MVSNet_Cascade,
        weights=weights,
        train=train,
        num_gpus=num_gpus,
    )
    return model
