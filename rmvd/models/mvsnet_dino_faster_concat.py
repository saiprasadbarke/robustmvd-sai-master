import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as transforms
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
from .blocks.vit_modules import VITDecoderStage4Single
import sys

sys.path.insert(0, "/home/barkes/robustmvd-internal-1/dino-vit-features")
from extractor import ViTExtractor as DINOExtractor

verbose = False


class MVSNet_DINO(nn.Module):
    def __init__(self):
        super().__init__()

        base_channels = 8

        self.num_sampling_points = 128
        self.sampling_type = "linear_depth"
        self.dino_stride = 8
        self.encoder = MVSNetEncoder(base_channels=base_channels)

        self.dino_extractor = DINOExtractor(
            model_type="dino_vits8", device="cuda:1", stride=self.dino_stride
        )
        self.decoder_vit = VITDecoderStage4Single(mode="concat")
        self.corr_block = PlanesweepCorrelation(normalize=False, corr_type="warponly")
        self.fusion_block = VarianceCostvolumeFusion()
        self.fusion_enc_block = MVSNetFusedCostvolumeEncoder(
            in_channels=96, base_channels=8, batch_norm=True
        )  # 96 = 64 + 32

        self.decoder = MVSNetDecoder(in_channels=64)

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
        N, _, H, W = images[0].shape
        vit_h, vit_w = int(H / 2), int(W / 2)
        print(f"images_key: {image_key.shape}") if verbose else None
        all_enc_key, enc_key = self.encoder(image_key)
        print(f"enc_key: {enc_key.shape}") if verbose else None
        print(
            {f"all_enc_key[{k}]": v.shape for k, v in all_enc_key.items()}
        ) if verbose else None
        print(f"images_source: {images_source[0].shape}") if verbose else None
        enc_sources = [self.encoder(image_source)[1] for image_source in images_source]
        print(f"enc_sources: {enc_sources[0].shape}") if verbose else None

        downscale_half_vit = transforms.Resize((vit_h, vit_w), antialias=True)
        image_key = downscale_half_vit(image_key)
        images_source = [
            downscale_half_vit(image_source) for image_source in images_source
        ]

        image_key = image_key.to("cuda:1")
        images_source = [image_source.to("cuda:1") for image_source in images_source]
        # DINO features
        with torch.no_grad():
            dino_features_key = self.dino_extractor.extract_descriptors(image_key)
            dino_features_sources = [
                self.dino_extractor.extract_descriptors(image_source)
                for image_source in images_source
            ]
            dino_saliency_key = self.dino_extractor.extract_saliency_maps(image_key)
            dino_saliency_sources = [
                self.dino_extractor.extract_saliency_maps(image_source)
                for image_source in images_source
            ]
        dino_features_key = (
            dino_features_key.reshape(
                N, vit_h // (self.dino_stride), vit_w // (self.dino_stride), -1
            )
            .permute(0, 3, 1, 2)
            .cuda(0)
        )
        dino_saliency_key = (
            dino_saliency_key.reshape(
                N, vit_h // (self.dino_stride), vit_w // (self.dino_stride), -1
            )
            .permute(0, 3, 1, 2)
            .cuda(0)
        )

        dino_features_sources = [
            dino_features_source.reshape(
                N, vit_h // (self.dino_stride), vit_w // (self.dino_stride), -1
            )
            .permute(0, 3, 1, 2)
            .cuda(0)
            for dino_features_source in dino_features_sources
        ]

        dino_saliency_sources = [
            dino_saliency_source.reshape(
                N, vit_h // (self.dino_stride), vit_w // (self.dino_stride), -1
            )
            .permute(0, 3, 1, 2)
            .cuda(0)
            for dino_saliency_source in dino_saliency_sources
        ]

        # DINO features

        dino_feat_key_attn = self.decoder_vit(dino_features_key, dino_saliency_key)
        dino_feat_sources_attn = [
            self.decoder_vit(dino_features_source, dino_saliency_source)
            for dino_features_source, dino_saliency_source in zip(
                dino_features_sources, dino_saliency_sources
            )
        ]

        concat_feat_key = torch.cat([enc_key, dino_feat_key_attn], dim=1)
        concat_feat_sources = [
            torch.cat([enc_source, dino_feat_source_attn], dim=1)
            for enc_source, dino_feat_source_attn in zip(
                enc_sources, dino_feat_sources_attn
            )
        ]

        del (
            dino_feat_key_attn,
            dino_feat_sources_attn,
            dino_features_key,
            dino_features_sources,
            dino_saliency_key,
            dino_saliency_sources,
            all_enc_key,
            enc_key,
            enc_sources,
        )
        corrs, masks, sampling_invdepths = self.corr_block(
            feat_key=concat_feat_key,
            intrinsics_key=intrinsics_key,
            feat_sources=concat_feat_sources,
            source_to_key_transforms=source_to_key_transforms,
            intrinsics_sources=intrinsics_source,
            num_sampling_points=self.num_sampling_points,
            sampling_type=self.sampling_type,
            min_depth=min_depth,
            max_depth=max_depth,
        )
        del concat_feat_sources
        fused_corr, _ = self.fusion_block(
            feat_key=concat_feat_key, corrs=corrs, masks=masks
        )
        print(f"fused_corr: {fused_corr.shape}") if verbose else None
        del corrs, masks, concat_feat_key
        all_enc_fused, enc_fused = self.fusion_enc_block(fused_corr=fused_corr)
        del fused_corr
        print(f"enc_fused: {enc_fused.shape}") if verbose else None
        print(
            {f"all_enc_fused[{k}]": v.shape for k, v in all_enc_fused.items()}
        ) if verbose else None
        dec = self.decoder(
            enc_fused=enc_fused,
            sampling_invdepths=sampling_invdepths,
            all_enc=all_enc_fused,
        )
        del enc_fused, all_enc_fused
        pred = {
            "depth": dec["depth"],
            "depth_uncertainty": dec["uncertainty"],
        }

        aux = dec
        del dec

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
def mvsnet_dino_faster_concat(
    pretrained=True, weights=None, train=False, num_gpus=1, **kwargs
):
    assert not (
        pretrained and weights is None
    ), "Pretrained weights are not available for this model."
    # weights = pretrained_weights if (pretrained and weights is None) else weights
    model = build_model_with_cfg(
        model_cls=MVSNet_DINO,
        weights=weights,
        train=train,
        num_gpus=num_gpus,
    )
    return model
