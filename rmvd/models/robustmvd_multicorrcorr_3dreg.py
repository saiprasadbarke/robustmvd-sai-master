import math

import torch
import torch.nn as nn
import numpy as np

from .registry import register_model
from .helpers import build_model_with_cfg
from .blocks.dispnet_context_encoder import DispnetContextEncoder
from .blocks.dispnet_encoder import DispnetEncoder
from .blocks.planesweep_corr import PlanesweepCorrelation
from .blocks.learned_fusion_3d import LearnedFusion3D
from .blocks.learned_fusion import LearnedFusion
from .blocks.mvsnet_fused_costvolume_encoder import MVSNetFusedCostvolumeEncoder
from .blocks.mvsnet_decoder import MVSNetDecoder

from rmvd.utils import (
    get_torch_model_device,
    to_numpy,
    to_torch,
    select_by_index,
    exclude_index,
)
from rmvd.data.transforms import ResizeInputs

verbose = False


class RobustMVDGroupWiseCorr(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_groups = 32
        self.num_sampling_points = 256
        self.sampling_type = "linear_depth"
        self.encoder = DispnetEncoder()
        self.context_encoder = DispnetContextEncoder()
        self.corr_block_groupwise = PlanesweepCorrelation(
            corr_type="groupwise_5D", normalize=False, num_groups=self.num_groups
        )
        self.corr_block_fullcorr = PlanesweepCorrelation()
        self.fusion_block_groupwise = LearnedFusion3D()
        self.fusion_block_fullcorr = LearnedFusion()
        self.fusion_enc_block = MVSNetFusedCostvolumeEncoder(
            in_channels=(self.num_groups + 1 + 32)
        )
        self.decoder = MVSNetDecoder()

        self.init_weights()

    def init_weights(self):
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

    def forward(self, images, poses, intrinsics, keyview_idx, **_):
        image_key = select_by_index(images, keyview_idx)
        images_source = exclude_index(images, keyview_idx)

        intrinsics_key = select_by_index(intrinsics, keyview_idx)
        intrinsics_source = exclude_index(intrinsics, keyview_idx)

        source_to_key_transforms = exclude_index(poses, keyview_idx)
        print(f"images_key: {image_key.shape}") if verbose else None
        all_enc_key, enc_key = self.encoder(image_key)
        del image_key
        print(f"enc_key: {enc_key.shape}") if verbose else None
        print(
            {f"all_enc_key[{k}]": v.shape for k, v in all_enc_key.items()}
        ) if verbose else None
        print(f"images_source: {images_source[0].shape}") if verbose else None
        enc_sources = [self.encoder(image_source)[1] for image_source in images_source]
        print(f"enc_sources: {enc_sources[0].shape}") if verbose else None
        ctx = self.context_encoder(enc_key)
        print(f"ctx: {ctx.shape}") if verbose else None

        (
            corrs_groupwise,
            masks_groupwise,
            sampling_invdepths,
        ) = self.corr_block_groupwise(
            feat_key=enc_key,
            intrinsics_key=intrinsics_key,
            feat_sources=enc_sources,
            source_to_key_transforms=source_to_key_transforms,
            intrinsics_sources=intrinsics_source,
            num_sampling_points=self.num_sampling_points,
            min_depth=0.4,
            max_depth=1000.0,
            sampling_type=self.sampling_type,
        )
        corrs_full, masks_full, _ = self.corr_block_fullcorr(
            feat_key=enc_key,
            intrinsics_key=intrinsics_key,
            feat_sources=enc_sources,
            source_to_key_transforms=source_to_key_transforms,
            intrinsics_sources=intrinsics_source,
            num_sampling_points=self.num_sampling_points,
            min_depth=0.4,
            max_depth=1000.0,
            sampling_type=self.sampling_type,
        )
        del enc_sources
        print(f"corrs_groupwise: {corrs_groupwise[0].shape}") if verbose else None
        print(f"masks_groupwise: {masks_groupwise[0].shape}") if verbose else None
        fused_corr_groupwise, _ = self.fusion_block_groupwise(
            corrs=corrs_groupwise, masks=masks_groupwise
        )
        print(
            f"fused_corr_groupwise: {fused_corr_groupwise.shape}"
        ) if verbose else None
        del enc_key, corrs_groupwise, masks_groupwise
        fused_corr_fullcorr, _ = self.fusion_block_fullcorr(
            corrs=corrs_full, masks=masks_full
        )
        del corrs_full, masks_full
        fused_corr_fullcorr = fused_corr_fullcorr.unsqueeze(1)
        ctx = ctx.unsqueeze(2).repeat(1, 1, self.num_sampling_points, 1, 1)
        fused_corr = torch.cat([fused_corr_groupwise, fused_corr_fullcorr, ctx], dim=1)
        fused_corr = fused_corr.permute(0, 2, 1, 3, 4)
        del fused_corr_groupwise, fused_corr_fullcorr, ctx
        all_enc_fused, enc_fused = self.fusion_enc_block(fused_corr=fused_corr)
        print(f"enc_fused: {enc_fused.shape}") if verbose else None
        print(
            {f"all_enc_fused[{k}]": v.shape for k, v in all_enc_fused.items()}
        ) if verbose else None
        dec = self.decoder(
            enc_fused=enc_fused,
            sampling_invdepths=sampling_invdepths,
            all_enc={**all_enc_key, **all_enc_fused},
        )

        pred = {
            "depth": dec["depth"],
            "depth_uncertainty": dec["uncertainty"],
        }

        aux = dec

        return pred, aux

    def input_adapter(self, images, keyview_idx, poses, intrinsics, **_):
        device = get_torch_model_device(self)

        orig_ht, orig_wd = images[0].shape[-2:]
        ht, wd = int(math.ceil(orig_ht / 64.0) * 64.0), int(
            math.ceil(orig_wd / 64.0) * 64.0
        )
        if (orig_ht != ht) or (orig_wd != wd):
            resized = ResizeInputs(size=(ht, wd))(
                {"images": images, "intrinsics": intrinsics}
            )
            images = resized["images"]
            intrinsics = resized["intrinsics"]

        # normalize images
        images = [image / 255.0 - 0.4 for image in images]

        # model works with relative intrinsics:
        scale_arr = np.array([[wd] * 3, [ht] * 3, [1.0] * 3], dtype=np.float32)  # 3, 3
        intrinsics = [intrinsic / scale_arr for intrinsic in intrinsics]

        images, keyview_idx, poses, intrinsics = to_torch(
            (images, keyview_idx, poses, intrinsics), device=device
        )

        sample = {
            "images": images,
            "keyview_idx": keyview_idx,
            "poses": poses,
            "intrinsics": intrinsics,
        }
        return sample

    def output_adapter(self, model_output):
        pred, aux = model_output
        return to_numpy(pred), to_numpy(aux)


@register_model
def robustmvd_multicorrcorr_3dreg(
    pretrained=True, weights=None, train=False, num_gpus=1, **kwargs
):
    model = build_model_with_cfg(
        model_cls=RobustMVDGroupWiseCorr,
        weights=weights,
        train=train,
        num_gpus=num_gpus,
    )
    return model
