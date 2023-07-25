import math

import torch
import torch.nn as nn
import numpy as np

from .registry import register_model
from .helpers import build_model_with_cfg
from .blocks.mvsnet_encoder import MVSNetEncoder as FeatEncoder
from .blocks.planesweep_corr import PlanesweepCorrelation as CorrBlock
from .blocks.variance_costvolume_fusion import (
    VarianceCostvolumeFusion as CostvolumeFusion,
)
from .blocks.mvsnet_fused_costvolume_encoder import (
    MVSNetFusedCostvolumeEncoder as CostvolumeEncoder,
)
from .blocks.mvsnet_decoder import MVSNetDecoder as CostvolumeDecoder

from rmvd.utils import (
    get_torch_model_device,
    to_numpy,
    to_torch,
    select_by_index,
    exclude_index,
)
from rmvd.data.transforms import ResizeInputs
from rmvd.data.transforms import (
    UpscaleInputsToNextMultipleOf,
    NormalizeIntrinsics,
    NormalizeImagesToMinMax,
    NormalizeImagesByShiftAndScale,
)

verbose = False


class MVSnetGroupWiseCorr(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_sampling_points = 128
        self.feat_encoder = FeatEncoder()
        self.corr_block_groupwise = CorrBlock(
            corr_type="groupwise_mvsnet", normalize=False
        )

        self.fusion_block = CostvolumeFusion()
        self.fusion_enc_block = CostvolumeEncoder()
        self.decoder = CostvolumeDecoder()

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

    def forward(self, images, poses, intrinsics, keyview_idx, depth_range, **_):
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
        all_enc_key, ctx_key = self.feat_encoder(image_key)
        print(
            {f"all_enc_key[{k}]": v.shape for k, v in all_enc_key.items()}
        ) if verbose else None
        print(f"ctx: {ctx_key.shape}") if verbose else None
        enc_final_key = all_enc_key["conv3"]
        del image_key
        print(f"enc_final: {enc_final_key.shape}") if verbose else None

        print(f"images_source: {images_source[0].shape}") if verbose else None
        enc_final_sources = []
        ctx_sources = []
        for image_source in images_source:
            all_enc_source, ctx_source = self.feat_encoder(image_source)
            enc_final_sources.append(all_enc_source["conv3"])
            ctx_sources.append(ctx_source)
        del images_source
        print(f"enc_sources: {enc_final_sources[0].shape}") if verbose else None
        print(f"ctx_sources: {ctx_sources[0].shape}") if verbose else None

        (
            corrs_groupwise,
            masks_groupwise,
            sampling_invdepths,
        ) = self.corr_block_groupwise(
            feat_key=enc_final_key,
            intrinsics_key=intrinsics_key,
            feat_sources=enc_final_sources,
            source_to_key_transforms=source_to_key_transforms,
            intrinsics_sources=intrinsics_source,
            num_sampling_points=self.num_sampling_points,
            min_depth=min_depth,
            max_depth=max_depth,
            sampling_type="linear_depth",
        )
        del (
            enc_final_key,
            enc_final_sources,
        )
        print(f"corrs_groupwise: {corrs_groupwise[0].shape}") if verbose else None
        print(f"masks_groupwise: {masks_groupwise[0].shape}") if verbose else None
        fused_corr_groupwise, _ = self.fusion_block(
            feat_key=ctx_key, corrs=corrs_groupwise, masks=masks_groupwise
        )
        del corrs_groupwise, masks_groupwise
        print(
            f"fused_corr_groupwise: {fused_corr_groupwise.shape}"
        ) if verbose else None
        del ctx_key, ctx_sources
        all_enc_fused, enc_fused = self.fusion_enc_block(
            fused_corr=fused_corr_groupwise
        )
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

    def input_adapter(self, images, keyview_idx, poses, intrinsics, depth_range, **_):
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
def mvsnet_groupwisecorr(
    pretrained=True, weights=None, train=False, num_gpus=1, **kwargs
):
    model = build_model_with_cfg(
        model_cls=MVSnetGroupWiseCorr,
        weights=weights,
        train=train,
        num_gpus=num_gpus,
    )
    return model
