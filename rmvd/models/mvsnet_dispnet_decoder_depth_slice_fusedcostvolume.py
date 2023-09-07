import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .registry import register_model
from .helpers import build_model_with_cfg
from .blocks.mvsnet_encoder import MVSNetEncoder
from .blocks.planesweep_corr import PlanesweepCorrelation
from .blocks.variance_costvolume_fusion import VarianceCostvolumeFusion
from .blocks.mvsnet_fused_costvolume_encoder import MVSNetFusedCostvolumeEncoder
from .blocks.dispnet_decoder_depthslice_mvsnet import DispnetDecoderForMvsnet

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


class MVSNetDispnetDecoder(nn.Module):
    def __init__(self, num_sampling_points=128, sampling_type="linear_depth"):
        super().__init__()

        base_channels = 8
        self.num_sampling_points = num_sampling_points
        self.sampling_type = sampling_type
        self.encoder = MVSNetEncoder(base_channels=base_channels)
        self.corr_block = PlanesweepCorrelation(normalize=False, corr_type="warponly")
        self.fusion_block = VarianceCostvolumeFusion()
        self.fusion_enc_block = MVSNetFusedCostvolumeEncoder(
            in_channels=32, base_channels=8, batch_norm=True
        )
        self.decoder = DispnetDecoderForMvsnet(arch="mvsnet")

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
        del image_key
        print(f"enc_key: {enc_key.shape}") if verbose else None
        print(
            {f"all_enc_key[{k}]": v.shape for k, v in all_enc_key.items()}
        ) if verbose else None
        print(f"images_source: {images_source[0].shape}") if verbose else None
        enc_sources = [self.encoder(image_source)[1] for image_source in images_source]
        del images_source
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
        del enc_sources
        fused_corr, _ = self.fusion_block(feat_key=enc_key, corrs=corrs, masks=masks)
        del corrs, masks, enc_key
        print(f"fused_corr: {fused_corr.shape}") if verbose else None
        all_enc_fused, enc_fused = self.fusion_enc_block(fused_corr=fused_corr)
        del fused_corr
        print(f"enc_fused: {enc_fused.shape}") if verbose else None
        print(
            {f"all_enc_fused[{k}]": v.shape for k, v in all_enc_fused.items()}
        ) if verbose else None
        enc_fused_depth = enc_fused.shape[2]
        all_enc_fused = {
            "conv5_1": all_enc_fused["3d_conv4"],
            "conv4_1": all_enc_fused["3d_conv2"],
            "conv3_1": all_enc_fused["3d_conv0"],
        }
        all_enc_fused_depth = [v.shape[2] for k, v in all_enc_fused.items()]
        all_depths = all_enc_fused_depth + [enc_fused_depth]
        # enc_fused = enc_fused.repeat(1, 1, int(max(all_depths) / enc_fused_depth), 1, 1)
        enc_fused_adjusted = F.interpolate(
            enc_fused,
            scale_factor=(int(max(all_depths) / enc_fused_depth), 1, 1),
            mode="trilinear",
            align_corners=False,
        )
        del enc_fused
        # all_enc_fused = {
        #     k: v.repeat(1, 1, int(max(all_depths) / all_enc_fused_depth[i]), 1, 1)
        #     for i, (k, v) in enumerate(all_enc_fused.items())
        # }
        all_enc_fused_adjusted = {
            k: F.interpolate(
                v,
                scale_factor=(
                    int(max(all_depths) / all_enc_fused_depth[i]),
                    1,
                    1,
                ),
                mode="trilinear",
                align_corners=False,
            )
            for i, (k, v) in enumerate(all_enc_fused.items())
        }
        del all_enc_fused
        decs = []
        for i in range(max(all_depths)):
            dec = self.decoder(
                enc_fused=enc_fused_adjusted[:, :, i, :, :],
                all_enc={
                    **all_enc_key,
                    **{k: v[:, :, i, :, :] for k, v in all_enc_fused_adjusted.items()},
                },
            )
            decs.append(dec)
        del enc_fused_adjusted
        del all_enc_fused_adjusted
        del all_enc_key
        prob = torch.stack([dec["invdepth"] for dec in decs], dim=2).squeeze(1)
        del decs
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
            d_indices = torch.arange(
                sampling_invdepths.shape[1], device=prob.device, dtype=torch.float
            )
            d_indices = d_indices.view(1, -1, 1, 1)
            pred_idx = torch.sum(prob * d_indices, dim=1, keepdim=True).long()  # N1HW
            # pred_idx = pred_idx.clamp(min=0, max=steps-1)
            # # the confidence is the 4-sum probability at this index:
            pred_depth_confidence = torch.gather(prob_sum4, 1, pred_idx)
        pred = {
            "depth": pred_depth,
            "depth_uncertainty": 1 - pred_depth_confidence,
        }
        aux = pred
        aux["invdepth"] = pred_invdepth
        aux["sampling_invdepths"] = sampling_invdepths
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
def mvsnet_dispnet_decoder_depth_slice_fusedcostvolume(
    pretrained=True, weights=None, train=False, num_gpus=1, **kwargs
):
    assert not (
        pretrained and weights is None
    ), "Pretrained weights are not available for this model."
    # weights = pretrained_weights if (pretrained and weights is None) else weights
    model = build_model_with_cfg(
        model_cls=MVSNetDispnetDecoder,
        weights=weights,
        train=train,
        num_gpus=num_gpus,
        num_sampling_points=128,
    )
    return model
