# Stadard Imports
import math

# Local imports
from .blocks.mvsnet_components import (
    FeatureNet,
    CostRegNet,
    homo_warp,
    depth_regression,
)
from .registry import register_model
from .helpers import build_model_with_cfg
from rmvd.utils import (
    get_torch_model_device,
    to_numpy,
    to_torch,
    select_by_index,
    exclude_index,
)
from rmvd.data.transforms import (
    NormalizeImagesByShiftAndScale,
    NormalizeImagesToMinMax,
    ResizeInputs,
    UpscaleInputsToNextMultipleOf,
)

# External Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T
import numpy as np


class MVSNet(nn.Module):
    def __init__(self, sample_in_inv_depth_space=False, num_sampling_steps=192):
        super().__init__()
        self.feature = FeatureNet()
        self.cost_regularization = CostRegNet()
        self.input_transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.num_sampling_steps = num_sampling_steps
        self.sample_in_inv_depth_space = sample_in_inv_depth_space

    def forward(self, images, poses, intrinsics, keyview_idx, depth_range, **_):
        N = images[0].shape[0]  # batch size
        # Create the depth samples from the minimum and maximum depth. If no depth range is provided, use the default range [0.2,100]
        if depth_range is None:
            if self.sample_in_inv_depth_space:
                depth_samples = (
                    1
                    / torch.linspace(
                        1 / 100, 1 / 0.2, self.num_sampling_steps, dtype=torch.float32
                    )[::-1]
                )
            else:
                depth_samples = torch.linspace(
                    0.2, 100, self.num_sampling_steps, dtype=torch.float32
                )
        else:
            min_depth, max_depth = depth_range
            if self.sample_in_inv_depth_space:
                depth_samples = (
                    1
                    / torch.linspace(
                        1 / max_depth[0],
                        1 / min_depth[0],
                        self.num_sampling_steps,
                        dtype=torch.float32,
                    )[::-1]
                )
            else:
                depth_samples = torch.linspace(
                    min_depth[0],
                    max_depth[0],
                    self.num_sampling_steps,
                    dtype=torch.float32,
                )

        depth_samples = torch.stack(N * [depth_samples])
        depth_samples = depth_samples.to(images[0].device)
        proj_mats = []
        for idx, (intrinsic_batch, pose_batch) in enumerate(zip(intrinsics, poses)):
            proj_mat_batch = []
            for intrinsic, pose, cur_keyview_idx in zip(
                intrinsic_batch, pose_batch, keyview_idx
            ):
                scale_arr = torch.tensor(
                    [[0.25] * 3, [0.25] * 3, [1.0] * 3], device=intrinsic.device
                )  # 3, 3
                intrinsic = (
                    intrinsic * scale_arr
                )  # scale intrinsics to 4x downsampling that happens within the model

                proj_mat = pose
                proj_mat[:3, :4] = torch.matmul(intrinsic, proj_mat[:3, :4])

                if idx == cur_keyview_idx:
                    proj_mat = torch.inverse(proj_mat)

                proj_mat_batch.append(proj_mat)

            proj_mat_batch = torch.stack(proj_mat_batch)
            proj_mats.append(proj_mat_batch)

        image_key = select_by_index(images, keyview_idx)
        images_source = exclude_index(images, keyview_idx)
        images = [image_key] + images_source

        proj_mat_key = select_by_index(proj_mats, keyview_idx)
        proj_mats_source = exclude_index(proj_mats, keyview_idx)
        proj_mats = [proj_mat_key] + proj_mats_source

        images = torch.stack(images, 1)  # N, num_views, 3, H, W
        proj_mats = torch.stack(proj_mats, 1)  # N, num_views, 4, 4
        proj_mats = proj_mats.to(device=images[0].device, dtype=torch.float32)
        ################## End of transformations.##################

        # images: (B, V, 3, H, W)
        # proj_mats: (B, V, 4, 4)
        # depth_samples: (B, D)
        B, V, _, H, W = images.shape
        D = depth_samples.shape[1]  # value of D should be 256

        # step 1. feature extraction
        # in: images; out: 32-channel feature maps
        images = images.reshape(B * V, 3, H, W)
        feats = self.feature(images)  # (B*V, F, h, w)
        del images
        feats = feats.reshape(B, V, *feats.shape[1:])  # (B, V, F, h, w)
        ref_feats, src_feats = feats[:, 0], feats[:, 1:]
        ref_proj, src_projs = proj_mats[:, 0], proj_mats[:, 1:]
        src_feats = src_feats.permute(1, 0, 2, 3, 4)  # (V-1, B, F, h, w)
        src_projs = src_projs.permute(1, 0, 2, 3)  # (V-1, B, 4, 4)

        # step 2. differentiable homograph, build cost volume
        ref_volume = ref_feats.unsqueeze(2).repeat(1, 1, D, 1, 1)  # (B, F, D, h, w)
        volume_sum = ref_volume
        volume_sq_sum = ref_volume**2
        del ref_volume

        for src_feat, src_proj in zip(src_feats, src_projs):
            warped_volume = homo_warp(src_feat, src_proj, ref_proj, depth_samples)
            volume_sum = volume_sum + warped_volume
            volume_sq_sum = volume_sq_sum + warped_volume**2
            del warped_volume
        # aggregate multiple feature volumes by variance
        volume_variance = volume_sq_sum.div_(V).sub_(volume_sum.div_(V).pow_(2))
        del volume_sq_sum, volume_sum

        # step 3. cost volume regularization
        cost_reg = self.cost_regularization(volume_variance).squeeze(1)
        prob_volume = F.softmax(cost_reg, 1)  # (B, D, h, w)
        depth = depth_regression(prob_volume, depth_samples)

        with torch.no_grad():
            # sum probability of 4 consecutive depth indices
            prob_volume_sum4 = 4 * F.avg_pool3d(
                F.pad(prob_volume.unsqueeze(1), pad=(0, 0, 0, 0, 1, 2)),
                (4, 1, 1),
                stride=1,
            ).squeeze(
                1
            )  # (B, D, h, w)

            depth_index = depth_regression(
                prob_volume,
                torch.arange(D, device=prob_volume.device, dtype=prob_volume.dtype),
            ).long()  # (B, h, w)

            confidence = torch.gather(
                prob_volume_sum4, 1, depth_index.unsqueeze(1)
            ).squeeze(
                1
            )  # (B, h, w)

        pred_depth_uncertainty = 1 - confidence

        pred_depth = depth.unsqueeze(1)
        pred_depth_uncertainty = pred_depth_uncertainty.unsqueeze(1)

        pred = {"depth": pred_depth, "depth_uncertainty": pred_depth_uncertainty}
        aux = {}

        return pred, aux

    def input_adapter(
        self,
        images,
        keyview_idx,
        poses=None,
        intrinsics=None,
        depth_range=None,
        masks=None,
    ):
        device = get_torch_model_device(self)

        resized = UpscaleInputsToNextMultipleOf(32)(
            {"images": images, "intrinsics": intrinsics}
        )
        resized = NormalizeImagesToMinMax(min_val=0.0, max_val=1.0)(resized)
        resized = NormalizeImagesByShiftAndScale(
            shift=[0.485, 0.456, 0.406], scale=[0.229, 0.224, 0.225]
        )(resized)
        images = resized["images"]
        intrinsics = resized["intrinsics"]

        images, keyview_idx, intrinsics, poses, depth_range, masks = to_torch(
            (images, keyview_idx, intrinsics, poses, depth_range, masks), device=device
        )

        sample = {
            "images": images,
            "poses": poses,
            "intrinsics": intrinsics,
            "keyview_idx": keyview_idx,
            "depth_range": depth_range,
            "masks": masks,
        }
        return sample

    def output_adapter(self, model_output):
        pred, aux = model_output
        return to_numpy(pred), to_numpy(aux)


@register_model(trainable=False)
def mvsnet_train(pretrained=True, weights=None, train=False, num_gpus=1, **kwargs):
    assert not (
        pretrained and weights is None
    ), "Pretrained weights are not available for this model."
    cfg = {"sample_in_inv_depth_space": False, "num_sampling_steps": 256}
    model = build_model_with_cfg(
        model_cls=MVSNet,
        cfg=cfg,
        weights=weights,
        train=train,
        num_gpus=num_gpus,
    )
    return model
