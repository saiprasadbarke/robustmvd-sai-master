import os.path as osp
import math

import torch
import torch.nn as nn
from torchvision import transforms as T
import numpy as np

from ..registry import register_model
from ..helpers import build_model_with_cfg
from rmvd.utils import (
    get_path,
    get_torch_model_device,
    to_numpy,
    to_torch,
    select_by_index,
    exclude_index,
)

class GMDepth_Wrapped(nn.Module):
    def __init__(self, checkpoint_name, reg_refine):
        super().__init__()

        import sys

        paths_file = self.get_paths_file()
        repo_path = get_path(paths_file, "gmdepth", "root")
        sys.path.insert(0, repo_path)

        from unimatch.unimatch import UniMatch
        from utils.utils import InputPadder

        # model parameters are the ones that are used when calling the script scripts/gmdepth_evaluate.sh
        # in the original repo
        self.model = UniMatch(
            feature_channels=128,
            num_scales=1,
            upsample_factor=8,
            num_head=1,
            ffn_dim_expansion=4,
            num_transformer_layers=6,
            reg_refine=reg_refine,
            task='depth',
        )
        # TODO: padding_factor=16, attn_type=swin, attn_splits_list=[2], prop_radius_list=[-1], num_reg_refine=1, num_depth_candidates=64, min_depth=0.5, max_depth=10
        state_dict = torch.load(osp.join(repo_path, "pretrained", checkpoint_name))["model"]
        self.model.load_state_dict(state_dict, strict=False)

        self.input_transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        # padder parameters are the ones that are used when calling the script scripts/gmdepth_evaluate.sh
        # in the original repo
        self.create_input_padder = lambda img_shape: InputPadder(img_shape, padding_factor=16, mode='kitti')
        
    def get_paths_file(self):
        rmvd_paths_file = osp.join(osp.dirname(osp.realpath(__file__)), "paths.toml")
        home_paths_file = osp.join(osp.expanduser('~'), 'rmvd_model_paths.toml')
    
        if osp.exists(rmvd_paths_file):
            paths_file = rmvd_paths_file
        elif osp.exists(home_paths_file):
            paths_file = home_paths_file
        else:
            raise FileNotFoundError("No paths.toml file found. Please create a paths.toml file as specified in the "
                                "rmvd/models/README.md file.")
            
        return paths_file

    def input_adapter(
        self, images, keyview_idx, poses=None, intrinsics=None, depth_range=None
    ):
        device = get_torch_model_device(self)

        for idx, image_batch in enumerate(images):
            tmp_images = []
            image_batch = image_batch.transpose(0, 2, 3, 1)
            for image in image_batch:
                image = self.input_transform(image.astype(np.uint8)).float()
                tmp_images.append(image)

            image_batch = torch.stack(tmp_images)
            images[idx] = image_batch

        if depth_range is None:
            depth_range = [0.2, 100]
        else:
            min_depths = depth_range[0]
            max_depths = depth_range[1]
            assert all(min_depths == min_depths[0]), "Model only works for constant depth range in the whole batch."
            assert all(max_depths == max_depths[0]), "Model only works for constant depth range in the whole batch."
            depth_range = [min_depths[0], max_depths[0]]

        images, keyview_idx, poses, intrinsics = to_torch((images, keyview_idx, poses, intrinsics), device=device)

        sample = {
            "images": images,
            "keyview_idx": keyview_idx,
            "poses": poses,
            "intrinsics": intrinsics,
            "depth_range": depth_range,
        }
        return sample

    def forward(self, images, keyview_idx, poses, intrinsics, depth_range, **_):
        image_key = select_by_index(images, keyview_idx)
        images_source = exclude_index(images, keyview_idx)
        image_source = images_source[0]

        intrinsics_key = select_by_index(intrinsics, keyview_idx)  # N, 3, 3
        poses_source = exclude_index(poses, keyview_idx)
        pose_source = poses_source[0]  # N, 4, 4

        # TODO: check depth range shape and type when given and when defaut
        # TODO: check ordering and default None's of params

        padder = self.create_input_padder(image_key.shape)
        image_key, image_source = padder.pad(image_key, image_source)

        pred_depth = self.model(image_key, image_source,
                               attn_type='swin',
                               attn_splits_list=[2],
                               prop_radius_list=[-1],
                               num_reg_refine=1,
                               intrinsics=intrinsics_key,
                               pose=pose_source,
                               min_depth=1. / depth_range[1],
                               max_depth=1. / depth_range[0],
                               num_depth_candidates=64,
                               task='depth',
                               )['flow_preds'][-1]  # [N, H, W]

        pred_depth = padder.unpad(pred_depth)
        pred_depth = pred_depth.unsqueeze(1)  # N, 1, H, W

        pred = {"depth": pred_depth}
        aux = {}

        return pred, aux

    def output_adapter(self, model_output):
        pred, aux = model_output
        return to_numpy(pred), to_numpy(aux)


@register_model
def gmdepth_scale1_scannet_wrapped(
    pretrained=True, weights=None, train=False, num_gpus=1, **kwargs
):
    assert pretrained and (
        weights is None
    ), "Model supports only pretrained=True, weights=None."

    # TODO: update cfg
    cfg = {
        "checkpoint_name": "gmdepth-scale1-scannet-d3d1efb5.pth",
        "reg_refine": False,
    }

    model = build_model_with_cfg(
        model_cls=GMDepth_Wrapped,
        cfg=cfg,
        weights=None,
        train=train,
        num_gpus=num_gpus,
    )
    return model


@register_model
def gmdepth_scale1_resumeflowthings_scannet_wrapped(
    pretrained=True, weights=None, train=False, num_gpus=1, **kwargs
):
    assert pretrained and (
        weights is None
    ), "Model supports only pretrained=True, weights=None."

    # TODO: update cfg
    cfg = {
        "checkpoint_name": "gmdepth-scale1-resumeflowthings-scannet-5d9d7964.pth",
        "reg_refine": False,
    }

    model = build_model_with_cfg(
        model_cls=GMDepth_Wrapped,
        cfg=cfg,
        weights=None,
        train=train,
        num_gpus=num_gpus,
    )
    return model


@register_model
def gmdepth_scale1_regrefine1_resumeflowthings_scannet_wrapped(
    pretrained=True, weights=None, train=False, num_gpus=1, **kwargs
):
    assert pretrained and (
        weights is None
    ), "Model supports only pretrained=True, weights=None."

    # TODO: update cfg
    cfg = {
        "checkpoint_name": "gmdepth-scale1-regrefine1-resumeflowthings-scannet-90325722.pth",
        "reg_refine": True,
    }

    model = build_model_with_cfg(
        model_cls=GMDepth_Wrapped,
        cfg=cfg,
        weights=None,
        train=train,
        num_gpus=num_gpus,
    )
    return model


@register_model
def gmdepth_scale1_demon_wrapped(
    pretrained=True, weights=None, train=False, num_gpus=1, **kwargs
):
    assert pretrained and (
        weights is None
    ), "Model supports only pretrained=True, weights=None."

    # TODO: update cfg
    cfg = {
        "checkpoint_name": "gmdepth-scale1-demon-bd64786e.pth",
        "reg_refine": False,
    }

    model = build_model_with_cfg(
        model_cls=GMDepth_Wrapped,
        cfg=cfg,
        weights=None,
        train=train,
        num_gpus=num_gpus,
    )
    return model


@register_model
def gmdepth_scale1_regrefine1_resumeflowthings_demon_wrapped(
    pretrained=True, weights=None, train=False, num_gpus=1, **kwargs
):
    assert pretrained and (
        weights is None
    ), "Model supports only pretrained=True, weights=None."

    # TODO: update cfg
    cfg = {
        "checkpoint_name": "gmdepth-scale1-regrefine1-resumeflowthings-demon-7c23f230.pth",
        "reg_refine": True,
    }

    model = build_model_with_cfg(
        model_cls=GMDepth_Wrapped,
        cfg=cfg,
        weights=None,
        train=train,
        num_gpus=num_gpus,
    )
    return model


@register_model
def gmdepth_scale1_resumeflowthings_demon_wrapped(
    pretrained=True, weights=None, train=False, num_gpus=1, **kwargs
):
    assert pretrained and (
        weights is None
    ), "Model supports only pretrained=True, weights=None."

    # TODO: update cfg
    cfg = {
        "checkpoint_name": "gmdepth-scale1-resumeflowthings-demon-a2fe127b.pth",
        "reg_refine": False,
    }

    model = build_model_with_cfg(
        model_cls=GMDepth_Wrapped,
        cfg=cfg,
        weights=None,
        train=train,
        num_gpus=num_gpus,
    )
    return model
