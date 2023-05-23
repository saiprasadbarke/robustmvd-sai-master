import os.path as osp

import numpy as np
from PIL import Image

from .dataset import Dataset, Sample
from .registry import register_dataset, register_default_dataset
from .layouts import MVDSequentialDefaultLayout, AllImagesLayout
from .kitti import KITTIDepth, KITTIImage, KITTISample
from .kitti_utils.provider import KittiProvider


class KITTIInitializer(Dataset):
    
    def _init_samples(self, conf_path, keyview_idx):
        self._init_samples_from_conf(conf_path=conf_path, keyview_idx=keyview_idx)
        self._write_samples_list()

    def _init_samples_from_conf(self, conf_path, keyview_idx):
        
        split_file = osp.join(conf_path, "split.txt")
        conf_path = osp.join(conf_path, "conf.toml")

        kitti = KittiProvider(paths_only=True)
        kitti.set_config(conf_path, split_file=split_file, kitti_root=self.root)
        
        for i, seq in enumerate(kitti.sequences):
            sample = KITTISample(name=seq.out_name + "/key{}".format(seq.view_nums[keyview_idx])) # e.g. '2011_09_30_drive_0033/key0'
            sample.data['images'] = []
            sample.data['poses'] = []
            sample.data['intrinsics'] = []
            sample.data['keyview_idx'] = keyview_idx
            
            for view_idx, view in enumerate(seq.views): # For every view of the sequence add the image and its data

                image = view['cam2']
                depth = view['depth_cam2']
                K = view['K_cam2'] # 3x3 mat
                pose = view['pose'] # 4x4 mat

                image = osp.relpath(image, self.root) #e.g. 0000.png in image_02
                sample.data['images'].append(KITTIImage(image)) 
                sample.data['poses'].append(pose)
                sample.data['intrinsics'].append(K)

                if view_idx == keyview_idx:
                    assert isinstance(depth, str)
                    depth = osp.relpath(depth, self.root)
                    sample.data['depth'] = KITTIDepth(depth)

            self.samples.append(sample)
                
                
class KITTIEigenDenseDepthTest(KITTIInitializer):
    # to go to rmvd root dir and do:
    # import rmvd.data.kitti_internal
    # rmvd.data.kitti_internal.KITTIEigenDenseDepthTest()

    base_dataset = 'kitti'
    split = 'eigen_dense_depth_test'
    dataset_type = 'mvd'

    def __init__(self, root=None, layouts=None, **kwargs):
        root = root if root is not None else self._get_path("kitti", "root")

        default_layouts = [
            MVDSequentialDefaultLayout("default", num_views=1, keyview_idx=0),
            AllImagesLayout("all_images", num_views=1),
        ]
        layouts = default_layouts + layouts if layouts is not None else default_layouts
        
        conf_path = osp.join(osp.dirname(osp.realpath(__file__)), "kitti_splits", "KITTIEigenDenseDepthTest")
        keyview_idx = 0

        super().__init__(root=root, layouts=layouts, conf_path=conf_path, keyview_idx=keyview_idx, **kwargs)
                
                
class KITTIEigenDenseDepthTrain(KITTIInitializer):
    # to go to rmvd root dir and do:
    # import rmvd.data.kitti_internal
    # rmvd.data.kitti_internal.KITTIEigenDenseDepthTrain()

    base_dataset = 'kitti'
    split = 'eigen_dense_depth_train'
    dataset_type = 'mvd'

    def __init__(self, root=None, layouts=None, **kwargs):
        root = root if root is not None else self._get_path("kitti", "root")

        default_layouts = [
            MVDSequentialDefaultLayout("default", num_views=1, keyview_idx=0),
            AllImagesLayout("all_images", num_views=1),
        ]
        layouts = default_layouts + layouts if layouts is not None else default_layouts
        
        conf_path = osp.join(osp.dirname(osp.realpath(__file__)), "kitti_splits", "KITTIEigenDenseDepthTrain")
        keyview_idx = 0

        super().__init__(root=root, layouts=layouts, conf_path=conf_path, keyview_idx=keyview_idx, **kwargs)
