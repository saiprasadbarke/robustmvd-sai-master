import os.path as osp

import pytoml as toml
import numpy as np


class PathConfig:
    def __init__(self, settings, kitti_root=None):
        if kitti_root is None:
            self.raw_dir = str(settings.get('raw_dir'))
            self.odometry_dir = str(settings.get('odometry_dir'))
            self.stereo2015_dir = str(settings.get('stereo2015_dir'))
            self.depthpred_dir = str(settings.get('depthpred_dir'))
            self.out_dir = str(settings.get('out_dir'))
        else:
            self.raw_dir = osp.join(kitti_root, "raw_data")
            self.odometry_dir = osp.join(kitti_root, "odometry", "dataset")
            self.stereo2015_dir = osp.join(kitti_root, "stereo_flow_sceneflow_2015")
            self.depthpred_dir = osp.join(kitti_root, "depth_completion_prediction")
            self.out_dir = osp.join("/tmp/out")  # we don't need this


class ProcConfig:
    def __init__(self, settings):
        self.proc_data = bool(settings.get('proc_data'))
        self.target_width = int(settings.get('target_width'))
        self.target_height = int(settings.get('target_height'))

        if 'target_K' in settings:
            self.target_K = str(settings.get('target_K'))
            self.target_K = np.load(self.target_K).astype(np.float32)
        else:
            self.target_K = None

        self.random_crop = bool(settings.get('random_crop', False))

        self.scale = float(settings.get('scale', 1.0))


class DataConfig:
    def __init__(self, settings, split_file=None):
        self.split_file = str(settings.get('split_file')) if split_file is None else split_file

        self.raw_cam0 = bool(settings.get('raw_cam0'))
        self.raw_cam1 = bool(settings.get('raw_cam1'))
        self.raw_cam2 = bool(settings.get('raw_cam2'))
        self.raw_cam3 = bool(settings.get('raw_cam3'))
        self.raw_velo = bool(settings.get('raw_velo'))
        self.raw_pose = bool(settings.get('raw_pose'))
        self.raw_K_cam0 = bool(settings.get('raw_K_cam0'))
        self.raw_K_cam1 = bool(settings.get('raw_K_cam1'))
        self.raw_K_cam2 = bool(settings.get('raw_K_cam2'))
        self.raw_K_cam3 = bool(settings.get('raw_K_cam3'))
        self.odom_pose = bool(settings.get('odom_pose'))
        self.depth_cam2gt = bool(settings.get('depth_cam2gt'))
        self.depth_cam3gt = bool(settings.get('depth_cam3gt'))
        self.depth_cam2velo = bool(settings.get('depth_cam2velo'))
        self.depth_cam3velo = bool(settings.get('depth_cam3velo'))

        self.pose_format = str(settings.get('pose_format'))
        self.velo_depth = bool(settings.get('velo_depth', False))


class Config:
    def __init__(self, settings_collection=None, path=None, split_file=None, kitti_root=None):
        if settings_collection is not None:
            settings = settings_collection[0].copy()
            for s in settings_collection:
                for c in s:
                    if c in settings and isinstance(settings[c], dict):
                        settings[c].update(s[c])
                    else:
                        settings[c] = s[c]
        else:
            settings = {}

        if 'paths' in settings:
            self.paths = PathConfig(settings.get('paths', {}), kitti_root=kitti_root)
        if 'proc' in settings:
            self.proc = ProcConfig(settings.get('proc', {}))
        if 'data' in settings:
            self.data = DataConfig(settings.get('data', {}), split_file=split_file)

        if path is not None:
            self.path = path

    def from_toml(self, filename, split_file=None, kitti_root=None):
        """
        Initialize this config from a list of TOML configuration files.
        Args:
            *filename: filename of TOML configuration file.

        Returns:
            Re-initialized config object.
        """

        settings = []
        with open(filename, 'rb') as fin:
            settings.append(toml.load(fin))

        return self.__init__(settings, filename, split_file=split_file, kitti_root=kitti_root)


CONFIG = Config()
# CONFIG.from_toml("/home/schroepp/workspace/kitti_utils/confs/default_conf.toml")
