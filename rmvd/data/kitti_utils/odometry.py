# copied and adapted from pykitti package
"""Provides 'odometry', which loads and parses odometry benchmark data."""

import datetime as dt
import glob
import os
from collections import namedtuple

import numpy as np

from .utils import *
from .config import CONFIG

__author__ = "Lee Clement"
__email__ = "lee.clement@robotics.utias.utoronto.ca"

odom_seqname_map = {
    '2011_10_03_drive_0027': '00',
    '2011_10_03_drive_0042': '01',
    '2011_10_03_drive_0034': '02',
    '2011_09_26_drive_0067': '03',
    '2011_09_30_drive_0016': '04',
    '2011_09_30_drive_0018': '05',
    '2011_09_30_drive_0020': '06',
    '2011_09_30_drive_0027': '07',
    '2011_09_30_drive_0028': '08',
    '2011_09_30_drive_0033': '09',
    '2011_09_30_drive_0034': '10',
}

odom_viewnum_map = {
    '2011_10_03_drive_0027': list(range(0, 4541)),
    '2011_10_03_drive_0042': list(range(0, 1101)),
    '2011_10_03_drive_0034': list(range(0, 4661)),
    '2011_09_26_drive_0067': list(range(0, 801)),
    '2011_09_30_drive_0016': list(range(0, 271)),
    '2011_09_30_drive_0018': list(range(0, 2761)),
    '2011_09_30_drive_0020': list(range(0, 1101)),
    '2011_09_30_drive_0027': list(range(0, 1101)),
    '2011_09_30_drive_0028': list(range(1100, 5171)),
    '2011_09_30_drive_0033': list(range(0, 1591)),
    '2011_09_30_drive_0034': list(range(0, 1201)),
}


def has_odometry(seq_name):
    return seq_name in odom_seqname_map.keys()


class OdometryDataset:
    def __init__(self):
        self.base_dir = CONFIG.paths.odometry_dir

    def get_seqs(self):
        seq_list = []
        for seq_name in odom_seqname_map.keys():
            seq_list.append(seq_name)
        return seq_list

    def get_seq_view_nums(self, seq_name):
        seq = odometry(seq_name)
        return seq.view_nums

    def write_split_file(self, path):
        with open(path, 'w') as f:
            for seq_name in self.get_seqs():
                seq_view_nums = self.get_seq_view_nums(seq_name)
                f.write(seq_name)
                for view_num in seq_view_nums:
                    f.write("; %d" % view_num)
                f.write("\n")

    def get_view_count(self):
        ctr = 0
        for seq_name in self.get_seqs():
            ctr += len(self.get_seq_view_nums(seq_name))
        return ctr


class odometry:
    """Load and parse odometry benchmark data into a usable format."""

    def __init__(self, seq_name, **kwargs):
        """Set the path."""
        base_path = CONFIG.paths.odometry_dir

        self.seq_name = seq_name
        self.sequence = odom_seqname_map[seq_name]
        self.sequence_path = os.path.join(base_path, 'sequences', self.sequence)
        self.pose_path = os.path.join(base_path, 'poses')

        view_nums_tmp = kwargs.get('view_nums', None)
        if view_nums_tmp is None:
            self.view_nums = odom_viewnum_map[seq_name]
        else:
            all_view_nums = odom_viewnum_map[seq_name]
            self.view_nums = []
            for view_num in view_nums_tmp:
                if view_num in all_view_nums:
                    self.view_nums.append(view_num)

        # Default image file extension is 'png'
        self.imtype = kwargs.get('imtype', 'png')

        # Find all the data files
        self._get_file_lists()

        # Pre-load data that isn't returned as a generator
        self._load_calib()
        self._load_timestamps()
        self._load_poses()

    def __len__(self):
        """Return the number of frames loaded."""
        return len(self.timestamps)

    def has_viewnum(self, view_num):
        return view_num in self.view_nums

    def pose(self, view_num):
        return self.get_pose_by_idx(self._viewnum_to_idx(view_num))

    def cam0(self, view_num):
        return self.get_cam0_by_idx(self._viewnum_to_idx(view_num))

    def cam1(self, view_num):
        return self.get_cam1_by_idx(self._viewnum_to_idx(view_num))

    def cam2(self, view_num):
        return self.get_cam2_by_idx(self._viewnum_to_idx(view_num))

    def cam3(self, view_num):
        return self.get_cam3_by_idx(self._viewnum_to_idx(view_num))

    def gray(self, view_num):
        return self.get_gray_by_idx(self._viewnum_to_idx(view_num))

    def rgb(self, view_num):
        return self.get_rgb_by_idx(self._viewnum_to_idx(view_num))

    def velo(self, view_num):
        return self.get_velo_by_idx(self._viewnum_to_idx(view_num))

    @property
    def poses(self):
        """Generator for poses."""
        for pose in self.pose_list:
            yield pose

    def get_pose_by_idx(self, idx):
        """Get pose at the specified index."""
        # return self.pose_list[idx]  # ref_to_rect0_transform
        ref_to_oldref_transform = invert_transform(self.pose_list[0])
        oldref_to_rect0_transform = self.pose_list[idx]
        ref_to_rect0_transform = np.dot(ref_to_oldref_transform, oldref_to_rect0_transform)
        return ref_to_rect0_transform

    @property
    def cam0s(self):
        """Generator to read image files for cam0 (monochrome left)."""
        return yield_images(self.cam0_files, mode='L')

    def get_cam0_by_idx(self, idx):
        """Read image file for cam0 (monochrome left) at the specified index."""
        return load_image(self.cam0_files[idx], mode='L')

    @property
    def cam1s(self):
        """Generator to read image files for cam1 (monochrome right)."""
        return yield_images(self.cam1_files, mode='L')

    def get_cam1_by_idx(self, idx):
        """Read image file for cam1 (monochrome right) at the specified index."""
        return load_image(self.cam1_files[idx], mode='L')

    @property
    def cam2s(self):
        """Generator to read image files for cam2 (RGB left)."""
        return yield_images(self.cam2_files, mode='RGB')

    def get_cam2_by_idx(self, idx):
        """Read image file for cam2 (RGB left) at the specified index."""
        return load_image(self.cam2_files[idx], mode='RGB')

    @property
    def cam3s(self):
        """Generator to read image files for cam0 (RGB right)."""
        return yield_images(self.cam3_files, mode='RGB')

    def get_cam3_by_idx(self, idx):
        """Read image file for cam3 (RGB right) at the specified index."""
        return load_image(self.cam3_files[idx], mode='RGB')

    @property
    def grays(self):
        """Generator to read monochrome stereo pairs from file.
        """
        return zip(self.cam0s, self.cam1s)

    def get_gray_by_idx(self, idx):
        """Read monochrome stereo pair at the specified index."""
        return (self.get_cam0_by_idx(idx), self.get_cam1_by_idx(idx))

    @property
    def rgbs(self):
        """Generator to read RGB stereo pairs from file.
        """
        return zip(self.cam2s, self.cam3s)

    def get_rgb_by_idx(self, idx):
        """Read RGB stereo pair at the specified index."""
        return (self.get_cam2_by_idx(idx), self.get_cam3_by_idx(idx))

    @property
    def velos(self):
        """Generator to read velodyne [x,y,z,reflectance] scan data from binary files."""
        # Return a generator yielding Velodyne scans.
        # Each scan is a Nx4 array of [x,y,z,reflectance]
        return yield_velo_scans(self.velo_files)

    def get_velo_by_idx(self, idx):
        """Read velodyne [x,y,z,reflectance] scan at the specified index."""
        return load_velo_scan(self.velo_files[idx])

    def _get_file_lists(self):
        """Find and list data files for each sensor."""
        self.cam0_files = sorted(glob.glob(
            os.path.join(self.sequence_path, 'image_0',
                         '*.{}'.format(self.imtype))))
        self.cam1_files = sorted(glob.glob(
            os.path.join(self.sequence_path, 'image_1',
                         '*.{}'.format(self.imtype))))
        self.cam2_files = sorted(glob.glob(
            os.path.join(self.sequence_path, 'image_2',
                         '*.{}'.format(self.imtype))))
        self.cam3_files = sorted(glob.glob(
            os.path.join(self.sequence_path, 'image_3',
                         '*.{}'.format(self.imtype))))
        self.velo_files = sorted(glob.glob(
            os.path.join(self.sequence_path, 'velodyne',
                         '*.bin')))

        # Subselect the chosen range of frames, if any
        if self.view_nums is not None:
            normalized_view_nums = [i-odom_viewnum_map[self.seq_name][0] for i in self.view_nums]

            self.cam0_files = subselect_odom_files_by_name(
                self.cam0_files, normalized_view_nums)
            self.cam1_files = subselect_odom_files_by_name(
                self.cam1_files, normalized_view_nums)
            self.cam2_files = subselect_odom_files_by_name(
                self.cam2_files, normalized_view_nums)
            self.cam3_files = subselect_odom_files_by_name(
                self.cam3_files, normalized_view_nums)
            self.velo_files = subselect_odom_files_by_name(
                self.velo_files, normalized_view_nums)

    def _load_calib(self):
        """Load and compute intrinsic and extrinsic calibration parameters."""
        # We'll build the calibration parameters as a dictionary, then
        # convert it to a namedtuple to prevent it from being modified later
        data = {}

        # Load the calibration file
        calib_filepath = os.path.join(self.sequence_path, 'calib.txt')
        filedata = read_calib_file(calib_filepath)

        # Create 3x4 projection matrices
        P_rect_00 = np.reshape(filedata['P0'], (3, 4))
        P_rect_10 = np.reshape(filedata['P1'], (3, 4))
        P_rect_20 = np.reshape(filedata['P2'], (3, 4))
        P_rect_30 = np.reshape(filedata['P3'], (3, 4))
        data['P_rect0_rect0'] = P_rect_00
        data['P_rect1_rect0'] = P_rect_10
        data['P_rect2_rect0'] = P_rect_20
        data['P_rect3_rect0'] = P_rect_30

        # Compute the camera intrinsics
        data['K_cam0'] = P_rect_00[0:3, 0:3]
        data['K_cam1'] = P_rect_10[0:3, 0:3]
        data['K_cam2'] = P_rect_20[0:3, 0:3]
        data['K_cam3'] = P_rect_30[0:3, 0:3]

        self.calib = namedtuple('CalibData', data.keys())(*data.values())

    def _load_timestamps(self):
        """Load timestamps from file."""
        timestamp_file = os.path.join(self.sequence_path, 'times.txt')

        # Read and parse the timestamps
        self.timestamps = []
        with open(timestamp_file, 'r') as f:
            for line in f.readlines():
                t = dt.timedelta(seconds=float(line))
                self.timestamps.append(t)

        # Subselect the chosen range of frames, if any
        if self.view_nums is not None:
            self.timestamps = [self.timestamps[i-odom_viewnum_map[self.seq_name][0]] for i in self.view_nums]

    def _load_poses(self):
        """Load ground truth poses as ref to rect0 transform ^ref_rect0_T from file."""
        pose_file = os.path.join(self.pose_path, self.sequence + '.txt')

        # Read and parse the poses
        poses = []
        try:
            with open(pose_file, 'r') as f:
                lines = f.readlines()
                if self.view_nums is not None:
                    lines = [lines[i-odom_viewnum_map[self.seq_name][0]] for i in self.view_nums]

                for line in lines:
                    ref_to_rect0_T = np.fromstring(line, dtype=float, sep=' ')
                    ref_to_rect0_T = ref_to_rect0_T.reshape(3, 4)
                    ref_to_rect0_T = np.vstack((ref_to_rect0_T, [0, 0, 0, 1]))
                    poses.append(ref_to_rect0_T)

        except FileNotFoundError:
            print('Ground truth poses are not available for sequence ' +
                  self.sequence + '.')

        self.pose_list = poses

    def _viewnum_to_idx(self, view_num):
        return self.view_nums.index(view_num)
