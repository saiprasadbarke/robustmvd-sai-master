# based on pykitti package
"""Provides 'depth', which loads and parses KITTI depth-from-single-view benchmark data."""

import datetime as dt
import glob
import os
from builtins import ValueError
from collections import namedtuple

import numpy as np

from .utils import *
from .config import CONFIG

__author__ = "Lee Clement"
__email__ = "lee.clement@robotics.utias.utoronto.ca"

train_seqs = ['2011_09_26_drive_0059', '2011_09_28_drive_0155', '2011_09_26_drive_0039', '2011_09_29_drive_0071',
              '2011_09_26_drive_0056', '2011_09_26_drive_0018', '2011_09_30_drive_0028', '2011_09_28_drive_0034',
              '2011_09_28_drive_0104', '2011_09_26_drive_0011', '2011_09_28_drive_0100', '2011_09_28_drive_0096',
              '2011_09_28_drive_0043', '2011_09_28_drive_0086', '2011_09_26_drive_0052', '2011_09_28_drive_0167',
              '2011_09_28_drive_0136', '2011_09_28_drive_0125', '2011_09_26_drive_0057', '2011_09_28_drive_0021',
              '2011_09_28_drive_0094', '2011_09_28_drive_0184', '2011_09_26_drive_0046', '2011_09_26_drive_0019',
              '2011_09_26_drive_0015', '2011_09_28_drive_0220', '2011_09_26_drive_0022', '2011_09_28_drive_0199',
              '2011_09_28_drive_0113', '2011_09_30_drive_0018', '2011_09_28_drive_0209', '2011_09_28_drive_0156',
              '2011_09_28_drive_0053', '2011_09_28_drive_0186', '2011_09_28_drive_0077', '2011_09_28_drive_0045',
              '2011_09_28_drive_0070', '2011_09_26_drive_0048', '2011_09_28_drive_0166', '2011_09_26_drive_0117',
              '2011_09_26_drive_0096', '2011_09_28_drive_0204', '2011_09_28_drive_0038', '2011_09_28_drive_0126',
              '2011_09_28_drive_0080', '2011_09_28_drive_0216', '2011_09_28_drive_0165', '2011_09_26_drive_0027',
              '2011_09_26_drive_0064', '2011_09_28_drive_0174', '2011_09_26_drive_0104', '2011_09_28_drive_0154',
              '2011_09_26_drive_0091', '2011_09_28_drive_0143', '2011_09_28_drive_0087', '2011_09_28_drive_0192',
              '2011_09_28_drive_0016', '2011_09_28_drive_0146', '2011_09_26_drive_0106', '2011_09_28_drive_0075',
              '2011_09_28_drive_0198', '2011_09_28_drive_0162', '2011_09_28_drive_0001', '2011_09_26_drive_0070',
              '2011_09_26_drive_0001', '2011_09_28_drive_0102', '2011_09_30_drive_0034', '2011_09_28_drive_0177',
              '2011_09_28_drive_0098', '2011_09_26_drive_0017', '2011_09_28_drive_0122', '2011_09_26_drive_0035',
              '2011_09_28_drive_0205', '2011_09_28_drive_0160', '2011_09_28_drive_0065', '2011_09_28_drive_0161',
              '2011_09_26_drive_0028', '2011_09_28_drive_0119', '2011_09_28_drive_0208', '2011_09_28_drive_0121',
              '2011_09_28_drive_0047', '2011_09_26_drive_0029', '2011_09_29_drive_0004', '2011_09_26_drive_0086',
              '2011_09_28_drive_0222', '2011_09_26_drive_0061', '2011_09_28_drive_0201', '2011_09_28_drive_0128',
              '2011_09_28_drive_0057', '2011_09_28_drive_0168', '2011_09_30_drive_0020', '2011_09_26_drive_0084',
              '2011_09_30_drive_0033', '2011_09_28_drive_0214', '2011_10_03_drive_0034', '2011_09_26_drive_0032',
              '2011_09_28_drive_0132', '2011_09_28_drive_0108', '2011_09_28_drive_0110', '2011_09_28_drive_0039',
              '2011_09_28_drive_0153', '2011_09_26_drive_0093', '2011_09_28_drive_0095', '2011_10_03_drive_0027',
              '2011_09_28_drive_0106', '2011_09_28_drive_0145', '2011_09_28_drive_0171', '2011_09_28_drive_0082',
              '2011_09_28_drive_0185', '2011_09_28_drive_0103', '2011_09_26_drive_0087', '2011_09_26_drive_0051',
              '2011_09_28_drive_0191', '2011_09_30_drive_0027', '2011_09_28_drive_0078', '2011_09_28_drive_0117',
              '2011_09_28_drive_0187', '2011_09_28_drive_0138', '2011_09_26_drive_0014', '2011_09_28_drive_0141',
              '2011_09_28_drive_0149', '2011_09_26_drive_0060', '2011_09_28_drive_0134', '2011_09_28_drive_0035',
              '2011_09_28_drive_0066', '2011_09_28_drive_0183', '2011_09_28_drive_0089', '2011_09_28_drive_0002',
              '2011_09_28_drive_0068', '2011_09_28_drive_0195', '2011_09_28_drive_0179', '2011_09_28_drive_0135',
              '2011_09_26_drive_0009', '2011_09_28_drive_0090', '2011_09_28_drive_0071', '2011_09_28_drive_0054',
              '2011_09_26_drive_0101', '2011_10_03_drive_0042']

val_seqs = ['2011_09_26_drive_0005', '2011_09_26_drive_0036', '2011_10_03_drive_0047', '2011_09_26_drive_0013',
            '2011_09_28_drive_0037', '2011_09_30_drive_0016', '2011_09_26_drive_0002', '2011_09_26_drive_0079',
            '2011_09_29_drive_0026', '2011_09_26_drive_0113', '2011_09_26_drive_0020', '2011_09_26_drive_0095',
            '2011_09_26_drive_0023']


def has_depth(seq_name):
    return seq_name in val_seqs or seq_name in train_seqs


class DepthSequence:
    """Load and parse depth-from-single-view benchmark data into a usable format.
    """

    def __init__(self, seq_name, **kwargs):
        """Set the path and pre-load calibration data and timestamps."""
        base_path = CONFIG.paths.depthpred_dir

        self.seq_name = seq_name
        if seq_name in train_seqs:
            split = "train"
        elif self.seq_name in val_seqs:
            split = "val"
        else:
            raise ValueError(
                'The sequence %s is not in the train or val split of the depth prediction benchmark' % self.seq_name)

        self.data_path = os.path.join(base_path, split, self.seq_name + '_sync')
        self.view_nums = kwargs.get('view_nums', None)

        # Find all the data files
        self._get_file_lists()

    def __len__(self):
        """Return the number of frames loaded."""
        return len(self.view_nums)

    def has_viewnum(self, view_num):
        return view_num in self.view_nums

    def cam2_gtdepth(self, view_num, path_only=False):
        return self.get_cam2_depth_by_idx(self._viewnum_to_idx(view_num), path_only)

    def cam3_gtdepth(self, view_num, path_only=False):
        return self.get_cam3_gtdepth_by_idx(self._viewnum_to_idx(view_num), path_only)

    @property
    def cam2_depths(self):
        """Generator to read gt depth for cam2 (RGB left)."""
        return yield_depths(self.cam2_gtdepth_files)

    def get_cam2_depth_by_idx(self, idx, path_only=False):
        """Read gt depth for cam2 (RGB left) at the specified index."""
        if path_only:
            return self.cam2_gtdepth_files[idx]
        else:
            return load_depth(self.cam2_gtdepth_files[idx])

    @property
    def cam3_gtdepths(self):
        """Generator to read gt depth for cam0 (RGB right)."""
        return yield_depths(self.cam3_gtdepth_files)

    def get_cam3_gtdepth_by_idx(self, idx, path_only=False):
        """Read gt depth for cam3 (RGB right) at the specified index."""
        if path_only:
            return self.cam3_gtdepth_files[idx]
        else:
            return load_depth(self.cam3_gtdepth_files[idx])

    def _get_file_lists(self):
        """Find and list data files for each sensor."""

        gtdepth_path = os.path.join(self.data_path, 'proj_depth', 'groundtruth')
        velodepth_path = os.path.join(self.data_path, 'proj_depth', 'velodyne_raw')

        self.cam2_gtdepth_files = sorted(glob.glob(os.path.join(gtdepth_path, 'image_02', '*.png')))
        self.cam3_gtdepth_files = sorted(glob.glob(os.path.join(gtdepth_path, 'image_03', '*.png')))

        # add self.cam2_velodepth_files = .. here when needed

        # Subselect the chosen range of frames, if any
        if self.view_nums is not None:
            self.cam2_gtdepth_files = subselect_depth_files_by_name(self.cam2_gtdepth_files, self.view_nums)
            self.cam3_gtdepth_files = subselect_depth_files_by_name(self.cam3_gtdepth_files, self.view_nums)

        # Set (if was None before), or update view_nums
        cam2_gtdepth_view_nums = [view_num_to_int(view_num_from_path(path)) for path in
                                  self.cam2_gtdepth_files]
        cam3_gtdepth_view_nums = [view_num_to_int(view_num_from_path(path)) for path in
                                  self.cam3_gtdepth_files]

        def filter_duplicates(view_nums):
            tmp_len = len(view_nums)
            view_nums = list(set(view_nums))
            if len(view_nums) != tmp_len:
                print(
                    "WARNING: %d duplicate views found in depth sequence %s." % (tmp_len - len(view_nums), self.drive))

        filter_duplicates(cam2_gtdepth_view_nums)
        filter_duplicates(cam3_gtdepth_view_nums)

        total_view_nums = sorted(list(set(cam2_gtdepth_view_nums + cam3_gtdepth_view_nums)))

        if not set(total_view_nums) == set(cam2_gtdepth_view_nums) == set(cam3_gtdepth_view_nums):
            print("WARNING: Depth data of sequence %s is not available for all views." % self.seq_name)
            print("View nums for GT depth for cam2: %d." % len(cam2_gtdepth_view_nums))
            print("View nums for GT depth for cam3: %d." % len(cam3_gtdepth_view_nums))

        if self.view_nums is not None and set(self.view_nums) != set(total_view_nums):
            print("WARNING (depth sequence %s): Given list of view nums "
                  "could not be successfully loaded." % self.seq_name)

        self.view_nums = total_view_nums

    def _viewnum_to_idx(self, view_num):
        return self.view_nums.index(view_num)
