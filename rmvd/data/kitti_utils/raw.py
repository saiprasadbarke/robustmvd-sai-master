# copied and adapted from pykitti package
"""Provides 'raw', which loads and parses raw KITTI data."""

import datetime as dt
import glob
import os
from collections import namedtuple, Counter

import numpy as np

from .utils import *
from .config import CONFIG

__author__ = "Lee Clement"
__email__ = "lee.clement@robotics.utias.utoronto.ca"


class RawDataset:
    def __init__(self):
        self.base_dir = CONFIG.paths.raw_dir

    def get_seqs(self):
        seq_list = []
        for dir in os.listdir(self.base_dir):
            if dir.startswith("2011_"):
                seq_list.extend([x for x in os.listdir(os.path.join(self.base_dir, dir)) if x.startswith("2011_")])

        return seq_list

    def get_seq_view_nums(self, seq_name):
        seq = RawSequence(*split_seqname(seq_name))
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


class RawSequence:
    """Load and parse data of a raw sequence into a usable format.
    """

    def __init__(self, date, drive, **kwargs):
        """Set the path and pre-load calibration data and timestamps."""
        base_path = CONFIG.paths.raw_dir

        self.dataset = kwargs.get('dataset', 'sync')
        self.drive = date + '_drive_' + drive + '_' + self.dataset
        self.calib_path = os.path.join(base_path, date)
        self.data_path = os.path.join(base_path, date, self.drive)
        self.view_nums = kwargs.get('view_nums', None)

        # Default image file extension is '.png'
        self.imtype = kwargs.get('imtype', 'png')

        # Find all the data files
        self._get_file_lists()

        # Pre-load data that isn't returned as a generator
        self._load_calib()
        self._load_timestamps()
        self._load_oxts()

    def __len__(self):
        """Return the number of frames loaded."""
        return len(self.timestamps)

    def has_viewnum(self, view_num):
        return view_num in self.view_nums

    def cam0(self, view_num):
        return self.get_cam0_by_idx(self._viewnum_to_idx(view_num))

    def cam1(self, view_num):
        return self.get_cam1_by_idx(self._viewnum_to_idx(view_num))

    def cam2(self, view_num, path_only=False):
        return self.get_cam2_by_idx(self._viewnum_to_idx(view_num), path_only)

    def cam3(self, view_num, path_only=False):
        return self.get_cam3_by_idx(self._viewnum_to_idx(view_num), path_only)

    def gray(self, view_num):
        return self.get_gray_by_idx(self._viewnum_to_idx(view_num))

    def rgb(self, view_num):
        return self.get_rgb_by_idx(self._viewnum_to_idx(view_num))

    def velo(self, view_num):
        return self.get_velo_by_idx(self._viewnum_to_idx(view_num))

    def velo_cam2(self, view_num):
        return self._project_velo_to_cam2(self.velo(view_num), view_num)

    def pose(self, view_num):
        return self.get_pose_by_idx(self._viewnum_to_idx(view_num))

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

    def get_cam2_by_idx(self, idx, path_only=False):
        """Read image file for cam2 (RGB left) at the specified index."""
        if path_only:
            return self.cam2_files[idx]
        else:
            return load_image(self.cam2_files[idx], mode='RGB')

    @property
    def cam3s(self):
        """Generator to read image files for cam0 (RGB right)."""
        return yield_images(self.cam3_files, mode='RGB')

    def get_cam3_by_idx(self, idx, path_only=False):
        """Read image file for cam3 (RGB right) at the specified index."""
        if path_only:
            return self.cam3_files[idx]
        else:
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

    @property
    def poses(self):
        """Generator for poses."""
        for view_num in self.view_nums:
            yield self.pose(view_num)

    def get_pose_by_idx(self, idx):
        """Get pose at the specified index."""
        return self._format_pose(self.oxts[idx].T_w_imu)

    def _project_velo_to_cam2(self, velo, view_num):
        size = self.cam2(view_num).size
        velo[:, 3] = 1.0  # homogenous

        # velo is np array of shape (NUM_POINTS, 4)
        P_rect2_velo = np.dot(np.dot(self.calib.P_rect2_rect0, self.calib.T_rect0_cam0), self.calib.T_cam0_velo)
        # P_rect2_velo = np.dot(self.calib.K_cam2, self.calib.T_rect2_velo[:3, :])  # just the same

        velo = velo[velo[:, 0] >= 0, :]  # remove all behind image plane (approximation)
        velo_cam2 = np.dot(P_rect2_velo, velo.T).T
        velo_cam2[:, :2] = velo_cam2[:, :2] / velo_cam2[:, 2][..., np.newaxis]

        if CONFIG.data.velo_depth:
            velo_cam2[:, 2] = velo[:, 0]

        velo_cam2[:, 0] = np.round(velo_cam2[:, 0]) - 1
        velo_cam2[:, 1] = np.round(velo_cam2[:, 1]) - 1
        val_inds = (velo_cam2[:, 0] >= 0) & (velo_cam2[:, 1] >= 0)
        val_inds = val_inds & (velo_cam2[:, 0] < size[0]) & (velo_cam2[:, 1] < size[1])
        velo_cam2 = velo_cam2[val_inds, :]

        # project to image
        depth = np.zeros((size[1], size[0]), dtype=np.float32)
        depth[velo_cam2[:, 1].astype(np.int), velo_cam2[:, 0].astype(np.int)] = velo_cam2[:, 2]

        # find the duplicate points and choose the closest depth
        m, n = depth.shape
        inds = velo_cam2[:, 1] * (n-1) + velo_cam2[:, 0] - 1
        dupe_inds = [item for item, count in Counter(inds).items() if count > 1]
        for dd in dupe_inds:
            pts = np.where(inds == dd)[0]
            x_loc = int(velo_cam2[pts[0], 0])
            y_loc = int(velo_cam2[pts[0], 1])
            depth[y_loc, x_loc] = velo_cam2[pts, 2].min()

        depth[depth < 0] = 0

        return depth

    def _format_pose(self, T_w_imu):

        if CONFIG.data.pose_format == 'ref_to_rect0_transform' or CONFIG.data.pose_format == 'rect2_to_ref_transform':
            # ref is the first rect0 frame
            T_imu_rect0 = invert_transform(self.calib.T_rect0_imu)
            T_w_ref = np.dot(self.oxts[0].T_w_imu, T_imu_rect0)
            T_ref_w = invert_transform(T_w_ref)
            T_ref_imu = np.dot(T_ref_w, T_w_imu)
            return np.dot(T_ref_imu, T_imu_rect0)

        elif CONFIG.data.pose_format == 'cam0_to_world_transform':  # as used in DeepV2D
            t_w_ref = trans_from_transform(self.oxts[0].T_w_imu)
            t_w_imu = trans_from_transform(T_w_imu)
            R_w_imu = rot_from_transform(T_w_imu)
            T_ref_imu = transform_from_rot_trans(R_w_imu, t_w_imu - t_w_ref)  # different ref as in my definition
            T_imu_ref = invert_transform(T_ref_imu)
            T_cam0_imu = np.dot(self.calib.T_cam0_velo, self.calib.T_velo_imu)
            T_cam0_ref = np.dot(T_cam0_imu, T_imu_ref)
            return T_cam0_ref

        # elif CONFIG.data.pose_format == 'ref_to_rect2_transform':
        #     T_imu_rect2 = invert_transform(self.calib.T_rect2_imu)
        #     T_w_ref = np.dot(self.oxts[0].T_w_imu, T_imu_rect2)
        #     T_ref_w = invert_transform(T_w_ref)
        #     T_ref_imu = np.dot(T_ref_w, T_w_imu)
        #     return np.dot(T_ref_imu, T_imu_rect2)

        # elif CONFIG.data.pose_format == 'rect2_to_ref_transform':
        #     T_imu_rect2 = invert_transform(self.calib.T_rect2_imu)
        #     T_w_ref = np.dot(self.oxts[0].T_w_imu, T_imu_rect2)
        #     T_ref_w = invert_transform(T_w_ref)
        #     T_ref_imu = np.dot(T_ref_w, T_w_imu)
        #     T_ref_rect2 = np.dot(T_ref_imu, T_imu_rect2)
        #     return invert_transform(T_ref_rect2)



    @property
    def _initial_pose(self):
        T_imu_cam0 = invert_transform(np.dot(self.calib.T_cam0_velo, self.calib.T_velo_imu))
        return np.dot(self.oxts[0].T_w_imu, T_imu_cam0)

    def _get_T_w_ref(self):
        T_imu_rect0 = invert_transform(self.calib.T_rect0_imu)
        T_w_ref = np.dot(self.oxts[0].T_w_imu, T_imu_rect0)
        return T_w_ref

    def _get_file_lists(self):
        """Find and list data files for each sensor."""
        self.oxts_files = sorted(glob.glob(
            os.path.join(self.data_path, 'oxts', 'data', '*.txt')))
        self.cam0_files = sorted(glob.glob(
            os.path.join(self.data_path, 'image_00',
                         'data', '*.{}'.format(self.imtype))))
        self.cam1_files = sorted(glob.glob(
            os.path.join(self.data_path, 'image_01',
                         'data', '*.{}'.format(self.imtype))))
        self.cam2_files = sorted(glob.glob(
            os.path.join(self.data_path, 'image_02',
                         'data', '*.{}'.format(self.imtype))))
        self.cam3_files = sorted(glob.glob(
            os.path.join(self.data_path, 'image_03',
                         'data', '*.{}'.format(self.imtype))))
        self.velo_files = sorted(glob.glob(
            os.path.join(self.data_path, 'velodyne_points',
                         'data', '*.bin')))

        # Subselect the chosen range of frames, if any
        if self.view_nums is not None:
            self.oxts_files = subselect_depth_files_by_name(
                self.oxts_files, self.view_nums)
            self.cam0_files = subselect_depth_files_by_name(
                self.cam0_files, self.view_nums)
            self.cam1_files = subselect_depth_files_by_name(
                self.cam1_files, self.view_nums)
            self.cam2_files = subselect_depth_files_by_name(
                self.cam2_files, self.view_nums)
            self.cam3_files = subselect_depth_files_by_name(
                self.cam3_files, self.view_nums)
            self.velo_files = subselect_depth_files_by_name(
                self.velo_files, self.view_nums)

        # Set (if was None before), or update view_nums
        oxts_view_nums = [view_num_to_int(view_num_from_path(path)) for path in self.oxts_files]
        cam0_view_nums = [view_num_to_int(view_num_from_path(path)) for path in self.cam0_files]
        cam1_view_nums = [view_num_to_int(view_num_from_path(path)) for path in self.cam1_files]
        cam2_view_nums = [view_num_to_int(view_num_from_path(path)) for path in self.cam2_files]
        cam3_view_nums = [view_num_to_int(view_num_from_path(path)) for path in self.cam3_files]
        velo_view_nums = [view_num_to_int(view_num_from_path(path)) for path in self.velo_files]

        def filter_duplicates(view_nums):
            tmp_len = len(view_nums)
            view_nums = list(set(view_nums))
            if len(view_nums) != tmp_len:
                print("WARNING: %d duplicate views found in raw sequence %s." % (tmp_len - len(view_nums), self.drive))

        filter_duplicates(oxts_view_nums)
        filter_duplicates(cam0_view_nums)
        filter_duplicates(cam1_view_nums)
        filter_duplicates(cam2_view_nums)
        filter_duplicates(cam3_view_nums)
        filter_duplicates(velo_view_nums)

        total_view_nums = sorted(list(
            set(oxts_view_nums + cam0_view_nums + cam1_view_nums + cam2_view_nums + cam3_view_nums + velo_view_nums)))

        if not set(total_view_nums) == set(oxts_view_nums) == set(cam0_view_nums) == set(cam1_view_nums) == set(
                cam2_view_nums) == set(cam3_view_nums) == set(velo_view_nums):
            print("WARNING: Raw data of sequence %s is not available for all views." % self.drive)
            print("View nums in oxts: %d." % len(oxts_view_nums))
            print("View nums in cam0: %d." % len(cam0_view_nums))
            print("View nums in cam1: %d." % len(cam1_view_nums))
            print("View nums in cam2: %d." % len(cam2_view_nums))
            print("View nums in cam3: %d." % len(cam3_view_nums))
            print("View nums in velo: %d." % len(velo_view_nums))

        if self.view_nums is not None and set(self.view_nums) != set(total_view_nums):
            print("WARNING (raw sequence %s): Given list of view nums could not be successfully loaded." % self.drive)

        self.oxts_view_nums = oxts_view_nums
        self.cam0_view_nums = cam0_view_nums
        self.cam1_view_nums = cam1_view_nums
        self.cam2_view_nums = cam2_view_nums
        self.cam3_view_nums = cam3_view_nums
        self.velo_view_nums = velo_view_nums
        self.view_nums = total_view_nums

    def _load_calib_rigid(self, filename):
        """Read a rigid transform calibration file as a numpy.array."""
        filepath = os.path.join(self.calib_path, filename)
        data = read_calib_file(filepath)
        return transform_from_rot_trans(data['R'], data['T'])

    def _load_calib_cam_to_cam(self, velo_to_cam_file, cam_to_cam_file):
        # We'll return the camera calibration as a dictionary
        data = {}

        # Load the rigid transformation from velodyne coordinates
        # to unrectified cam0 coordinates
        T_cam0_velo = self._load_calib_rigid(velo_to_cam_file)
        data['T_cam0_velo'] = T_cam0_velo

        # Load and parse the cam-to-cam calibration data
        cam_to_cam_filepath = os.path.join(self.calib_path, cam_to_cam_file)
        filedata = read_calib_file(cam_to_cam_filepath)

        # Create 3x4 projection matrices
        P_rect0_rect0 = np.reshape(filedata['P_rect_00'], (3, 4))
        P_rect1_rect0 = np.reshape(filedata['P_rect_01'], (3, 4))
        P_rect2_rect0 = np.reshape(filedata['P_rect_02'], (3, 4))
        P_rect3_rect0 = np.reshape(filedata['P_rect_03'], (3, 4))
        data['P_rect0_rect0'] = P_rect0_rect0
        data['P_rect1_rect0'] = P_rect1_rect0
        data['P_rect2_rect0'] = P_rect2_rect0
        data['P_rect3_rect0'] = P_rect3_rect0

        data['T_cam0_cam0'] = transform_from_rot_trans(filedata['R_00'], filedata['T_00'])
        data['T_cam1_cam0'] = transform_from_rot_trans(filedata['R_01'], filedata['T_01'])
        data['T_cam2_cam0'] = transform_from_rot_trans(filedata['R_02'], filedata['T_02'])
        data['T_cam3_cam0'] = transform_from_rot_trans(filedata['R_03'], filedata['T_03'])

        # Create 4x4 matrices from the rectifying rotation matrices
        T_rect0_cam0 = np.eye(4)
        T_rect0_cam0[0:3, 0:3] = np.reshape(filedata['R_rect_00'], (3, 3))
        T_rect1_cam1 = np.eye(4)
        T_rect1_cam1[0:3, 0:3] = np.reshape(filedata['R_rect_01'], (3, 3))
        T_rect2_cam2 = np.eye(4)
        T_rect2_cam2[0:3, 0:3] = np.reshape(filedata['R_rect_02'], (3, 3))
        T_rect3_cam3 = np.eye(4)
        T_rect3_cam3[0:3, 0:3] = np.reshape(filedata['R_rect_03'], (3, 3))

        data['T_rect0_cam0'] = T_rect0_cam0

        data['T_rect0_cam1'] = np.dot(data['T_rect0_cam0'], invert_transform(data['T_cam1_cam0']))
        data['T_rect0_cam2'] = np.dot(data['T_rect0_cam0'], invert_transform(data['T_cam2_cam0']))
        data['T_rect0_cam3'] = np.dot(data['T_rect0_cam0'], invert_transform(data['T_cam3_cam0']))

        data[
            'T_rect1_cam1'] = T_rect1_cam1  # also R_rectX_cam1, because all the rect systems have no rotation in between
        data[
            'T_rect2_cam2'] = T_rect2_cam2  # also R_rectX_cam2, because all the rect systems have no rotation in between
        data[
            'T_rect3_cam3'] = T_rect3_cam3  # also R_rectX_cam3, because all the rect systems have no rotation in between

        T_rect0_rect0 = np.eye(4)
        tz = P_rect0_rect0[2, 3]
        T_rect0_rect0[0, 3] = (P_rect0_rect0[0, 3] - P_rect0_rect0[0, 2] * tz) / P_rect0_rect0[0, 0]
        T_rect0_rect0[1, 3] = (P_rect0_rect0[1, 3] - P_rect0_rect0[1, 2] * tz) / P_rect0_rect0[1, 1]
        T_rect0_rect0[2, 3] = tz

        T_rect1_rect0 = np.eye(4)
        tz = P_rect1_rect0[2, 3]
        T_rect1_rect0[0, 3] = (P_rect1_rect0[0, 3] - P_rect1_rect0[0, 2] * tz) / P_rect1_rect0[0, 0]
        T_rect1_rect0[1, 3] = (P_rect1_rect0[1, 3] - P_rect1_rect0[1, 2] * tz) / P_rect1_rect0[1, 1]
        T_rect1_rect0[2, 3] = tz

        T_rect2_rect0 = np.eye(4)
        tz = P_rect2_rect0[2, 3]
        T_rect2_rect0[0, 3] = (P_rect2_rect0[0, 3] - P_rect2_rect0[0, 2] * tz) / P_rect2_rect0[0, 0]
        T_rect2_rect0[1, 3] = (P_rect2_rect0[1, 3] - P_rect2_rect0[1, 2] * tz) / P_rect2_rect0[1, 1]
        T_rect2_rect0[2, 3] = tz

        T_rect3_rect0 = np.eye(4)
        tz = P_rect3_rect0[2, 3]
        T_rect3_rect0[0, 3] = (P_rect3_rect0[0, 3] - P_rect3_rect0[0, 2] * tz) / P_rect3_rect0[0, 0]
        T_rect3_rect0[1, 3] = (P_rect3_rect0[1, 3] - P_rect3_rect0[1, 2] * tz) / P_rect3_rect0[1, 1]
        T_rect3_rect0[2, 3] = tz

        data['T_rect0_rect0'] = T_rect0_rect0
        data['T_rect1_rect0'] = T_rect1_rect0
        data['T_rect2_rect0'] = T_rect2_rect0
        data['T_rect3_rect0'] = T_rect3_rect0
        # to check that this is correct:
        # invert_transform(np.dot(seq.calib.T_rect0_cam3, invert_transform(seq.calib.T_rect3_cam3)))[0:3,3] - seq.calib.T_rect3_rect0[0:3,3]
        # invert_transform(np.dot(seq.calib.T_rect0_cam2, invert_transform(seq.calib.T_rect2_cam2)))[0:3,3] - seq.calib.T_rect2_rect0[0:3,3]
        # invert_transform(np.dot(seq.calib.T_rect0_cam1, invert_transform(seq.calib.T_rect1_cam1)))[0:3,3] - seq.calib.T_rect1_rect0[0:3,3]
        # invert_transform(np.dot(seq.calib.T_rect0_cam0, invert_transform(seq.calib.T_rect0_cam0)))[0:3,3] - seq.calib.T_rect0_rect0[0:3,3]

        # Compute the rectified camera coordinate to velodyne transforms
        data['T_rect0_velo'] = T_rect0_rect0.dot(T_rect0_cam0.dot(T_cam0_velo))
        data['T_rect1_velo'] = T_rect1_rect0.dot(T_rect0_cam0.dot(T_cam0_velo))
        data['T_rect2_velo'] = T_rect2_rect0.dot(T_rect0_cam0.dot(T_cam0_velo))
        data['T_rect3_velo'] = T_rect3_rect0.dot(T_rect0_cam0.dot(T_cam0_velo))

        # Compute the camera intrinsics
        data['K_cam0'] = P_rect0_rect0[0:3, 0:3]
        data['K_cam1'] = P_rect1_rect0[0:3, 0:3]
        data['K_cam2'] = P_rect2_rect0[0:3, 0:3]
        data['K_cam3'] = P_rect3_rect0[0:3, 0:3]

        # Compute the stereo baselines in meters by projecting the origin of
        # each camera frame into the velodyne frame and computing the distances
        # between them
        p_cam = np.array([0, 0, 0, 1])
        p_velo0 = np.linalg.inv(data['T_rect0_velo']).dot(p_cam)
        p_velo1 = np.linalg.inv(data['T_rect1_velo']).dot(p_cam)
        p_velo2 = np.linalg.inv(data['T_rect2_velo']).dot(p_cam)
        p_velo3 = np.linalg.inv(data['T_rect3_velo']).dot(p_cam)

        data['b_gray'] = np.linalg.norm(p_velo1 - p_velo0)  # gray baseline
        data['b_rgb'] = np.linalg.norm(p_velo3 - p_velo2)  # rgb baseline

        return data

    def _load_calib(self):
        """Load and compute intrinsic and extrinsic calibration parameters."""
        # We'll build the calibration parameters as a dictionary, then
        # convert it to a namedtuple to prevent it from being modified later
        data = {}

        # Load the rigid transformation from IMU to velodyne
        data['T_velo_imu'] = self._load_calib_rigid('calib_imu_to_velo.txt')

        # Load the camera intrinsics and extrinsics
        data.update(self._load_calib_cam_to_cam(
            'calib_velo_to_cam.txt', 'calib_cam_to_cam.txt'))

        # Pre-compute the IMU to rectified camera coordinate transforms
        data['T_rect0_imu'] = data['T_rect0_velo'].dot(data['T_velo_imu'])
        data['T_rect1_imu'] = data['T_rect1_velo'].dot(data['T_velo_imu'])
        data['T_rect2_imu'] = data['T_rect2_velo'].dot(data['T_velo_imu'])
        data['T_rect3_imu'] = data['T_rect3_velo'].dot(data['T_velo_imu'])

        self.calib = namedtuple('CalibData', data.keys())(*data.values())

    def _load_timestamps(self):
        """Load timestamps from file."""
        timestamp_file = os.path.join(
            self.data_path, 'oxts', 'timestamps.txt')

        # Read and parse the timestamps
        self.timestamps = []
        with open(timestamp_file, 'r') as f:
            for line in f.readlines():
                # NB: datetime only supports microseconds, but KITTI timestamps
                # give nanoseconds, so need to truncate last 4 characters to
                # get rid of \n (counts as 1) and extra 3 digits
                t = dt.datetime.strptime(line[:-4], '%Y-%m-%d %H:%M:%S.%f')
                self.timestamps.append(t)

        # Subselect the chosen range of frames, if any
        if self.view_nums is not None:
            self.timestamps = [self.timestamps[i] for i in self.view_nums]

    def _load_oxts(self):
        """Load OXTS data from file."""
        self.oxts = load_oxts_packets_and_poses(self.oxts_files)

    def _viewnum_to_idx(self, view_num):
        return self.view_nums.index(view_num)
