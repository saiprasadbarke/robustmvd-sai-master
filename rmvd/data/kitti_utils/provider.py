import csv

import numpy as np

from .config import CONFIG
from .utils import *
from .raw import RawSequence
from .odometry import has_odometry, odometry
from .depth import DepthSequence, has_depth
import matplotlib.pyplot as plt


class KittiProvider:
    def __init__(self, paths_only=False):
        self.paths_only = paths_only
        self.seqs = []

    @property
    def sequences(self):
        self._load_split_file()

        for seq in self.seqs:
            seq_name = seq['seq_name']
            seq_view_nums = seq['view_nums']
            part = seq.get('part', None)
            yield KittiSequenceProvider(seq_name=seq_name, view_nums=seq_view_nums, part=part, paths_only=self.paths_only)

    @staticmethod
    def set_config(conf_path, split_file=None, kitti_root=None):
        CONFIG.from_toml(conf_path, split_file=split_file, kitti_root=kitti_root)

    def _load_split_file(self):
        self.seqs = []
        split_file = CONFIG.data.split_file
        with open(split_file) as f:
            reader = csv.reader(f, delimiter=';')
            for x in reader:
                seq = {'seq_name': x[0], 'view_nums': [int(i) for i in x[1:]]}

                seq_name_ctr = 0
                for existing_name in [s['seq_name'] for s in self.seqs]:
                    if existing_name == seq['seq_name']:
                        seq_name_ctr += 1
                if seq_name_ctr > 0:
                    seq['part'] = seq_name_ctr + 1

                self.seqs.append(seq)

    def _sanity_check_gps_to_pose(self):
        self.set_config('/home/schroepp/workspace/kitti_utils/confs/sanity_check_gps_to_pose_conf.toml')
        mean_diff_ts = []
        for seq in self.sequences:
            mean_diff_t = seq._sanity_check_gps_to_pose()
            mean_diff_ts.append(mean_diff_t)

        mean_diff_t = np.zeros(3)
        for diff_t in mean_diff_ts:
            mean_diff_t += diff_t
        mean_diff_t /= len(mean_diff_ts)

        print("Overall:")
        print(mean_diff_t)


class KittiSequenceProvider:
    def __init__(self, seq_name, view_nums, part=None, paths_only=False):
        self.seq_name = seq_name

        self.out_name = self.seq_name + "_part" + str(part) if part is not None else self.seq_name

        self.raw = RawSequence(*split_seqname(seq_name), view_nums=view_nums)

        self.view_nums = self.raw.view_nums.copy()

        self.paths_only = paths_only

        if self._odom_required():
            if has_odometry(seq_name):
                self.odom = odometry(seq_name=seq_name, view_nums=view_nums)
                self.view_nums += self.odom.view_nums
            else:
                self.odom = None
                print("WARNING (sequence %s): No odom data available." % self.seq_name)
        else:
            self.odom = None

        if self._depth_required():
            if has_depth(seq_name):
                self.depth = DepthSequence(seq_name=seq_name, view_nums=view_nums)
                self.view_nums += self.depth.view_nums
            else:
                self.depth = None
                print("WARNING (sequence %s): No depth data available." % self.seq_name)
        else:
            self.depth = None

        if self.depth is not None:
            if not set(self.raw.view_nums) == set(self.depth.view_nums):
                print("WARNING (sequence %s): subsets have different number of views." % self.seq_name)
                print("\tRaw subset: %d" % len(self.raw.view_nums))
                print("\tDepth subset: %d" % len(self.depth.view_nums))

        if self.odom is not None:
            if not set(self.raw.view_nums) == set(self.odom.view_nums):
                print("WARNING (sequence %s): subsets have different number of views." % self.seq_name)
                print("\tRaw subset: %d" % len(self.raw.view_nums))
                print("\tOdom subset: %d" % len(self.odom.view_nums))

        self.view_nums = sorted(list(set(self.view_nums)))

    @property
    def views(self):
        for view_num in self.view_nums:
            yield self.view(view_num)

    def view(self, view_num):
        return self._construct_datablob(view_num)

    def _check_views(self):
        # only for debugging
        for view_num in self.view_nums:
            if self.depth is not None and not self.depth.has_viewnum(view_num):
                print('Depth data does not contain view %d for sequence %s.' % (view_num, self.seq_name))
            if self.odom is not None and not self.odom.has_viewnum(view_num):
                print('Odom data does not contain view %d for sequence %s.' % (view_num, self.seq_name))

    def _odom_required(self):
        return CONFIG.data.odom_pose and not CONFIG.data.raw_pose

    def _depth_required(self):
        return CONFIG.data.depth_cam2gt or CONFIG.data.depth_cam3gt

    def _construct_datablob(self, view_num):
        view = {}

        view['view_num'] = view_num

        if CONFIG.data.raw_cam2:
            view['cam2'] = self._get_raw_cam2(view_num)
        if CONFIG.data.raw_cam3:
            view['cam3'] = self._get_raw_cam3(view_num)
        if CONFIG.data.depth_cam2gt or CONFIG.data.depth_cam2velo:
            view['depth_cam2'] = self._get_depth_cam2(view_num)
        if CONFIG.data.odom_pose or CONFIG.data.raw_pose:
            view['pose'] = self._get_pose(view_num)
        if CONFIG.data.raw_K_cam2:
            view['K_cam2'] = self._get_K_cam2(view_num)
        if CONFIG.data.raw_K_cam3:
            view['K_cam3'] = self._get_K_cam3(view_num)

        if CONFIG.proc.proc_data:
            view = self._proc_data(view)

        return view

    def _get_raw_cam2(self, view_num):
        if self.raw.has_viewnum(view_num):
            return self.raw.cam2(view_num, self.paths_only)
        else:
            raise ValueError('Raw data does not contain view %d for sequence %s.' % (view_num, self.seq_name))

    def _get_raw_cam3(self, view_num):
        if self.raw.has_viewnum(view_num):
            return self.raw.cam3(view_num, self.paths_only)
        else:
            raise ValueError('Raw data does not contain view %d for sequence %s.' % (view_num, self.seq_name))

    def _get_K_cam2(self, view_num):
        return self.raw.calib.K_cam2

    def _get_K_cam3(self, view_num):
        return self.raw.calib.K_cam3

    def _get_depth_cam2(self, view_num):
        if CONFIG.data.depth_cam2gt and CONFIG.data.depth_cam2velo:
            if self.depth.has_viewnum(view_num):
                depth = self.depth.cam2_gtdepth(view_num, self.paths_only)
            elif self.raw.has_viewnum(view_num):
                depth = self.raw.velo_cam2(view_num)
                print("Using fallback depth from raw velo data for view %d from sequence %s." % (view_num, self.seq_name))
            else:
                raise ValueError('Depth data and raw velo data do not contain view %d for sequence %s.' % (view_num, self.seq_name))

        elif CONFIG.data.depth_cam2gt:
            if self.depth.has_viewnum(view_num):
                depth = self.depth.cam2_gtdepth(view_num)
            else:
                raise ValueError('Depth data does not contain view %d for sequence %s.' % (view_num, self.seq_name))

        elif CONFIG.data.depth_cam2velo:
            if self.raw.has_viewnum(view_num):
                depth = self.raw.velo_cam2(view_num)
            else:
                raise ValueError('Raw velo data does not contain view %d for sequence %s.' % (view_num, self.seq_name))

        else:
            raise ValueError(
                'Sequence %s, view %d: Not specified where to get cam2 depth from.' % (self.seq_name, view_num))

        if CONFIG.proc.scale != 1.0:
            depth *= CONFIG.proc.scale

        return depth

    def _get_pose(self, view_num):
        if CONFIG.data.odom_pose and CONFIG.data.raw_pose:
            raise ValueError("Odom poses and raw poses should not be mixed due to potential reference frame mismatch.")
        elif CONFIG.data.odom_pose:
            if self.odom.has_viewnum(view_num):
                pose = self.odom.pose(view_num)
            else:
                raise ValueError('Odom data does not contain view %d for sequence %s.' % (view_num, self.seq_name))
        elif CONFIG.data.raw_pose:
            if self.raw.has_viewnum(view_num):
                pose = self.raw.pose(view_num)
            else:
                raise ValueError('Raw data does not contain view %d for sequence %s.' % (view_num, self.seq_name))
        else:
            raise ValueError('Sequence %s, view %d: Not specified where to get pose from.' % (self.seq_name, view_num))

        return self._format_pose(pose)

    def _format_pose(self, pose):

        if CONFIG.data.pose_format == 'ref_to_rect0_transform':
            pose_out = pose

        elif CONFIG.data.pose_format == 'rect2_to_ref_transform':  # I am using this
            rect0_ref_transform = invert_transform(pose)  # pose is Ref to Rect0 transform ^Ref_R0_T -> we have to invert it
            pose = np.dot(self.raw.calib.T_rect2_rect0, rect0_ref_transform)
            pose_out = pose

        elif CONFIG.data.pose_format == 'cam0_to_world_transform':
            if CONFIG.data.odom_pose:
                raise ValueError("Can't retrieve cam0_t0_world_transform from odom benchmark.")
            elif CONFIG.data.raw_pose:
                pose_out = pose

        if CONFIG.proc.scale != 1.0:
            R = rot_from_transform(pose_out)
            t = trans_from_transform(pose_out)
            t *= CONFIG.proc.scale
            pose_out = transform_from_rot_trans(R, t)

        return pose_out

    def _proc_data(self, view):

        target_width = CONFIG.proc.target_width
        target_height = CONFIG.proc.target_height

        if CONFIG.proc.target_K is not None:

            target_K = CONFIG.proc.target_K

            cam2, depth_cam2 = proc.adjust_intrinsics(view['K_cam2'], target_K, target_width, target_height,
                                                      view['cam2'], view['depth_cam2'])
            view['cam2'] = cam2
            view['depth_cam2'] = depth_cam2
            view['K_cam2'] = target_K

        else:
            random_crop = CONFIG.proc.random_crop
            cam2, depth_cam2, K_cam2 = proc.adjust_resolution(target_width, target_height, view['cam2'],
                                                              view['depth_cam2'], view['K_cam2'],
                                                              random_crop=random_crop)

            view['cam2'] = cam2
            view['depth_cam2'] = depth_cam2
            view['K_cam2'] = K_cam2

        return view

    def _sanity_check_gps_to_pose(self):
        print()
        print("Sanity checking the GPS to Pose conversion for sequence %s." % self.seq_name)

        diff_ts = []
        raw_ts = []
        odom_ts = []
        diff_Rs = []

        for view_num in self.view_nums:
            raw_pose = self.raw.pose(view_num)
            odom_pose = self.odom.pose(view_num)

            raw_R = raw_pose[0:3, 0:3]
            raw_t = raw_pose[0:3, 3]
            odom_R = odom_pose[0:3, 0:3]
            odom_t = odom_pose[0:3, 3]

            diff_t = raw_t - odom_t
            diff_R = np.abs(raw_R - odom_R)

            print("\tView num %d:" % view_num)
            print("\tRaw t: ", raw_t)
            print("\tOdom t: ", odom_t)
            print("\tDiff t: ", diff_t)
            print("\tMagnitude raw t: ", np.linalg.norm(raw_t))
            print("\tMagnitude odom t: ", np.linalg.norm(odom_t))
            print("\tMagnitude absDiff t: ", np.linalg.norm(diff_t))
            print("\tRaw R: ")
            print(raw_R)
            print("\tOdom R: ")
            print(odom_R)
            print()

            diff_ts.append(diff_t)
            raw_ts.append(raw_t)
            odom_ts.append(odom_t)
            diff_Rs.append(diff_R)

        mean_diff_t = np.zeros(3)
        mean_diff_R = np.zeros((3, 3))
        for diff_t in diff_ts:
            mean_diff_t += diff_t
        mean_diff_t /= len(diff_ts)
        for diff_R in diff_Rs:
            mean_diff_R += diff_R
        mean_diff_R /= len(diff_Rs)
        print("Mean Diff t: ")
        print(mean_diff_t)
        print("Mean Diff R: ")
        print(mean_diff_R)
        print()
        plt.clf()
        plt.plot([a[0] for a in odom_ts], label='odom_tx')
        plt.plot([a[1] for a in odom_ts], label='odom_ty')
        plt.plot([a[2] for a in odom_ts], label='odom_tz')
        plt.plot([a[0] for a in raw_ts], label='raw_tx')
        plt.plot([a[1] for a in raw_ts], label='raw_ty')
        plt.plot([a[2] for a in raw_ts], label='raw_tz')
        plt.legend()
        plt.savefig('/home/schroepp/workspace/kitti_utils/tests/%s_translations.png' % self.seq_name)
        plt.clf()
        plt.plot([a[0] for a in diff_ts], label='diff_tx')
        plt.plot([a[1] for a in diff_ts], label='diff_ty')
        plt.plot([a[2] for a in diff_ts], label='diff_tz')
        plt.legend()
        plt.savefig('/home/schroepp/workspace/kitti_utils/tests/%s_translation_errors.png' % self.seq_name)

        return mean_diff_t
