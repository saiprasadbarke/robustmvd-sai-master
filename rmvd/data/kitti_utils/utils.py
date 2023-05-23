# copied and adapted from pykitti package
"""Provides helper methods for loading and parsing KITTI data."""

from collections import namedtuple

import numpy as np
from PIL import Image
import os
from collections import Counter

__author__ = "Lee Clement"
__email__ = "lee.clement@robotics.utias.utoronto.ca"

# Per dataformat.txt
OxtsPacket = namedtuple('OxtsPacket',
                        'lat, lon, alt, ' +
                        'roll, pitch, yaw, ' +
                        'vn, ve, vf, vl, vu, ' +
                        'ax, ay, az, af, al, au, ' +
                        'wx, wy, wz, wf, wl, wu, ' +
                        'pos_accuracy, vel_accuracy, ' +
                        'navstat, numsats, ' +
                        'posmode, velmode, orimode')

# Bundle into an easy-to-access structure
OxtsData = namedtuple('OxtsData', 'packet, T_w_imu')


def subselect_files(files, indices):
    try:
        files = [files[i] for i in indices]
    except:
        pass
    return files


def subselect_odom_files_by_name(files, indices):
    out_files = []
    for file in files:
        index = int(file[-10:-4])
        if index in indices:
            out_files.append(file)
    return out_files


def subselect_depth_files_by_name(files, indices):
    out_files = []
    for file in files:
        index = view_num_to_int(view_num_from_path(file))
        if index in indices:
            out_files.append(file)
    return out_files


def view_num_from_path(path):
    return path[-14:-4]


def view_num_to_int(view_num):
    return int(view_num)


def view_num_to_str(view_num):
    return str(view_num).zfill(10)


def split_seqname(seqname):
    """Splits e.g. 2011_09_29_drive_0108_sync to 2011_09_29 and 0108"""
    return seqname[0:10], seqname[17:21]


def rotx(t):
    """Rotation about the x-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s, c]])


def roty(t):
    """Rotation about the y-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]])


def rotz(t):
    """Rotation about the z-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s, 0],
                     [s, c, 0],
                     [0, 0, 1]])


def unit_transform():
    R = np.array([[1., 0, 0],
                  [0, 1., 0],
                  [0, 0, 1.]])

    t = np.array([0., 0., 0.])

    return transform_from_rot_trans(R, t)


def transform_from_rot_trans(R, t):
    """Transforation matrix from rotation matrix and translation vector."""
    R = R.reshape(3, 3)
    t = t.reshape(3, 1)
    return np.vstack((np.hstack([R, t]), [0, 0, 0, 1]))


def trans_from_transform(T):
    return T[0:3, 3]


def rot_from_transform(T):
    return T[0:3, 0:3]


def invert_transform(T):
    R = T[0:3, 0:3]
    t = T[0:3, 3]
    R_inv = R.T
    t_inv = np.dot(-R.T, t)
    return transform_from_rot_trans(R_inv, t_inv)


def sub2ind(matrixSize, rowSub, colSub):
    """Convert row, col matrix subscripts to linear indices
    (taken from https://github.com/nianticlabs/monodepth2/blob/master/kitti_utils.py )
    """
    m, n = matrixSize
    return rowSub * (n - 1) + colSub - 1


def read_calib_file(filepath):
    """Read in a calibration file and parse into a dictionary.
    (from pykitti)
    """
    data = {}

    with open(filepath, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass

    return data


# def read_calib_file(path):
#     """Read KITTI calibration file
#     (from https://github.com/hunse/kitti)
#     """
#     float_chars = set("0123456789.e+- ")
#     data = {}
#     with open(path, 'r') as f:
#         for line in f.readlines():
#             key, value = line.split(':', 1)
#             value = value.strip()
#             data[key] = value
#             if float_chars.issuperset(value):
#                 # try to cast to float array
#                 try:
#                     data[key] = np.array(list(map(float, value.split(' '))))
#                 except ValueError:
#                     # casting error: data[key] already eq. value, so pass
#                     pass
#
#     return data


def pose_from_oxts_packet(packet, scale):
    """Helper method to compute a SE(3) pose matrix from an OXTS packet (the world to imu transform).
    """
    er = 6378137.  # earth radius (approx.) in meters

    # Use a Mercator projection to get the translation vector
    tx = scale * packet.lon * np.pi * er / 180.
    ty = scale * er * np.log(np.tan((90. + packet.lat) * np.pi / 360.))
    tz = packet.alt
    t = np.array([tx, ty, tz])

    # Use the Euler angles to get the rotation matrix
    Rx = rotx(packet.roll)
    Ry = roty(packet.pitch)
    Rz = rotz(packet.yaw)
    R = Rz.dot(Ry.dot(Rx))

    # Combine the translation and rotation into a homogeneous transform
    return R, t


def load_oxts_packets_and_poses(oxts_files):
    """Generator to read OXTS ground truth data.

       Poses are given in an East-North-Up coordinate system 
       whose origin is the first GPS position.
    """
    # Scale for Mercator projection (from first lat value)
    scale = None
    # Origin of the global coordinate system (first GPS position)
    origin = None

    oxts = []

    for filename in oxts_files:
        with open(filename, 'r') as f:
            for line in f.readlines():
                line = line.split()
                # Last five entries are flags and counts
                line[:-5] = [float(x) for x in line[:-5]]
                line[-5:] = [int(float(x)) for x in line[-5:]]

                packet = OxtsPacket(*line)

                if scale is None:
                    scale = np.cos(packet.lat * np.pi / 180.)

                R, t = pose_from_oxts_packet(packet, scale)

                if origin is None:
                    origin = invert_transform(transform_from_rot_trans(R, t))

                # T_w_imu = np.dot(origin, transform_from_rot_trans(R, t))
                T_w_imu = transform_from_rot_trans(R, t)

                oxts.append(OxtsData(packet, T_w_imu))
                # T_w_imu is the world-to-IMU-transform and world is # the imu pose at the first frame

    return oxts


def load_image(file, mode):
    """Load an image from file."""
    return Image.open(file).convert(mode)


def yield_images(imfiles, mode):
    """Generator to read image files."""
    for file in imfiles:
        yield load_image(file, mode)


def load_velo_scan(file):
    """Load and parse a velodyne binary file."""
    scan = np.fromfile(file, dtype=np.float32)
    return scan.reshape((-1, 4))


def yield_velo_scans(velo_files):
    """Generator to parse velodyne binary files into arrays."""
    for file in velo_files:
        yield load_velo_scan(file)


def load_depth(file):  # copied from KITTI depth completion devkit
    # loads depth map D from png file
    # and returns it as a numpy array,
    # for details see readme.txt

    depth_png = np.array(Image.open(file), dtype=int)
    # make sure we have a proper 16bit depth map here.. not 8bit!
    assert (np.max(depth_png) > 255)

    depth = depth_png.astype(np.float) / 256.
    depth[depth_png == 0] = np.NaN

    depth = depth.astype(np.float32)

    invalid_values_mask = np.isinf(depth) | np.isnan(depth)
    depth[invalid_values_mask] = 0

    return depth


def yield_depths(depth_files):
    for file in depth_files:
        yield load_depth(file)


def load_velodyne_points(filename):
    """Load 3D point cloud from KITTI file format
    (taken from https://github.com/nianticlabs/monodepth2/blob/master/kitti_utils.py )
    (adapted from https://github.com/hunse/kitti)
    """
    points = np.fromfile(filename, dtype=np.float32).reshape(-1, 4)
    points[:, 3] = 1.0  # homogeneous
    return points


def generate_depth_map(calib_dir, velo_filename, cam=2, vel_depth=False):
    """Generate a depth map from velodyne data
    (taken from https://github.com/nianticlabs/monodepth2/blob/master/kitti_utils.py )
    """
    # load calibration files
    cam2cam = read_calib_file(os.path.join(calib_dir, 'calib_cam_to_cam.txt'))
    velo2cam = read_calib_file(os.path.join(calib_dir, 'calib_velo_to_cam.txt'))
    velo2cam = np.hstack((velo2cam['R'].reshape(3, 3), velo2cam['T'][..., np.newaxis]))
    velo2cam = np.vstack((velo2cam, np.array([0, 0, 0, 1.0])))

    # get image shape
    im_shape = cam2cam["S_rect_02"][::-1].astype(np.int32)

    # compute projection matrix velodyne->image plane
    R_cam2rect = np.eye(4)
    R_cam2rect[:3, :3] = cam2cam['R_rect_00'].reshape(3, 3)
    P_rect = cam2cam['P_rect_0' + str(cam)].reshape(3, 4)
    P_velo2im = np.dot(np.dot(P_rect, R_cam2rect), velo2cam)

    # load velodyne points and remove all behind image plane (approximation)
    # each row of the velodyne data is forward, left, up, reflectance
    velo = load_velodyne_points(velo_filename)
    velo = velo[velo[:, 0] >= 0, :]

    # project the points to the camera
    velo_pts_im = np.dot(P_velo2im, velo.T).T
    velo_pts_im[:, :2] = velo_pts_im[:, :2] / velo_pts_im[:, 2][..., np.newaxis]

    if vel_depth:
        velo_pts_im[:, 2] = velo[:, 0]

    # check if in bounds
    # use minus 1 to get the exact same value as KITTI matlab code
    velo_pts_im[:, 0] = np.round(velo_pts_im[:, 0]) - 1
    velo_pts_im[:, 1] = np.round(velo_pts_im[:, 1]) - 1
    val_inds = (velo_pts_im[:, 0] >= 0) & (velo_pts_im[:, 1] >= 0)
    val_inds = val_inds & (velo_pts_im[:, 0] < im_shape[1]) & (velo_pts_im[:, 1] < im_shape[0])
    velo_pts_im = velo_pts_im[val_inds, :]

    # project to image
    depth = np.zeros((im_shape[:2]))
    depth[velo_pts_im[:, 1].astype(np.int), velo_pts_im[:, 0].astype(np.int)] = velo_pts_im[:, 2]

    # find the duplicate points and choose the closest depth
    inds = sub2ind(depth.shape, velo_pts_im[:, 1], velo_pts_im[:, 0])
    dupe_inds = [item for item, count in Counter(inds).items() if count > 1]
    for dd in dupe_inds:
        pts = np.where(inds == dd)[0]
        x_loc = int(velo_pts_im[pts[0], 0])
        y_loc = int(velo_pts_im[pts[0], 1])
        depth[y_loc, x_loc] = velo_pts_im[pts, 2].min()
    depth[depth < 0] = 0

    return depth


# def convert_disps_to_depths_kitti(gt_disparities, pred_disparities):
# taken from https://raw.githubusercontent.com/mrharicot/monodepth/master/utils/evaluation_utils.py
#     gt_depths = []
#     pred_depths = []
#     pred_disparities_resized = []
#
#     for i in range(len(gt_disparities)):
#         gt_disp = gt_disparities[i]
#         height, width = gt_disp.shape
#
#         pred_disp = pred_disparities[i]
#         pred_disp = width * cv2.resize(pred_disp, (width, height), interpolation=cv2.INTER_LINEAR)
#
#         pred_disparities_resized.append(pred_disp)
#
#         mask = gt_disp > 0
#
#         gt_depth = width_to_focal[width] * 0.54 / (gt_disp + (1.0 - mask))
#         pred_depth = width_to_focal[width] * 0.54 / pred_disp
#
#         gt_depths.append(gt_depth)
#         pred_depths.append(pred_depth)
#     return gt_depths, pred_depths, pred_disparities_resized

# def lin_interp(shape, xyd):
#     # taken from https://github.com/hunse/kitti
#     m, n = shape
#     ij, d = xyd[:, 1::-1], xyd[:, 2]
#     f = LinearNDInterpolator(ij, d, fill_value=0)
#     J, I = np.meshgrid(np.arange(n), np.arange(m))
#     IJ = np.vstack([I.flatten(), J.flatten()]).T
#     disparity = f(IJ).reshape(shape)
#     return disparity


##### DEPTH UTILS ######

def interpolate_depth(depth):
    # taken from KITTI depth-from-single-view benchmark devkit
    height = depth.shape[0]
    width = depth.shape[1]

    # for each row do
    for v in range(height):
        count = 0

        # for each pixel in row do:
        for u in range(width):
            if depth[v][u] > 0:  # is valid

                # at least one pixel required interpolation
                if count >= 1:
                    # first and last value for interpolation
                    u1 = u - count
                    u2 = u - 1

                    # set pixel to min depth
                    if u1 > 0 and u2 < (width - 1):
                        d_ipol = min(depth[v][u1 - 1], depth[v][u2 + 1])
                        for u_cur in range(u1, u2 + 1):
                            depth[v][u_cur] = d_ipol

                # reset counter
                count = 0
            else:  # is not valid, increment counter
                count += 1

        # extrapolate to the left
        for u in range(width):
            if depth[v][u] > 0:  # is valid
                for u2 in range(u):
                    depth[v][u2] = depth[v][u]
                break

        # extrapolate to the right
        for u in reversed(range(0, width)):
            if depth[v][u] > 0:  # is valid
                for u2 in range(u + 1, width):
                    depth[v][u2] = depth[v][u]
                break

    # for each column do
    for u in range(width):

        # extrapolate to the top
        for v in range(height):
            if depth[v][u] > 0:  # is valid
                for v2 in range(v):
                    depth[v2][u] = depth[v][u]
                break

        # extrapolate to the bottom
        for v in reversed(range(0, height)):
            if depth[v][u] > 0:  # is valid
                for v2 in range(v + 1, height):
                    depth[v2][u] = depth[v][u]
                break

    return depth
