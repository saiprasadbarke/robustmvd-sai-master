import numpy as np
import torch
import torch.nn as nn

from rmvd.utils import to_torch


def normalize(x, dim=None, eps=1e-9):
    norm = torch.linalg.norm(x, dim=dim, keepdim=True)
    return x / (norm + eps)


def warp_multi(x, offsets=None, grids=None, padding_mode="border"):
    """
    warp an image/tensor (im2) back to im1, according to multiple offsets (optical flows), or at the given grid locations.
    x: [N, C, h_src, w_src] (im2)
    offsets: [N, S, 2, h_ref, w_ref] offsets
    grids: [N, S, 2, h_out, w_out] grids
    padding_mode: border or zeros

    Output: [N,S,C,h_out,w_out] ; [N, S, h_out, w_out] with h_out=h_ref, w_out=w_ref in case grids is None.
    """

    assert (offsets is None and grids is not None) or (
        grids is None and offsets is not None
    )

    locations = offsets if offsets is not None else grids
    N, S, _, H, W = locations.shape
    K2HW_shape = [N * S, 2, H, W]
    locations_K2HW = torch.reshape(locations, K2HW_shape)
    offsets_K2HW = locations_K2HW if offsets is not None else None
    grids_K2HW = locations_K2HW if grids is not None else None

    x_for_warping = torch.unsqueeze(x, 1).repeat(
        1, S, 1, 1, 1
    )  # NSCHW; costs: N*S*C*H*W*4 Bytes
    NSCHW_shape = x_for_warping.shape
    NSHW_shape = [N, S, H, W]
    N, S, C, H, W = NSCHW_shape
    KCHW_shape = [N * S, C, H, W]
    x_for_warping = torch.reshape(x_for_warping, KCHW_shape)

    x_warped, mask = warp(
        x_for_warping, offset=offsets_K2HW, grid=grids_K2HW, padding_mode=padding_mode
    )  # KCHW ; K1HW where HW is H_ref, W_ref or H_out, W_out
    x_warped = torch.reshape(x_warped, NSCHW_shape)
    mask = torch.reshape(mask, NSHW_shape)

    return x_warped, mask


def warp(
    x, offset=None, grid=None, padding_mode="border"
):  # based on PWC-Net Github Repo
    """
    warp an image/tensor (im2) back to im1, according to the offset (optical flow), or at the given grid locations.
    x: [N, C, H, W] (im2)
    offset: [N, 2, H_ref, W_ref] offset. h_ref/w_ref can differ from size H/W of x.
    grid: [N, 2, h_out, w_out] grid of sampling locations. h_out/w_out can differ from size H/W of x.
    padding_mode: border or zeros

    Output: [N,C,h_out,w_out] ; [N, 1, h_out, w_out] Sampled points from x and sampling masks (with h_out=h_ref, w_out=w_ref in case grids is None).
    """

    N = x.shape[0]

    assert (offset is None and grid is not None) or (
        grid is None and offset is not None
    )

    if offset is not None:
        h_ref, w_ref = offset.shape[-2:]

        with torch.no_grad():
            device = x.get_device()

            yy, xx = torch.meshgrid(
                torch.arange(h_ref), torch.arange(w_ref), indexing="ij"
            )  # both (h_ref, w_ref)
            xx = xx.to(device)
            yy = yy.to(device)
            xx = (xx + 0.5).unsqueeze_(0).unsqueeze_(0)  # (1, 1, h_ref, w_ref)
            yy = (yy + 0.5).unsqueeze_(0).unsqueeze_(0)  # (1, 1, h_ref, w_ref)
            xx = xx.repeat(N, 1, 1, 1)
            yy = yy.repeat(N, 1, 1, 1)
            grid = torch.cat((xx, yy), 1).float()  # N, 2, h_ref, w_ref

            grid.add_(offset)

    # scale grid to [-1,1]
    h_out, w_out = grid.shape[-2:]
    h_x, w_x = x.shape[-2:]

    grid = grid.permute(0, 2, 3, 1)  # N, h_out, w_out, 2  # K, 2, S, 1 -> K, S, 1, 2
    xgrid, ygrid = grid.split([1, 1], dim=-1)
    xgrid = 2 * xgrid / w_x - 1  # RAFT: 2*xgrid/(w_x-1) - 1
    ygrid = 2 * ygrid / h_x - 1  # RAFT: 2*ygrid/(h_x-1) - 1
    grid = torch.cat([xgrid, ygrid], dim=-1)

    output = nn.functional.grid_sample(
        x, grid, padding_mode=padding_mode, align_corners=False
    )  # N,C,h_out,w_out ; costs: N*S*C*H*W*4 Bytes

    if padding_mode == "border":
        mask = torch.ones(
            size=(N, 1, h_out, w_out), device=output.device, requires_grad=False
        )

    else:
        mask = torch.ones(
            size=(N, 1, h_x, w_x), device=output.device, requires_grad=False
        )
        mask = nn.functional.grid_sample(
            mask, grid, padding_mode="zeros", align_corners=False
        )  # N, 1, h_out, w_out
        mask[mask < 0.9999] = 0
        mask[mask > 0] = 1

    return output, mask  # N,C,h_out,w_out ; N,1,h_out,w_out


def compute_groupwise_correlation_5D(feat_src, feat_ref, num_groups):
    # Assume a and b are your 4D tensors with shape (batchsize, channels, height, width)
    # And channels is divisible by num_groups
    B, C, D, H, W = feat_src.shape
    group_channels = C // num_groups

    feat_src = feat_src.reshape(B, num_groups, group_channels, D, H, W)
    feat_ref = feat_ref.reshape(B, num_groups, group_channels, D, H, W)
    corr = (feat_src * feat_ref).sum(dim=2)
    return corr  # (B, num_groups, D, H, W)


class GroupWiseCorr_5D(nn.Module):
    def __init__(self, normalize=False, padding_mode="zeros", num_groups: int = 32):
        super().__init__()
        self.normalize = normalize
        self.padding_mode = padding_mode
        self.num_groups = num_groups

    def forward(self, feat_ref, feat_src, grids=None, mask=None):
        S = grids.shape[1]

        if self.normalize:
            feat_src = normalize(feat_src, dim=1)
            feat_ref = normalize(feat_ref, dim=1)
        warped_feat_src, warping_mask = warp_multi(
            x=feat_src, grids=grids, padding_mode=self.padding_mode
        )  # NSCHW, NSHW

        warped_feat_src_permuted = warped_feat_src.permute(0, 2, 1, 3, 4)

        feat_ref_NCSHW = feat_ref.unsqueeze(2).repeat(1, 1, S, 1, 1)
        groupwise_corr = compute_groupwise_correlation_5D(
            warped_feat_src_permuted,
            feat_ref_NCSHW,
            self.num_groups,  # N, num_groups, S, H, W
        )

        groupwise_corr = groupwise_corr * warping_mask.unsqueeze(1)

        if mask is not None:
            groupwise_corr = groupwise_corr * mask.unsqueeze(1)
            warping_mask = warping_mask * mask

        return groupwise_corr, warping_mask  # N,num_groups,S,H,W ; N,S,H,W


class Concat(nn.Module):
    def __init__(self, normalize=False, padding_mode="zeros"):
        super().__init__()
        self.normalize = normalize
        self.padding_mode = padding_mode

    def forward(self, feat_ref, feat_src, grids=None, mask=None):
        S = grids.shape[1]
        if self.normalize:
            feat_src = normalize(feat_src, dim=1)
            feat_ref = normalize(feat_ref, dim=1)
        warped_feat_src, warping_mask = warp_multi(
            x=feat_src, grids=grids, padding_mode=self.padding_mode
        )  # NSCHW, NSHW
        warped_feat_src = warped_feat_src.permute(0, 2, 1, 3, 4)
        feat_ref_NCSHW = feat_ref.unsqueeze(2).repeat(1, 1, S, 1, 1)

        concat_corr = torch.cat((warped_feat_src, feat_ref_NCSHW), dim=1)

        concat_corr = concat_corr * warping_mask.unsqueeze(1)

        if mask is not None:
            concat_corr = concat_corr * mask.unsqueeze(1)
            warping_mask = warping_mask * mask

        return concat_corr, warping_mask  # N,2*C,S,H,W ; N,S,H,W


class WarpOnlyCorr(nn.Module):
    def __init__(self, normalize=False, padding_mode="zeros"):
        super().__init__()
        self.normalize = normalize
        self.padding_mode = padding_mode

        if isinstance(self.normalize, str):
            assert (
                self.normalize == "dim"
                or self.normalize == "before"
                or self.normalize == "after"
            )

    def forward(self, feat_ref, feat_src, grids=None, mask=None):
        """
        Correlate features feat_ref and feat_src for points in feat_src located at the specified offsets.

        :param feat_ref: Reference feature map. NCHrWr. Unused!
        :param feat_src: Source feature map. NCHsWs.
        :param offsets: Offsets of the S correlation points in the source feature map. NS2HrWr.
        :param grids: Locations of the S correlation points in the source feature map. NS2HrWr.
        :param mask: Mask that marks invalid offsets/grid locations, where the correlation score should be set to 0. NSHrWr.
        :return: Warped source feature maps and warping mask. NSCHrWr and NSHrWr.
        """

        if self.normalize == "before":
            feat_src = normalize(feat_src, dim=1)

        warped_feat_src, warping_mask = warp_multi(
            x=feat_src, grids=grids, padding_mode=self.padding_mode
        )  # NSCHW, NSHW

        if self.normalize and self.normalize != "before" and self.normalize != "dim":
            warped_feat_src = normalize(
                warped_feat_src, dim=2
            )  # increases memory a lot

        warped_feat_src = warped_feat_src * warping_mask.unsqueeze(2)

        return warped_feat_src, warping_mask


class TorchCorr(nn.Module):  # based on RAFT
    def __init__(self, normalize=False, padding_mode="zeros"):
        super().__init__()
        self.normalize = normalize
        self.padding_mode = padding_mode

        if isinstance(self.normalize, str):
            assert self.normalize == "dim" or self.normalize == "before"

    def forward(self, feat_ref, feat_src, grids=None, mask=None):
        """
        Correlate features feat_ref and feat_src for points in feat_src located at the specified offsets.

        :param feat_ref: Reference feature map. NCHrWr.
        :param feat_src: Source feature map. NCHsWs.
        :param grids: Locations of the S correlation points in the source feature map. NS2HrWr.
        :param mask: Mask that marks invalid offsets/grid locations, where the correlation score should be set to 0. NSHrWr.
        :return: Volume of correlation scores and mask. Both NSHrWr.
        """

        N, C, h_ref, w_ref = feat_ref.shape
        S = grids.shape[1]
        h_src, w_src = feat_src.shape[-2:]

        feat_ref = feat_ref.view(N, C, h_ref * w_ref)
        feat_src = feat_src.view(N, C, h_src * w_src)

        if self.normalize and self.normalize != "dim":
            feat_ref = normalize(feat_ref, dim=1)
            feat_src = normalize(feat_src, dim=1)

        corr = torch.matmul(
            feat_ref.transpose(1, 2), feat_src
        )  # N, h_ref*w_ref, C @ N, C, h_src*w_src -> N, h_ref*w_ref, h_src*w_src

        if self.normalize == "dim":
            corr = corr / torch.sqrt(torch.tensor(C).float())

        corr = corr.reshape(N * h_ref * w_ref, 1, h_src, w_src)  # K, 1, h_src, w_src

        grids = grids.permute(0, 3, 4, 2, 1)  # N, h_ref, w_ref, 2, S
        grids = grids.reshape(N * h_ref * w_ref, 2, S).unsqueeze_(3)  # K, 2, S, 1

        corr, corr_mask = warp(
            corr, grid=grids, padding_mode=self.padding_mode
        )  # K, 1, S, 1 ; K, 1, S, 1
        corr = corr.reshape(N, h_ref, w_ref, -1).permute(
            0, 3, 1, 2
        )  # N, S, h_ref, w_ref
        corr_mask = corr_mask.reshape(N, h_ref, w_ref, -1).permute(
            0, 3, 1, 2
        )  # N, S, h_ref, w_ref
        corr = corr * corr_mask

        if mask is not None:
            corr = corr * mask  # N, S, h_ref, W_ref
            corr_mask = corr_mask * mask  # N, S, H_ref, W_ref

        return corr, corr_mask


class EpipolarCoeffs:
    def __init__(self, u_infs_h, v_infs_h, k_infs_h, m_u_h, m_v_h, m_k_h):
        self.height = u_infs_h.shape[2]  # resolution of the keyview
        self.width = u_infs_h.shape[3]  # resolution of the keyview

        self.u_infs_h = u_infs_h  # a_1_x, (N, 1, H, W)
        self.v_infs_h = v_infs_h  # a_1_y, (N, 1, H, W)
        self.k_infs_h = k_infs_h  # b_1, (N, 1, H, W)

        self.m_u_h = m_u_h  # a_0_x, (N, 1, 1, 1)
        self.m_v_h = m_v_h  # a_0_y, (N, 1, 1, 1)
        self.m_k_h = m_k_h  # b_0, (N, 1, 1, 1)

        self.eps = 1e-9

        self._u_infs = None
        # (N, 1, H, W)

        self._v_infs = None
        # (N, 1, H, W)

        self._z_poles = None
        # (N, 1, H, W)
        # distance where re-projected x and y coordinates run towards +-inf
        # (because distance is exactly in plane of the other camera center)

        self._epipole = None
        #  tuple of 2 x (N, 1, 1, 1) with x and y coordinates of the epipole
        #  can have inf and nan values (e.g. in stereo case)

    @staticmethod
    def from_calib(
        intrinsics_key,
        intrinsics_source,
        source_to_key_transform,
        height,
        width,
        height_source,
        width_source,
        device,
    ):
        with torch.no_grad():
            y, x = torch.meshgrid(
                torch.arange(height), torch.arange(width)
            )  # both (H, W)
            x = x.to(device)
            y = y.to(device)
            x = x + 0.5
            y = y + 0.5

            fx = (
                torch.unsqueeze(torch.unsqueeze(intrinsics_key[:, 0, 0], 1), 1) * width
            )  # N, 1, 1
            fy = (
                torch.unsqueeze(torch.unsqueeze(intrinsics_key[:, 1, 1], 1), 1) * height
            )  # N, 1, 1
            cx = (
                torch.unsqueeze(torch.unsqueeze(intrinsics_key[:, 0, 2], 1), 1) * width
            )  # N, 1, 1
            cy = (
                torch.unsqueeze(torch.unsqueeze(intrinsics_key[:, 1, 2], 1), 1) * height
            )  # N, 1, 1

            fxo = (
                torch.unsqueeze(torch.unsqueeze(intrinsics_source[:, 0, 0], 1), 1)
                * width_source
            )  # N, 1, 1
            fyo = (
                torch.unsqueeze(torch.unsqueeze(intrinsics_source[:, 1, 1], 1), 1)
                * height_source
            )  # N, 1, 1
            cxo = (
                torch.unsqueeze(torch.unsqueeze(intrinsics_source[:, 0, 2], 1), 1)
                * width_source
            )  # N, 1, 1
            cyo = (
                torch.unsqueeze(torch.unsqueeze(intrinsics_source[:, 1, 2], 1), 1)
                * height_source
            )  # N, 1, 1

            r11 = torch.unsqueeze(
                torch.unsqueeze(source_to_key_transform[:, 0, 0], 1), 1
            )  # N, 1, 1
            r12 = torch.unsqueeze(
                torch.unsqueeze(source_to_key_transform[:, 0, 1], 1), 1
            )  # N, 1, 1
            r13 = torch.unsqueeze(
                torch.unsqueeze(source_to_key_transform[:, 0, 2], 1), 1
            )  # N, 1, 1
            t1 = torch.unsqueeze(
                torch.unsqueeze(source_to_key_transform[:, 0, 3], 1), 1
            )  # N, 1, 1
            r21 = torch.unsqueeze(
                torch.unsqueeze(source_to_key_transform[:, 1, 0], 1), 1
            )  # N, 1, 1
            r22 = torch.unsqueeze(
                torch.unsqueeze(source_to_key_transform[:, 1, 1], 1), 1
            )  # N, 1, 1
            r23 = torch.unsqueeze(
                torch.unsqueeze(source_to_key_transform[:, 1, 2], 1), 1
            )  # N, 1, 1
            t2 = torch.unsqueeze(
                torch.unsqueeze(source_to_key_transform[:, 1, 3], 1), 1
            )  # N, 1, 1
            r31 = torch.unsqueeze(
                torch.unsqueeze(source_to_key_transform[:, 2, 0], 1), 1
            )  # N, 1, 1
            r32 = torch.unsqueeze(
                torch.unsqueeze(source_to_key_transform[:, 2, 1], 1), 1
            )  # N, 1, 1
            r33 = torch.unsqueeze(
                torch.unsqueeze(source_to_key_transform[:, 2, 2], 1), 1
            )  # N, 1, 1
            t3 = torch.unsqueeze(
                torch.unsqueeze(source_to_key_transform[:, 2, 3], 1), 1
            )  # N, 1, 1

            a = (fxo * r11 + cxo * r31) / fx
            b = (fxo * r12 + cxo * r32) / fy
            c = (
                -(cx * (fxo * r11 + cxo * r31) / fx)
                - (cy * (fxo * r12 + cxo * r32) / fy)
                + (fxo * r13 + cxo * r33)
            )
            e = fxo * t1 + cxo * t3

            f = (fyo * r21 + cyo * r31) / fx
            g = (fyo * r22 + cyo * r32) / fy
            h = (
                -(cx * (fyo * r21 + cyo * r31) / fx)
                - (cy * (fyo * r22 + cyo * r32) / fy)
                + (fyo * r23 + cyo * r33)
            )
            i = fyo * t2 + cyo * t3

            j = r31 / fx
            k = r32 / fy
            l = -cx * r31 / fx - cy * r32 / fy + r33
            m = t3

            n = a * x
            o = b * y
            p = j * x
            q = k * y
            r = f * x
            s = g * y

            u_infs_h = n + o + c
            m_u_h = e

            v_infs_h = r + s + h
            m_v_h = i

            k_infs_h = p + q + l
            m_k_h = m

            u_infs_h.unsqueeze_(1)  # (N, 1, H, W)
            m_u_h.unsqueeze_(1)  # (N, 1, 1, 1)
            v_infs_h.unsqueeze_(1)  # (N, 1, H, W)
            m_v_h.unsqueeze_(1)  # (N, 1, 1, 1)
            k_infs_h.unsqueeze_(1)  # (N, 1, H, W)
            m_k_h.unsqueeze_(1)  # (N, 1, 1, 1)

        coeffs = EpipolarCoeffs(u_infs_h, v_infs_h, k_infs_h, m_u_h, m_v_h, m_k_h)
        return coeffs

    @property
    def u_infs(self):
        if self._u_infs is None:
            self._u_infs = self.u_infs_h / self.k_infs_h
            self._u_infs[torch.isinf(self._u_infs)] = (
                1e9 * torch.sign(self._u_infs)[torch.isinf(self._u_infs)]
            )
            assert torch.isfinite(self._u_infs).all()
        return self._u_infs

    @property
    def v_infs(self):
        if self._v_infs is None:
            self._v_infs = self.v_infs_h / self.k_infs_h
            self._v_infs[torch.isinf(self._v_infs)] = (
                1e9 * torch.sign(self._v_infs)[torch.isinf(self._v_infs)]
            )
            assert torch.isfinite(self._v_infs).all()
        return self._v_infs

    @property
    def epipole(self):
        if self._epipole is None:
            self._epipole = (self.m_u_h / self.m_k_h, self.m_v_h / self.m_k_h)
            # stereo: m_u_h=-fx ; m_k_h = 0 ; m_v_h = 0 -> epi = (-inf, nan)
            # identity: m_u_h = m_v_h = m_k_h = 0 -> epi = (nan, nan)
            # 90 deg turn and fw motion: m_u_h = +fx; m_v_h = 0; m_k_h=0 -> epi = (inf, nan)
        return self._epipole

    @property
    def z_poles(self):
        if self._z_poles is None:
            self._z_poles = -(self.m_k_h / self.k_infs_h)
        return self._z_poles

    def us_from_ds(self, ds, replace_nonfinite=False):  # ds: (N, S, 1, 1)
        us = (self.u_infs_h + self.m_u_h * ds) / (self.k_infs_h + self.m_k_h * ds)

        if replace_nonfinite:
            us[torch.isinf(us)] = 1e9 * torch.sign(us)[torch.isinf(us)]
            us[torch.isnan(us)] = 1e9

        return us

    def vs_from_ds(self, ds, replace_nonfinite=False):  # ds: (S)
        vs = (self.v_infs_h + self.m_v_h * ds) / (self.k_infs_h + self.m_k_h * ds)

        if replace_nonfinite:
            vs[torch.isinf(vs)] = 1e9 * torch.sign(vs)[torch.isinf(vs)]
            vs[torch.isnan(vs)] = 1e9

        return vs


class EpipolarSamplingPoints:
    def __init__(self, us, vs):
        self.height = us.shape[2]
        self.width = us.shape[3]

        self.us = us  # (N, S, H, W)
        self.vs = vs  # (N, S, H, W)

        self.zs = None  # (N, S, H, W), depths
        self.mask = None  # (N, S, H, W)

    @property
    def num(self):
        return self.us.shape[1]  # scalar

    def get_uvs(self):
        return torch.stack((self.us, self.vs), 2)  # (N, S, 2, H, W)


class PlanesweepCorrelation(nn.Module):
    @torch.no_grad()
    def __init__(self, corr_type="torch", normalize="dim", num_groups: int = None):
        super().__init__()

        self.feat_key = None
        self.feat_key_width, self.feat_key_height = None, None
        self.intrinsics_key = None
        self.feat_sources = None
        self.intrinsics_sources = None
        self.source_to_key_transforms = None
        self.device = None

        if corr_type == "torch":
            self.corr_block = TorchCorr(normalize=normalize, padding_mode="zeros")
        elif corr_type == "warponly":
            self.corr_block = WarpOnlyCorr(normalize=normalize, padding_mode="zeros")
        elif corr_type == "groupwise_5D":
            assert (
                num_groups is not None
            ), "num_groups must be specified for groupwise correlation"
            self.corr_block = GroupWiseCorr_5D(
                normalize=normalize, padding_mode="zeros", num_groups=num_groups
            )
        elif corr_type == "concat":
            self.corr_block = Concat(normalize=normalize, padding_mode="zeros")

        self.coeffs = []

        self.sampling_points = []

        self.corrs = []
        self.masks = []

    def forward(
        self,
        feat_key,
        intrinsics_key,
        feat_sources,
        source_to_key_transforms,
        intrinsics_sources=None,
        num_sampling_points=None,
        min_depth=None,
        max_depth=None,
        sampling_invdepths=None,
        sampling_type="linear_invdepth",
    ):
        self.feat_key = feat_key
        self.feat_key_width, self.feat_key_height = feat_key.shape[3], feat_key.shape[2]
        self.intrinsics_key = intrinsics_key
        self.feat_sources = feat_sources
        self.source_to_key_transforms = source_to_key_transforms
        self.intrinsics_sources = intrinsics_sources
        self.device = feat_key.device

        self.reset()
        self.init_coeffs()

        sampling_invdepths = self.get_sampling_invdepths(
            num_sampling_points,
            min_depth,
            max_depth,
            sampling_invdepths,
            sampling_type,
        )
        self.get_plane_sweep_sampling_points(sampling_invdepths=sampling_invdepths)

        self.correlate()

        return self.corrs, self.masks, sampling_invdepths

    def reset(self):
        self.coeffs = []
        self.sampling_points = []

        self.corrs = []
        self.masks = []

    @torch.no_grad()
    def init_coeffs(self):
        width, height = self.feat_key.shape[3], self.feat_key.shape[2]
        device = self.device

        if self.intrinsics_sources is None:
            self.intrinsics_sources = [self.intrinsics_key] * len(self.feat_sources)

        assert (
            len(self.feat_sources)
            == len(self.source_to_key_transforms)
            == len(self.intrinsics_sources)
        )

        for idx, source_to_key_transform in enumerate(self.source_to_key_transforms):
            feat_source = self.feat_sources[idx]
            width_source, height_source = feat_source.shape[3], feat_source.shape[2]
            intrinsics_source = self.intrinsics_sources[idx]

            coeffs = EpipolarCoeffs.from_calib(
                intrinsics_key=self.intrinsics_key,
                intrinsics_source=intrinsics_source,
                source_to_key_transform=source_to_key_transform,
                height=height,
                width=width,
                height_source=height_source,
                width_source=width_source,
                device=device,
            )

            self.coeffs.append(coeffs)

    @torch.no_grad()
    def get_sampling_invdepths(
        self,
        num_sampling_points=None,
        min_depth=None,
        max_depth=None,
        sampling_invdepths=None,
        sampling_type="linear_invdepth",
    ):
        if min_depth is not None and max_depth is not None:
            assert sampling_invdepths is None
            assert num_sampling_points is not None
            assert sampling_type is not None
            sampling_invdepths = compute_sampling_invdepths(
                min_depth,
                max_depth,
                num_sampling_points,
                sampling_type,
            )
        else:
            assert (
                num_sampling_points is None and min_depth is None and max_depth is None
            )
            assert sampling_invdepths is not None

        sampling_invdepths = sampling_invdepths.to(self.device)
        assert sampling_invdepths.ndim >= 2  # (N, S) or (N, S, H) or (N, S, H, W)
        while sampling_invdepths.ndim < 4:
            sampling_invdepths = sampling_invdepths.unsqueeze(-1)

        return sampling_invdepths

    @torch.no_grad()
    def get_plane_sweep_sampling_points(self, sampling_invdepths):
        ds = sampling_invdepths
        zs = 1.0 / ds

        for coeffs in self.coeffs:
            us = coeffs.us_from_ds(ds=ds, replace_nonfinite=True)  # (N, S, H, W)
            vs = coeffs.vs_from_ds(ds=ds, replace_nonfinite=True)  # (N, S, H, W)
            zs = zs * torch.ones_like(us)

            visible_in_key = zs > 0
            visible_in_source = (
                ((coeffs.k_infs_h > 0) & (zs > coeffs.z_poles))
                | ((coeffs.k_infs_h < 0) & (zs < coeffs.z_poles))
                | ((coeffs.k_infs_h == 0) & (coeffs.m_k_h > 0))
            )

            mask = visible_in_key & visible_in_source

            sampling_points = EpipolarSamplingPoints(us=us, vs=vs)
            sampling_points.zs = zs
            sampling_points.mask = mask

            self.sampling_points.append(sampling_points)

    def correlate(self):
        for feat_source, sampling_points in zip(
            self.feat_sources, self.sampling_points
        ):
            corr, mask = self.corr_block(
                feat_ref=self.feat_key,
                feat_src=feat_source,
                grids=sampling_points.get_uvs(),
                mask=sampling_points.mask,
            )

            self.corrs.append(corr)
            self.masks.append(mask)


def compute_sampling_invdepths(
    min_depth, max_depth, num_samples, sampling_type="linear_invdepth"
):
    """Compute the inverse depth values for the sampling points.

    Args:
        min_depth: Minimum depth value. Can be: float, torch.Tensor of shape [N], numpy.ndarray of shape (N,).
        max_depth: Maximum depth value. Can be: float, torch.Tensor of shape [N], numpy.ndarray of shape (N,).
        num_samples (int): Number of sampling points.
    """
    if isinstance(min_depth, np.float32) or isinstance(min_depth, float):
        min_depth = torch.tensor([min_depth])
    if isinstance(max_depth, np.float32) or isinstance(max_depth, float):
        max_depth = torch.tensor([max_depth])

    min_depth = (
        to_torch(min_depth)[..., None] if len(min_depth.shape) == 1 else min_depth
    )  # shape [1, 1] or [N, 1]
    max_depth = (
        to_torch(max_depth)[..., None] if len(max_depth.shape) == 1 else max_depth
    )  # shape [1, 1] or [N, 1]
    min_invdepth = 1 / max_depth
    max_invdepth = 1 / min_depth
    steps = torch.arange(
        0, num_samples, dtype=min_invdepth.dtype, device=min_invdepth.device
    )
    steps = (
        steps.view(1, num_samples)
        if len(min_depth.shape) == 1
        else steps.view(1, num_samples, 1, 1)
    )

    if sampling_type == "linear_invdepth":
        sampling_invdepths = min_invdepth + steps * (max_invdepth - min_invdepth) / (
            num_samples - 1
        )  # shape [1, num_samples] or [N, num_samples]
    elif sampling_type == "linear_depth":
        sampling_invdepths = 1 / (
            min_depth + steps * (max_depth - min_depth) / (num_samples - 1) + 1e-9
        )  # shape [1, num_samples] or [N, num_samples]
        sampling_invdepths = sampling_invdepths.flip(1)

    return sampling_invdepths  # shape [1, num_samples] or [N, num_samples]
