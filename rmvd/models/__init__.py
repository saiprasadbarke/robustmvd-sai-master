# RobustMVD variants
from .robust_mvd import robust_mvd, robust_mvd_5M
from .robust_mvd_mvsnet_encoder import robust_mvd_mvsnet_encoder
from .robustmvd_groupwisecorr import robustmvd_groupwisecorr
from .robust_mvd_averagefusion import robust_mvd_averagefusion

# MVSNET variants
from .mvsnet import mvsnet_blendedmvs
from .mvsnet_learnedfusion_3d import mvsnet_learnedfusion_3d
from .mvsnet_averagefusion_3d import mvsnet_averagefusion_3d
from .mvsnet_dispnet_decoder_depth_slice_fusedcostvolume import (
    mvsnet_dispnet_decoder_depth_slice_fusedcostvolume,
)
from .mvsnet_dispnet_encoder import mvsnet_dispnet_encoder
from .mvsnet_groupwisecorr_learnedfusion import (
    mvsnet_groupwisecorr_learnedfusion,
)
from .mvsnet_groupwisecorr_averagefusion import mvsnet_groupwisecorr_averagefusion
from .mvsnet_multi_corr_learnedfusion import mvsnet_multicorr_learnedfusion
from .mvsnet_cascade import mvsnet_cascade
from .mvsnet_dino import mvsnet_dino
from .mvsnet_dino_faster_concat import mvsnet_dino_faster_concat
from .mvsnet_dino_faster_add import mvsnet_dino_faster_add
from .mvsnet_wo_depth import mvsnet_blendedmvs_wo_depth

from .dpt import dpt_large_kitti
from .supervised_monodepth2 import supervised_monodepth2
from .wrappers.monodepth2 import (
    monodepth2_mono_stereo_1024x320_wrapped,
    monodepth2_mono_stereo_wrapped,
    monodepth2_postuncertainty_mono_stereo_wrapped,
)
from .wrappers.mvsnet_pl import mvsnet_pl_wrapped
from .wrappers.midas import midas_big_v2_1_wrapped
from .wrappers.vis_mvsnet import vis_mvsnet_wrapped
from .wrappers.cvp_mvsnet import cvp_mvsnet_wrapped
from .wrappers.patchmatchnet import patchmatchnet_wrapped
from .wrappers.gmdepth import (
    gmdepth_scale1_scannet_wrapped,
    gmdepth_scale1_resumeflowthings_scannet_wrapped,
    gmdepth_scale1_regrefine1_resumeflowthings_scannet_wrapped,
    gmdepth_scale1_demon_wrapped,
    gmdepth_scale1_regrefine1_resumeflowthings_demon_wrapped,
    gmdepth_scale1_resumeflowthings_demon_wrapped,
)

from .factory import create_model, prepare_custom_model
from .registry import register_model, list_models, has_model
