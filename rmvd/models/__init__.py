# RobustMVD variants
from .robust_mvd import robust_mvd, robust_mvd_5M
from .robust_mvd_mvsnet_encoder import robust_mvd_mvsnet_encoder
from .robustmvd_groupwisecorr import robustmvd_groupwisecorr


# MVSNET variants
from .mvsnet import mvsnet_blendedmvs
from .mvsnet_dispnet_decoder_depth_slice_fusedcostvolume import (
    mvsnet_dispnet_decoder_depth_slice_fusedcostvolume,
)
from .mvsnet_dispnet_encoder import mvsnet_dispnet_encoder
from .mvsnet_groupwisecorr_variancefusion_withfinalenclayer import (
    mvsnet_groupwisecorr_variancefusion,
)
from .mvsnet_groupwisecorr_learnedfusion_withfinalenclayer import (
    mvsnet_groupwisecorr_learnedfusion_withfinalenclayer,
)
from .mvsnet_groupwisecorr_averagefusion_withfinalenclayer import (
    mvsnet_groupwisecorr_averagefusion_withfinalenclayer,
)
from .mvsnet_groupwisecorr_variancefusion_noctx import (
    mvsnet_groupwisecorr_variancefusion_noctx,
)
from .mvsenc_multi_corr_mvsencdec import mvsenc_multi_corr_mvsencdec


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
