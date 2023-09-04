from .factory import create_loss
from .registry import register_loss, list_losses, has_loss
from .multi_scale_uni_laplace import robust_mvd_loss
from .multi_scale_mae import supervised_monodepth2_loss
from .single_scale_mae import mvsnet_loss, dpt_loss
from .multi_scale_uni_laplace_cascade import robust_mvd_cascade_loss
