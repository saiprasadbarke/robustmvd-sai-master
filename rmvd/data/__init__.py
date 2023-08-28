from .factory import create_dataset, create_compound_dataset
from .registry import (
    register_dataset,
    list_base_datasets,
    list_dataset_types,
    list_splits,
    list_datasets,
    has_dataset,
)
from .registry import (
    register_augmentation,
    list_augmentations,
    has_augmentation,
    create_augmentation,
    register_batch_augmentation,
    list_batch_augmentations,
    has_batch_augmentation,
    create_batch_augmentation,
)

# import all datasets; this triggers the registration of the datasets in the registry
from .eth3d import ETH3DTrainRobustMVD
from .kitti import KITTIRobustMVD, KITTIEigenDenseDepthTest, KITTIEigenDenseDepthTrain
from .dtu import DTURobustMVD
from .scannet import ScanNetRobustMVD
from .tanks_and_temples import TanksAndTemplesTrainRobustMVD
from .flyingthings3d import FlyingThings3DSeq4Train
from .staticthings3d import StaticThings3DSeq4Train
from .blendedmvs import BlendedMVSSeq4TrainSmall

# import all augmentations; this triggers the registration of the augmentations in the registry
from .augmentations import (
    robust_mvd_augmentations_staticthings3d,
    robust_mvd_augmentations_blendedmvs,
    supervised_monodepth2_augmentations,
    mvsnet_augmentations,
    dpt_augmentations,
)
from .batch_augmentations import robust_mvd_batch_augmentations
from .batch_augmentations import mvsnet_batch_augmentations
