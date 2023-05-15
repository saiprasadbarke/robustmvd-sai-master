import torchvision.transforms
from .registry import register_augmentation
from .transforms import ResizeInputs, ResizeTargets, ColorJitter, Eraser, NormalizeImagesToMinMax, MaskDepth, SpatialAugmentation, NormalizeIntrinsics, NormalizeImagesByShiftAndScale


@register_augmentation
def robust_mvd_augmentations_staticthings3d(**kwargs):
    transforms = [
        ColorJitter(saturation=(0, 2), contrast=(0.01, 8), brightness=(0.01, 2.0), hue=0.5),
        SpatialAugmentation(size=(384, 768), p=1.0),
        NormalizeImagesToMinMax(min_val=-0.4, max_val=0.6),
        NormalizeIntrinsics(),
        Eraser(bounds=[250, 500], p=0.6),
        MaskDepth(min_depth=1/2.75, max_depth=1/0.009),
    ]
    transform = torchvision.transforms.Compose(transforms)
    return transform


@register_augmentation
def robust_mvd_augmentations_blendedmvs(**kwargs):
    transforms = [
        ColorJitter(saturation=(0, 2), contrast=(0.01, 8), brightness=(0.01, 2.0), hue=0.5),
        ResizeInputs(size=(384, 768)),
        ResizeTargets(size=(384, 768)),
        NormalizeImagesToMinMax(min_val=-0.4, max_val=0.6),
        NormalizeIntrinsics(),
        Eraser(bounds=[250, 500], p=0.6),
        # intentionally not masking depth
    ]
    transform = torchvision.transforms.Compose(transforms)
    return transform

@register_augmentation
def supervised_monodepth2_augmentations(**kwargs):
    transforms = [
        ResizeInputs(size=(384, 1280)),
        ResizeTargets(size=(384, 1280)),
        NormalizeImagesToMinMax(min_val=0., max_val=1.),
        NormalizeImagesByShiftAndScale(shift=[0.485, 0.456, 0.406], scale=[0.229, 0.224, 0.225]),
    ]
    transform = torchvision.transforms.Compose(transforms)
    return transform
