import torchvision.transforms
from .registry import register_batch_augmentation
from .batch_transforms import Scale3DEqualizedBatch, MaskDepthByMinMax


@register_batch_augmentation
def robust_mvd_batch_augmentations(**kwargs):
    transforms = [
        Scale3DEqualizedBatch(p=1, min_depth=1 / 2.75, max_depth=1 / 0.009),
        MaskDepthByMinMax(min_depth=1 / 2.75, max_depth=1 / 0.009),
    ]
    transform = torchvision.transforms.Compose(transforms)
    return transform


@register_batch_augmentation
def mvsnet_batch_augmentations(**kwargs):
    transforms = [
        Scale3DEqualizedBatch(p=1, min_depth=1 / 2.75, max_depth=1 / 0.009),
        MaskDepthByMinMax(min_depth=1 / 2.75, max_depth=1 / 0.009),
    ]
    transform = torchvision.transforms.Compose(transforms)
    return transform
