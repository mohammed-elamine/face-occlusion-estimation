"""Image transforms shared by config-driven Face Occlusion experiments.

We deliberately avoid augmentations that change face visibility
(RandomErasing, strong blur, heavy random crops, synthetic occlusion).
The original target reflects the *true* visibility of the face; perturbing
it would teach the model wrong things.
"""

from __future__ import annotations

from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_train_transform(cfg) -> transforms.Compose:
    aug = cfg.augmentation
    size = int(aug.resize)
    # Conservative pipeline: geometric flip + mild color + small rotation only.
    return transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(p=float(aug.horizontal_flip_p)),
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness=float(aug.brightness),
                        contrast=float(aug.contrast),
                        saturation=float(aug.saturation),
                        hue=0.0,
                    )
                ],
                p=float(aug.color_jitter_p),
            ),
            transforms.RandomRotation(degrees=float(aug.rotation_degrees)),
            transforms.ToTensor(),
            # timm backbones expect ImageNet-style normalization by default.
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_eval_transform(cfg) -> transforms.Compose:
    size = int(cfg.augmentation.resize)
    return transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
