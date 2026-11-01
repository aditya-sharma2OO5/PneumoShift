"""Image preprocessing and augmentation pipelines for training and evaluation."""

from typing import Tuple
from torchvision import transforms


def get_transforms(
    split: str = "train",
    image_size: int = 224,
    enable_horizontal_flip: bool = False,
    rotation_deg: float = 10.0,
    brightness_jitter: float = 0.1,
    contrast_jitter: float = 0.1,
) -> transforms.Compose:
    """Builds standard PyTorch torchvision transform pipeline.

    Args:
        split: 'train', 'val', or 'test'.
        image_size: Target square image dimension (e.g., 224 or 320).
        enable_horizontal_flip: Whether to apply RandomHorizontalFlip. Note: in chest X-rays,
            flipping changes anatomical laterality (dextrocardia, left vs right lung markers),
            so it is disabled or carefully flagged by default.
        rotation_deg: Maximum rotation in degrees (+/-).
        brightness_jitter: Maximum brightness jitter factor.
        contrast_jitter: Maximum contrast jitter factor.

    Returns:
        torchvision.transforms.Compose pipeline.
    """
    # ImageNet standard normalization constants
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if split == "train":
        transform_list = [
            transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0)),
            transforms.RandomRotation(degrees=(-rotation_deg, rotation_deg)),
            transforms.ColorJitter(brightness=brightness_jitter, contrast=contrast_jitter),
        ]
        if enable_horizontal_flip:
            transform_list.append(transforms.RandomHorizontalFlip(p=0.5))

        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    else:
        transform_list = [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]

    return transforms.Compose(transform_list)

