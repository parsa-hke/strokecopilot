"""2D slice classifier with an ImageNet-pretrained backbone, one logit for "any haemorrhage" (plan §8.2).

Backbones come from timm, which Kaggle images include, so pretrained weights download with Internet
on. Images arrive as uint8 height x width x 3 (brain, subdural, bone windows) and are normalised on
the GPU, which keeps the CPU workers free for decoding.
"""

import torch

MODELS = ("resnet50", "efficientnet_b3")
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def build(name="resnet50", pretrained=True):
    import timm

    if name not in MODELS:
        raise ValueError(f"Unknown model {name!r}; plan §8.2 allows {MODELS}.")
    return timm.create_model(name, pretrained=pretrained, num_classes=1)


def normalise(images, device):
    """uint8 N x H x W x 3 -> float N x 3 x H x W on device, channels-last, ImageNet-normalised."""
    x = images.to(device, non_blocking=True).permute(0, 3, 1, 2).float().div_(255)
    mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=device).view(1, 3, 1, 1)
    return ((x - mean) / std).contiguous(memory_format=torch.channels_last)


def parameters_millions(model):
    return round(sum(p.numel() for p in model.parameters()) / 1e6, 1)
