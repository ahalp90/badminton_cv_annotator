"""Pretrained video models with contact heads and shared clip augmentation."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .videomae import CHECKPOINT_REVISION


def build_model(
    *, pretrained: bool = True, model_name: str = "x3d",
    checkpoint_revision: str = CHECKPOINT_REVISION, stochastic_depth: float = 0.2,
) -> nn.Module:
    if model_name == "videomae":
        from .videomae import build_videomae

        return build_videomae(pretrained=pretrained, revision=checkpoint_revision, stochastic_depth=stochastic_depth)
    if model_name != "x3d":
        raise ValueError(f"Unknown model: {model_name}")
    from pytorchvideo.models.hub import x3d_m

    model = x3d_m(pretrained=pretrained)
    head = model.blocks[-1]
    # The released inference head applies softmax; cross-entropy needs logits.
    head.activation = None
    head.proj = nn.Linear(head.proj.in_features, 3)
    nn.init.normal_(head.proj.weight, std=0.01)
    nn.init.zeros_(head.proj.bias)
    return model


def prepare_clips(clips: torch.Tensor, *, training: bool, model_name: str = "x3d") -> torch.Tensor:
    """Augment each clip independently, using one transform for all its frames."""
    pixels = clips.float().div_(255)
    if training:
        batch, channels, frames, height, width = pixels.shape
        shape = (batch, 1, 1, 1, 1)
        colour = torch.rand(shape, device=pixels.device) < 0.5
        factors = 0.8 + 0.4 * torch.rand((3, *shape), device=pixels.device)
        brightness, contrast, saturation = torch.where(colour, factors, 1).unbind(0)
        # Keep the black letterbox fill fixed through colour transformations.
        content = pixels.any(dim=1, keepdim=True)
        pixels = (pixels * brightness).clamp_(0, 1)
        grey_weights = pixels.new_tensor([0.2989, 0.5870, 0.1140]).view(1, 3, 1, 1, 1)
        grey = (pixels * grey_weights).sum(1, keepdim=True)
        mean = (grey * content).sum((2, 3, 4), keepdim=True) / content.sum((2, 3, 4), keepdim=True).clamp_min(1)
        pixels = ((pixels - mean) * contrast + mean).clamp_(0, 1)
        grey = (pixels * grey_weights).sum(1, keepdim=True)
        pixels = ((pixels - grey) * saturation + grey).clamp_(0, 1) * content
        crop = torch.rand(batch, device=pixels.device) < 0.5
        scale = torch.where(crop, (0.8 + 0.2 * torch.rand(batch, device=pixels.device)).sqrt(), 1)
        flip = torch.where(torch.rand(batch, device=pixels.device) < 0.5, -1, 1)
        theta = pixels.new_zeros((batch, 2, 3))
        theta[:, 0, 0] = scale * flip
        theta[:, 1, 1] = scale
        theta[:, :, 2] = (2 * torch.rand((batch, 2), device=pixels.device) - 1) * (1 - scale[:, None])
        spatial = pixels.reshape(batch, channels * frames, height, width)
        grid = F.affine_grid(theta, spatial.shape, align_corners=False)
        pixels = F.grid_sample(spatial, grid, align_corners=False, padding_mode="border").reshape_as(pixels)
    if model_name == "videomae":
        batch, channels, frames, height, width = pixels.shape
        spatial = pixels.reshape(batch, channels * frames, height, width)
        pixels = F.interpolate(spatial, size=(224, 224), mode="bilinear", align_corners=False, antialias=True)
        pixels = pixels.reshape(batch, channels, frames, 224, 224)
        mean = pixels.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1)
        std = pixels.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1)
        return pixels.sub_(mean).div_(std).permute(0, 2, 1, 3, 4)
    return pixels.sub_(0.45).div_(0.225)
