"""RegNet/GSM/BiGRU adapted from E2E-Spot (see LICENSE.spot and LICENSE.gsm).

GSM arithmetic and channel shuffling follow the FBK implementation. Padding uses
input-allocated zeros so CPU inference and mixed precision use the correct dtype.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torchvision import transforms

from experiments.annotator.video_models_benchmark.model import prepare_clips

BACKBONES = {"200mf": "regnety_002.pycls_in1k", "800mf": "regnety_008.pycls_in1k"}
E2E_TRANSFORMS = nn.Sequential(
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply(nn.ModuleList([transforms.ColorJitter(hue=0.2)]), p=0.25),
    transforms.RandomApply(nn.ModuleList([transforms.ColorJitter(saturation=(0.7, 1.2))]), p=0.25),
    transforms.RandomApply(nn.ModuleList([transforms.ColorJitter(brightness=(0.7, 1.2))]), p=0.25),
    transforms.RandomApply(nn.ModuleList([transforms.ColorJitter(contrast=(0.7, 1.2))]), p=0.25),
    transforms.RandomApply(nn.ModuleList([transforms.GaussianBlur(5)]), p=0.25),
)
E2E_NORMALISE = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))


class GSM(nn.Module):
    def __init__(self, channels: int, segments: int) -> None:
        super().__init__()
        self.channels = channels
        self.segments = segments
        self.conv3D = nn.Conv3d(channels, 2, 3, padding=1, groups=2)
        nn.init.zeros_(self.conv3D.weight)
        nn.init.zeros_(self.conv3D.bias)
        self.bn = nn.BatchNorm3d(channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch = inputs.shape[0] // self.segments
        _, channels, height, width = inputs.shape
        pixels = inputs.reshape(batch, self.segments, channels, height, width).permute(0, 2, 1, 3, 4).contiguous()
        gates = self.conv3D(self.bn(pixels).relu()).tanh()
        first, second = pixels.chunk(2, dim=1)
        gated_first = gates[:, :1] * first
        gated_second = gates[:, 1:] * second
        zero = gated_first.new_zeros((batch, channels // 2, 1, height, width))
        first = torch.cat((gated_first[:, :, 1:], zero), dim=2) + (first - gated_first)
        second = torch.cat((zero, gated_second[:, :, :-1]), dim=2) + (second - gated_second)
        shuffled = []
        for group in (first, second):
            group = group.reshape(batch, 2, channels // 4, self.segments, height, width)
            shuffled.append(group.transpose(1, 2).contiguous().reshape(batch, channels // 2, self.segments, height, width))
        output = torch.cat(shuffled, dim=1)
        return output.permute(0, 2, 1, 3, 4).contiguous().reshape_as(inputs)


class GatedShift(nn.Module):
    def __init__(self, convolution: nn.Module, channels: int, segments: int) -> None:
        super().__init__()
        self.fold = math.ceil(channels // 4 / 4) * 4
        self.gsm = GSM(self.fold, segments)
        self.net = convolution

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat((self.gsm(inputs[:, :self.fold]), inputs[:, self.fold:]), dim=1))


class SpotModel(nn.Module):
    def __init__(self, backbone: str, clip_length: int, classes: int, *, pretrained: bool = True) -> None:
        super().__init__()
        import timm

        self.clip_length = clip_length
        self.features = timm.create_model(BACKBONES[backbone], pretrained=pretrained)
        feature_dim = self.features.head.fc.in_features
        self.features.head.fc = nn.Identity()
        for stage in (self.features.s1, self.features.s2, self.features.s3, self.features.s4):
            for block in stage.children():
                block.conv1 = GatedShift(block.conv1, block.conv1.conv.in_channels, clip_length)
        self.gru = nn.GRU(feature_dim, min(feature_dim, 768), batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(2 * min(feature_dim, 768), classes)

    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        batch, frames, channels, height, width = clips.shape
        if frames != self.clip_length:
            raise ValueError("Pad clips to the configured GSM clip length before inference")
        features = self.features(clips.reshape(batch * frames, channels, height, width))
        temporal, _ = self.gru(features.reshape(batch, frames, -1))
        return self.classifier(self.dropout(temporal))


def transform_clips(clips: torch.Tensor, valid_frames: torch.Tensor, *, training: bool,
                    augmentation: str = "hybrid") -> torch.Tensor:
    """Augment real frames, normalise, then zero artificial temporal padding."""
    if augmentation == "user":
        if clips.shape[-2:] != (256, 256):
            raise ValueError("User augmentation requires 256-square letterboxed clips")
        raw = clips.masked_fill(~valid_frames[:, None, :, None, None], 0)
        pixels = prepare_clips(raw, training=training, model_name="videomae")
        return pixels.masked_fill_(~valid_frames[:, :, None, None, None], 0)
    if clips.shape[-2:] != (224, 224):
        raise ValueError("Expected 224-square crops from height224 source frames")
    if augmentation == "e2e":
        # Each transform draws once per clip and operates on all its real frames.
        batch, _channels, frames, height, width = clips.shape
        pixels = torch.zeros((batch, frames, 3, height, width), device=clips.device, dtype=torch.float32)
        for index, valid in enumerate(valid_frames):
            real = clips[index].permute(1, 0, 2, 3)[valid].float().div_(255)
            if training:
                real = E2E_TRANSFORMS(real)
            pixels[index, valid] = E2E_NORMALISE(real)
        return pixels
    if augmentation != "hybrid":
        raise ValueError(f"Unknown augmentation recipe: {augmentation}")
    pixels = clips.float().div_(255)
    if training:
        batch = len(pixels)
        shape = (batch, 1, 1, 1, 1)
        flip = torch.rand(shape, device=pixels.device) < 0.5
        pixels = torch.where(flip, pixels.flip(-1), pixels)
        colour = torch.rand(shape, device=pixels.device) < 0.5
        factors = 0.8 + 0.4 * torch.rand((3, *shape), device=pixels.device)
        brightness, contrast, saturation = torch.where(colour, factors, 1).unbind(0)
        pixels = (pixels * brightness).clamp_(0, 1)
        grey_weights = pixels.new_tensor([0.2989, 0.5870, 0.1140]).view(1, 3, 1, 1, 1)
        grey = (pixels * grey_weights).sum(1, keepdim=True)
        frame_mask = valid_frames[:, None, :, None, None]
        real_pixels = valid_frames.sum(1).view(-1, 1, 1, 1, 1) * pixels.shape[-2] * pixels.shape[-1]
        mean = (grey * frame_mask).sum((2, 3, 4), keepdim=True) / real_pixels
        pixels = ((pixels - mean) * contrast + mean).clamp_(0, 1)
        grey = (pixels * grey_weights).sum(1, keepdim=True)
        pixels = ((pixels - grey) * saturation + grey).clamp_(0, 1)
    mean = pixels.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1)
    std = pixels.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1)
    pixels = pixels.sub_(mean).div_(std).permute(0, 2, 1, 3, 4)
    return pixels.masked_fill_(~valid_frames[:, :, None, None, None], 0)
