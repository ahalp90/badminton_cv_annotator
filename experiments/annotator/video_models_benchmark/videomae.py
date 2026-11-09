"""VideoMAE Base contact head, stochastic depth and layer-wise learning rates."""
from __future__ import annotations

import torch
from torch import nn

CHECKPOINT = "MCG-NJU/videomae-base-finetuned-kinetics"
CHECKPOINT_REVISION = "488eb9a0565f257b32866000305c8178965eb9f6"
HEAD_INITIAL_SCALE = 0.001


class DropPath(nn.Module):
    """Drop a residual branch independently for each clip, retaining its mean."""

    def __init__(self, probability: float) -> None:
        super().__init__()
        self.probability = probability

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if not self.training or self.probability == 0:
            return values
        keep = 1 - self.probability
        mask = values.new_empty((values.shape[0], 1, 1)).bernoulli_(keep)
        return values * mask / keep


class VideoMAE(nn.Module):
    def __init__(self, network: nn.Module, stochastic_depth: float) -> None:
        super().__init__()
        self.videomae = network.videomae
        self.fc_norm = network.fc_norm
        self.classifier = network.classifier
        blocks = self.videomae.encoder.layer
        probabilities = torch.linspace(0, stochastic_depth, len(blocks)).tolist()
        for block, probability in zip(blocks, probabilities, strict=True):
            # The pretrained architecture has query/value biases and no key bias.
            block.attention.attention.key.bias = None
            # These dropout outputs feed the two residual additions in Transformers.
            block.attention.output.dropout = nn.Sequential(block.attention.output.dropout, DropPath(probability))
            block.output.dropout = nn.Sequential(block.output.dropout, DropPath(probability))

    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        hidden = self.videomae(clips).last_hidden_state
        return self.classifier(self.fc_norm(hidden.mean(1)))


def build_videomae(*, pretrained: bool, revision: str, stochastic_depth: float) -> VideoMAE:
    from transformers import VideoMAEConfig, VideoMAEForVideoClassification

    # Base's architecture matches the pinned checkpoint; recovery is offline.
    config = VideoMAEConfig(norm_pix_loss=False)
    config.num_labels = 400 if pretrained else 3
    config._attn_implementation = "sdpa"
    model = VideoMAE(VideoMAEForVideoClassification(config), stochastic_depth)
    if pretrained:
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        path = hf_hub_download(CHECKPOINT, "model.safetensors", revision=revision)
        load_pretrained_weights(model, load_file(path))
        model.classifier = nn.Linear(config.hidden_size, 3)
        nn.init.trunc_normal_(model.classifier.weight, std=0.02)
        nn.init.zeros_(model.classifier.bias)
        # Follow the authors' fine-tuning head scale before updating the encoder.
        with torch.no_grad():
            model.classifier.weight.mul_(HEAD_INITIAL_SCALE)
    return model


def load_pretrained_weights(model: VideoMAE, state: dict[str, torch.Tensor]) -> None:
    """Translate the checkpoint's older bias names and require every weight."""
    translated = {}
    for name, value in state.items():
        name = name.replace(".attention.q_bias", ".attention.query.bias")
        name = name.replace(".attention.v_bias", ".attention.value.bias")
        translated[name] = value
    model.load_state_dict(translated, strict=True)


def parameter_groups(model: VideoMAE, layer_decay: float, weight_decay: float) -> list[dict]:
    """Give embeddings layer 0, transformer blocks 1..12 and the head layer 13."""
    last_layer = len(model.videomae.encoder.layer) + 1
    groups: dict[tuple[int, bool], dict] = {}
    for name, parameter in model.named_parameters():
        if name.startswith("videomae.embeddings."):
            layer = 0
        elif name.startswith("videomae.encoder.layer."):
            layer = int(name.split(".")[3]) + 1
        elif name.startswith(("fc_norm.", "classifier.")):
            layer = last_layer
        else:
            raise ValueError(f"No VideoMAE learning-rate assignment for {name}")
        decay = parameter.ndim > 1 and not name.endswith(".bias")
        key = layer, decay
        if key not in groups:
            groups[key] = {"params": [], "lr_scale": layer_decay ** (last_layer - layer),
                           "weight_decay": weight_decay if decay else 0.0}
        groups[key]["params"].append(parameter)
    return list(groups.values())
