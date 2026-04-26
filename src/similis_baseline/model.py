import warnings

import torch
import torch.nn as nn
from torchvision.models import (
    convnext_tiny,
    ConvNeXt_Tiny_Weights,
    resnet18,
    ResNet18_Weights,
    resnet50,
    ResNet50_Weights,
)


def _safe_load(builder, weights_enum, pretrained):
    if not pretrained:
        return builder(weights=None)
    try:
        return builder(weights=weights_enum.DEFAULT)
    except Exception as exc:
        warnings.warn(f"Failed to load pretrained weights ({exc}); falling back to random init.")
        return builder(weights=None)


class MultiTaskArtifactModel(nn.Module):
    def __init__(self, field_dims, backbone_name="resnet18", pretrained=True):
        super().__init__()
        self.field_dims = field_dims
        self.backbone_name = backbone_name

        if backbone_name == "resnet18":
            backbone = _safe_load(resnet18, ResNet18_Weights, pretrained)
            in_features = backbone.fc.in_features
            backbone.fc = nn.Identity()
            self.backbone = backbone
        elif backbone_name == "resnet50":
            backbone = _safe_load(resnet50, ResNet50_Weights, pretrained)
            in_features = backbone.fc.in_features
            backbone.fc = nn.Identity()
            self.backbone = backbone
        elif backbone_name == "convnext_tiny":
            backbone = _safe_load(convnext_tiny, ConvNeXt_Tiny_Weights, pretrained)
            in_features = backbone.classifier[-1].in_features
            backbone.classifier = nn.Identity()
            self.backbone = backbone
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        self.dropout = nn.Dropout(p=0.2)
        self.heads = nn.ModuleDict(
            {field: nn.Linear(in_features, n_classes) for field, n_classes in field_dims.items()}
        )

    def forward(self, x):
        features = self.backbone(x)
        if features.ndim > 2:
            features = torch.flatten(features, 1)
        features = self.dropout(features)
        return {field: head(features) for field, head in self.heads.items()}

    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())
