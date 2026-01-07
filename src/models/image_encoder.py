import torch
import torch.nn as nn
from torchvision import models


class DenseNetImageEncoder(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()

        self.backbone = models.densenet121(pretrained=pretrained)
        in_features = self.backbone.classifier.in_features

        # Replace classifier
        self.backbone.classifier = nn.Identity()

        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        feats = self.backbone(x)
        logits = self.classifier(feats)
        return logits
