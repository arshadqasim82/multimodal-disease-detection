import torch.nn as nn
from torchvision import models
from torchvision.models import DenseNet121_Weights
from transformers import AutoModel


class DenseNetFeatureExtractor(nn.Module):
    def __init__(self, weights=DenseNet121_Weights.DEFAULT):
        super().__init__()
        m = models.densenet121(weights=weights)
        self.backbone = m.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.out_dim = m.classifier.in_features

    def forward(self, x):
        f = self.backbone(x)
        f = self.pool(f).flatten(1)
        return f


class BertCLSFeatureExtractor(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.out_dim = self.encoder.config.hidden_size

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        return cls
