
import torch
import torch.nn as nn


class ConcatFusion(nn.Module):
    def __init__(self, img_dim: int, txt_dim: int, hidden: int = 256, num_classes: int = 2):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(img_dim + txt_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_classes)
        )

    def forward(self, img_feat, txt_feat):
        z = torch.cat([img_feat, txt_feat], dim=1)
        return self.fc(z)


class AttentionFusion(nn.Module):
    """
    Simple modality-attention: learn weights over {image,text} and fuse.
    """

    def __init__(self, img_dim: int, txt_dim: int, hidden: int = 256, num_classes: int = 2):
        super().__init__()
        self.img_proj = nn.Linear(img_dim, hidden)
        self.txt_proj = nn.Linear(txt_dim, hidden)

        self.attn = nn.Linear(hidden, 1)  # shared scorer
        self.classifier = nn.Linear(hidden, num_classes)

    def forward(self, img_feat, txt_feat):
        hi = torch.tanh(self.img_proj(img_feat))
        ht = torch.tanh(self.txt_proj(txt_feat))

        # scores -> weights
        si = self.attn(hi)  # [B,1]
        st = self.attn(ht)  # [B,1]
        w = torch.softmax(torch.cat([si, st], dim=1), dim=1)  # [B,2]

        fused = w[:, 0:1] * hi + w[:, 1:2] * ht
        logits = self.classifier(fused)
        return logits, w
