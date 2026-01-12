"""
Purpose
-------
Text classification model using a ClinicalBERT-style transformer encoder.

Architecture
-----------
- Transformer encoder (Bio_ClinicalBERT)
- [CLS] token embedding is used as pooled representation
- Linear classifier head -> 2 classes

Notes
-----
- Trained on a public biomedical QA dataset as a proxy text modality.
- Designed to be compatible with EHR-style notes with minimal changes.
"""

import torch.nn as nn
from transformers import AutoModel


class ClinicalBertClassifier(nn.Module):
    def __init__(self, model_name: str, num_classes: int = 2):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.classifier = nn.Linear(hidden, num_classes)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]  # [CLS]
        logits = self.classifier(cls)
        return logits
