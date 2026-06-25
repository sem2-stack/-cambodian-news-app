import torch.nn as nn
from transformers import RobertaModel

class NewsClassifier(nn.Module):
    def __init__(self, num_labels=5):
        super().__init__()
        self.encoder = RobertaModel.from_pretrained("roberta-base")
        self.head = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_labels),
        )

    def forward(self, input_ids, attention_mask=None):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.pooler_output
        return self.head(pooled)
