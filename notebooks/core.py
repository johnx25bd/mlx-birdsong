import whisper
import torch.nn as nn
from utils import get_device

class WhisperClassifier(nn.Module):
    def __init__(self, whisper_model: str, num_classes: int, projection_dim_factor: int, feedforward: bool, pooling: str):
        super().__init__()
        self.device = get_device()
        self.encoder = whisper.load_model(whisper_model).encoder.to(self.device)
        self.projection_dim = self.encoder.positional_embedding.size(-1) * projection_dim_factor
        self.feedforward = feedforward
        self.pooling = pooling
        if feedforward:
            self.classifier = nn.Sequential(
                nn.Linear(self.encoder.positional_embedding.size(-1), self.projection_dim),
                nn.ReLU(),
                nn.Linear(self.projection_dim, num_classes)
            ).to(self.device)
        else:
            self.classifier = nn.Linear(self.encoder.positional_embedding.size(-1), num_classes).to(self.device)
    
    def to(self, device):
        """Override to method to keep track of device"""
        self.device = device
        return super().to(device)
    
    def forward(self, mel_input):
        encoder_output = self.encoder(mel_input)
        if self.pooling == "mean":
            pooled_features = encoder_output.mean(dim=1)
        elif self.pooling == "max":
            pooled_features, _ = encoder_output.max(dim=1)
        logits = self.classifier(pooled_features)
        return logits