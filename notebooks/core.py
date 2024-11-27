import torch
import whisper
import torch.nn as nn
from utils import get_device


class WhisperClassifierBase(nn.Module):
    """Base class for all Whisper classifiers"""
    def __init__(self, whisper_model: str, num_classes: int):
        super().__init__()
        self.device = get_device()
        self.encoder = whisper.load_model(whisper_model).encoder.to(self.device)
    
    def to(self, device):
        """Override to method to keep track of device"""
        self.device = device
        return super().to(device)


# Mean Pooling
class WhisperClassifierMeanPool(WhisperClassifierBase):
    def __init__(self, whisper_model: str, num_classes: int):
        super().__init__(whisper_model, num_classes)
        self.classifier = nn.Linear(
            self.encoder.positional_embedding.size(-1), 
            num_classes
        ).to(self.device)
            # No non-linearity introduced here ...
    
    def forward(self, mel_input):
        encoder_output = self.encoder(mel_input)
        pooled_features = encoder_output.mean(dim=1)
        logits = self.classifier(pooled_features)
        return logits

# Mean Pooling + Feedforward
class WhisperClassifierMeanPoolFF(WhisperClassifierBase):
    def __init__(self, whisper_model: str, num_classes: int):
        super().__init__(whisper_model, num_classes)
        self.classifier = nn.Sequential(
            nn.Linear(self.encoder.positional_embedding.size(-1), num_classes * 3),
            nn.ReLU(),
            nn.Linear(num_classes * 3, num_classes)
        ).to(self.device)
    
    def forward(self, mel_input):
        encoder_output = self.encoder(mel_input)
        pooled_features = encoder_output.mean(dim=1)
        logits = self.classifier(pooled_features)
        return logits

# Max Pooling plus Feedforward
class WhisperClassifierMaxPoolFF(WhisperClassifierBase):
    def __init__(self, whisper_model: str, num_classes: int):
        super().__init__(whisper_model, num_classes)
        self.classifier = nn.Sequential(
            nn.Linear(self.encoder.positional_embedding.size(-1), num_classes * 3),
            nn.ReLU(),
            nn.Linear(num_classes * 3, num_classes)
        ).to(self.device)
    
    def forward(self, mel_input):
        encoder_output = self.encoder(mel_input)
        pooled_features = encoder_output.max(dim=1)
        logits = self.classifier(pooled_features)
        return logits