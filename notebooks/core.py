import whisper
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import get_device


class ClassificationBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(ClassificationBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.avg_pool(x.permute(0, 2, 1)).squeeze(-1)  # Average pooling
        x = self.fc1(x) # First linear layer
        x = F.relu(x)  # ReLU
        x = self.fc2(x)  # Second linear layer
        return x


class WhisperClassifier(nn.Module): # test
    def __init__(self, whisper_model: str, num_classes: int, projection_dim_factor: int, feedforward: bool, pooling: str, withcoords: bool):
        super().__init__()
        # Set up the encoder
        self.device = get_device()
        self.encoder = whisper.load_model(whisper_model).encoder.to(self.device)
        encoding_dim = self.encoder.positional_embedding.size(-1)

        # Set up the classifier
        self.projection_dim = encoding_dim * projection_dim_factor
        self.feedforward = feedforward
        self.pooling = pooling
        self.withcoords = withcoords

        proj_dim = encoding_dim * projection_dim_factor

        if self.withcoords:
            proj_dim += 2 # for longitude and latitude
        if feedforward:
            self.classifier = ClassificationBlock(encoding_dim, 
                                                  proj_dim, 
                                                  num_classes
                                                ).to(self.device)
        else:
            self.classifier = nn.Linear(self.encoder.positional_embedding.size(-1), num_classes).to(self.device)
    
    def to(self, device):
        """Override to method to keep track of device"""
        self.device = device
        return super().to(device)
    
    def forward(self, mel_input, coords):
        encoder_output = self.encoder(mel_input)
        if self.pooling == "mean":
            pooled_features = encoder_output.mean(dim=1)
        elif self.pooling == "max":
            pooled_features, _ = encoder_output.max(dim=1)

        print(pooled_features.shape, coords.shape)
        if self.withcoords:
            pooled_features = torch.cat((pooled_features, coords), dim=0)
        logits = self.classifier(pooled_features)

        return logits