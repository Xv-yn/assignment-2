"""Shared, randomly initialised CNN; no pretrained weights or external data."""

from torch import nn


class SmallCNN(nn.Module):
    def __init__(self, input_channels: int, channels: list[int]):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, channels[0], 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(channels[0], channels[1], 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Linear(channels[1] * 16, 3)

    def forward(self, images):
        return self.classifier(self.features(images).flatten(1))
