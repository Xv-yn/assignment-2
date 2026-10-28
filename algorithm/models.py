"""Shared, randomly initialized CNN; no external data or pretrained weights."""

import torch
import torch.nn as nn


class DeterministicAdaptivePool(nn.Module):
    """The same 3x3 averaging regions as adaptive pooling, with CUDA-safe backward.

    CUDA's adaptive_avg_pool2d backward rejects strict deterministic mode.
    Explicit slice means avoid its atomic accumulation kernel.
    """

    def forward(self, x):
        height, width = x.shape[-2:]
        return torch.cat([
            torch.cat([
                x[:, :, (i * height) // 3:((i + 1) * height + 2) // 3,
                  (j * width) // 3:((j + 1) * width + 2) // 3].mean((2, 3), keepdim=True)
                for j in range(3)
            ], dim=3)
            for i in range(3)
        ], dim=2)


class SmallCNN(nn.Module):
    def __init__(self, in_channels=1, channels=(32, 64, 128), num_classes=3):
        super().__init__()
        layers = []
        for width in channels:
            layers.extend([nn.Conv2d(in_channels, width, 3, padding=1),
                           nn.BatchNorm2d(width), nn.ReLU(inplace=True),
                           nn.Conv2d(width, width, 3, padding=1),
                           nn.BatchNorm2d(width), nn.ReLU(inplace=True),
                           nn.MaxPool2d(2)])
            in_channels = width
        self.features = nn.Sequential(*layers, DeterministicAdaptivePool())
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(in_channels * 9, 128),
                                        nn.ReLU(), nn.Dropout(0.25), nn.Linear(128, num_classes))

    def forward(self, images):
        return self.classifier(self.features(images))


def build_model(name, architecture):
    if architecture.get("name") != "small_cnn":
        raise ValueError("Only architecture.name=small_cnn is implemented")
    return SmallCNN(3 if name == "CIFAR" else 1,
                    architecture["channels"], architecture["num_classes"])
