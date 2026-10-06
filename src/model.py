"""Small CNN for 30x30 packet matrices (a lighter cousin of the 4-conv network in the DeepNFV paper)."""
import torch.nn as nn


class TrafficCNN(nn.Module):
    def __init__(self, n_classes: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 30 -> 15
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 15 -> 7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(32 * 7 * 7, 64), nn.ReLU(), nn.Linear(64, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))
