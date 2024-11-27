import torch
import torch.nn as nn
import torch.nn.functional as F


class XRayClassifier(nn.Module):
    def __init__(self, num_classes, num_layers=2, hidden_units=256):
        super(XRayClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        input_features = 64 * 56 * 56

        # Dynamiczne tworzenie w pełni połączonych warstw
        layers = []
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(input_features, hidden_units))
            layers.append(nn.ReLU())
            input_features = hidden_units
        layers.append(nn.Linear(input_features, num_classes))

        self.fc_layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

