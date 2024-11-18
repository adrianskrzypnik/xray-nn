import torch
import torch.nn as nn
import torch.nn.functional as F

class XRayClassifier(nn.Module):
    def __init__(self, num_classes):
        super(XRayClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # Rozmiar wejścia dla warstwy w pełni połączonej
        input_features = 64 * 56 * 56  # Liczba cech, która wejdzie do warstwy fc1
        self.fc1 = nn.Linear(input_features, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))


        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
