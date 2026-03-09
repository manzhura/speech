import torch
import torch.nn.functional as TF
from torch import nn
from thop import profile


class MelNet(nn.Module):
    def __init__(self, n_mels=80, n_classes=2, groups=1, ceil_mode=False):
        super().__init__()

        self.conv1 = nn.Conv1d(n_mels, 64, kernel_size=5, stride=1, padding=2, groups=groups)
        self.bn1   = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(2, ceil_mode=ceil_mode)

        self.conv2 = nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1, groups=groups)
        self.bn2   = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2, ceil_mode=ceil_mode)

        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1, groups=groups)
        self.bn3   = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(2, ceil_mode=ceil_mode)

        self.head_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, n_classes)

    def forward(self, x):
        if x.dim() == 4:
            x = x.squeeze(1)

        x = self.pool1(TF.relu(self.bn1(self.conv1(x))))
        x = self.pool2(TF.relu(self.bn2(self.conv2(x))))
        x = self.pool3(TF.relu(self.bn3(self.conv3(x))))

        x = self.head_pool(x).squeeze(-1)
        x = self.fc(x)
        return x

    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def num_flops(self, input_shape: tuple = (1, 80, 101)):
        dummy = torch.zeros(1, *input_shape)
        flops, _ = profile(self, inputs=(dummy,), verbose=False)
        return int(flops)