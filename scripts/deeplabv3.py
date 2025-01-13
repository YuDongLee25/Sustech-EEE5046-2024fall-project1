import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

class DeepLabV3(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(DeepLabV3, self).__init__()

        # 基础网络
        self.backbone = nn.Sequential(
            nn.Conv2d(n_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True)
        )
        
        # 空洞卷积（Atrous Convolution）
        self.conv1 = nn.Conv2d(64, 128, kernel_size=3, padding=2, dilation=2)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=3, padding=4, dilation=4)
        self.conv3 = nn.Conv2d(256, 512, kernel_size=3, padding=8, dilation=8)
        
        # 输出层
        self.classifier = nn.Conv2d(512, n_classes, kernel_size=1)

    def forward(self, x):
        x = self.backbone(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.classifier(x)
        return x