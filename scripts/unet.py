import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels  # 如果没有给出中间通道数，设置为输出通道数
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),  # 卷积层1
            nn.BatchNorm2d(mid_channels),  # 批归一化
            nn.ReLU(inplace=True),  # ReLU激活
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),  # 卷积层2
            nn.BatchNorm2d(out_channels),  # 批归一化
            nn.ReLU(inplace=True)  # ReLU激活
        )

    def forward(self, x):
        return self.double_conv(x)  # 前向传播


class Down(nn.Module):
    """下采样：使用最大池化然后进行双重卷积"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),  # 最大池化层，步幅为2
            DoubleConv(in_channels, out_channels)  # 双重卷积层
        )

    def forward(self, x):
        return self.maxpool_conv(x)  # 前向传播


class Up(nn.Module):
    """上采样：使用上采样或转置卷积后再进行双重卷积"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            # 如果使用双线性插值，使用正常卷积减少通道数
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)  # 双重卷积
        else:
            # 否则使用转置卷积进行上采样
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)  # 双重卷积

    def forward(self, x1, x2):
        # 上采样
        x1 = self.up(x1)

        # 填充，使得x1与x2的尺寸匹配
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        # 沿通道维度进行拼接
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)  # 前向传播


class OutConv(nn.Module):
    """输出卷积层，输出特定通道数的结果"""

    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)  # 1x1卷积层

    def forward(self, x):
        return self.conv(x)  # 前向传播


class UNet(nn.Module):
    """U-Net模型"""

    def __init__(self, n_channels, n_classes, base_channels=64):
        super(UNet, self).__init__()
        self.n_channels = n_channels  # 输入通道数
        self.n_classes = n_classes  # 输出类别数
        self.base_channels = base_channels  # 基础通道数

        # 定义U-Net各层
        self.inc = DoubleConv(n_channels, base_channels)  # 输入层，双重卷积
        self.down1 = Down(base_channels, base_channels * 2)  # 下采样层1
        self.down2 = Down(base_channels * 2, base_channels * 4)  # 下采样层2
        self.down3 = Down(base_channels * 4, base_channels * 8)  # 下采样层3
        self.down4 = Down(base_channels * 8, base_channels * 8)  # 下采样层4
        self.up1 = Up(base_channels * 16, base_channels * 4)  # 上采样层1
        self.up2 = Up(base_channels * 8, base_channels * 2)  # 上采样层2
        self.up3 = Up(base_channels * 4, base_channels)  # 上采样层3
        self.up4 = Up(base_channels * 2, base_channels)  # 上采样层4
        self.outc = OutConv(base_channels, n_classes)  # 输出卷积层

    def forward(self, x):
        x1 = self.inc(x)  # 输入数据经过输入层
        x2 = self.down1(x1)  # 下采样层1
        x3 = self.down2(x2)  # 下采样层2
        x4 = self.down3(x3)  # 下采样层3
        x5 = self.down4(x4)  # 下采样层4
        x = self.up1(x5, x4)  # 上采样层1
        x = self.up2(x, x3)  # 上采样层2
        x = self.up3(x, x2)  # 上采样层3
        x = self.up4(x, x1)  # 上采样层4
        logits = self.outc(x)  # 输出层
        return logits  # 返回原始输出