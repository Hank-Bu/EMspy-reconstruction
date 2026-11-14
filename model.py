"""Residual U-Net for multi-frequency EM leakage image reconstruction."""

import torch
from torch import nn


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        batch, channels, _, _ = x.size()
        weights = self.avg_pool(x).view(batch, channels)
        weights = self.fc(weights).view(batch, channels, 1, 1)
        return x * weights.expand_as(x)


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, act=True, reduction=16):
        super().__init__()
        self.act = act
        self.conv1 = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        if self.act:
            self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.se = SELayer(in_channels, reduction)

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        if self.act:
            out = self.relu(out)
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += x
        if self.act:
            out = self.relu(out)
        return out


class DownSampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, in_channels // 4, kernel_size=3,
            stride=2, padding=1, bias=False,
        )
        self.bn1 = nn.BatchNorm2d(in_channels // 4)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            in_channels // 4, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.skipconv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3,
            stride=2, padding=1, bias=False,
        )
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.bn3(self.skipconv(x))
        return out


class ReconstructionUNet(nn.Module):
    """EMspy reconstruction network with residual blocks and channel attention.

    The default input contains ten preprocessed in-phase and quadrature
    channels from five information-bearing frequencies, shaped (N, 10, H, W).
    The output is a grayscale image tensor shaped (N, 1, H, W).
    H and W must be positive multiples of 16. The default feature width is 64.
    """

    def __init__(self, in_channels=10, out_channels=1, feature_num=64):
        super().__init__()
        self.transform = nn.Conv2d(
            in_channels, feature_num, kernel_size=5, padding=2, bias=False
        )

        self.pool1 = DownSampleBlock(feature_num, feature_num)
        self.encoder1 = EncoderBlock(feature_num)
        self.pool2 = DownSampleBlock(feature_num, feature_num * 2)
        self.encoder2 = EncoderBlock(feature_num * 2)
        self.pool3 = DownSampleBlock(feature_num * 2, feature_num * 4)
        self.encoder3 = nn.Sequential(
            EncoderBlock(feature_num * 4),
            EncoderBlock(feature_num * 4),
        )
        self.pool4 = DownSampleBlock(feature_num * 4, feature_num * 8)
        self.encoder4 = EncoderBlock(feature_num * 8)

        self.bottleneck = nn.Sequential(
            EncoderBlock(feature_num * 8),
            EncoderBlock(feature_num * 8),
        )

        self.upconv4 = nn.ConvTranspose2d(
            feature_num * 8, feature_num, kernel_size=2, stride=2, bias=False
        )
        self.decoder4 = EncoderBlock(feature_num * 8)
        self.upconv3 = nn.ConvTranspose2d(
            feature_num * 2, feature_num, kernel_size=2, stride=2, bias=False
        )
        self.decoder3 = EncoderBlock(feature_num * 2)
        self.upconv2 = nn.ConvTranspose2d(
            feature_num * 2, feature_num // 2, kernel_size=2, stride=2, bias=False
        )
        self.decoder2 = EncoderBlock(feature_num * 2)
        self.upconv1 = nn.ConvTranspose2d(
            feature_num, feature_num // 4, kernel_size=2, stride=2, bias=False
        )
        self.decoder1 = EncoderBlock(feature_num)

        self.skipconv4 = nn.Conv2d(
            feature_num * 4, feature_num, kernel_size=3, padding=1, bias=False
        )
        self.skipconv3 = nn.Conv2d(
            feature_num * 2, feature_num, kernel_size=3, padding=1, bias=False
        )
        self.skipconv2 = nn.Conv2d(
            feature_num, feature_num // 2, kernel_size=3, padding=1, bias=False
        )
        self.skipconv1 = nn.Conv2d(
            feature_num, feature_num // 4, kernel_size=3, padding=1, bias=False
        )
        self.final_conv = nn.Conv2d(
            feature_num // 2, out_channels, kernel_size=1, bias=False
        )

    def forward(self, x):
        x = self.transform(x)
        x1 = self.encoder1(self.pool1(x))
        x2 = self.encoder2(self.pool2(x1))
        x3 = self.encoder3(self.pool3(x2))
        x4 = self.encoder4(self.pool4(x3))
        x5 = self.bottleneck(x4)

        x4 = torch.cat(
            [self.upconv4(self.decoder4(x5)), self.skipconv4(x3)], dim=1
        )
        x3 = torch.cat(
            [self.upconv3(self.decoder3(x4)), self.skipconv3(x2)], dim=1
        )
        x2 = torch.cat(
            [self.upconv2(self.decoder2(x3)), self.skipconv2(x1)], dim=1
        )
        x1 = torch.cat(
            [self.upconv1(self.decoder1(x2)), self.skipconv1(x)], dim=1
        )
        return torch.sigmoid(self.final_conv(x1))
