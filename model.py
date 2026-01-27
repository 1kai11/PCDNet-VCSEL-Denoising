import torch
from torch import nn
from plug_play.Attention import ECAAttention
from plug_play.Polar_DCN import PolarDeformConv2d


class PCDNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.initial = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(1, 64, kernel_size=7),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.down1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.res_blocks = nn.ModuleList([
            PCRB(256),
            PCRB(256),
            PCRB(256),
            PCRB(256),
            PCRB(256),
            PCRB(256)
        ])

        self.up1 = nn.Sequential(
            nn.Conv2d(256, 128 * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(128, 64 * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.output = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, 1, kernel_size=7),
            nn.Identity()
        )

    def forward(self, x):
        B, C, H, W = x.shape
        reduced_H, reduced_W = H // 4, W // 4
        yy, xx = torch.meshgrid(torch.arange(reduced_H), torch.arange(reduced_W), indexing='ij')
        radius_map = torch.sqrt((xx - reduced_W // 2) ** 2 + (yy - reduced_H // 2) ** 2).to(x.device).unsqueeze(
            0).expand(B, 1, reduced_H, reduced_W)
        x = self.initial(x)
        x = self.down1(x)
        x = self.down2(x)
        for block in self.res_blocks:
            x = block(x, radius_map)

        x = self.up1(x)
        x = self.up2(x)
        x = self.output(x)
        return x
class PCRB(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.conv1 = PolarDeformConv2d(channels, channels)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = PolarDeformConv2d(channels, channels)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.eca = ECAAttention(3)

    def forward(self, x, radius_map):
        residual = x
        x = self.conv1(x, radius_map)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x, radius_map)
        x = self.bn2(x)
        x = self.eca(x)
        return x + residual

