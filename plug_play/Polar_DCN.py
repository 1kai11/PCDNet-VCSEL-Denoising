import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class PolarDeformConv2d(nn.Module):
    """极坐标约束的可变形卷积（偏移方向沿径向/切向）"""

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.offset_conv = nn.Conv2d(in_channels + 1, 2 * kernel_size * kernel_size, kernel_size=3, padding=1)
        self.deform_conv = torchvision.ops.DeformConv2d(in_channels, out_channels, kernel_size, padding=self.padding)

    def forward(self, x, radius_map):
        B, C, H, W = x.shape
        kernel_size = self.kernel_size

        offset_input = torch.cat([x, radius_map], dim=1)
        offsets_polar = self.offset_conv(offset_input)
        offsets_polar = torch.tanh(offsets_polar)

        yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        theta = torch.atan2((yy - H // 2) / H, (xx - W // 2) / W).to(x.device)
        offsets_polar = offsets_polar.contiguous().view(B, 2, kernel_size * kernel_size, H, W)

        radial_offset = offsets_polar[:, 0]
        tangential_offset = offsets_polar[:, 1]

        offset_x = (radial_offset * torch.cos(theta) - tangential_offset * torch.sin(theta)).clamp(-2, 2)
        offset_y = (radial_offset * torch.sin(theta) + tangential_offset * torch.cos(theta)).clamp(-2, 2)

        offsets = torch.cat([offset_x, offset_y], dim=1)

        return self.deform_conv(x, offsets)




