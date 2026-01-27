import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16


class LossFunction(nn.Module):
    def __init__(self, lambda_l1=1.0,  lambda_edge=0.0, lambda_percep=0.0):
        super(LossFunction, self).__init__()
        self.lambda_l1 = lambda_l1
        # self.lambda_ssim = lambda_ssim
        self.lambda_edge = lambda_edge
        self.lambda_percep = lambda_percep

        self.l1_loss = nn.L1Loss()
        sobel_x = torch.tensor([[1, 0, -1],
                                [2, 0, -2],
                                [1, 0, -1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = sobel_x.transpose(2, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

        self.vgg = vgg16(pretrained=True).features[:9]
        for param in self.vgg.parameters():
            param.requires_grad = False
        self.vgg.eval()  # 固定权重

    def edge_loss(self, pred, target):
        grad_pred_x = F.conv2d(pred, self.sobel_x, padding=1)
        grad_pred_y = F.conv2d(pred, self.sobel_y, padding=1)
        grad_target_x = F.conv2d(target, self.sobel_x, padding=1)
        grad_target_y = F.conv2d(target, self.sobel_y, padding=1)

        grad_pred = torch.sqrt(grad_pred_x ** 2 + grad_pred_y ** 2 + 1e-6)
        grad_target = torch.sqrt(grad_target_x ** 2 + grad_target_y ** 2 + 1e-6)

        return F.l1_loss(grad_pred, grad_target)

    def perceptual_loss(self, pred, target):
        pred_rgb = pred.repeat(1, 3, 1, 1)
        target_rgb = target.repeat(1, 3, 1, 1)
        with torch.no_grad():
            pred_feat = self.vgg(pred_rgb)
            target_feat = self.vgg(target_rgb)
        return F.l1_loss(pred_feat, target_feat)

    def forward(self, pred, target):
        loss = 0.0

        if self.lambda_l1 > 0:
            l1 = self.l1_loss(pred, target)
            loss += self.lambda_l1 * l1

        if self.lambda_edge > 0:
            edge = self.edge_loss(pred, target)
            loss += self.lambda_edge * edge

        if self.lambda_percep > 0:
            percep = self.perceptual_loss(pred, target)
            loss += self.lambda_percep * percep

        return loss
