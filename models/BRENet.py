import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import ResNet
from .modules import BRE


class BRENet(nn.Module):
    """Boundary-Region Enhancement Network (BRENet)."""

    def __init__(self, backbone='resnet50', pretrained=True):
        super(BRENet, self).__init__()
        self.backbone = ResNet(backbone=backbone, pretrained=pretrained)
        self.bre = BRE(in_channels=2048, mid_channels=256)
        self.head = nn.Conv2d(256, 1, kernel_size=1)

    def forward(self, x):
        h, w = x.shape[2], x.shape[3]
        feats = self.backbone(x)
        out = self.bre(feats)
        out = self.head(out)
        out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=False)
        return out
