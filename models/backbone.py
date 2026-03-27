import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights, ResNet101_Weights


class ResNet(nn.Module):
    """ResNet backbone for feature extraction."""

    def __init__(self, backbone='resnet50', pretrained=True):
        super(ResNet, self).__init__()
        if backbone == 'resnet50':
            weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            net = models.resnet50(weights=weights)
        elif backbone == 'resnet101':
            weights = ResNet101_Weights.IMAGENET1K_V1 if pretrained else None
            net = models.resnet101(weights=weights)
        else:
            raise ValueError(f'Unsupported backbone: {backbone}')

        self.layer0 = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4

    def forward(self, x):
        x = self.layer0(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x4
