import torch
import torch.nn as nn
import torch.nn.functional as F
import math

try:
    from torchvision.ops import deform_conv2d
except ImportError:
    deform_conv2d = torch.ops.torchvision.deform_conv2d

class FusionUnit(nn.Module):
    def __init__(self, in_channels, out_channels, groups=8):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.gn = nn.GroupNorm(groups, out_channels)
        self.relu = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.relu(self.gn(self.conv(x)))

def autopad(kernel_size, padding):
    return padding if padding is not None else kernel_size // 2

class DCNv2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=None, groups=1, act=True, dilation=1, deformable_groups=1):
        super(DCNv2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = (stride, stride)
        
        default_padding = autopad(kernel_size, None)
        self.padding = (default_padding * dilation, default_padding * dilation)
        
        self.dilation = (dilation, dilation)
        self.groups = groups
        self.deformable_groups = deformable_groups
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, *self.kernel_size)
        )
        self.bias = nn.Parameter(torch.empty(out_channels))
        
        out_channels_offset_mask = (self.deformable_groups * 3 *
                                  self.kernel_size[0] * self.kernel_size[1])
        
        self.conv_offset_mask = nn.Conv2d(
            self.in_channels,
            out_channels_offset_mask,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=True,
        )
        
        self.gn = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        self.reset_parameters()

    def forward(self, x):
        with torch.amp.autocast('cuda', enabled=False):
            x = x.float()
            
            offset_mask = self.conv_offset_mask(x)
            
            if offset_mask.shape[2:] != x.shape[2:]:
                 offset_mask = F.interpolate(offset_mask, size=x.shape[2:], mode='bilinear', align_corners=False)
                    
            o1, o2, mask = torch.chunk(offset_mask, 3, dim=1)
            offset = torch.cat((torch.tanh(o1) * 3.0, torch.tanh(o2) * 3.0), dim=1)
            mask = torch.sigmoid(mask)
            
            weight = self.weight.float()
            bias = self.bias.float()
            
            output = torch.ops.torchvision.deform_conv2d(
                x, weight, offset, mask, bias,
                self.stride[0], self.stride[1],
                self.padding[0], self.padding[1],
                self.dilation[0], self.dilation[1],
                self.groups, self.deformable_groups, True
            )

        output = self.gn(output)
        output = self.act(output)
        return output

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        std = 1. / math.sqrt(n)
        self.weight.data.uniform_(-std, std)
        self.bias.data.zero_()
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()

class ADCB(nn.Module):
    def __init__(self, in_channels=6, base_channels=32, groups=8):
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels 
        self.output_channels = 32 
        
        self.conv1x1 = nn.Conv2d(in_channels, base_channels, kernel_size=1)
        self.fusion1 = FusionUnit(base_channels + in_channels, base_channels, groups)

        self.adaptive_conv = DCNv2(
            in_channels=base_channels,
            out_channels=base_channels,
            kernel_size=3,
            stride=1,
            padding=None,
            dilation=1,
            groups=groups,
            deformable_groups=1,
            act=True 
        )
        
        self.modulation_head = nn.Sequential(
            nn.Conv2d(base_channels, base_channels // 4, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(base_channels // 4, self.output_channels * 2, 1) 
        )

        if base_channels != self.output_channels:
            self.final_proj = nn.Conv2d(base_channels, self.output_channels, 1)
        else:
            self.final_proj = nn.Identity()

    def forward(self, inputA):
        x_init_feat = self.conv1x1(inputA)
        
        x_fused = self.fusion1(torch.cat([x_init_feat, inputA], dim=1))
        x_adaptive = self.adaptive_conv(x_fused)
        mod_params = self.modulation_head(x_fused)
        gamma, beta = mod_params.chunk(2, dim=1)
        x_adaptive_out = self.final_proj(x_adaptive) * (1 + gamma) + beta
        
        output = x_init_feat + x_adaptive_out

        return output
